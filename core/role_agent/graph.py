"""基于 LangGraph 的小说角色分配智能体。

流程（Plan-and-Execute + ReAct 校对）：
    segment      规则断句，得到带原文偏移量的片段
    plan_cast    Planner：通读全文，建立角色表（规范名/别名/性别/年龄/特征）与叙述视角
    label_chunk  Executor：按批次给片段编号打标签（类别 + 说话人），循环直到所有批次完成
    validate     确定性校验：漏标、未识别说话人、引号内被标旁白、低置信度、别名归一
    review_agent ReAct 校对员：带着问题清单调用工具查上下文、改标签、补/并角色
    review_tools 执行校对员的工具调用
    cast_voices  把角色映射到本地音色配置（规则匹配优先，其余由 LLM 按性别/年龄/特征选角）

各节点的提示词与阈值来自 skills/novel-role-assign/SKILL.md（见 skill.py）。
"""

import json
import operator
from typing import Annotated, Any, Callable, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from ..role_assigner import _normalize_name
from .llm import LLMSettings, accumulate_usage, build_chat_model, invoke_structured
from .schemas import (
    REVIEW_TOOLS, CastPlan, ChunkLabels, VoiceCasting,
)
from .segmenter import chunk_segments, split_segments
from .skill import Skill, load_skill

NARRATOR = '旁白'
SYSTEM = '系统'
UNKNOWN = '未识别角色'
SPECIAL_SPEAKERS = {NARRATOR, SYSTEM, UNKNOWN}
NARRATOR_ALIASES = {'旁白', '叙述', '叙述者', 'narrator', 'narration', 'voiceover'}
VALID_CATEGORIES = {'旁白', '对话', '心理', '系统'}

# 代词和泛称不能作为别名，否则会把 “我们” 之类模糊匹配到具体角色
PRONOUNS = {
    '我', '你', '他', '她', '它', '吾', '俺', '咱', '朕', '本王', '在下', '老子', '人家',
    '我们', '你们', '他们', '她们', '咱们', '我等', '尔等', '列位', '诸位', '大家', '众人',
}



class RoleAgentState(TypedDict, total=False):
    document_text: str
    segments: List[dict]
    chunks: List[List[int]]
    chunk_cursor: int
    voice_configs: List[dict]
    fallback_speaker: str
    roster: List[dict]
    plan: dict
    labels: Dict[int, dict]
    issues: List[dict]
    review_round: int
    review_steps: int
    review_done: bool
    messages: List[Any]
    voice_mapping: Dict[str, dict]
    # 首次校验时的快照，用于评估校对员的净收益与校验规则的命中率
    pre_review_labels: Dict[int, dict]
    first_issues: List[dict]
    trace: Annotated[List[dict], operator.add]



def _dump(data) -> str:
    return json.dumps(data, ensure_ascii=False)


def resolve_speaker(raw: str, roster: List[dict]) -> str:
    """把模型给出的称呼归一到角色表规范名；无法归一时返回空串。"""
    speaker = (raw or '').strip()
    if not speaker:
        return ''
    if speaker in SPECIAL_SPEAKERS:
        return speaker
    normalized = _normalize_name(speaker)
    if normalized in NARRATOR_ALIASES:
        return NARRATOR

    for character in roster:
        if speaker == character['name'] or speaker in character.get('aliases', []):
            return character['name']
    for character in roster:
        names = [character['name']] + list(character.get('aliases', []))
        if normalized in {_normalize_name(name) for name in names}:
            return character['name']

    if len(normalized) < 2:
        return ''
    fuzzy = [
        character['name'] for character in roster
        if any(
            len(_normalize_name(name)) >= 2 and (normalized in _normalize_name(name) or _normalize_name(name) in normalized)
            for name in [character['name']] + list(character.get('aliases', []))
        )
    ]
    return fuzzy[0] if len(set(fuzzy)) == 1 else ''


def _default_label(segment: dict) -> dict:
    if segment.get('in_quote'):
        return {'category': '对话', 'speaker': UNKNOWN, 'confidence': 0.0, 'reason': '模型漏标，按引号推断为对话'}
    return {'category': '旁白', 'speaker': NARRATOR, 'confidence': 0.0, 'reason': '模型漏标，按默认旁白处理'}


def _mark_reviewed(labels: Dict[int, dict], issues: List[dict], limit: int) -> Dict[int, dict]:
    """校对员主动结束时，视为已确认本轮交给它的片段（低置信度不再重复报告）。"""
    labels = {seg_id: dict(label) for seg_id, label in labels.items()}
    for issue in issues[:limit]:
        label = labels.get(issue['id'])
        if label is not None:
            label['reviewed'] = True
    return labels


class RoleAgent:
    """持有 LLM 与运行选项，节点方法共享同一份用量统计与进度回调。"""

    def __init__(
        self,
        settings: LLMSettings,
        progress: Optional[Callable[[str], None]] = None,
        max_review_rounds: int = 1,
        max_review_steps: int = 10,
        auto_cast: bool = True,
        llm=None,
        skill: Optional[Skill] = None,
    ):
        self.settings = settings
        self.skill = skill or load_skill()
        self.p = self.skill.params
        self.llm = llm or build_chat_model(settings)
        self.progress = progress or (lambda message: None)
        self.max_review_rounds = max_review_rounds
        self.max_review_steps = max_review_steps
        self.auto_cast = auto_cast
        # 调用次数 / token / 美元花费，by_node 按节点细分
        self.usage: Dict[str, Any] = {}
        self._tool_llm = None
        self._tools_supported = True

    # ---------- 节点 ----------

    def segment(self, state: RoleAgentState) -> dict:
        segments = split_segments(state['document_text'], self.p['speech_lead_verbs'])
        if not segments:
            raise ValueError("没有可分配的文本内容")
        chunks = chunk_segments(segments, self.p['chunk_max_items'], self.p['chunk_max_chars'])
        self.progress(f"断句完成：{len(segments)} 个片段")
        return {
            'segments': segments,
            'chunks': chunks,
            'chunk_cursor': 0,
            'labels': {},
            'review_round': 0,
            'trace': [{'node': 'segment', 'message': f"规则断句得到 {len(segments)} 个片段，分为 {len(chunks)} 批"}],
        }

    def plan_cast(self, state: RoleAgentState) -> dict:
        self.progress("规划角色表...")
        text = state['document_text']
        clipped = text[:self.p['max_plan_chars']]
        header = f"（全文共 {len(text)} 字，以下为前 {len(clipped)} 字，后文的新角色由标注员补充）\n" if len(text) > len(clipped) else ''
        messages = [SystemMessage(content=self.skill.prompts['planner']), HumanMessage(content=header + clipped)]
        plan = invoke_structured(self.llm, CastPlan, messages, self.usage, 'plan_cast')
        if not plan.characters and any(segment['in_quote'] for segment in state['segments']):
            # 有对白却没给出角色，多为模型把角色写进了 notes；追问一次
            messages.append(HumanMessage(
                content=f"你上次的 characters 为空，但正文中有对白。你的备注是：{plan.notes}\n请把所有说话角色填入 characters 重新输出。"
            ))
            plan = invoke_structured(self.llm, CastPlan, messages, self.usage, 'plan_cast')

        roster = []
        seen = set()
        for character in plan.characters:
            name = character.name.strip()
            if not name or name in SPECIAL_SPEAKERS or name in seen:
                continue
            seen.add(name)
            roster.append({
                'name': name,
                'aliases': [
                    alias.strip() for alias in character.aliases
                    if alias.strip() and alias.strip() != name and alias.strip() not in PRONOUNS
                ],
                'gender': character.gender,
                'age': character.age,
                'traits': character.traits,
                'discovered': False,
            })

        names = '、'.join(item['name'] for item in roster) or '无'
        return {
            'roster': roster,
            'plan': {'pov': plan.pov, 'narrator_character': plan.narrator_character, 'notes': plan.notes},
            'trace': [{'node': 'plan_cast', 'message': f"{plan.pov}；角色表：{names}" + (f"；备注：{plan.notes}" if plan.notes else '')}],
        }

    def label_chunk(self, state: RoleAgentState) -> dict:
        cursor = state['chunk_cursor']
        chunk_ids = state['chunks'][cursor]
        segment_map = {segment['id']: segment for segment in state['segments']}
        labels = dict(state['labels'])
        roster = [dict(item) for item in state['roster']]
        self.progress(f"标注第 {cursor + 1}/{len(state['chunks'])} 批...")

        first_id = chunk_ids[0]
        previous = [
            {'id': seg_id, 'text': segment_map[seg_id]['text'], **{k: labels[seg_id][k] for k in ('category', 'speaker')}}
            for seg_id in range(max(1, first_id - self.p['context_window']), first_id)
            if seg_id in labels
        ]
        payload = {
            'roster': [{'name': c['name'], 'aliases': c['aliases'], 'gender': c['gender']} for c in roster],
            **state['plan'],
            'previous_context': previous,
            'segments': [
                {'id': seg_id, 'text': segment_map[seg_id]['text'], 'quote': segment_map[seg_id]['in_quote']}
                for seg_id in chunk_ids
            ],
        }
        result = invoke_structured(self.llm, ChunkLabels, [
            SystemMessage(content=self.skill.prompts['executor']),
            HumanMessage(content=_dump(payload)),
        ], self.usage, 'label_chunk')

        returned = {item.id: item for item in result.labels if item.id in chunk_ids}
        discovered = []
        for seg_id in chunk_ids:
            item = returned.get(seg_id)
            if item is None:
                labels[seg_id] = {**_default_label(segment_map[seg_id]), 'auto': True}
                continue
            category = item.category if item.category in VALID_CATEGORIES else '旁白'
            speaker = item.speaker.strip()
            if category == '旁白':
                speaker = NARRATOR
            elif category == '系统':
                speaker = SYSTEM
            else:
                resolved = resolve_speaker(speaker, roster)
                if resolved:
                    speaker = resolved
                elif speaker and speaker not in SPECIAL_SPEAKERS:
                    # 角色表里没有的新角色，登记为执行阶段发现的角色
                    roster.append({'name': speaker, 'aliases': [], 'gender': '未知', 'age': '未知', 'traits': '', 'discovered': True})
                    discovered.append(speaker)
                else:
                    speaker = UNKNOWN
            labels[seg_id] = {
                'category': category,
                'speaker': speaker,
                'confidence': float(item.confidence),
                'reason': item.reason.strip(),
            }

        message = f"第 {cursor + 1} 批：{len(returned)}/{len(chunk_ids)} 个片段已标注"
        if discovered:
            message += f"；新发现角色：{'、'.join(discovered)}"
        return {
            'labels': labels,
            'roster': roster,
            'chunk_cursor': cursor + 1,
            'trace': [{'node': 'label_chunk', 'message': message}],
        }

    def validate(self, state: RoleAgentState) -> dict:
        roster = state['roster']
        labels = {seg_id: dict(label) for seg_id, label in state['labels'].items()}
        issues = []
        for segment in state['segments']:
            seg_id = segment['id']
            label = labels.get(seg_id)
            if label is None:
                label = labels[seg_id] = {**_default_label(segment), 'auto': True}

            # 别名归一（校对员可能新增了别名或合并了角色）
            if label['category'] in {'对话', '心理'}:
                resolved = resolve_speaker(label['speaker'], roster)
                if resolved and resolved != label['speaker']:
                    label['speaker'] = resolved

            problem = ''
            if label.get('auto') and not label.get('reviewed'):
                problem = '模型漏标'
            elif label['category'] in {'对话', '心理'} and label['speaker'] in {UNKNOWN, NARRATOR, ''}:
                problem = '说话人未识别'
            elif segment['in_quote'] and label['category'] == '旁白' and len(segment['text']) > 8:
                problem = '引号内文本被标为旁白'
            elif label['category'] in {'对话', '心理'} and not resolve_speaker(label['speaker'], roster):
                problem = '说话人不在角色表'
            elif label.get('confidence', 1) < self.p['low_confidence'] and not label.get('reviewed'):
                problem = f"置信度低({label.get('confidence', 0):.2f})"
            if problem:
                issues.append({'id': seg_id, 'problem': problem})

        update: dict = {'labels': labels, 'issues': issues}
        if 'first_issues' not in state:
            update['first_issues'] = issues
            update['pre_review_labels'] = {seg_id: dict(label) for seg_id, label in labels.items()}
        can_review = (
            issues
            and self._tools_supported
            and self.max_review_steps > 0
            and state.get('review_round', 0) < self.max_review_rounds
        )
        if can_review:
            update.update({
                'review_round': state.get('review_round', 0) + 1,
                'review_steps': 0,
                'review_done': False,
                'messages': [],
            })
        else:
            update['review_done'] = True

        summary = '、'.join(f"#{item['id']} {item['problem']}" for item in issues[:6])
        update['trace'] = [{
            'node': 'validate',
            'message': f"发现 {len(issues)} 个可疑片段" + (f"：{summary}" + ('…' if len(issues) > 6 else '') if issues else '，全部通过'),
        }]
        return update

    def review_agent(self, state: RoleAgentState) -> dict:
        if self._tool_llm is None:
            self._tool_llm = self.llm.bind_tools(REVIEW_TOOLS, max_tokens=self.p['review_max_tokens'])

        messages = list(state.get('messages') or [])
        if not messages:
            segment_map = {segment['id']: segment for segment in state['segments']}
            issue_lines = []
            for issue in state['issues'][:self.p['max_issues_in_prompt']]:
                label = state['labels'][issue['id']]
                issue_lines.append(
                    f"#{issue['id']} [{issue['problem']}] 当前={label['category']}/{label['speaker']} "
                    f"原文={segment_map[issue['id']]['text'][:60]}"
                )
            messages = [
                SystemMessage(content=self.skill.prompts['reviewer']),
                HumanMessage(content=(
                    f"叙述视角：{state['plan'].get('pov')}；叙述者：{state['plan'].get('narrator_character') or '无'}\n"
                    f"角色表：{'、'.join(c['name'] for c in state['roster']) or '无'}\n"
                    f"可疑片段（共 {len(state['issues'])} 个）：\n" + '\n'.join(issue_lines)
                )),
            ]

        self.progress(f"ReAct 校对第 {state.get('review_steps', 0) + 1} 步...")
        try:
            response = self._tool_llm.invoke(messages)
        except Exception as e:
            # 不支持工具调用的模型/服务：跳过校对，保留执行阶段结果
            self._tools_supported = False
            return {
                'review_done': True,
                'trace': [{'node': 'review_agent', 'message': f"当前模型不支持工具调用，跳过校对：{str(e)[:120]}"}],
            }
        accumulate_usage(self.usage, response, 'review_agent')

        calls = getattr(response, 'tool_calls', None) or []
        thought = (response.content if isinstance(response.content, str) else '').strip()
        if not calls and (getattr(response, 'response_metadata', None) or {}).get('finish_reason') == 'length':
            # 被输出上限截断：结束校对，但不能当作“已确认”
            return {
                'review_done': True,
                'trace': [{'node': 'review_agent', 'message': f"单步输出超过 {self.p['review_max_tokens']} token 上限，结束校对"}],
            }
        action = '、'.join(call['name'] for call in calls) or '结束'
        update = {
            'messages': messages + [response],
            'review_done': not calls,
            'trace': [{'node': 'review_agent', 'message': (f"思考：{thought[:80]} → " if thought else '') + f"行动：{action}"}],
        }
        if not calls:
            update['labels'] = _mark_reviewed(state['labels'], state['issues'], self.p['max_issues_in_prompt'])
        return update

    def review_tools(self, state: RoleAgentState) -> dict:
        messages = list(state['messages'])
        calls = messages[-1].tool_calls
        labels = {seg_id: dict(label) for seg_id, label in state['labels'].items()}
        roster = [dict(item, aliases=list(item.get('aliases', []))) for item in state['roster']]
        segment_map = {segment['id']: segment for segment in state['segments']}
        done = False
        observations = []

        for call in calls:
            name, args = call['name'], call.get('args') or {}
            try:
                if name == 'Finish':
                    done = True
                    result = '校对结束'
                    observations.append(f"Finish：{args.get('summary', '')[:80]}")
                else:
                    result = self._run_tool(name, args, labels, roster, segment_map)
                    observations.append(f"{name}{_dump(args)[:80]}")
            except Exception as e:
                result = f"工具执行失败：{e}"
            messages.append(ToolMessage(content=result, tool_call_id=call['id'], name=name))

        if done:
            labels = _mark_reviewed(labels, state['issues'], self.p['max_issues_in_prompt'])
        steps = state.get('review_steps', 0) + 1
        return {
            'messages': messages,
            'labels': labels,
            'roster': roster,
            'review_steps': steps,
            'review_done': done or steps >= self.max_review_steps,
            'trace': [{'node': 'review_tools', 'message': '；'.join(observations)}],
        }

    def _run_tool(self, name, args, labels, roster, segment_map) -> str:
        if name == 'ReadSegments':
            start = max(1, int(args.get('start_id', 1)))
            end = min(max(segment_map), int(args.get('end_id', start)), start + 39)
            lines = []
            for seg_id in range(start, end + 1):
                label = labels.get(seg_id, {})
                quote = '引号' if segment_map[seg_id]['in_quote'] else '    '
                lines.append(f"#{seg_id} {quote} {label.get('category', '?')}/{label.get('speaker', '?')} | {segment_map[seg_id]['text']}")
            return '\n'.join(lines) or '没有片段'

        if name == 'SearchText':
            keyword = str(args.get('keyword', '')).strip()
            if not keyword:
                return '关键词为空'
            hits = [f"#{seg_id} {segment['text'][:60]}" for seg_id, segment in segment_map.items() if keyword in segment['text']]
            return '\n'.join(hits[:20]) + (f"\n…共 {len(hits)} 处" if len(hits) > 20 else '') if hits else '未找到'

        if name == 'GetRoster':
            return _dump([{k: c[k] for k in ('name', 'aliases', 'gender', 'age', 'traits')} for c in roster])

        if name == 'Relabel':
            changed = []
            for update in args.get('updates') or []:
                seg_id = int(update.get('id', 0))
                if seg_id not in segment_map:
                    continue
                category = update.get('category') if update.get('category') in VALID_CATEGORIES else '旁白'
                speaker = str(update.get('speaker', '')).strip()
                if category == '旁白':
                    speaker = NARRATOR
                elif category == '系统':
                    speaker = SYSTEM
                else:
                    resolved = resolve_speaker(speaker, roster)
                    if not resolved:
                        return f"#{seg_id} 的说话人「{speaker}」不在角色表中，请先 AddCharacter 或改用规范名"
                    speaker = resolved
                labels[seg_id] = {
                    'category': category,
                    'speaker': speaker,
                    'confidence': 0.9,
                    'reason': ('校对：' + str(update.get('reason', '')).strip()).rstrip('：'),
                    'reviewed': True,
                }
                changed.append(f"#{seg_id}→{category}/{speaker}")
            return '已修改：' + '、'.join(changed) if changed else '没有可修改的片段'

        if name == 'AddCharacter':
            character_name = str(args.get('name', '')).strip()
            if not character_name or character_name in SPECIAL_SPEAKERS:
                return '角色名无效'
            if resolve_speaker(character_name, roster) == character_name:
                return f"{character_name} 已在角色表中"
            roster.append({
                'name': character_name,
                'aliases': [a for a in args.get('aliases') or [] if a and a != character_name and a not in PRONOUNS],
                'gender': args.get('gender', '未知'),
                'age': '未知',
                'traits': args.get('traits', ''),
                'discovered': True,
            })
            return f"已添加角色 {character_name}"

        if name == 'MergeCharacter':
            alias = str(args.get('alias', '')).strip()
            canonical = resolve_speaker(str(args.get('canonical', '')).strip(), roster)
            if not canonical:
                return '规范名不在角色表中'
            if alias == canonical:
                return '无需合并'
            target = next(c for c in roster if c['name'] == canonical)
            merged = next((c for c in roster if c['name'] == alias), None)
            if merged:
                roster.remove(merged)
                target['aliases'] = list(dict.fromkeys(target['aliases'] + [alias] + merged.get('aliases', [])))
            elif alias not in target['aliases']:
                target['aliases'].append(alias)
            count = 0
            for label in labels.values():
                if label['speaker'] == alias:
                    label['speaker'] = canonical
                    count += 1
            return f"已将 {alias} 合并到 {canonical}，更新 {count} 个片段"

        return f"未知工具 {name}"

    def cast_voices(self, state: RoleAgentState) -> dict:
        voice_configs = state['voice_configs']
        voice_names = [item['name'] for item in voice_configs]
        fallback = state['fallback_speaker']
        roster = state['roster']

        used = []
        for label in state['labels'].values():
            if label['speaker'] not in used:
                used.append(label['speaker'])

        mapping: Dict[str, dict] = {}
        pending = []
        normalized_voices = {_normalize_name(name): name for name in voice_names}
        for speaker in used:
            if speaker == NARRATOR or (speaker == SYSTEM and not self.p['cast_system_speaker']):
                mapping[speaker] = {'voice': fallback, 'reason': '旁白/系统使用默认角色', 'source': 'rule'}
                continue
            if speaker == UNKNOWN:
                continue
            default = {'name': speaker, 'aliases': []}
            if speaker == SYSTEM:
                default['traits'] = '系统提示、电子播报'
            character = next((c for c in roster if c['name'] == speaker), default)
            matched = ''
            for name in [speaker] + list(character.get('aliases', [])):
                matched = normalized_voices.get(_normalize_name(name), '')
                if matched:
                    break
            if matched:
                mapping[speaker] = {'voice': matched, 'reason': '音色配置名与角色名一致', 'source': 'rule'}
            else:
                pending.append(character)

        cast_note = ''
        if pending and self.auto_cast and voice_names:
            self.progress("为角色挑选音色...")
            payload = {
                'available_voice_configs': [
                    {
                        'name': item['name'],
                        'mode': item.get('mode', ''),
                        'instruct_text': (item.get('instruct_text') or '')[:60],
                        'prompt_text': (item.get('prompt_text') or '')[:40],
                    }
                    for item in voice_configs
                ],
                'narrator_voice': fallback,
                'already_mapped': {k: v['voice'] for k, v in mapping.items()},
                'characters': [
                    {k: c.get(k, '') for k in ('name', 'gender', 'age', 'traits')} for c in pending
                ],
            }
            try:
                casting = invoke_structured(self.llm, VoiceCasting, [
                    SystemMessage(content=self.skill.prompts['caster']),
                    HumanMessage(content=_dump(payload)),
                ], self.usage, 'cast_voices')
                pending_names = {c['name'] for c in pending}
                for item in casting.mappings:
                    character = resolve_speaker(item.character, roster) or item.character
                    if character in pending_names and item.voice_config in voice_names:
                        mapping[character] = {'voice': item.voice_config, 'reason': item.reason, 'source': 'llm'}
            except Exception as e:
                cast_note = f"（自动选角失败：{str(e)[:80]}）"

        mapped = '、'.join(f"{k}→{v['voice']}" for k, v in mapping.items()) or '无'
        unmapped = [c['name'] for c in pending if c['name'] not in mapping]
        message = f"音色映射：{mapped}" + (f"；待手动指定：{'、'.join(unmapped)}" if unmapped else '') + cast_note
        return {'voice_mapping': mapping, 'trace': [{'node': 'cast_voices', 'message': message}]}

    # ---------- 路由 ----------

    @staticmethod
    def route_after_label(state: RoleAgentState) -> str:
        return 'label_chunk' if state['chunk_cursor'] < len(state['chunks']) else 'validate'

    @staticmethod
    def route_after_validate(state: RoleAgentState) -> str:
        return 'cast_voices' if state.get('review_done') else 'review_agent'

    @staticmethod
    def route_after_review(state: RoleAgentState) -> str:
        if state.get('review_done'):
            return 'validate'
        return 'review_tools'

    @staticmethod
    def route_after_tools(state: RoleAgentState) -> str:
        return 'validate' if state.get('review_done') else 'review_agent'

    def build_graph(self):
        graph = StateGraph(RoleAgentState)
        graph.add_node('segment', self.segment)
        graph.add_node('plan_cast', self.plan_cast)
        graph.add_node('label_chunk', self.label_chunk)
        graph.add_node('validate', self.validate)
        graph.add_node('review_agent', self.review_agent)
        graph.add_node('review_tools', self.review_tools)
        graph.add_node('cast_voices', self.cast_voices)

        graph.add_edge(START, 'segment')
        graph.add_edge('segment', 'plan_cast')
        graph.add_edge('plan_cast', 'label_chunk')
        graph.add_conditional_edges('label_chunk', self.route_after_label, ['label_chunk', 'validate'])
        graph.add_conditional_edges('validate', self.route_after_validate, ['review_agent', 'cast_voices'])
        graph.add_conditional_edges('review_agent', self.route_after_review, ['review_tools', 'validate'])
        graph.add_conditional_edges('review_tools', self.route_after_tools, ['review_agent', 'validate'])
        graph.add_edge('cast_voices', END)
        return graph.compile()

    def run(self, document_text: str, voice_configs: List[dict], fallback_speaker: str) -> RoleAgentState:
        app = self.build_graph()
        return app.invoke(
            {
                'document_text': document_text,
                'voice_configs': voice_configs,
                'fallback_speaker': fallback_speaker,
                'trace': [],
            },
            config={'recursion_limit': 300},
        )
