"""基于 LangGraph 的小说角色分配智能体。

流程（Plan-and-Execute + ReAct 校对 + 重规划 + 人工确认）：
    segment       规则断句，得到带原文偏移量的片段
    plan_cast     Planner：通读全文，建立角色表（规范名/别名/性别/年龄/特征/语气）与叙述视角
    label_chunk   Executor：按批次队列给片段编号打标签（类别 + 说话人），直到队列清空
    validate      确定性校验：漏标、未识别说话人、引号内被标旁白、提示语指向不一致、群体说单人台词、
                  低置信度、别名归一；能从提示语确定说话人时附修正建议；收尾时移除从未发言的角色
    review_agent  ReAct 校对员：带着问题清单调用工具查上下文、改标签、补/并角色
    review_tools  执行校对员的工具调用
    replan_cast   重规划：校对员/标注员新增了角色，或用户提出意见时，修订角色表并只重标受影响的批次
    roster_review 人工确认（interactive）：暂停并交出角色名单，用户确认或提出意见后继续
    cast_voices   可选：把角色映射到已有的本地音色配置（没有音色配置时跳过）

interactive 模式需要 checkpointer（见 session.py），中途失败可以从最近的检查点续跑。
各节点的提示词与阈值来自 skills/novel-role-assign/SKILL.md（见 skill.py）。
"""

import json
import operator
import re
from typing import Annotated, Any, Callable, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ..role_assigner import _normalize_name
from .llm import LLMSettings, accumulate_usage, build_chat_model, invoke_structured
from .schemas import (
    REVIEW_TOOLS, CastPlan, ChunkLabels, RosterRevision, VoiceCasting,
)
from . import checks
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
PROFILE_FIELD_NAMES = {'gender': '性别', 'age': '年龄', 'traits': '特征', 'voice_style': '语气'}
PROFILE_FIELDS = tuple(PROFILE_FIELD_NAMES)
SPEAKING = {'对话', '心理'}


class RoleAgentState(TypedDict, total=False):
    document_text: str
    segments: List[dict]
    chunks: List[List[int]]
    # 待标注的批次下标；首轮为全部批次，重规划后只放受影响的批次
    chunk_queue: List[int]
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
    # 重规划与人工确认
    user_feedback: List[str]
    pending_feedback: str
    recheck: List[str]
    replan_reply: str
    auto_replans: int
    approved: bool
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


def roster_entries(characters, discovered: bool = False) -> List[dict]:
    """把模型给出的 Character 列表整理成角色表：去重、去掉特殊说话人与代词别名。"""
    roster, seen = [], set()
    for character in characters:
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
            'voice_style': character.voice_style,
            'discovered': discovered,
        })
    return roster


def mentioned_segments(feedback: str, segments: List[dict], max_hits: int = 3) -> List[int]:
    """意见里点名的片段：#编号，以及「引文」/“引文”能在原文中定位到（最多 max_hits 处）的片段。"""
    ids = {int(n) for n in re.findall(r'#\s*(\d+)', feedback or '')}
    valid = {segment['id'] for segment in segments}
    for quote in re.findall(r'[「“"『‘]([^」”"』’]{2,})[」”"』’]', feedback or ''):
        needle = _squash(quote)
        hits = [s['id'] for s in segments if needle and needle in _squash(s['text'])]
        if len(hits) <= max_hits:
            ids.update(hits)
    return sorted(ids & valid)


def _squash(text: str) -> str:
    """去掉空白与标点，只比较文字本身（用户引用时常省略或改动标点）。"""
    return re.sub(r'[\W_]+', '', text or '')


def character_stats(roster: List[dict], labels: Dict[int, dict], segments: List[dict], samples: int = 3) -> Dict[str, dict]:
    """每个角色的台词数与前几句台词（编号 + 原文），供重规划与用户确认时参考。"""
    segment_map = {segment['id']: segment for segment in segments}
    stats = {c['name']: {'lines': 0, 'samples': []} for c in roster}
    for seg_id in sorted(labels):
        label = labels[seg_id]
        item = stats.get(label['speaker'])
        if item is None or label['category'] not in SPEAKING:
            continue
        item['lines'] += 1
        if len(item['samples']) < samples:
            item['samples'].append({'id': seg_id, 'text': segment_map[seg_id]['text'][:50]})
    return stats


def apply_roster_edits(roster: List[dict], labels: Dict[int, dict], edits: List[dict]):
    """用户在名单里直接改的字段：性别/年龄/特征/语气，以及改名（改成已有角色名即合并）。"""
    roster = [dict(c, aliases=list(c.get('aliases', []))) for c in roster]
    labels = {seg_id: dict(label) for seg_id, label in labels.items()}
    changes = []
    for edit in edits or []:
        character = next((c for c in roster if c['name'] == edit.get('name')), None)
        if character is None:
            continue
        for field in PROFILE_FIELDS:
            if field in edit and str(edit[field]) != str(character.get(field, '')):
                character[field] = str(edit[field])
                changes.append(f"{character['name']}的{PROFILE_FIELD_NAMES[field]}")
        new_name = str(edit.get('new_name') or '').strip()
        old_name = character['name']
        if not new_name or new_name == old_name or new_name in SPECIAL_SPEAKERS:
            continue
        target = next((c for c in roster if c['name'] == new_name), None)
        if target:
            roster.remove(character)
            target['aliases'] = list(dict.fromkeys(target['aliases'] + [old_name] + character['aliases']))
            changes.append(f"{old_name} 并入 {new_name}")
        else:
            character['aliases'] = list(dict.fromkeys(character['aliases'] + [old_name]))
            character['name'] = new_name
            changes.append(f"{old_name} 改名 {new_name}")
        for label in labels.values():
            if label['speaker'] == old_name:
                label['speaker'] = new_name
    return roster, labels, changes


class RoleAgent:
    """持有 LLM 与运行选项，节点方法共享同一份用量统计与进度回调。"""

    def __init__(
        self,
        settings: LLMSettings,
        progress: Optional[Callable[[str], None]] = None,
        max_review_rounds: int = 2,
        max_review_steps: int = 10,
        auto_cast: bool = True,
        llm=None,
        skill: Optional[Skill] = None,
        interactive: bool = False,
    ):
        self.settings = settings
        self.skill = skill or load_skill()
        self.p = self.skill.params
        self.llm = llm or build_chat_model(settings)
        self._llm_injected = llm is not None
        self.progress = progress or (lambda message: None)
        self.max_review_rounds = max_review_rounds
        self.max_review_steps = max_review_steps
        self.auto_cast = auto_cast
        # 校对结束后在 roster_review 暂停，等用户确认角色名单或提出意见（需要 checkpointer）
        self.interactive = interactive
        # 调用次数 / token / 美元花费，by_node 按节点细分
        self.usage: Dict[str, Any] = {}
        self._tool_llm = None
        self._replan_llm = None
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
            'chunk_queue': list(range(len(chunks))),
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

        roster = roster_entries(plan.characters)
        names = '、'.join(item['name'] for item in roster) or '无'
        return {
            'roster': roster,
            'plan': {'pov': plan.pov, 'narrator_character': plan.narrator_character, 'notes': plan.notes},
            'trace': [{'node': 'plan_cast', 'message': f"{plan.pov}；角色表：{names}" + (f"；备注：{plan.notes}" if plan.notes else '')}],
        }

    def label_chunk(self, state: RoleAgentState) -> dict:
        queue = list(state['chunk_queue'])
        index = queue.pop(0)
        chunk_ids = state['chunks'][index]
        segment_map = {segment['id']: segment for segment in state['segments']}
        labels = dict(state['labels'])
        roster = [dict(item) for item in state['roster']]
        relabeling = any(seg_id in labels for seg_id in chunk_ids)
        recheck = set(state.get('recheck') or [])
        verb = '重标' if relabeling else '标注'
        self.progress(f"{verb}第 {index + 1}/{len(state['chunks'])} 批...")

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
        if state.get('user_feedback'):
            # 用户对上一版结果提出的意见，标注时同样要遵守
            payload['user_feedback'] = state['user_feedback']
        result = invoke_structured(self.llm, ChunkLabels, [
            SystemMessage(content=self.skill.prompts['executor']),
            HumanMessage(content=_dump(payload)),
        ], self.usage, 'label_chunk')

        returned = {item.id: item for item in result.labels if item.id in chunk_ids}
        discovered = []
        kept = 0
        for seg_id in chunk_ids:
            old = labels.get(seg_id)
            if old and (old.get('pinned') or (old.get('reviewed') and old['speaker'] not in recheck)):
                # 重标时保留用户指定和已核实的标签，只有需要复查的角色例外
                kept += 1
                continue
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
                    roster.append({'name': speaker, 'aliases': [], 'gender': '未知', 'age': '未知', 'traits': '',
                                   'voice_style': '', 'discovered': True})
                    discovered.append(speaker)
                else:
                    speaker = UNKNOWN
            labels[seg_id] = {
                'category': category,
                'speaker': speaker,
                'confidence': float(item.confidence),
                'reason': item.reason.strip(),
            }

        message = f"{'重标' if relabeling else ''}第 {index + 1} 批：{len(returned)}/{len(chunk_ids)} 个片段已标注"
        if kept:
            message += f"，保留 {kept} 个已确认标签"
        if discovered:
            message += f"；新发现角色：{'、'.join(discovered)}"
        return {
            'labels': labels,
            'roster': roster,
            'chunk_queue': queue,
            'chunk_cursor': state.get('chunk_cursor', 0) + 1,
            'trace': [{'node': 'label_chunk', 'message': message}],
        }

    def validate(self, state: RoleAgentState) -> dict:
        roster = state['roster']
        labels = {seg_id: dict(label) for seg_id, label in state['labels'].items()}
        cues = checks.attribution_cues(state['segments'], roster)
        groups = {c['name'] for c in roster if checks.is_group(c)}
        issues = []
        for segment in state['segments']:
            seg_id = segment['id']
            label = labels.get(seg_id)
            if label is None:
                label = labels[seg_id] = {**_default_label(segment), 'auto': True}

            # 别名归一（校对员可能新增了别名或合并了角色）
            speaking = label['category'] in {'对话', '心理'}
            if speaking:
                resolved = resolve_speaker(label['speaker'], roster)
                if resolved and resolved != label['speaker']:
                    label['speaker'] = resolved

            cue = cues.get(seg_id)
            reviewed = label.get('reviewed')
            kind = problem = ''
            if label.get('auto') and not reviewed:
                kind, problem = 'missing', '模型漏标'
            elif speaking and label['speaker'] in {UNKNOWN, NARRATOR, ''}:
                kind, problem = 'unknown_speaker', '说话人未识别'
            elif segment['in_quote'] and label['category'] == '旁白' and len(segment['text']) > 8:
                kind, problem = 'quote_as_narration', '引号内文本被标为旁白'
            elif speaking and not resolve_speaker(label['speaker'], roster):
                kind, problem = 'not_in_roster', '说话人不在角色表'
            elif reviewed:
                pass
            elif speaking and cue and cue[0] != label['speaker']:
                kind, problem = 'cue_mismatch', f"提示语「{cue[1]}」指向 {cue[0]}"
            elif speaking and label['speaker'] in groups and segment['in_quote'] and checks.singular_line(segment['text']):
                kind, problem = 'group_singular', '群体角色说了含“我”的单人台词'
            elif label.get('confidence', 1) < self.p['low_confidence']:
                kind, problem = 'low_confidence', f"置信度低({label.get('confidence', 0):.2f})"
            if not kind:
                continue
            issue = {'id': seg_id, 'kind': kind, 'problem': problem}
            if cue and cue[0] != label['speaker'] and kind != 'low_confidence':
                # 提示语给出了明确的说话人：附上修正建议，校对员核实后可一次性采纳
                issue['suggest'] = {'category': label['category'] if speaking else '对话', 'speaker': cue[0]}
            issues.append(issue)

        update: dict = {'labels': labels, 'issues': issues}
        if 'first_issues' not in state:
            update['first_issues'] = issues
            update['pre_review_labels'] = {seg_id: dict(label) for seg_id, label in labels.items()}

        auto_fixed = []
        if self.p['auto_apply_suggestions']:
            # 提示语给出的说话人可靠性很高：直接修复，只把需要判断的疑点留给校对员
            for issue in [i for i in issues if i.get('suggest')]:
                labels[issue['id']] = {
                    **issue['suggest'],
                    'confidence': 0.9,
                    'reason': f"自动修复：{issue['problem']}"[:30],
                    'reviewed': True,
                }
                auto_fixed.append(f"#{issue['id']}→{issue['suggest']['speaker']}")
            issues = [i for i in issues if not i.get('suggest')]
            update['issues'] = issues
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
        message = f"发现 {len(issues)} 个可疑片段" + (f"：{summary}" + ('…' if len(issues) > 6 else '') if issues else '，全部通过')
        if auto_fixed:
            message = f"按提示语自动修复 {len(auto_fixed)} 处（{'、'.join(auto_fixed[:6])}）；" + message
        if update['review_done']:
            # 收尾：移除从未发言的角色（故事里被讲述的人物、规划误报），不再为它们选角
            unused = checks.unused_characters(roster, labels)
            if unused:
                update['roster'] = [c for c in roster if c['name'] not in unused]
                message += f"；移除未发言角色：{'、'.join(unused)}"
        update['trace'] = [{'node': 'validate', 'message': message}]
        return update

    def review_agent(self, state: RoleAgentState) -> dict:
        if self._tool_llm is None:
            base = self.llm
            if self.p['review_reasoning_effort'] and not self._llm_injected:
                base = build_chat_model(self.settings, max_tokens=self.p['review_max_tokens'],
                                        reasoning_effort=self.p['review_reasoning_effort'])
            self._tool_llm = base.bind_tools(REVIEW_TOOLS, max_tokens=self.p['review_max_tokens'])

        messages = list(state.get('messages') or [])
        if not messages:
            segment_map = {segment['id']: segment for segment in state['segments']}
            issue_lines = []
            for issue in state['issues'][:self.p['max_issues_in_prompt']]:
                label = state['labels'][issue['id']]
                suggest = issue.get('suggest')
                issue_lines.append(
                    f"#{issue['id']} [{issue['problem']}] 当前={label['category']}/{label['speaker']} "
                    + (f"建议={suggest['category']}/{suggest['speaker']} " if suggest else '')
                    + f"原文={segment_map[issue['id']]['text'][:60]}"
                )
            messages = [
                SystemMessage(content=self.skill.prompts['reviewer']),
                HumanMessage(content=(
                    f"叙述视角：{state['plan'].get('pov')}；叙述者：{state['plan'].get('narrator_character') or '无'}\n"
                    f"角色表：{'、'.join(c['name'] for c in state['roster']) or '无'}\n"
                    + (f"用户意见（必须遵守）：{'；'.join(state['user_feedback'])}\n" if state.get('user_feedback') else '')
                    + f"可疑片段（共 {len(state['issues'])} 个）：\n" + '\n'.join(issue_lines)
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
            # 被输出上限截断：结束校对且不再开新一轮（重来多半同样被截断），也不能当作“已确认”
            return {
                'review_done': True,
                'review_round': self.max_review_rounds,
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
        suggestions = {issue['id']: issue['suggest'] for issue in state.get('issues', []) if issue.get('suggest')}
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
                    result = self._run_tool(name, args, labels, roster, segment_map, suggestions)
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

    def _run_tool(self, name, args, labels, roster, segment_map, suggestions=None) -> str:
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

        if name == 'ApplySuggestions':
            changed = []
            for raw in args.get('ids') or []:
                seg_id = int(raw)
                suggest = (suggestions or {}).get(seg_id)
                if not suggest:
                    continue
                labels[seg_id] = {
                    'category': suggest['category'],
                    'speaker': suggest['speaker'],
                    'confidence': 0.9,
                    'reason': ('校对：' + (str(args.get('reason', '')).strip() or '采纳提示语建议'))[:30],
                    'reviewed': True,
                }
                changed.append(f"#{seg_id}→{suggest['category']}/{suggest['speaker']}")
            return '已采纳：' + '、'.join(changed) if changed else '这些编号没有可采纳的建议'

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
                'voice_style': '',
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

    def replan_cast(self, state: RoleAgentState) -> dict:
        """修订角色表：补全执行/校对阶段新增角色的档案，或按用户意见增删改角色、改点名的片段，
        然后只把受影响的批次放回标注队列。"""
        feedback = (state.get('pending_feedback') or '').strip()
        by_user = bool(feedback)
        self.progress("按意见修订角色表..." if by_user else "补全新增角色的信息...")
        roster = state['roster']
        labels = {seg_id: dict(label) for seg_id, label in state['labels'].items()}
        segment_map = {segment['id']: segment for segment in state['segments']}
        stats = character_stats(roster, labels, state['segments'], samples=2)

        payload = {
            'trigger': '用户意见' if by_user else '标注/校对时新增了角色',
            'user_feedback': feedback,
            'earlier_feedback': state.get('user_feedback') or [],
            **state['plan'],
            'roster': [
                {**{k: c.get(k, '') for k in ('name', 'aliases', *PROFILE_FIELDS)},
                 'new': bool(c.get('discovered')), **stats.get(c['name'], {})}
                for c in roster
            ],
            'unknown_lines': [
                {'id': seg_id, 'text': segment_map[seg_id]['text'][:40]}
                for seg_id, label in sorted(labels.items()) if label['speaker'] == UNKNOWN
            ][:20],
        }
        if by_user:
            payload['mentioned_segments'] = [
                {'id': seg_id, 'text': segment_map[seg_id]['text'][:80],
                 'category': labels[seg_id]['category'], 'speaker': labels[seg_id]['speaker']}
                for seg_id in mentioned_segments(feedback, state['segments'])
            ]
        prompt = self.skill.prompts.get('replanner') or self.skill.prompts['planner']
        if self._replan_llm is None:
            effort = self.p['replan_reasoning_effort']
            self._replan_llm = (build_chat_model(self.settings, reasoning_effort=effort)
                                if effort and not self._llm_injected else self.llm)
        revision = invoke_structured(self._replan_llm, RosterRevision, [
            SystemMessage(content=prompt), HumanMessage(content=_dump(payload)),
        ], self.usage, 'replan_cast')

        new_roster = roster_entries(revision.characters)
        new_names = {c['name'] for c in new_roster}
        old_names = [c['name'] for c in roster]
        merged, removed = {}, []
        for character in roster:
            if character['name'] in new_names:
                continue
            target = resolve_speaker(character['name'], new_roster)
            if target:
                merged[character['name']] = target
            elif by_user and any(n in feedback for n in [character['name']] + character.get('aliases', [])):
                removed.append(character['name'])
            else:
                # 意见里没提到的角色不许删：模型漏写的原样保留
                new_roster.append({**character, 'discovered': False})
        for old_name, target in merged.items():
            for label in labels.values():
                if label['speaker'] == old_name:
                    label['speaker'] = target

        added = [c['name'] for c in new_roster if c['name'] not in old_names]
        recheck = {resolve_speaker(name, new_roster) for name in revision.recheck} - {''}
        recheck |= set(added) | set(removed)

        pinned = []
        for update in revision.relabels:
            if update.id not in segment_map:
                continue
            category = update.category if update.category in VALID_CATEGORIES else '旁白'
            speaker = {'旁白': NARRATOR, '系统': SYSTEM}.get(category) or resolve_speaker(update.speaker, new_roster)
            if not speaker:
                continue
            labels[update.id] = {
                'category': category, 'speaker': speaker, 'confidence': 0.95,
                'reason': ('用户意见' if by_user else '修订角色表') + (f"：{update.reason}" if update.reason else ''),
                'reviewed': True, 'pinned': by_user,
            }
            pinned.append(update.id)

        # 受影响的片段：需要复查的角色的台词；有角色增删时连同未识别的台词、提到新角色名字的片段
        affected = {seg_id for seg_id, label in labels.items() if label['speaker'] in recheck}
        if recheck:
            affected |= {seg_id for seg_id, label in labels.items() if label['speaker'] == UNKNOWN}
            names = [n for c in new_roster if c['name'] in added for n in [c['name']] + c['aliases'] if len(n) >= 2]
            affected |= {s['id'] for s in state['segments'] if any(n in s['text'] for n in names)}
        affected -= set(pinned)
        queue = [index for index, chunk in enumerate(state['chunks']) if affected & set(chunk)]

        parts = []
        if added:
            parts.append(f"新增 {'、'.join(added)}")
        if merged:
            parts.append('合并 ' + '、'.join(f"{a}→{b}" for a, b in merged.items()))
        if removed:
            parts.append(f"移除 {'、'.join(removed)}")
        if pinned:
            parts.append(f"改正 {len(pinned)} 个片段")
        parts.append(f"重标 {len(queue)} 批" if queue else '无需重标')
        update = {
            'roster': new_roster,
            'labels': labels,
            'plan': {**state['plan'], 'notes': revision.notes or state['plan'].get('notes', '')},
            'chunk_queue': queue,
            'recheck': sorted(recheck),
            'pending_feedback': '',
            'replan_reply': revision.reply.strip(),
            'trace': [{'node': 'replan_cast', 'message': ('按用户意见' if by_user else '补全新增角色，') + '修订角色表：'
                       + '；'.join(parts) + (f"（{revision.reply.strip()[:60]}）" if revision.reply.strip() else '')}],
        }
        if by_user:
            update['user_feedback'] = list(state.get('user_feedback') or []) + [feedback]
            # 用户意见带来的改动再给校对员一轮复核
            update['review_round'] = max(0, self.max_review_rounds - 1)
            update['approved'] = False
        else:
            update['auto_replans'] = state.get('auto_replans', 0) + 1
        return update

    def roster_payload(self, state: RoleAgentState) -> dict:
        """交给用户确认的角色名单：每个角色的档案、台词数与示例台词，以及仍存疑的片段。"""
        stats = character_stats(state['roster'], state['labels'], state['segments'])
        segment_map = {segment['id']: segment for segment in state['segments']}
        roster = [
            {**{k: c.get(k, '') for k in ('name', *PROFILE_FIELDS)}, 'aliases': list(c.get('aliases', [])),
             **stats[c['name']]}
            for c in state['roster']
        ]
        roster.sort(key=lambda c: -c['lines'])
        counts = {}
        for label in state['labels'].values():
            counts[label['speaker']] = counts.get(label['speaker'], 0) + 1
        return {
            'roster': roster,
            'pov': state['plan'].get('pov', ''),
            'segments': len(state['segments']),
            'narration_lines': counts.get(NARRATOR, 0),
            'system_lines': counts.get(SYSTEM, 0),
            'unknown_lines': counts.get(UNKNOWN, 0),
            'issues': [
                {'id': issue['id'], 'problem': issue['problem'], 'text': segment_map[issue['id']]['text'][:40]}
                for issue in state.get('issues') or []
            ],
            'reply': state.get('replan_reply', ''),
            'feedback_history': list(state.get('user_feedback') or []),
        }

    def roster_review(self, state: RoleAgentState) -> dict:
        """暂停等用户确认名单。恢复值：{'action': 'approve'|'revise', 'feedback': 意见原文, 'edits': 直接修改}"""
        decision = interrupt(self.roster_payload(state)) or {}
        roster, labels, changes = apply_roster_edits(state['roster'], state['labels'], decision.get('edits') or [])
        feedback = str(decision.get('feedback') or '').strip()
        revise = decision.get('action') == 'revise' and bool(feedback)
        message = f"用户提出意见：{feedback[:60]}" if revise else '用户确认角色名单'
        if changes:
            message += f"；直接修改：{'、'.join(changes)}"
        return {
            'roster': roster,
            'labels': labels,
            'pending_feedback': feedback if revise else '',
            'approved': not revise,
            'trace': [{'node': 'roster_review', 'message': message}],
        }

    def cast_voices(self, state: RoleAgentState) -> dict:
        voice_configs = state.get('voice_configs') or []
        voice_names = [item['name'] for item in voice_configs]
        fallback = state.get('fallback_speaker', '')
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
                    {k: c.get(k, '') for k in ('name', *PROFILE_FIELDS)} for c in pending
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
        return 'label_chunk' if state.get('chunk_queue') else 'validate'

    def _finish(self, state: RoleAgentState) -> str:
        return 'cast_voices' if self.auto_cast and state.get('voice_configs') else END

    def route_after_validate(self, state: RoleAgentState) -> str:
        if not state.get('review_done'):
            return 'review_agent'
        if (self.p['auto_replan'] and state.get('auto_replans', 0) < self.p['max_replan']
                and any(c.get('discovered') for c in state['roster'])):
            return 'replan_cast'
        if self.interactive and not state.get('approved'):
            return 'roster_review'
        return self._finish(state)

    def route_after_replan(self, state: RoleAgentState) -> str:
        return 'label_chunk' if state.get('chunk_queue') else 'validate'

    def route_after_roster_review(self, state: RoleAgentState) -> str:
        return 'replan_cast' if state.get('pending_feedback') else self._finish(state)

    @staticmethod
    def route_after_review(state: RoleAgentState) -> str:
        if state.get('review_done'):
            return 'validate'
        return 'review_tools'

    @staticmethod
    def route_after_tools(state: RoleAgentState) -> str:
        return 'validate' if state.get('review_done') else 'review_agent'

    def build_graph(self, checkpointer=None):
        graph = StateGraph(RoleAgentState)
        graph.add_node('segment', self.segment)
        graph.add_node('plan_cast', self.plan_cast)
        graph.add_node('label_chunk', self.label_chunk)
        graph.add_node('validate', self.validate)
        graph.add_node('review_agent', self.review_agent)
        graph.add_node('review_tools', self.review_tools)
        graph.add_node('replan_cast', self.replan_cast)
        graph.add_node('roster_review', self.roster_review)
        graph.add_node('cast_voices', self.cast_voices)

        graph.add_edge(START, 'segment')
        graph.add_edge('segment', 'plan_cast')
        graph.add_edge('plan_cast', 'label_chunk')
        graph.add_conditional_edges('label_chunk', self.route_after_label, ['label_chunk', 'validate'])
        graph.add_conditional_edges('validate', self.route_after_validate,
                                    ['review_agent', 'replan_cast', 'roster_review', 'cast_voices', END])
        graph.add_conditional_edges('review_agent', self.route_after_review, ['review_tools', 'validate'])
        graph.add_conditional_edges('review_tools', self.route_after_tools, ['review_agent', 'validate'])
        graph.add_conditional_edges('replan_cast', self.route_after_replan, ['label_chunk', 'validate'])
        graph.add_conditional_edges('roster_review', self.route_after_roster_review, ['replan_cast', 'cast_voices', END])
        graph.add_edge('cast_voices', END)
        return graph.compile(checkpointer=checkpointer)

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
