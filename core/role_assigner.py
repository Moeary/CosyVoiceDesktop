import json
import re
from typing import Dict, List, Any

import requests


class RoleAssignmentError(Exception):
    """角色分配异常"""


def _normalize_name(name: str) -> str:
    text = (name or "").strip().lower()
    return re.sub(r'[\s\-_()（）\[\]【】"\'“”‘’:：,，.。!！？?、/\\]+', '', text)


def _build_chat_completions_url(base_url: str) -> str:
    base = (base_url or "").strip().rstrip('/')
    if not base:
        raise RoleAssignmentError("请先在设置页面填写 LLM Base URL")

    if base.endswith('/chat/completions'):
        return base
    if base.endswith('/v1'):
        return f"{base}/chat/completions"
    if base.endswith('/openai'):
        return f"{base}/v1/chat/completions"
    return f"{base}/v1/chat/completions"


def _extract_message_content(response_data: dict) -> str:
    choices = response_data.get('choices') or []
    if not choices:
        raise RoleAssignmentError("LLM 返回中缺少 choices 字段")

    message = choices[0].get('message') or {}
    content = message.get('content', '')

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get('type') == 'text':
                    parts.append(item.get('text', ''))
                elif 'text' in item:
                    parts.append(str(item.get('text', '')))
            else:
                parts.append(str(item))
        return ''.join(parts).strip()

    return str(content).strip()


def _extract_json_payload(text: str) -> Any:
    content = (text or "").strip()
    if not content:
        raise RoleAssignmentError("LLM 返回为空")

    fenced = re.search(r'```(?:json)?\s*([\s\S]*?)```', content, re.IGNORECASE)
    candidates = []
    if fenced:
        candidates.append(fenced.group(1).strip())
    candidates.append(content)

    array_match = re.search(r'(\[[\s\S]*\])', content)
    if array_match:
        candidates.append(array_match.group(1))

    object_match = re.search(r'(\{[\s\S]*\})', content)
    if object_match:
        candidates.append(object_match.group(1))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise RoleAssignmentError("无法从 LLM 返回中解析 JSON")


def _normalize_category(value: str) -> str:
    text = (value or "").strip().lower()
    if not text:
        return ""

    mapping = {
        '旁白': 'narration',
        '叙述': 'narration',
        '叙事': 'narration',
        '描写': 'narration',
        'narrator': 'narration',
        'narration': 'narration',
        '对白': 'dialogue',
        '对话': 'dialogue',
        '台词': 'dialogue',
        '发言': 'dialogue',
        'dialog': 'dialogue',
        'dialogue': 'dialogue',
        'speech': 'dialogue',
        '心理': 'thought',
        '心声': 'thought',
        '独白': 'thought',
        '内心独白': 'thought',
        'thought': 'thought',
        'inner': 'thought',
        'system': 'system',
        '系统': 'system',
        '说明': 'system',
        '提示': 'system',
    }
    return mapping.get(text, text if text in {'narration', 'dialogue', 'thought', 'system'} else "")


def _build_group_name(category: str, speaker_label: str) -> str:
    normalized_category = _normalize_category(category) or 'narration'
    speaker = (speaker_label or "").strip()
    if normalized_category in {'dialogue', 'thought'}:
        return f"{normalized_category} / {speaker or '未识别角色'}"
    if normalized_category == 'system' and speaker and _normalize_name(speaker) not in {'system', '系统'}:
        return f"system / {speaker}"
    return normalized_category


def _merge_reason_with_speaker(category: str, speaker_label: str, reason: str) -> str:
    speaker = (speaker_label or "").strip()
    message = (reason or "").strip()
    normalized_category = _normalize_category(category)

    if normalized_category == 'dialogue' and speaker:
        prefix = f"说话人: {speaker}"
    elif normalized_category == 'thought' and speaker:
        prefix = f"思考人: {speaker}"
    else:
        prefix = ""

    if prefix and prefix not in message:
        return f"{prefix} | {message}" if message else prefix
    return message


def _compact_text_with_map(text: str):
    compact_chars = []
    index_map = []
    for index, char in enumerate(text):
        if char.isspace():
            continue
        compact_chars.append(char)
        index_map.append(index)
    return ''.join(compact_chars), index_map


def _extract_segment_text(item: dict) -> str:
    return str(
        item.get('text')
        or item.get('segment_text')
        or item.get('content')
        or item.get('quote')
        or item.get('sentence')
        or ''
    ).strip()


def _align_raw_segments_to_text(full_text: str, raw_segments: List[dict]) -> List[dict]:
    compact_full_text, index_map = _compact_text_with_map(full_text)
    if not compact_full_text:
        raise RoleAssignmentError("原文为空，无法对齐 AI 分句结果")

    compact_cursor = 0
    aligned_segments = []

    for order, item in enumerate(raw_segments, 1):
        segment_text = _extract_segment_text(item)
        compact_segment = re.sub(r'\s+', '', segment_text)
        if not compact_segment:
            continue

        compact_pos = compact_full_text.find(compact_segment, compact_cursor)
        if compact_pos < 0:
            preview = segment_text[:60]
            raise RoleAssignmentError(f"AI 返回的第 {order} 个分句无法在原文中定位: {preview}")

        start = index_map[compact_pos]
        end = index_map[compact_pos + len(compact_segment) - 1] + 1
        merged_item = dict(item)
        merged_item['text'] = full_text[start:end]
        merged_item['start'] = start
        merged_item['end'] = end
        merged_item['_segment_order'] = order
        aligned_segments.append(merged_item)
        compact_cursor = compact_pos + len(compact_segment)

    return aligned_segments


class RoleAssignmentService:
    """OpenAI 兼容 LLM 角色分配服务"""

    def __init__(self, config_manager):
        self.config_manager = config_manager

    def _get_default_speaker(self, speaker_names: List[str]) -> str:
        configured = (self.config_manager.get('default_speaker_name', '') or '').strip()
        if configured in speaker_names:
            return configured
        return speaker_names[0]

    def _match_voice_config(self, raw_speaker: str, speaker_names: List[str]) -> str:
        speaker = (raw_speaker or '').strip()
        if speaker in speaker_names:
            return speaker

        normalized = _normalize_name(speaker)
        if not normalized:
            return ""

        normalized_map = {_normalize_name(name): name for name in speaker_names}
        if normalized in normalized_map:
            return normalized_map[normalized]

        fuzzy_matches = []
        for name in speaker_names:
            normalized_name = _normalize_name(name)
            if normalized_name and (normalized in normalized_name or normalized_name in normalized):
                fuzzy_matches.append(name)

        if len(fuzzy_matches) == 1:
            return fuzzy_matches[0]

        return ""

    def _suggest_voice_mapping(
        self,
        speaker_label: str,
        category: str,
        speaker_names: List[str],
        fallback_speaker: str,
        current_speaker: str,
    ) -> str:
        current = (current_speaker or "").strip()
        if current in speaker_names:
            return current

        matched = self._match_voice_config(speaker_label, speaker_names)
        if matched:
            return matched

        normalized_category = _normalize_category(category) or 'narration'
        if normalized_category in {'narration', 'system'} and fallback_speaker in speaker_names:
            return fallback_speaker

        normalized_speaker = _normalize_name(speaker_label)
        if normalized_speaker in {'旁白', 'narrator', 'voiceover'} and fallback_speaker in speaker_names:
            return fallback_speaker

        return ""

    def _build_document_text(self, document_text: str) -> str:
        return (document_text or '').strip()

    def _build_messages(self, document_text: str, speaker_names: List[str], fallback_speaker: str) -> List[dict]:
        document_text = self._build_document_text(document_text)

        system_prompt = (
            "你是中文多角色有声书文本分析助手。"
            "你会先完整阅读整篇文本，再自动断句，最后按阅读顺序输出结构化 JSON。"
            "available_voice_configs 只是界面里的本地配音配置名称，只用于后续人工映射，不是正文角色名。"
            "除非这些名字真的出现在正文里，否则不要把正文角色写成本地配音配置名称。"
            "你要输出的是正文中的真实角色或群体标识，例如 悟空、四猴、众猴、巡海夜叉、旁白。"
            "你必须主动断句，不要把整段长文本合并成一个 item。"
            "通常应在句号、问号、感叹号、分号、说话切换、引号结束、叙述视角切换处断开；一段里有多句时，正常应拆成多个 segments。"
            "如果一段文字里既有旁白又有对白，必须继续细分，不要整段只给一个 narration。"
            "带引号的台词、冒号后的发言、以及包含“说、问、答、喊、叫、笑道、低声、怒道、回应、嘀咕、说道、问道、答道、叫道、喝道”等提示词的句子，应优先判为 dialogue。"
            "即使一句里夹杂少量动作描写，只要核心内容是人物发言，仍应判为 dialogue。"
            "心理活动、内心独白、自语、自思、自忖优先判为 thought，并尽量指出是谁在想。"
            "纯叙述、环境描写、动作描写判为 narration；系统提示、舞台说明判为 system。"
            "对于 dialogue 和 thought，speaker_label 必须尽量填写具体人物或群体名称。"
            "如果能从前后文推断出说话人或思考人，就不要留空。"
            "如果连续多段明显属于同一角色发言或思考，应保持 speaker_label 一致。"
            "如果 dialogue 或 thought 实在无法判断人物，也要写 speaker_label 为“未识别角色”，不要误判成 narration。"
            "每个 segment.text 必须是原文中的连续原句，尽量保持字面一致，不要改写，不要总结，不要省略。"
            "所有 segment.text 按顺序拼接后，应该覆盖全文正文。"
            "请严格返回 JSON，不要附加解释。"
        )

        user_prompt = {
            'task': '先通读全文，自动断句，再为每个分句输出 category、speaker_label、text、reason、confidence。',
            'available_voice_configs': speaker_names,
            'default_narration_voice_config': fallback_speaker,
            'decision_rules': [
                '你必须把整篇全文作为一个连续故事来判断，再自行切成适合配音的片段。',
                '如果输入是一整段长文本，输出必须拆成多个 segments，不能只返回一个总段落。',
                'dialogue 和 thought 尽量填具体 speaker_label，例如 悟空、四猴、众猴；不要直接写成本地配音配置名。',
                '如果是对话，请在 reason 里简短写出依据，并明确说话人是谁。',
                '如果是 thought，请在 reason 里简短写出依据，并明确是谁的内心活动。',
                '每个 segments[i].text 都必须是原文连续摘录，不能改写。',
                'segments 数组需要覆盖全文，不能漏掉中间句子。',
            ],
            'output_schema': {
                'segments': [
                    {
                        'text': '俺也去。',
                        'category': 'dialogue',
                        'speaker_label': '悟空',
                        'confidence': 0.95,
                        'reason': '说话人: 悟空 | 根据前文“悟空道”判断'
                    }
                ]
            },
            'full_text': document_text,
        }

        return [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': json.dumps(user_prompt, ensure_ascii=False)},
        ]

    def _post_chat_completions(self, url: str, headers: dict, payload: dict, timeout: int) -> dict:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if response.ok:
            return response.json()

        # 一些 OpenAI 兼容服务不支持 response_format，这里自动降级重试一次
        fallback_payload = dict(payload)
        fallback_payload.pop('response_format', None)
        fallback_response = requests.post(url, headers=headers, json=fallback_payload, timeout=timeout)
        if fallback_response.ok:
            return fallback_response.json()

        try:
            error_text = fallback_response.text[:300]
        except Exception:
            error_text = str(fallback_response.status_code)
        raise RoleAssignmentError(f"LLM 请求失败: {fallback_response.status_code} {error_text}")

    def assign_roles(self, segments: List[dict], document_text: str, voice_configs: Dict[str, Any]) -> dict:
        if not (document_text or '').strip():
            raise RoleAssignmentError("没有可分配的文本内容")

        speaker_names = list(voice_configs.keys())
        if not speaker_names:
            raise RoleAssignmentError("请先在语音设置中添加角色配置")

        base_url = (self.config_manager.get('llm_base_url', '') or '').strip()
        model = (self.config_manager.get('llm_model', '') or '').strip()
        api_key = (self.config_manager.get('llm_api_key', '') or '').strip()
        timeout = int(self.config_manager.get('llm_timeout_sec', 60) or 60)

        if not base_url:
            raise RoleAssignmentError("请先在设置页面填写 LLM Base URL")
        if not model:
            raise RoleAssignmentError("请先在设置页面填写 LLM 模型名称")

        url = _build_chat_completions_url(base_url)
        fallback_speaker = self._get_default_speaker(speaker_names)

        headers = {'Content-Type': 'application/json'}
        if api_key and not api_key.lower().startswith('http'):
            headers['Authorization'] = f'Bearer {api_key}'

        payload = {
            'model': model,
            'messages': self._build_messages(document_text, speaker_names, fallback_speaker),
            'temperature': 0.1,
            'response_format': {'type': 'json_object'},
        }

        response_data = self._post_chat_completions(url, headers, payload, timeout)
        content = _extract_message_content(response_data)
        parsed_payload = _extract_json_payload(content)

        if isinstance(parsed_payload, list):
            raw_assignments = parsed_payload
        else:
            raw_assignments = (
                parsed_payload.get('segments')
                or parsed_payload.get('items')
                or parsed_payload.get('assignments')
                or parsed_payload.get('results')
                or parsed_payload.get('data')
                or []
            )

        if not raw_assignments:
            raise RoleAssignmentError("LLM 已返回结果，但分句结果为空")

        raw_items = [item for item in raw_assignments if isinstance(item, dict)]
        if not raw_items:
            raise RoleAssignmentError("LLM 返回的分句结果格式无效")

        def current_speaker_for_range(start: int, end: int) -> str:
            for segment in segments:
                segment_start = segment.get('start')
                segment_end = segment.get('end')
                if isinstance(segment_start, int) and isinstance(segment_end, int):
                    if segment_start <= start < segment_end or segment_start < end <= segment_end:
                        return str(segment.get('current_speaker', '') or '').strip()
            return ""

        if any(_extract_segment_text(item) for item in raw_items):
            source_segments = _align_raw_segments_to_text(document_text, raw_items)
            if not source_segments:
                raise RoleAssignmentError("AI 没有返回可用的分句结果")
        else:
            if not segments:
                raise RoleAssignmentError("AI 没有返回 text 字段，且无法回退到原始段落")

            assignment_map = {}
            for item in raw_items:
                try:
                    index = int(item.get('index'))
                except (TypeError, ValueError):
                    continue
                assignment_map[index] = item

            source_segments = []
            for segment in segments:
                item = assignment_map.get(segment['index'], {})
                source_item = dict(item)
                source_item['text'] = segment['text']
                source_item['start'] = segment.get('start')
                source_item['end'] = segment.get('end')
                source_item['_segment_order'] = segment['index']
                source_segments.append(source_item)

        normalized_assignments = []
        for order, item in enumerate(source_segments, 1):
            raw_speaker = str(
                item.get('speaker_label')
                or item.get('speaker')
                or item.get('speaker_name')
                or item.get('character')
                or item.get('role')
                or ''
            ).strip()
            category = _normalize_category(str(item.get('category', '') or '').strip())
            if not category:
                category = 'dialogue' if raw_speaker else 'narration'
            if category == 'narration' and not raw_speaker:
                raw_speaker = '旁白'
            elif category in {'dialogue', 'thought'} and not raw_speaker:
                raw_speaker = '未识别角色'
            elif category == 'system' and not raw_speaker:
                raw_speaker = '系统'

            confidence = item.get('confidence', None)
            try:
                confidence = float(confidence) if confidence is not None else None
            except (TypeError, ValueError):
                confidence = None

            reason = _merge_reason_with_speaker(
                category,
                raw_speaker,
                str(item.get('reason', '') or '').strip(),
            )
            current_speaker = current_speaker_for_range(item.get('start', -1), item.get('end', -1))
            suggested_voice = str(item.get('suggested_voice') or item.get('mapped_speaker') or '').strip()
            if suggested_voice not in speaker_names:
                suggested_voice = self._suggest_voice_mapping(
                    raw_speaker,
                    category,
                    speaker_names,
                    fallback_speaker,
                    current_speaker,
                )

            normalized_assignments.append({
                'index': order,
                'speaker': raw_speaker,
                'speaker_label': raw_speaker,
                'raw_speaker': raw_speaker,
                'group_name': str(item.get('group_name') or item.get('group') or '').strip() or _build_group_name(category, raw_speaker),
                'suggested_voice': suggested_voice,
                'category': category,
                'confidence': confidence,
                'reason': reason or ('模型未返回该段说明，已按当前分类展示' if item else '模型未返回该段，已按默认分类补齐'),
                'text': str(item.get('text') or '').strip(),
                'current_speaker': current_speaker,
                'start': item.get('start'),
                'end': item.get('end'),
            })

        return {
            'assignments': normalized_assignments,
            'auto_apply': bool(self.config_manager.get('llm_auto_apply', False)),
            'fallback_speaker': fallback_speaker,
            'raw_response': content,
            'request_url': url,
        }
