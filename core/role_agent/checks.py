"""确定性一致性检查：为 validate 节点提供带证据的疑点和修正建议，供 ReAct 校对员核实后采纳。

只用原文结构和角色表，不调用模型：
- attribution_cues  同一段落内紧贴引号的提示语（“……”王二婶嗓门最大 / 王会计站起来：“……”）指向谁
- is_group          角色是否为群体
- singular_line     引号内是否是单人口吻（含“我”但不是“我们”）
- unused_characters 角色表里从未发言的角色（多为故事里被讲述的人物或规划阶段的误报）
"""

from typing import Dict, List, Optional, Tuple

# 提示语开头可跳过的连接词
LEAD_ADVERBS = ('这时', '于是', '随即', '接着', '然后', '只见', '忽然', '突然', '最后', '此时')
# 无名多人发言的主语
CROWD_SUBJECTS = ('众人', '大家', '人群里有人', '人群中有人', '有人', '人群', '众')
# 引号前的提示语以这些标点结尾才算“提示语 + 引号”
LEAD_ENDINGS = ('：', ':', '，', ',')


def is_group(character: dict) -> bool:
    name = character.get('name', '')
    return (character.get('gender') == '群体' or name.startswith('众') or name.endswith('们')
            or name in ('大家', '人群', '群众'))


def _strip_adverbs(text: str) -> str:
    text = text.lstrip()
    for adverb in LEAD_ADVERBS:
        if text.startswith(adverb):
            return text[len(adverb):].lstrip('，, ')
    return text


def subject_at_start(text: str, roster: List[dict]) -> str:
    """片段开头的主语对应的角色规范名；开头不是角色名（或是“某某的”所有格）时返回空串。"""
    text = _strip_adverbs(text)
    best, best_len = '', 0
    for character in roster:
        for name in [character['name'], *character.get('aliases', [])]:
            if not name or len(name) <= best_len or not text.startswith(name):
                continue
            rest = text[len(name):len(name) + 1]
            # “我们”“赵德厚的脸” 不是该角色在说话
            if rest in ('的', '们') or (len(name) == 1 and rest and rest in '们俩'):
                continue
            best, best_len = character['name'], len(name)
    if best:
        return best
    groups = [c['name'] for c in roster if is_group(c)]
    if len(groups) == 1 and any(text.startswith(word) for word in CROWD_SUBJECTS):
        return groups[0]
    return ''


def attribution_cues(segments: List[dict], roster: List[dict]) -> Dict[int, Tuple[str, str]]:
    """引号片段 → (提示语指向的角色, 证据原文)。

    只看同一段落内紧挨引号的旁白片段，优先级：冒号提示语 > 引号后的主语 > 逗号提示语。
    """
    cues: Dict[int, Tuple[str, str]] = {}
    for index, segment in enumerate(segments):
        if not segment.get('in_quote'):
            continue
        prev = segments[index - 1] if index > 0 else None
        nxt = segments[index + 1] if index + 1 < len(segments) else None
        candidates = []
        if prev and not prev['in_quote'] and prev['paragraph'] == segment['paragraph']:
            lead = prev['text'].rstrip()
            if lead.endswith(LEAD_ENDINGS):
                rank = 0 if lead.endswith(('：', ':')) else 2
                candidates.append((rank, subject_at_start(lead, roster), lead))
        if nxt and not nxt['in_quote'] and nxt['paragraph'] == segment['paragraph']:
            candidates.append((1, subject_at_start(nxt['text'], roster), nxt['text']))
        candidates = sorted(c for c in candidates if c[1])
        if candidates:
            _, speaker, evidence = candidates[0]
            cues[segment['id']] = (speaker, evidence.strip()[:20])
    return cues


def singular_line(text: str) -> bool:
    """单人口吻：出现“我”且不是“我们/我俩/我等”。"""
    for index, char in enumerate(text):
        if char == '我' and text[index + 1:index + 2] not in ('们', '俩', '等'):
            return True
    return False


def unused_characters(roster: List[dict], labels: Dict[int, dict]) -> List[str]:
    speakers = {label['speaker'] for label in labels.values() if label['category'] in ('对话', '心理')}
    return [c['name'] for c in roster if c['name'] not in speakers]


def find_character(roster: List[dict], name: str) -> Optional[dict]:
    return next((c for c in roster if c['name'] == name), None)
