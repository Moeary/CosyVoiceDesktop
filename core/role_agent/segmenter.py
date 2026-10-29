"""规则断句：在本地把全文切成带原文偏移量的片段，LLM 只需按编号打标签，不再复述原文。"""

from typing import List

# 引号配对（ASCII 双引号开合同形，按栈顶判断）
QUOTE_PAIRS = {'“': '”', '「': '」', '『': '』', '"': '"'}
TERMINATORS = set('。！？!?；;…')
TRAILING_CLOSERS = set('）)】]’\'》>')
DIALOGUE_LEADS = set('：:，,。！？!?；;…—')
# 冒号前以这些字收尾时，视为无引号发言/心理的提示语（Skill 参数 speech_lead_verbs 可覆盖）
SPEECH_VERBS = '道说想问答叫喊骂笑叹念忖思'
PUNCT_ONLY = set('，,、。！？!?；;：:…—-～~·”」』’）)】]》"\' \t　')

# 引号内文本超过该长度时，允许在句末标点处继续切分
MAX_QUOTE_LEN = 120
# 引号外无句末标点的超长文本，在逗号处兜底切分
MAX_PLAIN_LEN = 200
# 叙述中被引号括起的短称谓不单独成段
MAX_INLINE_QUOTE_LEN = 10


def _is_punct_only(text: str) -> bool:
    return all(char in PUNCT_ONLY for char in text)


def _is_speech_lead(full_text: str, seg_start: int, colon: int, verbs: str = SPEECH_VERBS) -> bool:
    """冒号前是否为“某某道/想/问”之类的发言提示语，且冒号后还有内容。"""
    if colon + 1 >= len(full_text) or full_text[colon + 1] in '\n' or full_text[colon + 1] in QUOTE_PAIRS:
        return False
    lead = full_text[seg_start:colon].rstrip()
    return bool(lead) and lead[-1] in verbs


def _inline_quote_end(full_text: str, index: int, line_start: int, line_end: int) -> int:
    """判断 index 处的引号是否只是叙述中引用的短称谓（如 来到“水帘洞”前）。

    是则返回闭合引号之后的位置，否则返回 0。
    """
    prev = index - 1
    while prev >= line_start and full_text[prev].isspace():
        prev -= 1
    # 行首或紧跟标点（如 “道：”）的引号视为发言
    if prev < line_start or full_text[prev] in DIALOGUE_LEADS:
        return 0
    close = full_text.find(QUOTE_PAIRS[full_text[index]], index + 1, line_end)
    if close < 0:
        return 0
    inner = full_text[index + 1:close]
    if len(inner) > MAX_INLINE_QUOTE_LEN or any(char in TERMINATORS or char in '，,' for char in inner):
        return 0
    return close + 1


def split_segments(full_text: str, speech_verbs: str = SPEECH_VERBS) -> List[dict]:
    """按段落、引号与句末标点断句。

    返回的每个片段包含 id(从 1 开始)、start、end（full_text 中的半开区间）、
    text、paragraph（段落序号）以及 in_quote（是否为引号内文本）。
    """
    segments: List[dict] = []
    paragraph = 0
    position = 0

    def flush(start: int, end: int, in_quote: bool):
        # 去掉首尾空白，偏移量随之收缩
        while start < end and full_text[start].isspace():
            start += 1
        while end > start and full_text[end - 1].isspace():
            end -= 1
        if start >= end:
            return
        text = full_text[start:end]
        # 纯标点碎片并入同段落的前一片段，避免出现单独的 “。”
        if _is_punct_only(text) and segments and segments[-1]['paragraph'] == paragraph:
            prev = segments[-1]
            prev['end'] = end
            prev['text'] = full_text[prev['start']:end]
            return
        segments.append({
            'id': len(segments) + 1,
            'start': start,
            'end': end,
            'text': text,
            'paragraph': paragraph,
            'in_quote': in_quote,
        })

    for line in full_text.split('\n'):
        line_start = position
        line_end = position + len(line)
        position = line_end + 1
        if not line.strip():
            continue
        paragraph += 1

        seg_start = line_start
        stack: List[str] = []
        index = line_start
        while index < line_end:
            char = full_text[index]

            if stack:
                if char == stack[-1]:
                    stack.pop()
                    if not stack:
                        flush(seg_start, index + 1, True)
                        seg_start = index + 1
                elif char in QUOTE_PAIRS and char != '"':
                    stack.append(QUOTE_PAIRS[char])
                elif char in TERMINATORS and index + 1 - seg_start > MAX_QUOTE_LEN:
                    end = index + 1
                    while end < line_end and full_text[end] in TERMINATORS:
                        end += 1
                    flush(seg_start, end, True)
                    seg_start = end
                    index = end
                    continue
                index += 1
                continue

            if char in QUOTE_PAIRS:
                inline_end = _inline_quote_end(full_text, index, line_start, line_end)
                if inline_end:
                    index = inline_end
                    continue
                flush(seg_start, index, False)
                seg_start = index
                stack.append(QUOTE_PAIRS[char])
            elif char in TERMINATORS or (char == '.' and (index + 1 >= line_end or full_text[index + 1].isspace())):
                end = index + 1
                while end < line_end and (full_text[end] in TERMINATORS or full_text[end] in TRAILING_CLOSERS):
                    end += 1
                flush(seg_start, end, False)
                seg_start = end
                index = end
                continue
            elif char in '：:' and _is_speech_lead(full_text, seg_start, index, speech_verbs):
                # “石猴暗想：这里……” 这类无引号的发言/心理，在冒号后断开，提示语归旁白
                flush(seg_start, index + 1, False)
                seg_start = index + 1
            elif char in '，,' and index + 1 - seg_start > MAX_PLAIN_LEN:
                flush(seg_start, index + 1, False)
                seg_start = index + 1
            index += 1

        # 行尾收尾；未闭合的引号按引号内文本处理
        flush(seg_start, line_end, bool(stack))

    return segments


def chunk_segments(segments: List[dict], max_items: int = 40, max_chars: int = 2400) -> List[List[int]]:
    """把片段按数量和字数切成若干批，返回每批的片段 id 列表。"""
    chunks: List[List[int]] = []
    current: List[int] = []
    current_chars = 0
    for segment in segments:
        length = len(segment['text'])
        over = len(current) >= max_items or current_chars + length > max_chars
        if current and over:
            chunks.append(current)
            current, current_chars = [], 0
        current.append(segment['id'])
        current_chars += length
    if current:
        chunks.append(current)
    return chunks
