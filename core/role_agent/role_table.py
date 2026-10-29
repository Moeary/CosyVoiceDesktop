"""角色名单 → 角色表（每个角色一条本地语音配置），参考音频可以手动填写，也可以从目录里自动匹配。

参考音频目录支持常见的几种组织方式：
- 按说话人分文件夹：<目录>/<说话人>/<任意名>.wav，同名 .txt / .lab 是参考文本；
- 标注清单：*.list / *.lst，每行 “音频路径|说话人|语言|文本”（GPT-SoVITS / Bert-VITS2 格式）；
- 平铺文件：<说话人>_<参考文本>.wav 或 <说话人>-<参考文本>.wav。
同一说话人有多条音频时，优先选有参考文本、时长 3~12 秒的那条。
"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from ..role_assigner import _normalize_name
from .llm import invoke_structured
from .schemas import LibraryCasting

AUDIO_EXTS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
TEXT_EXTS = ('.txt', '.lab', '.normalized.txt')
MAX_FILES = 5000
MAX_VOICES_IN_PROMPT = 150
IDEAL_DURATION = (3.0, 12.0)
NARRATOR = '旁白'
SYSTEM = '系统'
PALETTE = ['#FFD166', '#06D6A0', '#118AB2', '#EF476F', '#8338EC', '#FF9F1C', '#2EC4B6', '#E71D36',
           '#9B5DE5', '#00BBF9', '#F15BB5', '#90BE6D', '#F9844A', '#4D908E', '#577590', '#F94144']

LIBRARIAN_PROMPT = (
    "你是有声书选角导演。请为每个角色从 voices 中挑选一位说话人作为参考音色，voice_id 必须原样使用列表中的 id。\n"
    "依据角色的性别、年龄、身份特征、语气，与说话人的名称、所在文件夹、参考文本做匹配：性别必须一致；"
    "年龄段尽量一致（老人用老人音色，孩子用孩子音色）；同一篇里的主要角色不要共用同一位说话人，也不要用旁白的音色。"
    "实在没有合适的就把 voice_id 留空，交给用户手动指定。"
)


@dataclass
class VoiceCandidate:
    speaker: str
    path: str
    text: str = ''
    duration: Optional[float] = None
    hints: List[str] = field(default_factory=list)


def _audio_duration(path: Path) -> Optional[float]:
    try:
        import soundfile
        return float(soundfile.info(str(path)).duration)
    except Exception:
        return None


def _read_sidecar(path: Path) -> str:
    for ext in TEXT_EXTS:
        sidecar = path.with_suffix(ext)
        if sidecar.exists():
            try:
                return sidecar.read_text(encoding='utf-8').strip().splitlines()[0].strip()
            except (OSError, UnicodeDecodeError, IndexError):
                return ''
    return ''


def _parse_list_files(root: Path) -> Dict[str, tuple]:
    """标注清单：音频路径 → (说话人, 文本)。清单里的路径可能是别的机器上的绝对路径，按文件名兜底匹配。"""
    entries = {}
    for list_file in list(root.rglob('*.list'))[:20] + list(root.rglob('*.lst'))[:20]:
        try:
            lines = list_file.read_text(encoding='utf-8').splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for line in lines:
            parts = line.strip().split('|')
            if len(parts) >= 4 and parts[0]:
                entries[Path(parts[0]).name] = (parts[1].strip(), parts[3].strip())
    return entries


def _split_stem(stem: str):
    """平铺文件名：说话人_参考文本 / 说话人-参考文本；去掉 【开心】(1) 这类标记。"""
    match = re.match(r'^([^_\-—\s]{1,12})[_\-—\s]+(.+)$', stem)
    if not match:
        return stem, ''
    text = re.sub(r'[【\[(（][^】\])）]*[】\])）]', '', match.group(2)).strip()
    return match.group(1), text if re.search(r'[一-鿿]{4,}', text) else ''


def scan_voice_library(root: str, max_files: int = MAX_FILES) -> List[VoiceCandidate]:
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"参考音频目录不存在: {root}")
    listed = _parse_list_files(root_path)

    candidates = []
    for dirpath, _, filenames in os.walk(root_path):
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            if path.suffix.lower() not in AUDIO_EXTS:
                continue
            relative = path.relative_to(root_path)
            hints = list(relative.parts[:-1])
            if filename in listed:
                speaker, text = listed[filename]
            elif hints:
                speaker, text = hints[-1], ''
            else:
                speaker, text = _split_stem(path.stem)
            text = _read_sidecar(path) or text
            candidates.append(VoiceCandidate(speaker=speaker, path=str(path), text=text, hints=hints))
            if len(candidates) >= max_files:
                return candidates
    return candidates


def best_clip_per_speaker(candidates: List[VoiceCandidate]) -> Dict[str, VoiceCandidate]:
    """每位说话人挑一条参考音频：有参考文本优先，其次时长落在 3~12 秒。"""
    grouped: Dict[str, List[VoiceCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.speaker, []).append(candidate)

    best = {}
    low, high = IDEAL_DURATION
    for speaker, items in grouped.items():
        with_text = [c for c in items if c.text] or items
        # 只为入围的少数文件读时长，大目录也不会卡
        for candidate in with_text[:8]:
            if candidate.duration is None:
                candidate.duration = _audio_duration(Path(candidate.path))

        def score(c: VoiceCandidate):
            d = c.duration
            if d is None:
                return 1.0
            return 0.0 if low <= d <= high else min(abs(d - low), abs(d - high))

        best[speaker] = min(with_text[:8], key=score)
    return best


def match_characters(characters: List[dict], voices: Dict[str, VoiceCandidate], llm=None, usage=None,
                     prompt: str = '', progress: Optional[Callable[[str], None]] = None) -> Dict[str, dict]:
    """角色 → 说话人。说话人名与角色名/别名一致的直接匹配，其余交给 LLM 按性别、年龄、语气挑选。"""
    by_name = {_normalize_name(speaker): speaker for speaker in voices}
    matches: Dict[str, dict] = {}
    pending = []
    for character in characters:
        names = [character['name']] + list(character.get('aliases', []))
        speaker = next((by_name[_normalize_name(n)] for n in names if _normalize_name(n) in by_name), '')
        if speaker:
            matches[character['name']] = {'speaker': speaker, 'reason': '说话人名与角色名一致', 'source': 'rule'}
        else:
            pending.append(character)

    if not pending or llm is None or not voices:
        return matches
    if progress:
        progress("为角色匹配参考音频...")
    taken = {m['speaker'] for m in matches.values()}
    listing = [
        {'id': speaker, 'folders': '/'.join(clip.hints[-3:]), 'sample_text': clip.text[:30]}
        for speaker, clip in voices.items() if speaker not in taken
    ][:MAX_VOICES_IN_PROMPT]
    payload = {
        'voices': listing,
        'already_cast': {name: m['speaker'] for name, m in matches.items()},
        'characters': [
            {k: c.get(k, '') for k in ('name', 'gender', 'age', 'traits', 'voice_style')} | {'lines': c.get('lines', 0)}
            for c in pending
        ],
    }
    casting = invoke_structured(llm, LibraryCasting, [
        SystemMessage(content=prompt or LIBRARIAN_PROMPT),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ], usage, 'match_library')
    valid = {item['id'] for item in listing}
    pending_names = {c['name'] for c in pending}
    for pick in casting.picks:
        if pick.character in pending_names and pick.voice_id in valid:
            matches[pick.character] = {'speaker': pick.voice_id, 'reason': pick.reason, 'source': 'llm'}
    return matches


def style_instruction(voice_style: str) -> str:
    style = (voice_style or '').strip().rstrip('。')
    return f"请用{style}的语气说这句话。" if style else ''


def build_role_table(characters: List[dict], matches: Optional[Dict[str, dict]] = None,
                     voices: Optional[Dict[str, VoiceCandidate]] = None, narrator_config: str = '',
                     include_system: bool = False, existing_names=()) -> List[dict]:
    """生成角色表行（与 VoiceConfig.to_dict 同字段，另带 character / source / reason）。

    模式：有语气 → 指令控制（参考音频定音色，语气指令定说法）；有参考文本 → 零样本复制；只有音频 → 精细控制。
    旁白已有默认音色配置时不再新建，直接复用。
    """
    matches = matches or {}
    voices = voices or {}
    rows = []
    existing = set(existing_names)

    def add_row(name, voice_style='', reason=''):
        match = matches.get(name)
        clip = voices.get(match['speaker']) if match else None
        audio, text = (clip.path, clip.text) if clip else ('', '')
        instruct = style_instruction(voice_style)
        if instruct and audio:
            mode = '指令控制'
        elif audio and not text:
            mode = '精细控制'
        else:
            mode = '零样本复制'
        rows.append({
            'name': name,
            'mode': mode,
            'prompt_text': text,
            'prompt_audio': audio,
            'instruct_text': instruct,
            'color': PALETTE[(len(existing) + len(rows)) % len(PALETTE)],
            'character': name,
            'source': (match or {}).get('source', 'manual'),
            'reason': (match or {}).get('reason', reason),
            'voice_speaker': (match or {}).get('speaker', ''),
        })

    if not narrator_config:
        add_row(NARRATOR, reason='旁白')
    if include_system:
        add_row(SYSTEM, voice_style='平稳的电子播报', reason='系统提示')
    for character in characters:
        add_row(character['name'], character.get('voice_style', ''))
    return rows


def speaker_to_config(rows: List[dict], narrator_config: str = '') -> Dict[str, str]:
    """分角结果里的说话人 → 角色表里的配置名；旁白（以及没有单独配置的系统）用默认旁白配置。"""
    mapping = {row['character']: row['name'] for row in rows}
    narrator = mapping.get(NARRATOR) or narrator_config
    mapping.setdefault(NARRATOR, narrator)
    mapping.setdefault(SYSTEM, narrator)
    return mapping
