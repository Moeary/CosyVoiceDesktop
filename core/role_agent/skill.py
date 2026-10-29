"""分角 Skill 加载：提示词与参数集中在 SKILL.md（YAML 头 + 二级标题分段），改字不用动代码。"""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

DEFAULT_SKILL_PATH = Path(__file__).resolve().parent / 'skills' / 'novel-role-assign' / 'SKILL.md'
REQUIRED_SECTIONS = ('planner', 'executor', 'reviewer', 'caster')
DEFAULT_PARAMS: Dict[str, Any] = {
    'low_confidence': 0.6,
    'chunk_max_items': 40,
    'chunk_max_chars': 2400,
    'context_window': 8,
    'max_plan_chars': 30000,
    'max_issues_in_prompt': 40,
    'review_max_tokens': 2048,
    'speech_lead_verbs': '道说想问答叫喊骂笑叹念忖思',
    # 为 系统 单独选角（系统流的电子音）；关闭时 系统 与旁白共用默认音色
    'cast_system_speaker': False,
    # 疑点带有提示语给出的修正建议时，validate 直接采纳，不再交给校对员
    'auto_apply_suggestions': False,
    # 校对员的推理强度（none/low/medium/high，空为模型默认）；推理过长会挤占工具调用的输出空间
    'review_reasoning_effort': '',
    # 标注/校对时新增了角色，校对结束后交给重规划补全档案、合并重复角色，必要时重标受影响的批次
    'auto_replan': False,
    'max_replan': 1,
    # 重规划的推理强度（同 review_reasoning_effort）
    'replan_reasoning_effort': '',
}


class SkillError(Exception):
    """Skill 文件缺失或格式错误"""


@dataclass
class Skill:
    name: str
    version: str
    path: str
    digest: str
    params: Dict[str, Any] = field(default_factory=dict)
    prompts: Dict[str, str] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.name}@v{self.version}#{self.digest}"

    def info(self) -> dict:
        return {'name': self.name, 'version': self.version, 'digest': self.digest, 'path': self.path, 'params': self.params}


def parse_markdown_sections(text: str) -> Tuple[dict, Dict[str, str]]:
    """拆出 YAML 头和 `## 名称` 分段；第一个二级标题之前的正文视为说明，不返回。"""
    meta = {}
    body = text
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n', text, re.S)
    if match:
        meta = yaml.safe_load(match.group(1)) or {}
        body = text[match.end():]

    sections: Dict[str, str] = {}
    current = None
    lines = []
    for line in body.splitlines():
        heading = re.match(r'^##\s+(\S+)\s*$', line)
        if heading:
            if current:
                sections[current] = '\n'.join(lines).strip()
            current = heading.group(1).lower()
            lines = []
        elif current:
            lines.append(line)
    if current:
        sections[current] = '\n'.join(lines).strip()
    return meta, sections


def load_skill(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Skill:
    skill_path = Path(path) if path else DEFAULT_SKILL_PATH
    if not skill_path.exists():
        raise SkillError(f"找不到 Skill 文件: {skill_path}")
    text = skill_path.read_text(encoding='utf-8')
    meta, sections = parse_markdown_sections(text)

    missing = [name for name in REQUIRED_SECTIONS if not sections.get(name)]
    if missing:
        raise SkillError(f"Skill 缺少段落: {', '.join(missing)}（{skill_path}）")

    params = dict(DEFAULT_PARAMS)
    params.update(meta.get('params') or {})
    params.update(overrides or {})

    return Skill(
        name=str(meta.get('name') or skill_path.parent.name),
        version=str(meta.get('version', '0')),
        path=str(skill_path),
        digest=hashlib.sha256(text.encode('utf-8')).hexdigest()[:8],
        params=params,
        prompts={name: sections[name] for name in sections},
    )
