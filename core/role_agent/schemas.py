"""智能体各节点的结构化输出与 ReAct 工具参数定义。"""

from typing import List, Literal

from pydantic import BaseModel, Field

Category = Literal['旁白', '对话', '心理', '系统']


# ---------- Planner：角色表 ----------

class Character(BaseModel):
    name: str = Field(description="正文中的规范角色名，群体也可作为角色，如 众猴")
    aliases: List[str] = Field(default_factory=list, description="正文里出现过的别名、称呼、外号，如 美猴王；不含 我/他/我等 这类代词泛称")
    gender: Literal['男', '女', '群体', '未知'] = '未知'
    age: Literal['儿童', '少年', '青年', '中年', '老年', '未知'] = '未知'
    traits: str = Field(default='', description="不超过 20 字的身份/性格/嗓音特征")


class CastPlan(BaseModel):
    characters: List[Character] = Field(description="所有会说话或有心理活动的角色")
    pov: Literal['第一人称', '第三人称'] = '第三人称'
    narrator_character: str = Field(default='', description="第一人称叙述时“我”对应的角色名，否则留空")
    notes: str = Field(default='', description="给后续标注员的注意事项，不超过 80 字")


# ---------- Executor：分批标注 ----------

class SegmentLabel(BaseModel):
    id: int
    category: Category
    speaker: str = Field(description="对话/心理填角色表中的规范名；旁白填 旁白；系统填 系统；无法判断填 未识别角色")
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(default='', description="不超过 20 字的依据")


class ChunkLabels(BaseModel):
    labels: List[SegmentLabel]


# ---------- Caster：角色 → 本地音色 ----------

class VoiceMapping(BaseModel):
    character: str
    voice_config: str = Field(description="必须是 available_voice_configs 中的名称；没有合适的留空")
    reason: str = ''


class VoiceCasting(BaseModel):
    mappings: List[VoiceMapping] = Field(description="每个角色一条")


# ---------- Reviewer：ReAct 工具 ----------

class ReadSegments(BaseModel):
    """查看编号区间内片段的原文和当前标签，用于结合上下文判断说话人。"""
    start_id: int
    end_id: int


class SearchText(BaseModel):
    """在全文片段中搜索关键词，返回命中的片段编号和原文。"""
    keyword: str


class GetRoster(BaseModel):
    """查看当前角色表（规范名、别名、性别、特征）。"""


class LabelUpdate(BaseModel):
    id: int
    category: Category
    speaker: str
    reason: str = ''


class Relabel(BaseModel):
    """修正一个或多个片段的类别与说话人。speaker 必须是角色表中的规范名、旁白、系统或未识别角色。"""
    updates: List[LabelUpdate]


class AddCharacter(BaseModel):
    """向角色表添加漏掉的角色。"""
    name: str
    aliases: List[str] = Field(default_factory=list)
    gender: Literal['男', '女', '群体', '未知'] = '未知'
    traits: str = ''


class MergeCharacter(BaseModel):
    """把重复/别名角色合并到规范名下，所有相关片段会一并改名。"""
    alias: str
    canonical: str


class Finish(BaseModel):
    """所有问题处理完毕（或确认无需修改）时调用，结束校对。"""
    summary: str


REVIEW_TOOLS = [ReadSegments, SearchText, GetRoster, Relabel, AddCharacter, MergeCharacter, Finish]
