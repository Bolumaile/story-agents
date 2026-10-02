"""Pydantic 数据模型层：把「模型输出」从"靠手工兜底的裸 dict"变成"有 schema 的对象"。

对应优化建议第 6 条（伏笔状态机 + schema 校验）与第 9 条（强类型约束）。
两处共用一套模型，避免各写一份校验逻辑。

设计取舍（与本项目一贯的 fail-open 哲学一致）
------------------------------------------------
1. **缺字段落默认值，不抛异常。** 模型少写一个键，不该让已经写好的整章作废
   ——这正是 `nodes._normalize_comments()` 当初存在的理由，这里把它系统化。
2. **非法枚举值归一到合法值并告警，而不是直接拒绝整条数据。**
   对比 inkos 的"坏数据直接拒绝"：那套更适合服务端 API，而这里是创作流水线，
   丢掉一条伏笔会让长篇的伏笔链永久断裂，比"值不合法"更伤。所以策略是
   **收敛到默认态 + 提示用户**，坏值不会继续传播，数据也不丢。
3. 所有模型都能 `model_dump()` 成纯 dict，可直接塞进 State 和 JSON 文件。
"""
from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, BeforeValidator, Field, field_validator, model_validator

import llm

# ── 基础类型：任何脏输入都先收敛，再交给 Pydantic 校验 ──────────────

def _s(v: Any, default: str = "") -> str:
    """任意值 → 字符串。None 给默认值，列表/字典拼接而不是报错。"""
    if v is None:
        return default
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, (list, tuple)):
        return "；".join(_s(x) for x in v if x is not None)
    if isinstance(v, dict):
        return "；".join(f"{k}：{_s(x)}" for k, x in v.items())
    return str(v)


def _i(v: Any, default: int = 0) -> int:
    """任意值 → 整数。"3" / "3.0" / 3.5 都能收，实在转不了给默认值。"""
    if v is None or isinstance(v, bool):
        return default
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    if isinstance(v, str):
        t = v.strip()
        if not t:
            return default
        try:
            return int(float(t))
        except ValueError:
            return default
    return default


def _as_list(v: Any) -> list:
    """None → []；单元素 → [元素]；列表原样返回。"""
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, tuple):
        return list(v)
    return [v]


def _dict_items(v: Any, key_hint: str) -> list:
    """收敛成 list[dict]。元素是字符串时，按 key_hint 包成单字段对象。

    例：characters 写成 ["林岸", "老崔"] → [{"name": "林岸"}, {"name": "老崔"}]
    """
    items = []
    for it in _as_list(v):
        if isinstance(it, dict):
            items.append(it)
        elif isinstance(it, str) and it.strip():
            items.append({key_hint: it.strip()})
        # 其余（None / 数字 / 嵌套列表）直接丢弃，不阻断解析
    return items


Str = Annotated[str, BeforeValidator(_s)]
Int = Annotated[int, BeforeValidator(_i)]

# ── 告警：同一类问题只提醒一次，避免刷屏 ──────────────────────────
_REPORTED: set = set()


def _warn(key: str, message: str) -> None:
    if key in _REPORTED:
        return
    _REPORTED.add(key)
    try:
        llm.emit_notice("warn", message)
    except Exception:                                    # noqa: BLE001
        pass


def reset_warnings() -> None:
    """清空告警去重表（测试用；生产环境跨章保留更合理）。"""
    _REPORTED.clear()


# ── 枚举归一 ─────────────────────────────────────────────────────

FORESHADOW_STATUS = Literal["open", "progressing", "deferred", "resolved"]

_STATUS_ALIASES = {
    "open": "open", "埋下": "open", "未回收": "open", "待回收": "open", "未解决": "open",
    "progressing": "progressing", "in_progress": "progressing", "inprogress": "progressing",
    "推进中": "progressing", "发展中": "progressing", "进行中": "progressing",
    "deferred": "deferred", "搁置": "deferred", "暂缓": "deferred", "延后": "deferred",
    "resolved": "resolved", "closed": "resolved", "done": "resolved",
    "已回收": "resolved", "回收": "resolved", "已解决": "resolved", "已完成": "resolved",
}


def _norm_status(v: Any) -> str:
    raw = _s(v).strip()
    key = raw.lower()
    if key in _STATUS_ALIASES:
        return _STATUS_ALIASES[key]
    if raw in _STATUS_ALIASES:
        return _STATUS_ALIASES[raw]
    _warn(f"foreshadow.status.{key}",
          f"记忆结算给出的伏笔状态「{raw or '空'}」不是合法值"
          f"（只接受 open / progressing / deferred / resolved），已按 open 处理。")
    return "open"


_TYPE_ALIASES = {
    "ooc": "ooc", "人物": "ooc", "人设": "ooc", "人物一致性": "ooc", "角色": "ooc",
    "logic": "logic", "逻辑": "logic", "剧情": "logic", "因果": "logic",
    "setting": "setting", "设定": "setting", "世界观": "setting",
    "timeline": "timeline", "时间线": "timeline", "时间": "timeline", "前后矛盾": "timeline",
    "pacing": "pacing", "节奏": "pacing", "注水": "pacing", "篇幅": "pacing",
}


def _norm_type(v: Any) -> str:
    raw = _s(v).strip()
    return _TYPE_ALIASES.get(raw.lower()) or _TYPE_ALIASES.get(raw) or "logic"


def _norm_severity(v: Any) -> str:
    """判定从严：只有明确写 critical 才算严重问题（沿用原 _normalize_comments 的策略）。"""
    return "critical" if _s(v).strip().lower().startswith("crit") else "minor"


# ── 模型 ─────────────────────────────────────────────────────────

class SceneBeat(BaseModel):
    """场景节拍：一章内部的一次推进单元（优化建议第 4 条）。"""
    goal: Str = ""           # 本场景推进什么
    conflict: Str = ""       # 阻力 / 冲突
    turn: Str = ""           # 情绪或信息转折
    words: Int = 0           # 建议字数
    emotion: Int = 3         # 情绪温度 1-5

    @field_validator("emotion", mode="after")
    @classmethod
    def _clamp_emotion(cls, v: int) -> int:
        return min(5, max(1, v))

    @field_validator("words", mode="after")
    @classmethod
    def _clamp_words(cls, v: int) -> int:
        return min(10000, max(0, v))


class Chapter(BaseModel):
    index: Int = 0
    title: Str = ""
    outline: Str = ""
    turning_point: Str = ""
    notes: Str = ""
    beats: List[SceneBeat] = Field(default_factory=list)

    @field_validator("beats", mode="before")
    @classmethod
    def _coerce_beats(cls, v):
        return _dict_items(v, "goal")


class Character(BaseModel):
    name: Str = ""
    personality: Str = ""
    motivation: Str = ""
    taboo: Str = ""
    potential_conflicts: Str = ""


class Outline(BaseModel):
    title: Str = ""
    theme: Str = ""
    characters: List[Character] = Field(default_factory=list)
    chapters: List[Chapter] = Field(default_factory=list)
    core_conflict: Str = ""
    ending_direction: Str = ""

    @field_validator("characters", mode="before")
    @classmethod
    def _coerce_characters(cls, v):
        return _dict_items(v, "name")

    @field_validator("chapters", mode="before")
    @classmethod
    def _coerce_chapters(cls, v):
        return _dict_items(v, "title")

    @model_validator(mode="after")
    def _fill_chapter_index(self):
        """模型常常漏写或写错 index。缺省/非法时按列表位置补 1-based 序号。

        这比事后在 nodes 里比对更靠前，能保证 index 全局唯一、连续。
        """
        for pos, ch in enumerate(self.chapters, start=1):
            if ch.index <= 0:
                _warn("chapter.index.missing",
                      "策划案里有章节没写 index（或写了非正整数），已按顺序自动补齐。")
                ch.index = pos
        return self


class ReviewComment(BaseModel):
    type: Str = "logic"
    severity: Literal["critical", "minor"] = "minor"
    quote: Str = ""
    issue: Str = ""
    suggestion: Str = ""

    @field_validator("type", mode="before")
    @classmethod
    def _coerce_type(cls, v):
        return _norm_type(v)

    @field_validator("severity", mode="before")
    @classmethod
    def _coerce_severity(cls, v):
        return _norm_severity(v)


class Foreshadow(BaseModel):
    """伏笔状态机（优化建议第 6 条）。

    相比原来的 `{id, desc, chapter, status}`：
    - status 从 open/resolved 两态扩展为 open / progressing / deferred / resolved
    - 新增 last_advanced_chapter：最后一次被"推进"（提及、加深、兑现）的章节
    - 新增 deadline_chapter：期望回收的章节，超期可在看板提示
    """
    id: Str = ""
    desc: Str = ""
    chapter: Int = 0                       # 埋下的章节
    status: FORESHADOW_STATUS = "open"
    last_advanced_chapter: Int = 0         # 最后一次有推进的章节
    deadline_chapter: Optional[int] = None

    @field_validator("status", mode="before")
    @classmethod
    def _coerce_status(cls, v):
        return _norm_status(v)

    @field_validator("deadline_chapter", mode="before")
    @classmethod
    def _coerce_deadline(cls, v):
        if v is None or v == "" or v == 0:
            return None
        n = _i(v, 0)
        return n if n > 0 else None


class CharacterState(BaseModel):
    name: Str = ""
    state: Str = ""


class ChapterSummary(BaseModel):
    chapter: Int = 0
    summary: Str = ""


class Memory(BaseModel):
    """故事记忆库（全量权威数据）。"""
    chapter_summaries: List[ChapterSummary] = Field(default_factory=list)
    foreshadow_pool: List[Foreshadow] = Field(default_factory=list)
    character_states: List[CharacterState] = Field(default_factory=list)

    @field_validator("chapter_summaries", "foreshadow_pool", "character_states", mode="before")
    @classmethod
    def _coerce_lists(cls, v):
        return _as_list(v)


class MemoryDelta(BaseModel):
    """记忆结算员每章输出的增量。"""
    summary: Str = ""
    new_foreshadows: List[Foreshadow] = Field(default_factory=list)
    resolved_foreshadows: List[str] = Field(default_factory=list)
    advanced_foreshadows: List[str] = Field(default_factory=list)
    character_updates: List[CharacterState] = Field(default_factory=list)

    @field_validator("new_foreshadows", mode="before")
    @classmethod
    def _coerce_new_fs(cls, v):
        return _dict_items(v, "desc")

    @field_validator("character_updates", mode="before")
    @classmethod
    def _coerce_updates(cls, v):
        return _dict_items(v, "name")

    @field_validator("resolved_foreshadows", "advanced_foreshadows", mode="before")
    @classmethod
    def _coerce_id_list(cls, v):
        """id 列表：容忍写成 [{"id": "F1"}] 或 "F1,F2" 这类形态。"""
        out = []
        for it in _as_list(v):
            if isinstance(it, dict):
                out.append(_s(it.get("id")).strip())
            else:
                for part in _s(it).replace("，", ",").split(","):
                    if part.strip():
                        out.append(part.strip())
        return [x for x in out if x]


# ── 解析入口：节点统一走这几个函数，不再自己写兜底 ────────────────

def parse_outline(data: Dict[str, Any]) -> Outline:
    return Outline.model_validate(data or {})


def parse_memory(data: Dict[str, Any]) -> Memory:
    return Memory.model_validate(data or {})


def parse_memory_delta(data: Dict[str, Any]) -> MemoryDelta:
    return MemoryDelta.model_validate(data or {})


def parse_comments(items: Any) -> List[ReviewComment]:
    return [ReviewComment.model_validate(x) for x in _dict_items(items, "issue")]


def dump(items) -> List[Dict[str, Any]]:
    """把模型列表转成纯 dict 列表（可直接进 State / JSON）。"""
    out = []
    for x in (items or []):
        out.append(x.model_dump() if isinstance(x, BaseModel) else x)
    return out
