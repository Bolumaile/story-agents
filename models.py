"""Pydantic 数据模型层：把「模型输出」从"靠手工兜底的裸 dict"变成"有 schema 的对象"。

对应优化建议第 6 条（伏笔状态机 + schema 校验）与第 9 条（强类型约束）。
两处共用一套模型，避免各写一份校验逻辑。

后续补充（首次真模型实跑暴露的问题）：
- `Item` / `ItemChange`：物资道具账本。原先记忆库只有 伏笔/人物/摘要 三个键，
  物资不在内，撰稿人每章凭摘要自编，导致"吃掉的压缩饼干又出现""凭空多出黄桃罐头"，
  3 章烧掉 3 轮重写。现在物资进账本，撰稿端只能用账本内的东西。
- `canon_name()`：角色名归一。原先按 name 精确匹配合并人物快照，
  模型换个称呼（「橘猫（流浪猫）」/「橘猫」/「猫」）就新建一条，同一角色裂成多条。

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
import re
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


# ── 物资状态归一 ─────────────────────────────────────────────────

ITEM_STATUS = Literal["available", "consumed", "lost"]

_ITEM_STATUS_ALIASES = {
    "available": "available", "usable": "available", "held": "available",
    "可用": "available", "持有": "available", "在身": "available", "完好": "available",
    "consumed": "consumed", "used": "consumed", "used_up": "consumed", "empty": "consumed",
    "耗尽": "consumed", "已耗尽": "consumed", "用完": "consumed", "已用完": "consumed",
    "吃掉": "consumed", "吃光": "consumed", "消耗": "consumed", "空": "consumed",
    "lost": "lost", "missing": "lost", "gone": "lost",
    "丢失": "lost", "遗失": "lost", "被抢": "lost", "不在": "lost",
}


def _norm_item_status(v: Any) -> Optional[str]:
    """物资状态归一。空值返回 None = 「本次不修改状态」。

    注意 "空" 归到 consumed（"罐子空了"），与"未提供"（None）是两回事：
    前者是明确的消耗动作，后者是模型没提这件事——混起来会让已吃完的东西复活。
    """
    raw = _s(v).strip()
    if not raw:
        return None
    key = raw.lower()
    if key in _ITEM_STATUS_ALIASES:
        return _ITEM_STATUS_ALIASES[key]
    if raw in _ITEM_STATUS_ALIASES:
        return _ITEM_STATUS_ALIASES[raw]
    _warn(f"item.status.{key}",
          f"记忆结算给出的物资状态「{raw}」不是合法值"
          f"（只接受 available / consumed / lost），本次不改动该物资状态。")
    return None


# ── 角色名归一（人物快照防裂条）──────────────────────────────────

_BRACKET_RE = re.compile(r"[（(【\[][^）)】\]]*[）)】\]]")
_NAME_NOISE_RE = re.compile(r"[\s·・.,，。、;；:：!！?？\"'“”‘’\-—_]+")


def canon_name(name: Any) -> str:
    """把角色名归一成可比较的键：去括号备注 → 去空白标点 → 转小写。

    例：「橘猫（流浪猫）」/「橘猫 」/「橘猫」→ 都是 "橘猫"。
    不做子串包含匹配（「林岸」⊂「林岸的父亲」会误合并），
    那类漂移靠结算提示词「从给定名单中选名字」从源头抑制。
    """
    s = _s(name).strip()
    if not s:
        return ""
    s = _BRACKET_RE.sub("", s)
    s = _NAME_NOISE_RE.sub("", s)
    return s.lower()


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


# ── 物资道具账本 ─────────────────────────────────────────────────
# 首次真模型实跑（2026-10-02）暴露的头号问题：记忆库只有 伏笔 / 人物 / 摘要 三个键，
# 物资不在内，撰稿人每章凭摘要自编——"吃完的压缩饼干又出现""凭空多出黄桃罐头"
# "凭空多出牛肉干"，3 章里 2 章因这个被打回，烧掉 3 轮重写。

_ITEM_NAME_KEYS = ("item", "物品", "道具", "物件", "名称", "thing")


class Item(BaseModel):
    """账本里的一件物资（权威记录；撰稿端只能用账本内的东西）。

    为什么单独立账本而不是塞进人物快照：人物快照写的是"他是谁、他在想什么"，
    物资写的是"他手上还剩什么"。混在一起模型就会用写人物状态的笔法写物资，
    余量、数量这类关键信息全丢——而那恰恰是跨章比对要用的。
    """
    name: Str = ""
    qty: Str = ""                    # 当前存量描述（"半罐""1 瓶""约两天口粮"）
    status: ITEM_STATUS = "available"
    note: Str = ""                   # 来源 / 存放位置 / 其他说明
    chapter: Int = 0                 # 首次登记章节
    last_changed_chapter: Int = 0    # 最后一次变动章节

    @field_validator("status", mode="before")
    @classmethod
    def _coerce_status(cls, v):
        # 账本内的条目必须有确定状态；空值/非法值一律当 available（fail-open）
        return _norm_item_status(v) or "available"


class ItemChange(BaseModel):
    """记忆结算给出的单条物资变动。留空的字段 = 本次不改动该项。

    与 Item 分开的原因：
    - "这个罐头还剩半罐"（改存量）和"这个罐头吃完了"（改状态）是两件事，
      合成一个模型就会出现"只改存量时状态被默认值覆盖成 available"的复活 bug。
    - 新物品也走这里：name 不在账本里 → 新建条目。
    """
    name: Str = ""
    qty: Str = ""                          # 变更后的存量描述；空 = 不改动
    status: Optional[ITEM_STATUS] = None   # 变更后的状态；None = 不改动
    note: Str = ""

    @model_validator(mode="before")
    @classmethod
    def _alias_name(cls, data):
        """容忍模型把物品名写在别的键上（item / 物品 / 道具…）。"""
        if isinstance(data, dict) and not _s(data.get("name")).strip():
            for k in _ITEM_NAME_KEYS:
                if _s(data.get(k)).strip():
                    return {**data, "name": data.get(k)}
        return data

    @field_validator("status", mode="before")
    @classmethod
    def _coerce_status(cls, v):
        return _norm_item_status(v)


class Outline(BaseModel):
    title: Str = ""
    theme: Str = ""
    characters: List[Character] = Field(default_factory=list)
    chapters: List[Chapter] = Field(default_factory=list)
    core_conflict: Str = ""
    ending_direction: Str = ""
    # 开局物资清单：给第 1 章的撰稿人一个起点。
    # 为什么放策划案里：实测最伤的是"第 1 章背包清单自身就矛盾（列了 4 项又摸出罐头），
    # 而第 1 章 0 轮重写通过 → 内伤被当正典固化，后续每章都要重新审一遍"。
    # 让策划先把清单立住，撰稿端从第一章起就有账本可比对。
    initial_items: List[Item] = Field(default_factory=list)

    @field_validator("characters", mode="before")
    @classmethod
    def _coerce_characters(cls, v):
        return _dict_items(v, "name")

    @field_validator("chapters", mode="before")
    @classmethod
    def _coerce_chapters(cls, v):
        return _dict_items(v, "title")

    @field_validator("initial_items", mode="before")
    @classmethod
    def _coerce_items(cls, v):
        return _dict_items(v, "name")

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
    items: List[Item] = Field(default_factory=list)     # 物资道具账本

    @field_validator("chapter_summaries", "foreshadow_pool",
                     "character_states", "items", mode="before")
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
    item_changes: List[ItemChange] = Field(default_factory=list)

    @field_validator("new_foreshadows", mode="before")
    @classmethod
    def _coerce_new_fs(cls, v):
        return _dict_items(v, "desc")

    @field_validator("character_updates", mode="before")
    @classmethod
    def _coerce_updates(cls, v):
        return _dict_items(v, "name")

    @field_validator("item_changes", mode="before")
    @classmethod
    def _coerce_item_changes(cls, v):
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
