"""Pydantic 数据模型层：把「模型输出」从"靠手工兜底的裸 dict"变成"有 schema 的对象"。

对应优化建议第 6 条（伏笔状态机 + schema 校验）与第 9 条（强类型约束）。
两处共用一套模型，避免各写一份校验逻辑。

后续补充（首次真模型实跑暴露的问题）：
- `Item` / `ItemChange`：物资道具账本。原先记忆库只有 伏笔/人物/摘要 三个键，
  物资不在内，撰稿人每章凭摘要自编，导致"吃掉的压缩饼干又出现""凭空多出黄桃罐头"，
  3 章烧掉 3 轮重写。现在物资进账本，撰稿端只能用账本内的东西。
- `canon_name()`：角色名归一。原先按 name 精确匹配合并人物快照，
  模型换个称呼（「橘猫（流浪猫）」/「橘猫」/「猫」）就新建一条，同一角色裂成多条。
- `Setting` / `SettingChange`：设定档案（第四本账）。原先前三本账只覆盖
  伏笔 / 人物 / 物资，**地点与物理设定不在任何一本里**——校对手上没有"与前文一致"
  的依据，于是把第 1 章早已确立的「店面卷帘门 + 后门铁门」连着两轮判成 critical
  设定矛盾（正文没按它改是对的）。设定进档后，撰稿端知道这是既成事实、
  逻辑校对端有了比对基准，误报从源头掐掉。

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
import threading
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


# ── 存量可计数化：账本要「能对账」，就必须有机器可比的数字 ────────
# 第二次真模型实跑（2026-10-02）暴露的头号成本项：策划给的 initial_items.qty 是
# 「约五天份」「半袋」「约两周份」这类**自由文本**，撰稿人每章得自己换算成瓶/块，
# 于是每章算出不同的数、逻辑校对每章都报 —— 7 个重写轮次里 ch2/ch3 各撞 3 轮上限，
# 约六成 critical 是账目类。这是"凭空多物"被止住之后**迁移出来的新形态**。
#
# 解法分两层：
#   1. 提示词要求策划直接给 count（整数）+ unit（瓶/块/罐），模糊描述挪进 qty；
#   2. 拿到手之后仍然尽力**从描述里提取数字**（"六块"→6、"5 瓶"→5），
#      这样老存档、mock、以及不听话的模型输出也能参与对账，而不是直接放弃。
_QTY_UNITS = "瓶块罐包个根把支盒袋张条枚口桶听片台册"

_CN_NUM = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3,
           "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def _cn_to_int(tok: str) -> Optional[int]:
    """中文数字 → 整数（只处理百以内，账本够用）：「六」→6、「十五」→15、「二十」→20。"""
    if not tok:
        return None
    if "十" in tok:
        left, _, right = tok.partition("十")
        tens = _CN_NUM.get(left, 1) if left else 1
        ones = _CN_NUM.get(right, 0) if right else 0
        return tens * 10 + ones
    n = 0
    for ch in tok:
        d = _CN_NUM.get(ch)
        if d is None:
            return None
        n = n * 10 + d
    return n


_QTY_PREFIX = "约剩还只余仅大概有近摸多差不多"


def _qty_to_count(qty: Any) -> Optional[int]:
    """从存量描述里提取「件数」。提取不出返回 None（**不猜**）。

    「5 瓶」「六块」「只剩两口」→ 数字；「约五天份」「半袋」「每天一瓶」→ None。

    必须**锚定在描述开头**（只允许少量前缀词），否则会误伤：
    「约五天份，每天一瓶」里的"每天一瓶"会被当成存量 1 瓶 —— 那是速率不是余量，
    一个错的数字比对账的伤害比"没有数字"更大（会立刻触发误报）。
    """
    s = _s(qty).strip()
    if not s:
        return None
    pre = rf"^[{_QTY_PREFIX}]{{0,3}}\s*"
    m = re.match(pre + rf"(\d+)\s*([{_QTY_UNITS}])", s)
    if m:
        return int(m.group(1))
    m = re.match(pre + rf"([零〇一二两三四五六七八九十]+)\s*([{_QTY_UNITS}])", s)
    if m:
        return _cn_to_int(m.group(1))
    return None


def _norm_count(v: Any) -> Optional[int]:
    """任意值 → 可计数存量。无法确定就 None（**不猜、不填 0**）。"""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v if v >= 0 else None
    if isinstance(v, float):
        return int(v) if v >= 0 else None
    s = _s(v).strip()
    if not s:
        return None
    if re.fullmatch(r"\d+", s):
        return int(s)
    return _qty_to_count(s)


OptCount = Annotated[Optional[int], BeforeValidator(_norm_count)]

# ── 告警：同一类问题只提醒一次，避免刷屏 ──────────────────────────
_REPORTED: set = set()
# 「检查后写入」必须原子：三路校对是并行跑的，两路同时调用 _warn 时
# 都能通过 `key in _REPORTED` 检查，结果同一条告警被推两遍。
_REPORTED_LOCK = threading.Lock()


def _warn(key: str, message: str) -> None:
    with _REPORTED_LOCK:
        if key in _REPORTED:
            return
        _REPORTED.add(key)
    # 回调放在锁外：它会把消息塞进队列，不该占着去重表的锁
    try:
        llm.emit_notice("warn", message)
    except Exception:                                    # noqa: BLE001
        pass


def reset_warnings() -> None:
    """清空告警去重表。

    进程级共享的表，key 只与"值名"相关（如 `item.status.xxx`）。**新建故事时必须
    调用**：否则上一个故事触发过的同类告警，会在下一个故事里被永久屏蔽
    （原先它只被测试调用，网页端「新建故事」不会清）。
    """
    with _REPORTED_LOCK:
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


# ── 设定档案的状态与类型归一 ─────────────────────────────────────
# 设定档案记的是"这个地方/这条规则长什么样"这类**既成事实**，与物资最大的区别是
# 它没有"消耗"概念，只有"还算不算数"：便利店被烧了、某个组织解散了，那条设定就作废。
# 于是状态只有两个值。空值在**变更条目**里表示"本次不改动"（与物资一致），
# 在**账本条目**里表示"有效"（fail-open，老存档/漏写都按有效处理）。

SETTING_STATUS = Literal["active", "retired"]

_SETTING_STATUS_ALIASES = {
    "active": "active", "established": "active", "canon": "active", "valid": "active",
    "有效": "active", "已确立": "active", "确立": "active", "仍成立": "active",
    "retired": "retired", "gone": "retired", "destroyed": "retired", "abandoned": "retired",
    "废弃": "retired", "作废": "retired", "失效": "retired", "已毁": "retired",
    "毁坏": "retired", "烧毁": "retired", "拆除": "retired", "不再成立": "retired",
}


def _norm_setting_status(v: Any) -> Optional[str]:
    """设定状态归一。空值返回 None（= 本次不改动 / 交由上层补默认值）。"""
    raw = _s(v).strip()
    if not raw:
        return None
    key = raw.lower()
    if key in _SETTING_STATUS_ALIASES:
        return _SETTING_STATUS_ALIASES[key]
    if raw in _SETTING_STATUS_ALIASES:
        return _SETTING_STATUS_ALIASES[raw]
    _warn(f"setting.status.{key}",
          f"设定档案给出的状态「{raw}」不是合法值（只接受 active / retired），"
          f"本次不改动该设定状态。")
    return None


SETTING_KIND = Literal["place", "object", "rule", "relation", "other"]

_SETTING_KIND_ALIASES = {
    "place": "place", "location": "place", "地点": "place", "场所": "place",
    "位置": "place", "场景": "place", "地方": "place", "环境": "place",
    "object": "object", "物件": "object", "物品": "object", "道具": "object",
    "设施": "object", "建筑": "object", "固定物件": "object", "地形": "object",
    "rule": "rule", "规则": "rule", "世界规则": "rule", "力量规则": "rule",
    "机制": "rule", "约束": "rule", "设定法则": "rule",
    "relation": "relation", "关系": "relation", "组织": "relation",
    "势力": "relation", "人物关系": "relation", "阵营": "relation",
    "other": "other", "其他": "other", "其它": "other", "misc": "other",
}


def _norm_setting_kind(v: Any) -> str:
    """设定类型归一。空值返回 ""（= 本次不改动）；无法识别归 other 并告警。

    为什么不把无法识别的一律当 other 静默处理：类型会直接影响校对怎么用它
    ——把「规则」当「地点」，校对就会拿场景描写的标准去比，白报一串。
    """
    raw = _s(v).strip()
    if not raw:
        return ""
    key = raw.lower()
    k = _SETTING_KIND_ALIASES.get(key) or _SETTING_KIND_ALIASES.get(raw)
    if k:
        return k
    _warn(f"setting.kind.{key}",
          f"设定档案给出的类型「{raw}」不是合法值"
          f"（只接受 place / object / rule / relation / other），已按 other 处理。")
    return "other"


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

_ITEM_FIELD_ALIASES = {
    "name": ("item", "物品", "道具", "物件", "名称", "thing"),
    "count": ("数量", "件数", "余量", "num"),
    "unit": ("单位", "units"),
    "qty": ("存量", "剩余", "剩余量"),
    "note": ("说明", "备注", "理由", "原因"),
}


def _fill_aliases(data, aliases: dict):
    """把写在同义词键上的值搬到规范字段（规范字段已有值则不动）。

    模型给字段换名字是常态（item / 物品 / 名称，数量 / 件数，理由 / 原因…），
    这里统一收口，免得每个字段都写一遍 alias 逻辑。
    """
    if not isinstance(data, dict):
        return data
    out = dict(data)
    for canon, alts in aliases.items():
        if _s(out.get(canon)).strip():
            continue
        for a in alts:
            if _s(out.get(a)).strip():
                out[canon] = out[a]
                break
    return out


def _fill_item_aliases(data):
    return _fill_aliases(data, _ITEM_FIELD_ALIASES)


_SETTING_FIELD_ALIASES = {
    "name": ("setting", "地点", "场所", "场景", "设定", "名称", "location", "thing"),
    "detail": ("描述", "说明", "内容", "详情", "设定内容", "特征"),
    "kind": ("类型", "类别", "category", "type"),
    "status": ("状态",),
    "note": ("备注", "理由", "原因"),
}


def _fill_setting_aliases(data):
    return _fill_aliases(data, _SETTING_FIELD_ALIASES)


class Item(BaseModel):
    """账本里的一件物资（权威记录；撰稿端只能用账本内的东西）。

    为什么单独立账本而不是塞进人物快照：人物快照写的是"他是谁、他在想什么"，
    物资写的是"他手上还剩什么"。混在一起模型就会用写人物状态的笔法写物资，
    余量、数量这类关键信息全丢——而那恰恰是跨章比对要用的。

    count / unit 与 qty 的分工（2026-10-02 补，治"账目反复打回"）：
    - count + unit 是**机器可比**的件数（5 瓶 / 6 块），跨章对账只认这个；
    - qty 是给人看的描述，可写换算由来或"约三天份"这类模糊量。
    - 只给 qty 不给 count 时仍尽力提取（"六块" → 6），提取不到才留 None。
      留 None 不等于 0：它的意思是"这件东西数不清"，不要拿去参与加减。
    """
    name: Str = ""
    count: OptCount = None           # 可计数存量（件数）；None = 数不清
    unit: Str = ""                   # 件数单位：瓶/块/罐/包…
    qty: Str = ""                    # 当前存量描述（"半罐""约两天口粮"）
    status: ITEM_STATUS = "available"
    note: Str = ""                   # 来源 / 存放位置 / 其他说明
    chapter: Int = 0                 # 首次登记章节
    last_changed_chapter: Int = 0    # 最后一次变动章节

    @model_validator(mode="before")
    @classmethod
    def _aliases(cls, data):
        return _fill_item_aliases(data)

    @field_validator("status", mode="before")
    @classmethod
    def _coerce_status(cls, v):
        # 账本内的条目必须有确定状态；空值/非法值一律当 available（fail-open）
        return _norm_item_status(v) or "available"

    @model_validator(mode="after")
    def _derive_count(self):
        """只写了描述（"六块"）没写 count 时，把数字补上，让老存档/mock 也能对账。"""
        if self.count is None and self.qty:
            self.count = _qty_to_count(self.qty)
        return self


class ItemChange(BaseModel):
    """记忆结算给出的单条物资变动。留空的字段 = 本次不改动该项。

    与 Item 分开的原因：
    - "这个罐头还剩半罐"（改存量）和"这个罐头吃完了"（改状态）是两件事，
      合成一个模型就会出现"只改存量时状态被默认值覆盖成 available"的复活 bug。
    - 新物品也走这里：name 不在账本里 → 新建条目。
    """
    name: Str = ""
    count: OptCount = None                 # 变更后的件数；None = 清单没给数字
    unit: Str = ""                         # 件数单位（账本已有则沿用账本的）
    qty: Str = ""                          # 变更后的存量描述；空 = 不改动
    status: Optional[ITEM_STATUS] = None   # 变更后的状态；None = 不改动
    note: Str = ""                         # 变动原因（消耗/给出去向…），对账要用

    @model_validator(mode="before")
    @classmethod
    def _aliases(cls, data):
        return _fill_item_aliases(data)

    @field_validator("status", mode="before")
    @classmethod
    def _coerce_status(cls, v):
        return _norm_item_status(v)

    @model_validator(mode="after")
    def _derive_count(self):
        """模型常只给描述（"两块"）不给件数，这里把数字提出来供对账。"""
        if self.count is None and self.qty:
            self.count = _qty_to_count(self.qty)
        return self


# ── 设定档案（第四本账）─────────────────────────────────────────
# 第四次真模型实跑（2026-10-02）暴露的缺口：地点、物理设施、世界规则不属于
# 伏笔 / 人物 / 物资任何一本账。前情摘要只记情节，top-k 又只带上一章，
# 于是逻辑校对手里**没有判断"与前文一致"的依据**——它连着两轮把
# 「便利店卷帘门在落」判成 critical 设定矛盾，而第 1 章早就建立了卷帘门 + 后门铁门
# （正文没按它改是对的）。这是纯粹的误报，代价却是白烧重写轮次。
#
# 为什么不复用伏笔池：伏笔有的是"悬念"，会 open → progressing → resolved；
# 设定自确立起就**始终成立**，没有"揭晓"这回事，只在被销毁/废弃时才失效
# （status: active → retired）。两者的生命周期不同，混在一本账里，
# 伏笔的超期巡检会把所有地点都当成"埋太久没推进"报出来。

class Setting(BaseModel):
    """设定档案里的一条既成设定（权威记录）。

    与 Item 一样以 name 作主键（走 canon_name 归一匹配）：
    「便利店（镇口）」与「便利店」是同一处，不该裂成两条。
    """
    name: Str = ""
    detail: Str = ""                   # 具体描述：位置、物理特征、规则内容
    kind: SETTING_KIND = "other"       # place / object / rule / relation / other
    status: SETTING_STATUS = "active"  # active = 仍成立；retired = 已废弃/已毁
    note: Str = ""
    chapter: Int = 0                   # 确立章节
    last_mentioned_chapter: Int = 0    # 最后一次被写下或核对的章节

    @model_validator(mode="before")
    @classmethod
    def _aliases(cls, data):
        return _fill_setting_aliases(data)

    @field_validator("kind", mode="before")
    @classmethod
    def _coerce_kind(cls, v):
        # 账本条目必须有确定的类型；空值/非法值一律当 other（fail-open，不拒绝整条）
        return _norm_setting_kind(v) or "other"

    @field_validator("status", mode="before")
    @classmethod
    def _coerce_status(cls, v):
        return _norm_setting_status(v) or "active"


class SettingChange(BaseModel):
    """记忆结算给出的单条设定变动。留空的字段 = 本次不改动该项。

    与 Setting 分开的理由同 ItemChange：
    "这个地点补一条描写"（改 detail）与"这个地点被烧了"（改 status）是两件事，
    合在一起就会出现"只补描写时状态被默认值覆盖回 active"的复活 bug。
    """
    name: Str = ""
    detail: Str = ""                          # 变更后的描述；空 = 不改动
    kind: Str = ""                            # 变更后的类型；空 = 不改动
    status: Optional[SETTING_STATUS] = None   # 变更后的状态；None = 不改动
    note: Str = ""                            # 变动原因（本章新设/被毁/补充细节…）

    @model_validator(mode="before")
    @classmethod
    def _aliases(cls, data):
        return _fill_setting_aliases(data)

    @field_validator("kind", mode="before")
    @classmethod
    def _coerce_kind(cls, v):
        return _norm_setting_kind(v)          # 空 => ""（不改动）

    @field_validator("status", mode="before")
    @classmethod
    def _coerce_status(cls, v):
        return _norm_setting_status(v)        # 空 => None（不改动）


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
    # 开局设定档案：地点、固定设施、世界规则。
    # 为什么也要在策划阶段立：与物资同理——第 1 章自己"现编"的设定一旦被后续章节
    # 当正典复用，前后不一致就要等到很晚才被发现；先立档，第 1 章的撰稿人就有基准。
    initial_settings: List[Setting] = Field(default_factory=list)

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

    @field_validator("initial_settings", mode="before")
    @classmethod
    def _coerce_settings(cls, v):
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


# 每个记忆列表的"元素写成裸字符串时该归到哪个字段"。
# 为什么必须做这一层：**记忆库面板是让用户直接编辑 JSON 的**，
# 而 `{"items": ["矿泉水"]}` 这种简写会让 List[Item] 直接抛 ValidationError。
# 抛在 memory_settler_node 的解析处（那一步在 try 之外）就是整章崩掉——
# 与 models.py 的 fail-open 哲学正好相反：脏数据该被收敛，不该终止创作。
# 注意必须放模块级：类属性以 `_` 开头会被 Pydantic 认成私有属性（ModelPrivateAttr），
# 取出来不是 dict。
_MEMORY_LIST_HINTS = {"chapter_summaries": "summary", "foreshadow_pool": "desc",
                      "character_states": "name", "items": "name", "settings": "name"}


class Memory(BaseModel):
    """故事记忆库（全量权威数据）。"""
    chapter_summaries: List[ChapterSummary] = Field(default_factory=list)
    foreshadow_pool: List[Foreshadow] = Field(default_factory=list)
    character_states: List[CharacterState] = Field(default_factory=list)
    items: List[Item] = Field(default_factory=list)     # 物资道具账本
    settings: List[Setting] = Field(default_factory=list)  # 设定档案（地点/设施/规则）

    @field_validator("chapter_summaries", "foreshadow_pool",
                     "character_states", "items", "settings", mode="before")
    @classmethod
    def _coerce_lists(cls, v, info):
        return _dict_items(v, _MEMORY_LIST_HINTS[info.field_name])


class MemoryDelta(BaseModel):
    """记忆结算员每章输出的增量。"""
    summary: Str = ""
    new_foreshadows: List[Foreshadow] = Field(default_factory=list)
    resolved_foreshadows: List[str] = Field(default_factory=list)
    advanced_foreshadows: List[str] = Field(default_factory=list)
    character_updates: List[CharacterState] = Field(default_factory=list)
    item_changes: List[ItemChange] = Field(default_factory=list)
    setting_updates: List[SettingChange] = Field(default_factory=list)

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

    @field_validator("setting_updates", mode="before")
    @classmethod
    def _coerce_setting_updates(cls, v):
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
