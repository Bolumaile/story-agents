"""全局共享状态：所有 Agent 共读共写这份数据，LangGraph 负责在节点间传递。

本文件同时提供 **唯一的** 初始状态构造与「每章重置」定义（`new_state` /
`reset_chapter_fields`）。CLI（main.py）与网页端（web/server.py）都从这里取，
不再各写一份 —— 历史上「漏清空三路校对 key 导致上一章意见串进下一章」
「漏同步导致标题变《第2章》」都是重复维护造成的。

注意：`StoryState` 的键集合就是 LangGraph 的**通道白名单**。节点返回了
schema 里没有的键会直接报错，所以新增 state key 必须在这里登记。
"""
from typing import TypedDict, List, Dict, Any, Optional


class StoryState(TypedDict):
    # —— 用户输入 ——
    user_prompt: str              # 用户原始需求

    # —— 策划输出（全局故事信息，写多章时持续复用）——
    outline: Dict[str, Any]       # 大纲 + 人物卡 + 分章大纲

    # —— 单章流程 ——
    chapter_index: int            # 当前写的章节序号（从 1 起）
    chapter_draft: str            # 撰稿草稿
    review_comments: List[Dict]   # 校对反馈清单（三路合并后的最终结果）
    review_verdict: str           # "pass" / "fail"
    revision_round: int           # 当前章已重写轮数
    final_chapter: str            # 润色后定稿章节

    # —— 校对拆分为三路并行 specialist ——
    # 每个节点只写自己这一个 key。并行节点若同时写同一个 key 会互相覆盖，
    # 所以让它们各写各的，再由 review_merge 汇总成上面的 review_comments。
    review_comments_ooc: List[Dict]
    review_comments_logic: List[Dict]
    review_comments_pacing: List[Dict]

    # 每路各自的判定结果（"pass" / "fail" / "" = 该路本次没跑成）。
    # 为什么必须单独记录：模型明确判 fail、但 comments 一条都没解析出来时，
    # 只看 severity=="critical" 的合并逻辑会把它当成 pass 静默放行——
    # 而 llm._salvage_objects 专门从残缺 JSON 里抢救过这个 verdict。
    # 三个键分开写是刻意的：并行节点写同一个键会互相覆盖。
    review_verdict_ooc: str
    review_verdict_logic: str
    review_verdict_pacing: str

    # —— 去 AI 味检测报告（仅提示，不参与流程判定）——
    style_report: Dict[str, Any]

    # —— 可选进阶材料（表单收集，撰写/校对/润色共用）——
    meta: Dict[str, Any]          # {style_sample, word_count, foreshadow, cliffhanger, forbidden_list}

    # —— 故事记忆（每章定稿后由记忆结算员更新）——
    # {chapter_summaries, foreshadow_pool, character_states, items, settings}
    # 四本账都属「必带项」，写每章时必定带上：
    #   items    物资账本 —— 撰稿端只能用账本内的物资，已耗尽/已丢失的不许再用
    #   settings 设定档案 —— 已确立的地点/设施/世界规则，是判断"与前文是否一致"的依据
    memory: Dict[str, Any]

    # —— 累积记忆 ——
    final_chapters: List[Dict]    # 已定稿章节 [{"index", "title", "text"}]


# 每个新故事都要有的字段与初值。放成一份数据，new_state 与下面「每章重置」
# 各取所需，新增字段时只改这里、CLI 与 Web 自动同步。
_INITIAL_FIELDS: Dict[str, Any] = {
    "outline": {},
    "chapter_index": 1,
    "chapter_draft": "",
    "review_comments": [],
    "review_verdict": "pass",
    "review_comments_ooc": [],
    "review_comments_logic": [],
    "review_comments_pacing": [],
    "review_verdict_ooc": "pass",
    "review_verdict_logic": "pass",
    "review_verdict_pacing": "pass",
    "style_report": {},
    "revision_round": 0,
    "final_chapter": "",
    "meta": {},
    "memory": {},
    "final_chapters": [],
}


def _fresh(v: Any) -> Any:
    """复制容器初值：绝不把模块级那份可变对象（列表/字典）共享给调用方。"""
    if isinstance(v, dict):
        return dict(v)
    if isinstance(v, list):
        return list(v)
    return v


def new_state(user_prompt: str, meta: Optional[Dict[str, Any]] = None) -> StoryState:
    """构造一份全新的故事状态（CLI 与网页端共用）。

    之前 `main.build_initial_state` 与 `web.server._fresh_state` 各写一份，
    两边字段一旦不同步，就会出现「只在其中一个入口复现」的怪 bug。
    """
    state: Dict[str, Any] = {k: _fresh(v) for k, v in _INITIAL_FIELDS.items()}
    state["user_prompt"] = user_prompt
    state["meta"] = dict(meta or {})
    return state  # type: ignore[return-value]


def reset_chapter_fields(idx: int) -> Dict[str, Any]:
    """进入新一章前必须清空的字段。

    `review_comments_ooc/_logic/_pacing` 一定要一起清：三路 specialist 各写各的
    key，留下上一章的意见会被 merge_reviews 当成这一章的结论，直接误导撰稿人。
    """
    return {
        "chapter_index": idx,
        "chapter_draft": "",
        "review_comments": [],
        "review_verdict": "pass",
        "review_comments_ooc": [],
        "review_comments_logic": [],
        "review_comments_pacing": [],
        "review_verdict_ooc": "pass",
        "review_verdict_logic": "pass",
        "review_verdict_pacing": "pass",
        "style_report": {},
        "revision_round": 0,
        "final_chapter": "",
    }
