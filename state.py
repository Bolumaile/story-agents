"""全局共享状态：所有 Agent 共读共写这份数据，LangGraph 负责在节点间传递。"""
from typing import TypedDict, List, Dict, Any


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

    # —— 去 AI 味检测报告（仅提示，不参与流程判定）——
    style_report: Dict[str, Any]

    # —— 可选进阶材料（表单收集，撰写/校对/润色共用）——
    meta: Dict[str, Any]          # {style_sample, word_count, foreshadow, cliffhanger, forbidden_list}

    # —— 故事记忆（每章定稿后由记忆结算员更新）——
    memory: Dict[str, Any]        # {chapter_summaries, foreshadow_pool, character_states}

    # —— 累积记忆 ——
    final_chapters: List[Dict]    # 已定稿章节 [{"index", "title", "text"}]
