"""调度器：LangGraph StateGraph 组装流水线 + 校对回退路由。

流转图（多章连载共用一张图，入口路由保证策划只跑一次）：

  START ─(无策划案)→ planner ─┐
        └(已有策划案)─────────┴→ chapter_planner → writer ─┬→ reviewer_ooc   ─┐
                                                           ├→ reviewer_logic ─┼→ merge_reviews
                                                           └→ reviewer_pacing ─┘        │
                                                                                        ├─(fail 且未超轮)→ bump_round → writer
                                                                                        └─(pass 或超轮)──→ polisher → memory_settler → END

chapter_planner 的分工：策划案只覆盖到第 N 章。续写超出 N 时由它为本章补一份大纲
（标题 / 梗概 / 转折 / 场景节拍）；本章已在策划案内时它什么都不做。
为什么不写在 _chapter_plan() 里：那个函数每章被调用 7 次，在里面发 LLM 请求会重复规划。

校对为什么拆成三路并行（优化建议第 7 条）：
原先一个 prompt 里塞 4 个检查维度，再叠加 comments ≤ 5 的篇幅上限，
维度之间互相挤占输出额度，模型只会报最显眼的一两个问题，其余维度形同虚设。
拆开后每路只盯一个维度，额度专款专用。

**每个 specialist 只写自己的 state key**（review_comments_ooc / _logic / _pacing）：
并行节点若同时写同一个 key 会互相覆盖，所以各写各的，再由 merge_reviews 汇合。
"""
import config
import nodes
from state import StoryState
from langgraph.graph import StateGraph, START, END


def route_entry(state: StoryState) -> str:
    """入口路由：策划案还没生成 → 先策划；已生成 → 先去核对本章是否在策划案范围内。

    注意这里**不再直接进 writer**：续写时 chapter_index 可能超出策划案的章节数，
    需要先让 chapter_planner 给本章补大纲（否则标题会退化成「第N章」、正文也没有分章梗概）。
    策划案里已有本章时，chapter_planner 是个空操作，不产生额外调用。
    """
    return "planner" if not state.get("outline") else "chapter_planner"


def route_after_review(state: StoryState) -> str:
    """三路校对汇合后的路由：有严重问题打回撰稿；超轮数保护强制放行。"""
    if (state.get("review_verdict") == "fail"
            and state.get("revision_round", 0) < config.MAX_REVISION_ROUNDS):
        return "rewrite"
    return "polish"


def bump_revision_round(state: StoryState) -> dict:
    """被打回时轮数 +1（作为独立节点，保证状态更新次序清晰）。"""
    return {"revision_round": state.get("revision_round", 0) + 1}


def build_graph():
    g = StateGraph(StoryState)
    g.add_node("planner", nodes.planner_node)
    g.add_node("chapter_planner", nodes.chapter_planner_node)
    g.add_node("writer", nodes.writer_node)
    g.add_node("reviewer_ooc", nodes.reviewer_ooc_node)
    g.add_node("reviewer_logic", nodes.reviewer_logic_node)
    g.add_node("reviewer_pacing", nodes.reviewer_pacing_node)
    g.add_node("merge_reviews", nodes.merge_reviews_node)
    g.add_node("bump_round", bump_revision_round)
    g.add_node("polisher", nodes.polisher_node)
    g.add_node("memory_settler", nodes.memory_settler_node)

    g.add_conditional_edges(START, route_entry,
                            {"planner": "planner",
                             "chapter_planner": "chapter_planner"})
    g.add_edge("planner", "chapter_planner")
    # chapter_planner 只管"本章有没有大纲"，有就直接放行，没有才补。
    # 重写循环（bump_round → writer）刻意绕开它：此时大纲早已补好，不必再判断一次。
    g.add_edge("chapter_planner", "writer")

    # 三路校对并行：writer 同时指向三个 specialist
    g.add_edge("writer", "reviewer_ooc")
    g.add_edge("writer", "reviewer_logic")
    g.add_edge("writer", "reviewer_pacing")
    # 扇入：三路都跑完，merge_reviews 才会执行
    g.add_edge("reviewer_ooc", "merge_reviews")
    g.add_edge("reviewer_logic", "merge_reviews")
    g.add_edge("reviewer_pacing", "merge_reviews")

    g.add_conditional_edges("merge_reviews", route_after_review,
                            {"rewrite": "bump_round", "polish": "polisher"})
    g.add_edge("bump_round", "writer")          # 回退：重写循环
    g.add_edge("polisher", "memory_settler")    # 定稿后结算记忆
    g.add_edge("memory_settler", END)
    return g.compile()


def run_chapter(app, state: StoryState) -> StoryState:
    """跑完单章流水线（策划已定稿时可复用 state['outline']）。"""
    final = app.invoke(state)
    return final
