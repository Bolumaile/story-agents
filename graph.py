"""调度器：LangGraph StateGraph 组装流水线 + 校对回退路由。

流转图（多章连载共用一张图，入口路由保证策划只跑一次）：
  START ─(无策划案)→ planner → writer → reviewer ─(fail 且未超轮数)→ bump_round → writer（重写循环）
        └(已有策划案)→ writer                └─(pass 或超轮数)→ polisher → memory_settler → END
"""
import config
import nodes
from state import StoryState
from langgraph.graph import StateGraph, START, END


def route_entry(state: StoryState) -> str:
    """入口路由：策划案还没生成 → 先策划；已生成 → 直接撰稿（策划案即全局记忆）。"""
    return "planner" if not state.get("outline") else "writer"


def route_after_review(state: StoryState) -> str:
    """校对后的路由：严重问题打回撰稿；超轮数保护强制放行。"""
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
    g.add_node("writer", nodes.writer_node)
    g.add_node("reviewer", nodes.reviewer_node)
    g.add_node("bump_round", bump_revision_round)
    g.add_node("polisher", nodes.polisher_node)
    g.add_node("memory_settler", nodes.memory_settler_node)

    g.add_conditional_edges(START, route_entry,
                            {"planner": "planner", "writer": "writer"})
    g.add_edge("planner", "writer")
    g.add_edge("writer", "reviewer")
    g.add_conditional_edges("reviewer", route_after_review,
                            {"rewrite": "bump_round", "polish": "polisher"})
    g.add_edge("bump_round", "writer")          # 回退：重写循环
    g.add_edge("polisher", "memory_settler")    # 定稿后结算记忆
    g.add_edge("memory_settler", END)
    return g.compile()


def run_chapter(app, state: StoryState) -> StoryState:
    """跑完单章流水线（策划已定稿时可复用 state['outline']）。"""
    final = app.invoke(state)
    return final
