"""钉住「空操作节点上报 None」这个坑 —— 2026-10-02 web 端实跑中断的根因。

现象：看板刚打出「🧭 策划完成」，紧接着就是「⚠ 出错中断」，
      `TypeError: 'NoneType' object is not iterable`，run 目录一片空白。

机理：LangGraph 在 stream_mode="updates" 下，对「一个字段都没写」的节点
      上报的是 `{node: None}`，而不是空字典。而本项目存在合法的空操作节点：

        · chapter_planner_node —— 本章已在策划案范围内时 `return {}`
        · memory_settler_node  —— 记忆结算失败降级时 `return {}`

      原先 web/server.py 的消费循环直接 `state.update(delta)`，
      于是 `dict.update(None)` → TypeError，整条流水线当场断掉。

为什么离线测试当时全绿：CLI（main.py）走的是 `app.invoke`，根本不经过
「逐节点消费 updates」这条路径，踩不到。所以这里必须直接测
`web.server._merge_update` 与「复刻 worker 的消费循环」两件事。
"""
import pytest

import graph as graph_mod
import llm
import web.server as server

from conftest import initial_state


def _consume(app, state, chapters):
    """复刻 web/server.py worker 里的消费循环（含事件分支与空值守卫）。

    刻意与 server 源码保持同构：这样它就不再是「另写一份」，而是那条真实
    路径的离线替身——消费逻辑一旦改回 state.update(delta) 直接调用，这里会红。
    """
    events = []
    for idx in range(1, chapters + 1):
        state.update({
            "chapter_index": idx, "chapter_draft": "",
            "review_comments": [], "review_verdict": "pass",
            "review_comments_ooc": [], "review_comments_logic": [],
            "review_comments_pacing": [], "style_report": {},
            "revision_round": 0, "final_chapter": "",
        })
        events.append(("chapter_start", idx))
        for update in app.stream(state, stream_mode="updates"):
            for node, delta in update.items():
                if not server._merge_update(state, delta):
                    events.append((node, delta))          # 空操作：跳过，不并 state
                    continue
                if node == "planner":
                    events.append(("planner_done", state["outline"]["title"]))
                elif node == "writer":
                    events.append(("writer_done", idx))
                elif node == "merge_reviews":
                    events.append(("review", state["review_verdict"]))
                elif node == "bump_round":
                    events.append(("rewrite", state["revision_round"]))
        events.append(("chapter_done", state["final_chapters"][-1]["title"]))
    return events


# ── 消费函数本身 ────────────────────────────────────────────────

def test_merge_update_tolerates_none():
    """LangGraph 给空操作节点的 delta 是 None：不能抛，且 state 不许被改动。"""
    state = {"a": 1}
    assert server._merge_update(state, None) is False
    assert state == {"a": 1}


def test_merge_update_tolerates_empty_dict():
    """空字典同义（不同 LangGraph 版本可能给 {} 而不是 None），同样跳过。"""
    state = {"a": 1}
    assert server._merge_update(state, {}) is False
    assert state == {"a": 1}


def test_merge_update_applies_real_delta():
    state = {"a": 1}
    assert server._merge_update(state, {"b": 2}) is True
    assert state == {"a": 1, "b": 2}


# ── 真实图的消费路径 ───────────────────────────────────────────

def test_stream_reports_none_for_noop_chapter_planner(mock_mode):
    """空操作节点确实会上报 None —— 这是测试有意义的前提（不是空跑）。"""
    state = initial_state()
    # 本章已在策划案内 → chapter_planner 走 return {} 的空操作分支
    state["outline"] = {"title": "已有策划案",
                        "chapters": [{"index": 1, "title": "第一章"}]}
    app = graph_mod.build_graph()

    noop = []
    for update in app.stream(state, stream_mode="updates"):
        for node, delta in update.items():
            if not delta:
                noop.append((node, delta))
            state.update(delta or {})

    assert ("chapter_planner", None) in noop, (
        "LangGraph 的上报形态变了（不再给 None）：请确认消费端的守卫是否仍必要")


def test_web_consumer_survives_noop_node(mock_mode):
    """修复点：空操作节点不再让 web 端消费循环崩掉，本章照常定稿。"""
    state = initial_state()
    state["outline"] = {"title": "已有策划案",
                        "chapters": [{"index": 1, "title": "第一章"}]}
    app = graph_mod.build_graph()

    # 修复前这里会抛 TypeError: 'NoneType' object is not iterable
    events = _consume(app, state, chapters=1)

    assert ("chapter_planner", None) in events        # 空操作被跳过
    assert ("chapter_done", "第一章") in events
    assert len(state["final_chapters"]) == 1


def test_web_consumer_full_pipeline_from_scratch(mock_mode):
    """从零跑 3 章（策划案只给 1 章，后两章走补纲），消费循环全程不崩。"""
    state = initial_state()
    app = graph_mod.build_graph()

    events = _consume(app, state, chapters=3)

    titles = [v for (k, v) in events if k == "chapter_done"]
    assert len(titles) == 3
    assert titles[0] == "雨夜来客"                    # 策划案里的
    assert titles[1] != "第2章" and titles[2] != "第3章"   # 不再退化成流水号
    # 每章都必须真的走过：撰稿 → 校对汇合 → 定稿
    for idx in (1, 2, 3):
        assert ("writer_done", idx) in events
        assert ("chapter_done", titles[idx - 1]) in events
    assert any(k == "review" for k, _ in events)
    # 物资账本随章推进（本仓库的账本四段贯通用例之一）
    assert state["memory"]["items"]
