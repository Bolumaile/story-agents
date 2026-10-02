"""端到端与降级路径测试（全离线，不调用任何真实模型）。

用 MockLLM 跑完整流水线，验证：
  · 两章能正常跑完，章节与记忆都入库
  · 第一章被校对打回后确实重写了一次（回退链路真的通）
  · 策划 Agent 只跑一次（续写不重跑策划）
  · 校对/记忆结算失败时按设计降级，不毁掉已写好的正文
"""
import config
import graph as graph_mod
import llm
import nodes
import pytest
from llm import JSONParseError

from conftest import initial_state


def _run_chapters(app, state, n):
    """按 main.py 的方式逐章推进（每章前重置单章字段）。"""
    for idx in range(1, n + 1):
        state.update({
            "chapter_index": idx,
            "chapter_draft": "",
            "review_comments": [],
            "review_verdict": "pass",
            "revision_round": 0,
            "final_chapter": "",
        })
        state = graph_mod.run_chapter(app, state)
    return state


# ── 全流程（Mock） ────────────────────────────────────────────
def test_mock_pipeline_runs_two_chapters(mock_mode):
    app = graph_mod.build_graph()
    state = _run_chapters(app, initial_state("写一个旧堤防悬疑故事"), 2)

    assert state["outline"].get("title")
    assert len(state["final_chapters"]) == 2
    for ch in state["final_chapters"]:
        assert ch["title"] and ch["text"].strip(), "定稿章节不能是空的"
    assert [c["index"] for c in state["final_chapters"]] == [1, 2]


def test_first_chapter_actually_gets_rewritten(mock_mode):
    """MockLLM 在 revision_round=0 时判 fail，所以第一章必然被重写一轮。

    这条是回退链路的取证：如果哪天路由被改坏（比如 fail 也直接放行），
    这里立刻会红。
    """
    app = graph_mod.build_graph()
    state = _run_chapters(app, initial_state(), 1)
    assert state["revision_round"] >= 1, "校对判 fail 后没有打回重写"


def test_memory_settled_after_chapter(mock_mode):
    app = graph_mod.build_graph()
    state = _run_chapters(app, initial_state(), 1)
    mem = state["memory"]
    assert mem["chapter_summaries"], "缺少章节摘要"
    assert mem["chapter_summaries"][0]["chapter"] == 1
    assert mem["foreshadow_pool"], "缺少伏笔记录"
    assert mem["character_states"], "缺少人物状态快照"


def test_outline_carries_over_to_next_chapter(mock_mode):
    """第二章必须复用第一章的策划案（同一本书的设定不能变）。"""
    app = graph_mod.build_graph()
    state = _run_chapters(app, initial_state(), 2)
    assert state["outline"]["chapters"][0]["index"] == 1
    assert state["outline"]["title"] == "测试小说"


def test_planner_runs_only_once_for_two_chapters(mock_mode, monkeypatch):
    """入口路由的核心保证：策划 Agent 全书只跑一次，续写不重跑。"""
    calls = {"n": 0}
    real_planner = nodes.planner_node

    def spy(state):
        calls["n"] += 1
        return real_planner(state)

    monkeypatch.setattr(nodes, "planner_node", spy)
    app = graph_mod.build_graph()
    _run_chapters(app, initial_state(), 2)
    assert calls["n"] == 1, f"策划跑了 {calls['n']} 次，应为 1 次"


def test_no_planner_when_outline_already_present(mock_mode, monkeypatch):
    """已有策划案直接进 writer（等价于"续写"场景）。"""
    calls = {"n": 0}

    def spy(state):
        calls["n"] += 1
        return {"outline": {}}

    monkeypatch.setattr(nodes, "planner_node", spy)
    app = graph_mod.build_graph()
    state = initial_state()
    state["outline"] = {"title": "已存在", "characters": [], "chapters": [{"index": 1, "title": "甲"}]}
    _run_chapters(app, state, 1)
    assert calls["n"] == 0


def test_mock_never_touches_network(mock_mode):
    """Mock 模式下不应构造真实 client——否则测试会依赖网络/Key。"""
    assert mock_mode.USE_MOCK is True
    assert mock_mode.chat("你是一位小说撰稿人（Writer）", "写一段") != ""


# ── 降级路径 ──────────────────────────────────────────────────
class _Boom:
    """替身：调用即抛 JSONParseError，模拟"模型输出救不回来"。"""

    def __init__(self, *a, **kw):
        raise JSONParseError("模拟无法解析的 JSON", "{ 坏掉的输出")


def test_reviewer_fails_open_by_default(monkeypatch):
    """校对结果解析不了 → 默认放行（不毁掉已写好的整章），并推一条告警。"""
    monkeypatch.setattr(config, "REVIEW_FAIL_OPEN", True)
    monkeypatch.setattr(llm, "chat_json", _Boom)

    notices = []
    llm.set_notice_cb(lambda level, msg: notices.append((level, msg)))

    out = nodes.reviewer_node({
        "chapter_index": 1,
        "chapter_draft": "正文",
        "outline": {"characters": []},
    })
    assert out["review_verdict"] == "pass"
    assert out["review_comments"] == []
    assert notices and notices[0][0] == "warn"
    assert "跳过校对" in notices[0][1]


def test_reviewer_raises_when_fail_open_disabled(monkeypatch):
    """REVIEW_FAIL_OPEN=0 时必须严格失败，不能悄悄放行。"""
    monkeypatch.setattr(config, "REVIEW_FAIL_OPEN", False)
    monkeypatch.setattr(llm, "chat_json", _Boom)

    with pytest.raises(JSONParseError):
        nodes.reviewer_node({
            "chapter_index": 1,
            "chapter_draft": "正文",
            "outline": {"characters": []},
        })


def test_reviewer_fail_without_critical_becomes_pass(monkeypatch):
    """判了 fail 却一条 critical 都没有 → 无可执行修改点，按 pass 处理（避免空转重写）。"""
    monkeypatch.setattr(llm, "chat_json", lambda *a, **kw: {
        "verdict": "fail",
        "comments": [{"type": "logic", "severity": "minor", "issue": "小问题"}],
    })
    notices = []
    llm.set_notice_cb(lambda level, msg: notices.append((level, msg)))

    out = nodes.reviewer_node({
        "chapter_index": 2,
        "chapter_draft": "正文",
        "outline": {"characters": []},
    })
    assert out["review_verdict"] == "pass"
    assert len(out["review_comments"]) == 1
    assert any("未指出任何严重问题" in m for _, m in notices)


def test_memory_settler_failure_does_not_break_chapter(monkeypatch):
    """记忆结算失败只告警，本章已定稿的值必须原样保留。"""
    monkeypatch.setattr(llm, "chat_json", _Boom)
    notices = []
    llm.set_notice_cb(lambda level, msg: notices.append((level, msg)))

    out = nodes.memory_settler_node({
        "chapter_index": 3,
        "final_chapter": "本章正文",
        "outline": {"chapters": [{"index": 3, "title": "第三章"}]},
    })
    assert out == {}                       # 不改动任何状态
    assert notices and "记忆结算失败" in notices[0][1]
