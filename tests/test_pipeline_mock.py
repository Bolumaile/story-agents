"""端到端与降级路径测试（全离线，不调用任何真实模型）。

用 MockLLM 跑完整流水线，验证：
  · 两章能正常跑完，章节与记忆都入库
  · 第一章被校对打回后确实重写了一次（回退链路真的通）
  · 策划 Agent 只跑一次（续写不重跑策划）
  · 三路并行校对确实都在跑
  · 校对/记忆结算失败时按设计降级，不毁掉已写好的正文
  · 伏笔状态机：去重、推进、回收、超期巡检
"""
import config
import graph as graph_mod
import llm
import nodes
import pytest
from llm import JSONParseError

from conftest import initial_state


def _run_chapters(app, state, n):
    """按 web/server.py 的方式逐章推进（每章前重置单章字段）。

    三个 review_comments_* 是并行校对各写各的 key，**必须随章清空**，
    否则上一章的意见会漏进下一章的合并结果里。
    """
    for idx in range(1, n + 1):
        state.update({
            "chapter_index": idx,
            "chapter_draft": "",
            "review_comments": [],
            "review_verdict": "pass",
            "review_comments_ooc": [],
            "review_comments_logic": [],
            "review_comments_pacing": [],
            "style_report": {},
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


def test_all_three_reviewers_actually_run(mock_mode, monkeypatch):
    """三路并行校对必须都执行。

    少跑一路不会有任何报错，只表现为「校对好像没以前严了」——
    这种退化最难被发现，所以用 spy 直接盯住。
    """
    calls = []
    for name in ("reviewer_ooc_node", "reviewer_logic_node", "reviewer_pacing_node"):
        real = getattr(nodes, name)

        def spy(state, _real=real, _name=name):
            calls.append(_name)
            return _real(state)

        monkeypatch.setattr(nodes, name, spy)

    app = graph_mod.build_graph()
    _run_chapters(app, initial_state(), 1)
    assert sorted(set(calls)) == [
        "reviewer_logic_node", "reviewer_ooc_node", "reviewer_pacing_node"], \
        f"三路校对没有全部执行，实际跑了：{sorted(set(calls))}"


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


# ── 场景节拍真的送到了撰稿人手里 ──────────────────────────────
def test_beats_reach_the_writer_prompt(mock_mode, monkeypatch):
    """策划给出的节拍必须出现在撰稿请求里，否则这层等于白加。"""
    seen = {}
    real_chat = llm.chat

    def spy(system, user, *a, **kw):
        if "Writer" in system:
            seen["prompt"] = user
        return real_chat(system, user, *a, **kw)

    monkeypatch.setattr(llm, "chat", spy)
    app = graph_mod.build_graph()
    _run_chapters(app, initial_state(), 1)
    # Mock 策划案里第一个 beat 的目标
    assert "场景节拍" in seen.get("prompt", ""), "撰稿请求里没有节拍段落"
    assert "林岸在雨夜独处" in seen["prompt"], "节拍内容没有传下去"


# ── 降级路径 ──────────────────────────────────────────────────
class _Boom:
    """替身：调用即抛 JSONParseError，模拟"模型输出救不回来"。"""

    def __init__(self, *a, **kw):
        raise JSONParseError("模拟无法解析的 JSON", "{ 坏掉的输出")


def test_one_route_failing_does_not_kill_the_chapter(monkeypatch):
    """某一路校对解析不了 → 该路跳过、其余照常，不毁掉已写好的整章。"""
    monkeypatch.setattr(config, "REVIEW_FAIL_OPEN", True)
    monkeypatch.setattr(llm, "chat_json", _Boom)

    notices = []
    llm.set_notice_cb(lambda level, msg: notices.append((level, msg)))

    out = nodes.reviewer_ooc_node({
        "chapter_index": 1,
        "chapter_draft": "正文",
        "outline": {"characters": []},
    })
    assert out["review_comments_ooc"] == []
    assert notices and notices[0][0] == "warn"
    assert "跳过" in notices[0][1]


def test_route_raises_when_fail_open_disabled(monkeypatch):
    """REVIEW_FAIL_OPEN=0 时必须严格失败，不能悄悄放行。"""
    monkeypatch.setattr(config, "REVIEW_FAIL_OPEN", False)
    monkeypatch.setattr(llm, "chat_json", _Boom)

    with pytest.raises(JSONParseError):
        nodes.reviewer_logic_node({
            "chapter_index": 1,
            "chapter_draft": "正文",
            "outline": {"characters": []},
        })


def test_one_route_down_still_lets_others_report(monkeypatch):
    """一路挂了，另外两路的意见仍要能汇总上来。"""
    monkeypatch.setattr(config, "REVIEW_FAIL_OPEN", True)

    def half_broken(system, user, *a, **kw):
        if "ReviewerOOC" in system:
            raise JSONParseError("这一路坏了", "{}")
        return {"verdict": "pass", "comments": [
            {"type": "logic", "severity": "minor", "issue": "来自逻辑路"}]}

    monkeypatch.setattr(llm, "chat_json", half_broken)
    llm.set_notice_cb(lambda level, msg: None)

    ooc = nodes.reviewer_ooc_node({
        "chapter_index": 1, "chapter_draft": "正文",
        "outline": {"characters": []}})
    logic = nodes.reviewer_logic_node({
        "chapter_index": 1, "chapter_draft": "正文",
        "outline": {"characters": []}})

    merged = nodes.merge_reviews_node({
        "chapter_index": 1, "outline": {"chapters": []},
        "review_comments_ooc": ooc["review_comments_ooc"],
        "review_comments_logic": logic["review_comments_logic"],
        "review_comments_pacing": [],
    })
    assert merged["review_verdict"] == "pass"
    assert len(merged["review_comments"]) == 1
    assert merged["review_comments"][0]["from"] == "逻辑/时间线/设定"


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


# ── 伏笔状态机 ────────────────────────────────────────────────
def _settler_with(delta, memory, idx=2):
    import llm as _llm
    orig = _llm.chat_json
    _llm.chat_json = lambda *a, **kw: delta
    try:
        return nodes.memory_settler_node({
            "chapter_index": idx,
            "final_chapter": "正文",
            "outline": {"chapters": [{"index": idx, "title": f"第{idx}章"}]},
            "memory": memory,
        })
    finally:
        _llm.chat_json = orig


def test_foreshadow_ids_are_not_duplicated(mock_mode):
    """Mock 每章都会返回同形的伏笔 id，去重逻辑必须生效。

    重复 id 会让"回收/推进"指向歧义条目，长篇的伏笔链就乱了。
    """
    app = graph_mod.build_graph()
    state = _run_chapters(app, initial_state(), 3)
    ids = [f["id"] for f in state["memory"]["foreshadow_pool"]]
    assert len(ids) == len(set(ids)), f"伏笔 id 重复：{ids}"
    assert ids == ["F1", "F2", "F3"]


def test_advanced_foreshadow_gets_marked_progressing():
    """出现在 advanced_foreshadows 里的伏笔：open → progressing，并记录推进章。"""
    out = _settler_with(
        {"summary": "s", "new_foreshadows": [], "resolved_foreshadows": [],
         "advanced_foreshadows": ["F1"], "character_updates": []},
        {"chapter_summaries": [], "character_states": [],
         "foreshadow_pool": [{"id": "F1", "desc": "d", "chapter": 1, "status": "open"}]})
    f = out["memory"]["foreshadow_pool"][0]
    assert f["status"] == "progressing"
    assert f["last_advanced_chapter"] == 2


def test_resolved_foreshadow_is_closed():
    out = _settler_with(
        {"summary": "s", "new_foreshadows": [], "advanced_foreshadows": [],
         "resolved_foreshadows": ["F1"], "character_updates": []},
        {"chapter_summaries": [], "character_states": [],
         "foreshadow_pool": [{"id": "F1", "desc": "d", "chapter": 1, "status": "progressing"}]})
    assert out["memory"]["foreshadow_pool"][0]["status"] == "resolved"


def test_new_foreshadow_records_burying_chapter():
    out = _settler_with(
        {"summary": "s", "new_foreshadows": [{"id": "F9", "desc": "新埋的"}],
         "resolved_foreshadows": [], "advanced_foreshadows": [], "character_updates": []},
        {"chapter_summaries": [], "foreshadow_pool": [], "character_states": []}, idx=4)
    f = out["memory"]["foreshadow_pool"][0]
    assert (f["id"], f["chapter"], f["status"]) == ("F9", 4, "open")
    assert f["last_advanced_chapter"] == 4, "刚埋下就算一次推进，否则下一章立刻被判超期"


def test_bad_foreshadow_status_is_normalized_not_propagated():
    """记忆里混进非法 status → 收敛成 open 并告警，绝不把坏值写回 state。"""
    notices = []
    llm.set_notice_cb(lambda level, msg: notices.append((level, msg)))
    out = _settler_with(
        {"summary": "s", "new_foreshadows": [], "resolved_foreshadows": [],
         "advanced_foreshadows": [], "character_updates": []},
        {"chapter_summaries": [], "character_states": [],
         "foreshadow_pool": [{"id": "F1", "desc": "d", "chapter": 1,
                              "status": "乱七八糟的状态"}]})
    assert out["memory"]["foreshadow_pool"][0]["status"] == "open"
    assert any("不是合法值" in m for _, m in notices)


def test_stale_foreshadow_triggers_warning(mock_mode):
    """跑够章数后，「埋太久没推进」的伏笔必须报警——这是防烂尾的兜底。

    Mock 每章新增一条伏笔、且从不推进旧伏笔，所以第 1 章埋下的那条
    会在第 (阈值+1) 章触发提醒。
    """
    notices = []
    llm.set_notice_cb(lambda level, msg: notices.append((level, msg)))
    app = graph_mod.build_graph()
    _run_chapters(app, initial_state(), config.FORESHADOW_STALE_CHAPTERS + 1)
    msgs = [m for _, m in notices]
    assert any("未推进" in m for m in msgs), f"没触发伏笔超期告警，实际告警：{msgs}"


def test_style_report_is_produced(mock_mode):
    """去 AI 味规则层要产出统计（即使没超阈值也要有数据，供看板展示）。"""
    app = graph_mod.build_graph()
    state = _run_chapters(app, initial_state(), 1)
    rep = state.get("style_report") or {}
    assert rep, "没有产出风格体检报告"
    assert rep["stats"]["chars"] > 0
