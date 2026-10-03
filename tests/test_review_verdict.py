"""校对 verdict 不再被丢弃（P0-3）。

缺陷回顾：`_run_review` 只取 `result["comments"]`，把模型明确给出的 `verdict`
整条丢掉；而 `llm._salvage_objects` 还专门从残缺 JSON 里把 `"verdict":"fail"`
抢救出来。于是**模型判「不合格、要重写」的章节，只要 comments 没解析出来，
就被静默放行进入润色** —— 用户侧连一条 warning 都没有。这是核心质检闸门的漏洞。

现在：每路把 verdict 写进自己的 state key（并行节点共写一个键会互相覆盖），
`merge_reviews` 把「有 critical」或「任一路明确 fail」都判 fail，并对
"报了 fail 却给不出具体意见"的情况显式告警。

同时钉住一条边界：**某一路解析失败返回 `""` ≠ fail**。那是"本次没跑成"，
`REVIEW_FAIL_OPEN=True` 的语义就是跳过它，不能因此把整章打回。
"""
import pytest

import config
import llm
import nodes

from conftest import initial_state


@pytest.fixture
def notices():
    """收集 emit_notice 出来的告警。"""
    box = []
    llm.set_notice_cb(lambda level, msg: box.append((level, msg)))
    yield box
    llm.set_notice_cb(None)


def _run_route(route: str, result, raises: bool = False):
    """把某一路的 LLM 输出钉死，直接跑那个节点。"""
    orig = llm.chat_json

    def fake(*a, **kw):
        if raises:
            raise llm.JSONParseError("模型输出的内容不是合法 JSON", "…")
        return result

    llm.chat_json = fake
    try:
        state = initial_state()
        state["chapter_draft"] = "正文"
        state["outline"] = {"chapters": [{"index": 1, "title": "第一章"}]}
        return getattr(nodes, f"reviewer_{route}_node")(state)
    finally:
        llm.chat_json = orig


def _merge(**kw):
    state = initial_state()
    state.update(kw)
    return nodes.merge_reviews_node(state)


CRITICAL = {"type": "ooc", "severity": "critical",
            "quote": "林岸说出了家庭住址", "issue": "违反禁忌", "suggestion": "改为沉默"}


# ── 每路都把自己的 verdict 写进 state ──────────────────────────

def test_each_route_records_its_verdict():
    assert _run_route("ooc", {"verdict": "pass", "comments": []})["review_verdict_ooc"] == "pass"
    assert _run_route("logic", {"verdict": "fail", "comments": []})["review_verdict_logic"] == "fail"
    assert _run_route("pacing", {"verdict": "pass", "comments": []})["review_verdict_pacing"] == "pass"


def test_verdict_is_normalized_case_insensitively():
    assert _run_route("logic", {"verdict": " FAIL ", "comments": []})["review_verdict_logic"] == "fail"
    assert _run_route("logic", {"verdict": "PASS"})["review_verdict_logic"] == "pass"


def test_missing_or_garbled_verdict_becomes_unstated():
    """模型没写或写不清楚 → 空串（未表态），不能瞎猜成 pass 或 fail。"""
    assert _run_route("logic", {"comments": []})["review_verdict_logic"] == ""
    assert _run_route("logic", {"verdict": "还行吧"})["review_verdict_logic"] == ""


def test_route_failure_yields_empty_verdict_and_no_comments():
    """解析失败走 fail-open：comments 空 + verdict 空（≠ pass，也≠ fail）。"""
    out = _run_route("logic", None, raises=True)
    assert out == {"review_comments_logic": [], "review_verdict_logic": ""}


def test_pacing_disabled_route_is_unstated():
    orig = config.REVIEW_PACING_ENABLED
    config.REVIEW_PACING_ENABLED = False
    try:
        assert _run_route("pacing", {"verdict": "fail"}) == {
            "review_comments_pacing": [], "review_verdict_pacing": ""}
    finally:
        config.REVIEW_PACING_ENABLED = orig


# ── 合并判定 ────────────────────────────────────────────────────

def test_critical_still_forces_fail(notices):
    out = _merge(review_comments_ooc=[CRITICAL], review_verdict_ooc="fail")
    assert out["review_verdict"] == "fail"
    assert any("严重问题" in m for _lv, m in notices)


def test_route_fail_without_any_comment_is_not_silently_passed(notices):
    """P0-3 的核心用例：这就是修复前会被静默放行的那种输入。"""
    out = _merge(review_verdict_logic="fail", review_comments_logic=[])
    assert out["review_verdict"] == "fail", "模型明确判 fail，不能因为没有 critical 就放行"
    assert out["review_comments"] == []
    # 而且必须留下痕述，不能又变成一次静默
    assert any("但没有给出任何具体问题" in m for _lv, m in notices)
    assert any(lv == "warn" for lv, _m in notices)


def test_unstated_route_does_not_force_fail():
    """'' = 本次没跑成（fail-open 跳过），不能算 fail。"""
    out = _merge(review_verdict_ooc="", review_verdict_logic="", review_verdict_pacing="",
                 review_comments_ooc=[], review_comments_logic=[], review_comments_pacing=[])
    assert out["review_verdict"] == "pass"


def test_missing_verdict_keys_default_to_pass():
    """老存档 / 不带新键的 state：不能因为读不到 key 就判 fail。"""
    assert _merge()["review_verdict"] == "pass"


def test_minor_only_comments_do_not_fail():
    out = _merge(review_comments_ooc=[{"type": "ooc", "severity": "minor", "issue": "小问题"}],
                 review_verdict_ooc="pass")
    assert out["review_verdict"] == "pass"


def test_merge_marks_source_route_and_orders_critical_first():
    out = _merge(
        review_comments_logic=[{"type": "logic", "severity": "minor", "issue": "次要"}],
        review_verdict_logic="pass",
        review_comments_pacing=[{**CRITICAL, "type": "pacing"}],
        review_verdict_pacing="fail")
    assert out["review_comments"][0]["severity"] == "critical"
    assert out["review_comments"][0]["from"] == "节奏/篇幅"
    assert out["review_comments"][1]["from"] == "逻辑/时间线/设定"


def test_critical_and_unstated_route_reports_both(notices):
    out = _merge(review_comments_ooc=[CRITICAL], review_verdict_ooc="fail",
                 review_verdict_pacing="fail", review_comments_pacing=[])
    assert out["review_verdict"] == "fail"
    assert any("未给出具体意见" in m for _lv, m in notices)


# ── 端到端：fail 真的能驱动重写 ─────────────────────────────────

def test_silent_fail_drives_rewrite_through_graph(monkeypatch):
    """把三路都钉成"fail 但没意见"，图必须走重写分支而不是直接去润色。"""
    import graph as graph_mod

    monkeypatch.setattr(config, "MAX_REVISION_ROUNDS", 3)
    state = initial_state()
    state["chapter_draft"] = "正文"
    state["outline"] = {"title": "书", "chapters": [{"index": 1, "title": "第一章"}]}

    state.update({
        "review_comments_ooc": [], "review_verdict_ooc": "fail",
        "review_comments_logic": [], "review_verdict_logic": "fail",
        "review_comments_pacing": [], "review_verdict_pacing": "fail",
    })
    out = nodes.merge_reviews_node(state)
    state.update(out)
    assert graph_mod.route_after_review(state) == "rewrite"
