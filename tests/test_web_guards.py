"""网页端的并发、取消与生命周期守卫（P0-5 / P1-1 / P1-2 / P1-4 / P1-6）。

这些缺陷有个共同点：**只在并发或真实操作下才暴露**，静态读代码看不出来，
单跑一次"正常流程"也永远碰不到。所以用例尽量贴着真实路径写：
真去抢那把锁、真去调端点、真跑一遍 SSE 流水线（mock，不联网）。

覆盖：
- P0-5  运行中不许有别的端点改写全局模型配置（现在是明确的 409，不是静默写坏稿子）
- P1-1  「保存记忆修改」真的落盘到 _session.json
- P1-2  「停止生成」在章与章之间生效，且已定稿的章节保留
- P1-4  「新建故事」与运行中的任务不再竞态（旧会话不会"复活"）
- P1-6  告警去重表并发安全，且新建故事时清空（否则下个故事被永久屏蔽）
- P1-5  CLI 与网页端的初始状态 / 每章重置字段集必须一致（防止再次漂移）
"""
import json
import pathlib
import re
import threading

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import llm
import models
import state as state_mod
import web.server as server


@pytest.fixture(autouse=True)
def _clean_server_state(tmp_path, monkeypatch):
    """每个用例都用干净的会话、产物目录与取消信号，避免相互污染。"""
    monkeypatch.setattr(server, "OUTPUTS_DIR", str(tmp_path / "outputs_web"))
    monkeypatch.setattr(server, "SESSION_PATH", str(tmp_path / "outputs_web" / "_session.json"))
    old = dict(server._session)
    server._session.update({"state": None, "out_dir": None, "saved_at": None})
    server._cancel.clear()
    models.reset_warnings()
    yield
    server._session.update(old)
    server._cancel.clear()
    models.reset_warnings()


def _locked() -> bool:
    """当前是否有人持着「创作独占锁」（= 正在生成）。"""
    return server._lock.locked()


# ── P0-5：运行中不许改写全局配置 ─────────────────────────────────

def test_other_endpoints_get_409_while_generating():
    """worker 全程持锁，别的端点必须被明确拒绝，而不是悄悄改配置。"""
    assert server._lock.acquire(blocking=False)
    try:
        with pytest.raises(HTTPException) as ei:
            server._acquire_or_409("润色需求")
        assert ei.value.status_code == 409
        assert "正在生成中" in str(ei.value.detail)
    finally:
        server._lock.release()


def test_lock_is_free_again_after_the_run_finishes():
    """拒绝不是把锁"吃掉"：用完必须还回去，否则后面所有功能都不可用。"""
    assert server._lock.acquire(blocking=False)
    with pytest.raises(HTTPException):
        server._acquire_or_409("预览大纲")
    server._lock.release()
    server._acquire_or_409("预览大纲")          # 不抛 = 可用
    server._lock.release()


def test_session_clear_is_refused_while_generating():
    """P1-4：运行中点「新建故事」以前会让旧会话"复活"，现在直接拒绝。"""
    server._session["state"] = {"outline": {"title": "进行中的故事"}, "final_chapters": []}
    assert server._lock.acquire(blocking=False)
    try:
        with pytest.raises(HTTPException) as ei:
            server.api_session_clear()
        assert ei.value.status_code == 409
        assert server._session["state"] is not None, "被拒绝时不许真的清掉会话"
    finally:
        server._lock.release()


def test_generate_applies_settings_while_holding_the_lock(monkeypatch):
    """配置必须在 worker 的锁内应用。

    若有人在 `api_generate` 里（启动线程之前）改回"先 apply_llm_settings"，
    那么在 apply 与 worker 抢到锁之间的窗口里，另一个请求就能把配置换掉 ——
    正是"写到第 3 章时点一下润色，中途换模型/换 Key"的成因。
    """
    seen = {}
    real = server.apply_llm_settings

    def spy(*a, **kw):
        seen["locked"] = _locked()
        return real(*a, **kw)

    monkeypatch.setattr(server, "apply_llm_settings", spy)
    with TestClient(server.app) as client:
        client.post("/api/generate", json={"requirement": "测试", "chapters": 1,
                                           "mock": True, "mode": "new"})
    assert seen.get("locked") is True, "apply_llm_settings 又跑回锁外了"


# ── P1-2：停止生成 ──────────────────────────────────────────────

def test_stop_endpoint_sets_the_cancel_flag():
    assert not server._cancel.is_set()
    assert server.api_generate_stop()["ok"] is True
    assert server._cancel.is_set()


def test_stop_endpoint_works_while_generating():
    """停止端点刻意不取锁：不然它会在最需要的时候被自己的锁挡在门外。"""
    assert server._lock.acquire(blocking=False)
    try:
        assert server.api_generate_stop()["ok"] is True
        assert server._cancel.is_set()
    finally:
        server._lock.release()


def test_stop_takes_effect_at_chapter_boundary(monkeypatch):
    """点了停止之后不该再开新的一章，但要留下已定稿的内容。

    用「每章落盘时置位取消信号」来模拟用户在第一（或第二）章定稿后点停止：
    这是唯一能稳定复现"章与章之间退出"的时点。
    """
    real = server.products.write_chapter_products
    calls = []

    def spy(out_dir, state):
        real(out_dir, state)
        calls.append(1)
        server._cancel.set()

    monkeypatch.setattr(server.products, "write_chapter_products", spy)

    with TestClient(server.app) as client:
        body = client.post("/api/generate", json={
            "requirement": "测试", "chapters": 3, "mock": True, "mode": "new"}).text

    assert len(calls) == 1, "取消信号置位后不该再开始下一章"
    assert '"type": "stopped"' in body
    assert '"type": "all_done"' not in body
    assert '"total_chapters": 1' in body

    # 已定稿的那一章必须完整落盘
    runs = [p for p in pathlib.Path(server.OUTPUTS_DIR).iterdir() if p.is_dir()]
    assert len(runs) == 1
    assert sorted(p.name for p in runs[0].glob("chapter_*.md")) == ["chapter_01.md"]


def test_completed_run_still_reports_all_done():
    """没点停止时必须照旧报 all_done —— 别把正常收尾也说成"已停止"。"""
    with TestClient(server.app) as client:
        body = client.post("/api/generate", json={
            "requirement": "测试", "chapters": 1, "mock": True, "mode": "new"}).text
    assert '"type": "all_done"' in body
    assert '"type": "stopped"' not in body


# ── 端到端：产物与下发路径 ───────────────────────────────────────

def _sse_events(body: str):
    out = []
    for line in body.splitlines():
        if line.startswith("data: "):
            out.append(json.loads(line[6:]))
    return out


def test_generate_end_to_end_mock(tmp_path):
    """真跑一遍 SSE 端点：事件齐全、产物齐全、产物无破坏、路径不外泄。"""
    with TestClient(server.app) as client:
        body = client.post("/api/generate", json={
            "requirement": "短篇软末世，内向少年与流浪猫",
            "chapters": 2, "mock": True, "mode": "new"}).text

    events = _sse_events(body)
    kinds = [e["type"] for e in events]
    assert "start" in kinds and "chapter_done" in kinds and "all_done" in kinds
    assert kinds.count("chapter_done") == 2

    done = [e for e in events if e["type"] == "all_done"][0]
    assert done["total_chapters"] == 2

    # 界面下发的产物路径绝不能含盘符 / 用户名（历史上截图就是这么外泄的）
    assert not re.search(r"[A-Za-z]:[\\/]", done["out_dir"]), done["out_dir"]
    assert "Users" not in done["out_dir"] and "小项目" not in done["out_dir"]

    import pathlib
    runs = [p for p in pathlib.Path(server.OUTPUTS_DIR).iterdir() if p.is_dir()]
    assert len(runs) == 1
    run = runs[0]
    for name in ("outline.json", "memory.json", "chapter_01.md", "chapter_02.md", "final.md"):
        assert (run / name).exists(), f"缺产物 {name}"

    # 每章只能有一个一级标题（双标题 bug 的回归）
    for i in (1, 2):
        md = (run / f"chapter_{i:02d}.md").read_text(encoding="utf-8")
        assert len([ln for ln in md.splitlines() if ln.startswith("# ")]) == 1, md[:120]

    # 中途没有 .tmp 残留
    assert not list(run.glob("*.tmp"))

    # 会话已落盘，可续写
    saved = json.loads(pathlib.Path(server.SESSION_PATH).read_text(encoding="utf-8"))
    assert len(saved["state"]["final_chapters"]) == 2


def test_generate_refuses_without_requirement():
    with TestClient(server.app) as client:
        r = client.post("/api/generate", json={"requirement": "  ", "mock": True})
    assert r.status_code == 400


# ── P1-1：记忆库保存真的落盘 ─────────────────────────────────────

def test_session_memory_endpoint_persists_to_disk():
    server._session.update({
        "state": {"outline": {"title": "书"}, "final_chapters": []},
        "out_dir": None, "saved_at": None})

    r = server.api_session_memory(server.MemoryEditReq(
        memory={"items": ["矿泉水"], "foreshadow_pool": []}))

    assert r["ok"] is True
    assert server._session["state"]["memory"]["items"] == ["矿泉水"]
    saved = json.loads(open(server.SESSION_PATH, encoding="utf-8").read())
    assert saved["state"]["memory"]["items"] == ["矿泉水"], "必须真的写进 _session.json"
    assert saved["saved_at"]


def test_session_memory_endpoint_errors_without_a_session():
    with pytest.raises(HTTPException) as ei:
        server.api_session_memory(server.MemoryEditReq(memory={"items": []}))
    assert ei.value.status_code == 400
    assert not _locked(), "报错路径也要把锁还回去"


# ── P1-6：告警去重表 ────────────────────────────────────────────

def test_warning_dedup_is_reset_when_starting_a_new_story():
    seen = []
    llm.set_notice_cb(lambda level, msg: seen.append(msg))
    try:
        models._warn("item.status.xyz", "第一次提醒")
        models._warn("item.status.xyz", "同一个 key 第二次（应被去重挡住）")
        assert seen == ["第一次提醒"]

        server.api_session_clear()

        models._warn("item.status.xyz", "新故事里的同类告警")
        assert seen == ["第一次提醒", "新故事里的同类告警"], \
            "跨故事必须重新提醒，否则上个故事的告警会把下个故事永久屏蔽"
    finally:
        llm.set_notice_cb(None)


def test_warning_dedup_is_atomic_under_threads():
    """check-then-act 不加锁时，两线程会同时通过检查 → 同一条告警推两遍。"""
    seen = []
    llm.set_notice_cb(lambda level, msg: seen.append(msg))
    barrier = threading.Barrier(8)
    try:
        def work():
            barrier.wait()
            models._warn("concurrent.key", "同一条告警")

        threads = [threading.Thread(target=work) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert seen == ["同一条告警"]
    finally:
        llm.set_notice_cb(None)


# ── P1-5：CLI 与网页端的字段集不许漂 ───────────────────────────

def test_cli_and_web_initial_states_have_identical_keys():
    web = server._fresh_state(server.GenReq(requirement="测试"))
    cli = state_mod.new_state("测试")
    assert set(web) == set(cli)
    assert set(cli) == set(state_mod.StoryState.__annotations__), \
        "state.new_state 与 StoryState 的键集漂了（LangGraph 会拒绝未登记的键）"


def test_reset_chapter_fields_covers_every_per_chapter_field():
    """每章重置必须清掉三路校对 key 与三路 verdict —— 漏一个就会串章。"""
    reset = state_mod.reset_chapter_fields(3)
    assert reset["chapter_index"] == 3
    for key in ("chapter_draft", "review_comments", "review_verdict",
                "review_comments_ooc", "review_comments_logic", "review_comments_pacing",
                "review_verdict_ooc", "review_verdict_logic", "review_verdict_pacing",
                "style_report", "revision_round", "final_chapter"):
        assert key in reset, f"reset_chapter_fields 漏了 {key}"
    assert set(reset) <= set(state_mod.StoryState.__annotations__)


def test_reset_chapter_fields_hands_out_fresh_containers():
    """两次调用不能共享同一个列表实例，否则一章的清空会波及另一章。"""
    a, b = state_mod.reset_chapter_fields(1), state_mod.reset_chapter_fields(2)
    a["review_comments"].append({"x": 1})
    assert b["review_comments"] == []


def test_new_state_does_not_share_containers_between_instances():
    a, b = state_mod.new_state("甲"), state_mod.new_state("乙")
    a["final_chapters"].append({"index": 1})
    a["memory"]["items"] = ["矿泉水"]
    assert b["final_chapters"] == []
    assert b["memory"] == {}
    assert a["user_prompt"] == "甲" and b["user_prompt"] == "乙"
