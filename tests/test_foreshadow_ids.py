"""伏笔自动编号不许撞号（P0-6）。

缺陷回顾：`memory_settler_node` 里自动编号写的是 `f"F{len(pool) + 1}"`，
隐含「编号连续且无缺口」。但模型完全可以给出 `F1`、`F3` 这种不连续编号，
或经过去重/手工编辑后留下缺口。此时 `len + 1` 会撞上既有编号，走进
「重复 id」分支 —— **改写旧伏笔的描述、丢弃新伏笔，而且全程没有任何告警**。

伏笔账本是长篇的"埋线"，被悄悄换掉要等读到后面才会发现，所以这条必须有回归。
"""
import llm
import nodes

from conftest import initial_state


def _mem(pool_ids):
    return {
        "chapter_summaries": [],
        "foreshadow_pool": [
            {"id": i, "desc": f"{i} 的原描述", "chapter": 1, "status": "open"}
            for i in pool_ids],
        "character_states": [],
        "items": [],
        "settings": [],
    }


def _settle(new_foreshadows, pool_ids, idx=2):
    """把结算员的 LLM 输出钉成固定 delta，直接跑 memory_settler_node。"""
    orig = llm.chat_json
    llm.chat_json = lambda *a, **kw: {
        "summary": f"第{idx}章摘要",
        "new_foreshadows": new_foreshadows,
        "resolved_foreshadows": [], "advanced_foreshadows": [],
        "character_updates": [], "item_changes": [], "setting_updates": [],
    }
    try:
        state = initial_state()
        state.update({
            "chapter_index": idx,
            "final_chapter": "正文",
            "outline": {"chapters": [{"index": idx, "title": f"第{idx}章"}]},
            "memory": _mem(pool_ids),
        })
        out = nodes.memory_settler_node(state)
        return {f["id"]: f for f in out["memory"]["foreshadow_pool"]}
    finally:
        llm.chat_json = orig


# ── 自动编号找空位 ──────────────────────────────────────────────

def test_auto_id_skips_existing_gap_instead_of_overwriting():
    """池里有 F1、F3（缺口在 F2）：len+1=3 会撞上 F3，必须改找空位。"""
    pool = _settle([{"desc": "本章新埋的悬念"}], ["F1", "F3"])

    assert "F1" in pool and "F3" in pool
    assert pool["F3"]["desc"] == "F3 的原描述", "旧伏笔被改写 = 静默篡改，绝不允许"
    assert "F2" in pool, f"新伏笔应落到空位 F2，实际 id 集合 {sorted(pool)}"
    assert pool["F2"]["desc"] == "本章新埋的悬念"
    assert len(pool) == 3, "新伏笔必须真的被记下，不能丢"


def test_auto_id_fills_the_actual_gap():
    """编号从 F1 开始找第一个空位：池里只有 F2 时，新 id 应是 F1。"""
    pool = _settle([{"desc": "悬念"}], ["F2"])
    assert "F1" in pool and pool["F1"]["desc"] == "悬念"
    assert pool["F2"]["desc"] == "F2 的原描述"


def test_auto_ids_are_unique_within_one_batch():
    pool = _settle([{"desc": "第一条"}, {"desc": "第二条"}, {"desc": "第三条"}], ["F1", "F3"])
    assert sorted(pool) == ["F1", "F2", "F3", "F4", "F5"]
    assert len(pool) == 5, "三条新伏笔一条都不能丢"


def test_auto_id_from_empty_pool_starts_at_f1():
    pool = _settle([{"desc": "第一条"}], [])
    assert "F1" in pool and len(pool) == 1


def test_no_foreshadows_is_a_noop():
    assert sorted(_settle([], ["F1", "F2"])) == ["F1", "F2"]
    assert _settle([], ["F1"])["F1"]["desc"] == "F1 的原描述"


# ── 模型显式给出编号时，语义不变 ────────────────────────────────

def test_explicit_free_id_is_kept_as_given():
    pool = _settle([{"id": "F9", "desc": "自定义编号"}], ["F1"])
    assert pool["F9"]["desc"] == "自定义编号"


def test_explicit_duplicate_id_updates_desc_without_adding_entry():
    """同 id 重复埋设是既有的合法语义：以新描述为准，但不产生重复条目。"""
    pool = _settle([{"id": "F1", "desc": "补充后的描述"}], ["F1", "F2"])
    assert len(pool) == 2
    assert pool["F1"]["desc"] == "补充后的描述"


def test_explicit_duplicate_with_same_desc_keeps_entry():
    pool = _settle([{"id": "F1", "desc": "F1 的原描述"}], ["F1"])
    assert len(pool) == 1
    assert pool["F1"]["desc"] == "F1 的原描述"


def test_new_foreshadow_is_recorded_with_chapter_and_open_status():
    pool = _settle([{"desc": "悬念"}], ["F1"], idx=4)
    fresh = pool["F2"]
    assert fresh["chapter"] == 4
    assert fresh["status"] == "open"
    assert fresh["last_advanced_chapter"] == 4
