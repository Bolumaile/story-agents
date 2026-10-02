"""Pydantic 模型层的容错与归一（优化建议第 6、9 条）。

这一层的核心承诺是「模型给的脏数据不会打断流水线，也不会把坏值传下去」，
所以用例集中在各种畸形输入上——真实模型输出比想象中离谱得多。
"""
import llm
import models


# ── 缺字段与类型收敛 ─────────────────────────────────────────
def test_missing_fields_get_defaults():
    o = models.parse_outline({})
    assert o.title == "" and o.chapters == [] and o.characters == []


def test_none_becomes_empty_string():
    o = models.parse_outline({"title": None, "core_conflict": None})
    assert o.title == "" and o.core_conflict == ""


def test_number_fields_accept_strings_and_floats():
    assert models.Chapter.model_validate({"index": "3"}).index == 3
    assert models.Chapter.model_validate({"index": 2.9}).index == 2


def test_unparsable_number_falls_back_to_zero():
    assert models.Chapter.model_validate({"index": "不是数字"}).index == 0


def test_bool_is_not_silently_treated_as_int():
    """True 是 int 的子类，当成 1 用会掩盖错误——这里明确按默认值处理。"""
    assert models.Chapter.model_validate({"index": True}).index == 0


def test_list_field_wraps_a_single_object():
    """characters 写成单个对象而不是数组时也要收下。"""
    o = models.parse_outline({"characters": {"name": "林岸"}})
    assert [c.name for c in o.characters] == ["林岸"]


def test_string_items_are_wrapped_by_key_hint():
    assert [c.name for c in models.parse_outline(
        {"characters": ["林岸", "老崔"]}).characters] == ["林岸", "老崔"]


def test_junk_items_are_dropped_without_failing():
    o = models.parse_outline({"characters": ["林岸", None, 42, []]})
    assert [c.name for c in o.characters] == ["林岸"]


# ── 章节序号补齐 ─────────────────────────────────────────────
def test_missing_chapter_index_is_filled_by_position():
    o = models.parse_outline({"chapters": [{"title": "一"}, {"title": "二"}]})
    assert [c.index for c in o.chapters] == [1, 2]


def test_valid_chapter_index_is_kept():
    o = models.parse_outline({"chapters": [{"index": 7, "title": "七"}]})
    assert o.chapters[0].index == 7


# ── 场景节拍 ─────────────────────────────────────────────────
def test_beats_are_parsed():
    c = models.Chapter.model_validate({"beats": [
        {"goal": "推进", "conflict": "阻力", "turn": "转折", "words": 600, "emotion": 4}]})
    b = c.beats[0]
    assert (b.goal, b.conflict, b.turn, b.words, b.emotion) == ("推进", "阻力", "转折", 600, 4)


def test_beats_default_to_empty():
    assert models.Chapter.model_validate({}).beats == []


def test_emotion_is_clamped_to_1_5():
    assert models.SceneBeat.model_validate({"emotion": 99}).emotion == 5
    assert models.SceneBeat.model_validate({"emotion": -3}).emotion == 1


def test_string_beats_are_wrapped_as_goal():
    assert models.Chapter.model_validate({"beats": ["开场"]}).beats[0].goal == "开场"


# ── 校对意见 ─────────────────────────────────────────────────
def test_comment_defaults_to_minor():
    """没写严重度一律按 minor——避免把"没写"误解成严重问题而触发无意义重写。"""
    assert models.parse_comments([{"issue": "只写了 issue"}])[0].severity == "minor"


def test_comment_severity_judgement():
    for raw, want in [("critical", "critical"), ("CRITICAL", "critical"),
                      ("严重", "minor"), ("minor", "minor"), ("", "minor")]:
        assert models.parse_comments([{"severity": raw}])[0].severity == want


def test_comment_type_is_normalized_from_chinese():
    assert models.parse_comments([{"type": "人物一致性"}])[0].type == "ooc"
    assert models.parse_comments([{"type": "时间线"}])[0].type == "timeline"
    assert models.parse_comments([{"type": "节奏"}])[0].type == "pacing"
    assert models.parse_comments([{"type": "没见过的类型"}])[0].type == "logic"


def test_comments_handle_none_and_garbage():
    assert models.parse_comments(None) == []
    assert models.parse_comments([None, 3, []]) == []
    assert len(models.parse_comments(["只有一句话"])) == 1


# ── 伏笔状态机 ───────────────────────────────────────────────
def test_foreshadow_status_aliases():
    for raw, want in [("open", "open"), ("已回收", "resolved"), ("推进中", "progressing"),
                      ("搁置", "deferred"), ("resolved", "resolved")]:
        assert models.Foreshadow.model_validate({"status": raw}).status == want


def test_invalid_foreshadow_status_is_converged_with_warning():
    notices = []
    llm.set_notice_cb(lambda level, msg: notices.append((level, msg)))
    models.reset_warnings()
    try:
        f = models.Foreshadow.model_validate({"status": "胡说八道"})
    finally:
        llm.set_notice_cb(None)
    assert f.status == "open"
    assert any("不是合法值" in m for _, m in notices)


def test_same_bad_value_warns_only_once():
    """同一类问题只提醒一次，避免几十条坏数据刷屏。"""
    notices = []
    llm.set_notice_cb(lambda level, msg: notices.append((level, msg)))
    models.reset_warnings()
    try:
        for _ in range(5):
            models.Foreshadow.model_validate({"status": "还是错的"})
    finally:
        llm.set_notice_cb(None)
    assert len(notices) == 1


def test_foreshadow_deadline_accepts_empty_forms():
    assert models.Foreshadow.model_validate({"deadline_chapter": ""}).deadline_chapter is None
    assert models.Foreshadow.model_validate({"deadline_chapter": 0}).deadline_chapter is None
    assert models.Foreshadow.model_validate({"deadline_chapter": "6"}).deadline_chapter == 6


# ── 向后兼容：旧存档必须能读 ─────────────────────────────────
def _old_memory():
    return {"chapter_summaries": [{"chapter": 1, "summary": "s"}],
            "foreshadow_pool": [{"id": "F1", "desc": "d", "chapter": 1, "status": "open"}],
            "character_states": [{"name": "林岸", "state": "戒备"}]}


def test_old_memory_without_new_fields_still_loads():
    """P1 之前生成的 _session.json 不能被新 schema 拒之门外。"""
    f = models.parse_memory(_old_memory()).foreshadow_pool[0]
    assert f.last_advanced_chapter == 0 and f.deadline_chapter is None


def test_memory_dump_roundtrip_is_stable():
    """转 dict 再读回来必须完全一致，否则每次落盘都会产生虚假 diff。"""
    once = models.parse_memory(_old_memory()).model_dump()
    twice = models.parse_memory(once).model_dump()
    assert once == twice


# ── 记忆增量 ─────────────────────────────────────────────────
def test_delta_id_lists_accept_many_shapes():
    d = models.parse_memory_delta({
        "resolved_foreshadows": [{"id": "F1"}, "F2,F3", "F4，F5"],
        "advanced_foreshadows": "F6"})
    assert d.resolved_foreshadows == ["F1", "F2", "F3", "F4", "F5"]
    assert d.advanced_foreshadows == ["F6"]


def test_delta_empty_is_all_defaults():
    d = models.parse_memory_delta({})
    assert d.summary == "" and d.new_foreshadows == [] and d.resolved_foreshadows == []


def test_dump_returns_plain_dicts():
    out = models.dump(models.parse_comments([{"issue": "i"}]))
    assert isinstance(out[0], dict) and out[0]["issue"] == "i"
