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


# ── 物资账本（首次真模型实跑暴露的第 1 号缺口）────────────────
def test_item_status_accepts_chinese_aliases():
    m = models.parse_memory({"items": [{"name": "饼干", "status": "已用完"},
                                       {"name": "刀", "status": "被抢"}]})
    assert [i.status for i in m.items] == ["consumed", "lost"]


def test_illegal_item_status_falls_back_to_available():
    """账本条目必须有确定状态；模型乱写时收敛成 available 并告警。"""
    notices = []
    llm.set_notice_cb(lambda level, msg: notices.append(msg))
    m = models.parse_memory({"items": [{"name": "x", "status": "不知道什么状态"}]})
    assert m.items[0].status == "available"
    assert any("不是合法值" in x for x in notices)


def test_item_change_accepts_alternate_name_keys():
    """模型常把物品名写在 item / 物品 上，不能因此丢掉整条变动。"""
    d = models.parse_memory_delta({"item_changes": [{"item": "罐头", "qty": "半罐"}]})
    assert d.item_changes[0].name == "罐头"


def test_blank_item_change_status_means_unchanged():
    """空状态 = 「状态没变」，必须与 available 区分开。

    混起来会让已经吃完的东西在下一章被结算复活——这正是账本要防的事。
    """
    d = models.parse_memory_delta({"item_changes": [{"name": "罐头", "qty": "半罐"}]})
    assert d.item_changes[0].status is None


def test_item_change_tolerates_plain_string_entry():
    d = models.parse_memory_delta({"item_changes": ["手电筒"]})
    assert d.item_changes[0].name == "手电筒"


def test_old_memory_without_items_still_loads():
    """老存档没有 items 键：加载成空账本，而不是报错或塞进垃圾。"""
    m = models.parse_memory({"chapter_summaries": [{"chapter": 1, "summary": "s"}]})
    assert m.items == []


def test_outline_initial_items_are_parsed():
    o = models.parse_outline({"initial_items": [{"name": "水", "qty": "1 瓶"}, "干粮"]})
    assert [i.name for i in o.initial_items] == ["水", "干粮"]
    assert o.initial_items[0].status == "available"


# ── 角色名归一（人物快照防裂条）──────────────────────────────
def test_canon_name_strips_bracket_note_and_space():
    assert models.canon_name("橘猫（流浪猫）") == models.canon_name("橘猫")
    assert models.canon_name("橘猫 ") == models.canon_name("橘猫")


def test_canon_name_keeps_genuinely_different_names_apart():
    """不能把「林岸」和「林岸的父亲」判成同一个人——所以不做子串匹配。"""
    assert models.canon_name("林岸") != models.canon_name("林岸的父亲")


def test_canon_name_handles_empty_input():
    assert models.canon_name(None) == ""
    assert models.canon_name("   ") == ""
