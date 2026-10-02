"""设定档案（第四本账）的模型归一、立档、渲染、结算与误报防线。

为什么要有这一本账（第四次真模型实跑，2026-10-02）：
前三本账只覆盖 伏笔 / 人物 / 物资，**地点与物理设定不在任何一本里**。
前情摘要只记情节，top-k 又只带上一章，于是逻辑校对手上没有任何"与前文一致"的
判断依据——它连着两轮把第 1 章早已确立的「店面卷帘门 + 后门铁门」判成 critical
设定矛盾（正文没按它改是对的，是纯误报）。代价是白烧重写轮次。

所以这个文件里最要紧的两条是：
  · test_logic_review_prompt_carries_settings_bible —— 判断基准真的送到了校对手里
  · test_settler_does_not_overwrite_with_empty_fields —— 空字段不覆盖（复活 bug 的设定版）
其余是 schema 归一与渲染取舍的常规回归。
"""
import config
import graph as graph_mod
import llm
import models
import nodes
import prompts

from conftest import initial_state


def _mem(**kw):
    """一份完整的 memory dict（缺的键补空，避免 _memory_context 走"第一章"分支）。"""
    base = {"chapter_summaries": [], "foreshadow_pool": [],
            "character_states": [], "items": [], "settings": []}
    base.update(kw)
    return base


def _settle(delta, memory, idx=2):
    """把结算员的 LLM 输出钉成固定 delta，直接跑 memory_settler_node。"""
    orig = llm.chat_json
    llm.chat_json = lambda *a, **kw: delta
    try:
        return nodes.memory_settler_node({
            "chapter_index": idx,
            "final_chapter": "正文",
            "outline": {"chapters": [{"index": idx, "title": f"第{idx}章"}]},
            "memory": memory,
        })
    finally:
        llm.chat_json = orig


# ── ① 模型归一：Setting ────────────────────────────────────────
def test_setting_parses_with_field_aliases():
    """字段换名字是常态（地点/场所/描述/类型），必须收口到规范字段。"""
    m = models.parse_memory({"settings": [
        {"地点": "便利店", "描述": "门脸是卷帘门，后面还有一扇铁门", "类型": "设施"}]})
    s = m.settings[0]
    assert (s.name, s.kind) == ("便利店", "object")
    assert s.detail == "门脸是卷帘门，后面还有一扇铁门"


def test_setting_string_entry_becomes_name_only():
    """策划把 initial_settings 写成 ["老堤防"] 也要能收（与 items/characters 同策）。"""
    m = models.parse_memory({"settings": ["老堤防"]})
    s = m.settings[0]
    assert s.name == "老堤防"
    assert s.kind == "other" and s.status == "active" and s.detail == ""


def test_setting_kind_aliases_normalized():
    cases = {"地点": "place", "场所": "place", "设施": "object", "建筑": "object",
             "规则": "rule", "机制": "rule", "组织": "relation", "势力": "relation",
             "其他": "other", "PLACE": "place"}
    m = models.parse_memory({"settings": [
        {"name": f"S{i}", "kind": k} for i, k in enumerate(cases)]})
    assert [s.kind for s in m.settings] == list(cases.values())


def test_setting_kind_unknown_warns_and_falls_back_to_other():
    notices = []
    llm.set_notice_cb(lambda level, msg: notices.append((level, msg)))
    m = models.parse_memory({"settings": [{"name": "x", "kind": "什么都不是"}]})
    assert m.settings[0].kind == "other"
    assert any("不是合法值" in msg for _, msg in notices), \
        f"非法类型被静默吞掉了：{notices}"


def test_setting_status_aliases_normalized():
    m = models.parse_memory({"settings": [
        {"name": "a", "status": "已毁"}, {"name": "b", "status": "仍成立"},
        {"name": "c", "status": "destroyed"}]})
    assert [s.status for s in m.settings] == ["retired", "active", "retired"]


def test_setting_status_defaults_to_active():
    """账本条目的状态空值按「有效」处理——老存档里没有 status 是常态。"""
    assert models.parse_memory({"settings": [{"name": "老堤防"}]}).settings[0].status == "active"


def test_setting_status_illegal_value_warns():
    notices = []
    llm.set_notice_cb(lambda level, msg: notices.append((level, msg)))
    m = models.parse_memory({"settings": [{"name": "x", "status": "半死不活"}]})
    assert m.settings[0].status == "active"
    assert any("不是合法值" in msg for _, msg in notices)


# ── ② 模型归一：SettingChange（空 = 不改动）────────────────────
def test_setting_change_empty_means_no_change():
    """留空 = 本次不改动。这是与 Setting 分成两个模型的分界线。"""
    d = models.parse_memory_delta({"summary": "s", "setting_updates": [{"地点": "便利店"}]})
    c = d.setting_updates[0]
    assert c.name == "便利店"
    assert c.detail == "" and c.kind == "" and c.status is None


def test_setting_change_status_only():
    d = models.parse_memory_delta({"summary": "s", "setting_updates": [
        {"name": "废弃水文站", "status": "已毁", "note": "夜里塌了"}]})
    c = d.setting_updates[0]
    assert c.status == "retired" and c.detail == "" and c.kind == ""
    assert c.note == "夜里塌了"


def test_setting_change_kind_illegal_warns_but_status_stays_none():
    notices = []
    llm.set_notice_cb(lambda level, msg: notices.append((level, msg)))
    d = models.parse_memory_delta({"summary": "s", "setting_updates": [
        {"name": "x", "kind": "胡说", "status": "也许"}]})
    c = d.setting_updates[0]
    assert c.kind == "other" and c.status is None, "非法状态应保持 None（不改动），不能复活"
    assert len(notices) == 2, f"两处非法值应各告警一次：{notices}"


def test_memory_delta_coerces_setting_updates_from_strings():
    d = models.parse_memory_delta({"summary": "s", "setting_updates": ["老堤防"]})
    assert d.setting_updates[0].name == "老堤防"


def test_old_archive_without_settings_key_still_parses():
    """老存档（第四次实跑产物）里没有 settings 键，不能因此报错或丢数据。"""
    m = models.parse_memory({"chapter_summaries": [{"chapter": 1, "summary": "s"}],
                             "items": [{"name": "矿泉水"}], "character_states": []})
    assert m.settings == [] and len(m.items) == 1


def test_memory_lists_accept_bare_strings():
    """记忆库面板是让用户直接编辑 JSON 的：`{"items": ["矿泉水"]}` 这种简写
    以前会让 List[Item] 直接抛 ValidationError，而解析点在 memory_settler_node 的
    try 之外 → 整章崩掉。脏数据该被收敛，不该终止创作。
    """
    m = models.parse_memory({
        "items": ["矿泉水"], "settings": ["老堤防"], "character_states": ["林岸"],
        "foreshadow_pool": ["半页笔记"], "chapter_summaries": ["第一章发生了什么"]})
    assert m.items[0].name == "矿泉水"
    assert m.settings[0].name == "老堤防"
    assert m.character_states[0].name == "林岸"
    assert m.foreshadow_pool[0].desc == "半页笔记"
    assert m.chapter_summaries[0].summary == "第一章发生了什么"


def test_outline_carries_initial_settings():
    o = models.parse_outline({"title": "T", "initial_settings": [
        {"name": "老堤防", "kind": "地点", "detail": "堤面窄"}]})
    assert o.initial_settings[0].kind == "place"


# ── ③ 策划阶段立档 ────────────────────────────────────────────
def test_planner_seeds_settings_into_memory(monkeypatch):
    monkeypatch.setattr(llm, "chat_json", lambda *a, **kw: {
        "title": "T", "characters": [], "chapters": [{"index": 1, "title": "甲"}],
        "initial_settings": [
            {"name": "便利店", "kind": "object", "detail": "卷帘门 + 后门铁门"},
            {"name": "  "},                                   # 空名字要丢掉
            "老堤防"]})
    out = nodes.planner_node({"user_prompt": "悬疑", "memory": {}})
    names = [s["name"] for s in out["memory"]["settings"]]
    assert names == ["便利店", "老堤防"], f"开局设定立档不对：{names}"
    assert all(s["chapter"] == 1 and s["last_mentioned_chapter"] == 1
               for s in out["memory"]["settings"])


def test_planner_does_not_overwrite_existing_settings(monkeypatch):
    """续写时策划（若被重跑）不得把已经长出来的档案冲掉。"""
    monkeypatch.setattr(llm, "chat_json", lambda *a, **kw: {
        "title": "T", "characters": [], "chapters": [{"index": 1, "title": "甲"}],
        "initial_settings": [{"name": "新地点"}]})
    out = nodes.planner_node({"user_prompt": "悬疑", "memory": _mem(
        settings=[{"name": "便利店", "chapter": 2}])})
    assert [s["name"] for s in out["memory"]["settings"]] == ["便利店"]


# ── ④ 渲染成第四本必带账 ──────────────────────────────────────
def test_memory_context_lists_established_settings():
    ctx = nodes._memory_context({"chapter_index": 3, "memory": _mem(settings=[
        {"name": "便利店", "kind": "object", "detail": "门脸是卷帘门，后面还有一扇铁门",
         "note": "镇口", "chapter": 1}])})
    assert "已确立设定" in ctx
    assert "卷帘门" in ctx and "第1章确立" in ctx and "镇口" in ctx
    assert "【设施】" in ctx, "类型标签丢了，校对就分不清这是地点还是规则"


def test_memory_context_marks_retired_settings():
    ctx = nodes._memory_context({"chapter_index": 4, "memory": _mem(settings=[
        {"name": "废弃水文站", "kind": "place", "status": "retired", "chapter": 2}])})
    assert "已废弃" in ctx
    assert "不得再当作有效设定或场景使用" in ctx


def test_memory_context_ranks_least_recently_mentioned_first(monkeypatch):
    """最久没被提到的设定排最前面。

    依据是"边际价值"：撰稿与校对每章都能拿到上一章全文，越新近出现过的设定越可能
    已经在那份全文里；第 1 章立下、此后再没提过的，除了这本档案没有任何别的来源
    ——它恰恰是最容易被误判成"与前文矛盾"的那一类。
    """
    monkeypatch.setattr(config, "MEMORY_SETTING_MAX", 1)
    ctx = nodes._memory_context({"chapter_index": 9, "memory": _mem(settings=[
        {"name": "后立的地方", "chapter": 8, "last_mentioned_chapter": 8},
        {"name": "第1章立的地方", "chapter": 1, "last_mentioned_chapter": 1}])})
    assert "第1章立的地方（第1章确立）" in ctx
    assert "另有 1 条较早设定未列描述：后立的地方" in ctx


def test_memory_context_setting_cap_declares_hidden_still_valid(monkeypatch):
    """被截断的设定不能"消失"——它们同样有效，必须写明"以设定为准、不要判为矛盾"。

    设定档案是没有终点的线性项（一处地点写一次就永远在档），所以必须像伏笔一样加上界；
    但截断的代价和伏笔不同：漏掉的伏笔是"没人推进"，漏掉的设定是"下一章直接误报"。
    所以被省略的不仅要留下名字，还必须明确告诉模型它们仍然成立。
    """
    monkeypatch.setattr(config, "MEMORY_SETTING_MAX", 2)
    ctx = nodes._memory_context({"chapter_index": 9, "memory": _mem(settings=[
        {"name": f"地{i}", "chapter": i, "last_mentioned_chapter": i} for i in range(1, 6)])})
    assert ctx.count("- 地") == 2, "上限没生效"
    assert "另有 3 条较早设定未列描述：地3、地4、地5" in ctx
    assert "不要判为矛盾" in ctx


def test_memory_context_settings_alone_is_not_treated_as_empty():
    """只有设定档案时不能说"这是第一章，暂无故事记忆"——那会让校对回到无依据状态。"""
    ctx = nodes._memory_context({"chapter_index": 5, "memory": _mem(settings=[
        {"name": "便利店", "chapter": 1}])})
    assert "暂无故事记忆" not in ctx
    assert "便利店" in ctx


# ── ⑤ 判断基准真的送到校对与撰稿手里（核心 regression）────────
def test_logic_review_prompt_carries_settings_bible(mock_mode, monkeypatch):
    """逻辑校对必须拿到「已确立设定」。

    这条直接钉住第四次实跑的误报根因：校对手里没有判断基准时，会把第 1 章
    早已确立的「卷帘门 + 后门铁门」当成这一章的设定矛盾判 critical。
    """
    seen = {}
    real = llm.chat_json

    def spy(system, user, *a, **kw):
        if "ReviewerLogic" in system:
            seen["msg"] = user
        return real(system, user, *a, **kw)

    monkeypatch.setattr(llm, "chat_json", spy)
    nodes.reviewer_logic_node({
        "chapter_index": 4,
        "chapter_draft": "他把卷帘门落下一半，从后门出去了。",
        "outline": {"chapters": [{"index": 4, "title": "甲"}]},
        "memory": _mem(settings=[{"name": "便利店", "kind": "object",
                                  "detail": "门脸是卷帘门，后面还有一扇铁门",
                                  "chapter": 1}]),
    })
    assert "已确立设定" in seen["msg"], "校对请求里没有设定档案"
    assert "卷帘门" in seen["msg"], "设定档案里最关键的那句描写没带过去"


def test_writer_prompt_carries_settings_bible(mock_mode, monkeypatch):
    """撰稿人也要看得到——他得知道那两扇门是既成事实，不能改写。"""
    seen = {}
    real = llm.chat

    def spy(system, user, *a, **kw):
        if "Writer" in system:
            seen["prompt"] = user
        return real(system, user, *a, **kw)

    monkeypatch.setattr(llm, "chat", spy)
    nodes.writer_node({
        "chapter_index": 2, "revision_round": 0,
        "outline": {"title": "T", "characters": [],
                    "chapters": [{"index": 2, "title": "乙"}]},
        "memory": _mem(settings=[{"name": "便利店", "kind": "object",
                                  "detail": "门脸是卷帘门，后面还有一扇铁门",
                                  "chapter": 1}]),
    })
    assert "已确立设定" in seen["prompt"] and "卷帘门" in seen["prompt"]


def test_prompts_declare_the_setting_contract():
    """提示词侧的契约：四个角色都要知道这本账存在（少一个，账就断了）。"""
    assert "initial_settings" in prompts.PLANNER_SYSTEM
    assert "setting_updates" in prompts.MEMORY_SYSTEM
    assert "设定档案" in prompts.WRITER_SYSTEM
    assert "已确立设定" in prompts.REVIEWER_LOGIC_SYSTEM


def test_logic_reviewer_is_told_established_settings_are_not_contradictions():
    """判定尺度里必须明写"清单里已确立的不算矛盾"，否则模型照样会按自己的直觉报。"""
    sys_prompt = prompts.REVIEWER_LOGIC_SYSTEM
    assert "不是矛盾，是延续" in sys_prompt
    assert "只有前者才判 critical" in sys_prompt


# ── ⑥ 结算合并 ────────────────────────────────────────────────
def test_settler_adds_new_setting():
    out = _settle({"summary": "s", "setting_updates": [
        {"name": "废弃水文站", "kind": "place", "detail": "锁着的砖房", "note": "第2章走近"}]},
        _mem(), idx=2)
    s = out["memory"]["settings"][0]
    assert (s["name"], s["kind"], s["chapter"]) == ("废弃水文站", "place", 2)
    assert s["status"] == "active" and s["last_mentioned_chapter"] == 2


def test_settler_bare_name_mention_bumps_last_mentioned():
    """只报名字 = "这条设定还活着"，不改内容、只更新时间戳。"""
    out = _settle({"summary": "s", "setting_updates": [{"name": "便利店"}]},
                  _mem(settings=[{"name": "便利店", "detail": "卷帘门 + 后门铁门",
                                  "chapter": 1, "last_mentioned_chapter": 1}]), idx=5)
    s = out["memory"]["settings"][0]
    assert s["last_mentioned_chapter"] == 5
    assert s["detail"] == "卷帘门 + 后门铁门", "空字段不许把已有描述冲掉"


def test_settler_does_not_overwrite_with_empty_fields():
    """空字段不覆盖——这是"复活 bug"的设定版。

    只补 detail 时把 status 刷回 active，就等于把一个已经废弃的地点悄悄恢复，
    后文再写它就变成了"凭空出现"。
    """
    out = _settle({"summary": "s", "setting_updates": [
        {"name": "废弃水文站", "detail": "塌了一角"}]},
        _mem(settings=[{"name": "废弃水文站", "kind": "place", "detail": "锁着的砖房",
                        "status": "retired", "chapter": 2}]), idx=3)
    s = out["memory"]["settings"][0]
    assert s["status"] == "retired", "已废弃的设定被刷回有效了"
    assert s["kind"] == "place", "没给 kind 时把已有类型冲掉了"
    assert s["detail"] == "塌了一角"


def test_settler_setting_name_matched_after_normalization():
    """「便利店（镇口）」要并进已有的「便利店」，不能裂成两处。"""
    out = _settle({"summary": "s", "setting_updates": [
        {"name": "便利店（镇口）", "detail": "卷帘门坏了一半"}]},
        _mem(settings=[{"name": "便利店", "detail": "卷帘门 + 后门铁门", "chapter": 1}]), idx=4)
    ss = out["memory"]["settings"]
    assert len(ss) == 1, f"同一处设定被拆成两条：{[s['name'] for s in ss]}"
    assert ss[0]["name"] == "便利店", "命中已有条目时应保留档案里的原名"


def test_settler_ignores_bare_name_not_in_bible():
    """空壳条目不入档：报了个档案里没有、又什么信息都没带的名字，只是顺口一提。

    放进去会往档案里塞空壳、把真设定挤掉——而它们同样要占必带额度。
    """
    out = _settle({"summary": "s", "setting_updates": [
        {"name": "路边一个摊子"}, {"name": "废弃水文站", "detail": "砖房"}]},
        _mem(), idx=2)
    assert [s["name"] for s in out["memory"]["settings"]] == ["废弃水文站"]


def test_settler_retire_without_reason_warns():
    """某个地点悄悄失效，后文就会莫名把它写没了——必须提醒作者。"""
    notices = []
    llm.set_notice_cb(lambda level, msg: notices.append((level, msg)))
    _settle({"summary": "s", "setting_updates": [
        {"name": "便利店", "status": "retired"}]},
        _mem(settings=[{"name": "便利店", "chapter": 1}]), idx=3)
    warn = [m for lv, m in notices if lv == "warn"]
    assert warn and "没写废弃原因" in warn[0], f"静默废弃没被抓出来：{notices}"


def test_settler_retire_with_reason_is_informational():
    """写了原因就只提示（info），不告警——要抓的是"静默"，不是"变化"本身。"""
    notices = []
    llm.set_notice_cb(lambda level, msg: notices.append((level, msg)))
    out = _settle({"summary": "s", "setting_updates": [
        {"name": "便利店", "status": "retired", "note": "第3章被烧了"}]},
        _mem(settings=[{"name": "便利店", "chapter": 1}]), idx=3)
    assert out["memory"]["settings"][0]["status"] == "retired"
    assert [lv for lv, _ in notices] == ["info"], f"交代了原因还告警会制造噪音：{notices}"
    assert "被标记为已废弃" in notices[0][1]


def test_settler_prompt_lists_current_settings():
    """结算请求要带上当前档案——模型没有可对齐的依据就会自由发挥、另立新条目。"""
    seen = {}

    def fake_chat_json(system, user, *a, **kw):
        seen["msg"] = user
        return {"summary": "s"}

    orig = llm.chat_json
    llm.chat_json = fake_chat_json
    try:
        nodes.memory_settler_node({
            "chapter_index": 3, "final_chapter": "正文",
            "outline": {"chapters": [{"index": 3, "title": "第3章"}]},
            "memory": _mem(settings=[{"name": "便利店", "kind": "object",
                                      "detail": "卷帘门 + 后门铁门",
                                      "status": "active", "chapter": 1}]),
        })
    finally:
        llm.chat_json = orig
    assert "当前设定档案" in seen["msg"] and "便利店" in seen["msg"]
    assert "卷帘门" in seen["msg"]


# ── ⑦ 端到端 ─────────────────────────────────────────────────
def test_mock_pipeline_builds_settings_ledger(mock_mode):
    """三章跑完：立档 → 新增 → 废弃，三条路径都留下痕迹。"""
    app = graph_mod.build_graph()
    state = initial_state()
    for idx in (1, 2, 3):
        state.update({"chapter_index": idx, "chapter_draft": "", "review_comments": [],
                      "review_verdict": "pass", "review_comments_ooc": [],
                      "review_comments_logic": [], "review_comments_pacing": [],
                      "style_report": {}, "revision_round": 0, "final_chapter": ""})
        state = graph_mod.run_chapter(app, state)

    by_name = {s["name"]: s for s in state["memory"]["settings"]}
    assert "便利店" in by_name and "卷帘门" in by_name["便利店"]["detail"], \
        "策划给的开局设定没有立进档案"
    assert by_name["便利店"]["chapter"] == 1
    assert by_name["废弃水文站"]["status"] == "retired", "第 3 章的废弃没结算进去"


def test_mock_pipeline_reaches_next_chapter_prompt(mock_mode):
    """跑过的设定要出现在下一章的撰稿请求里——账本立了不带过去等于没立。"""
    app = graph_mod.build_graph()
    state = initial_state()
    for idx in (1, 2):
        state.update({"chapter_index": idx, "chapter_draft": "", "review_comments": [],
                      "review_verdict": "pass", "review_comments_ooc": [],
                      "review_comments_logic": [], "review_comments_pacing": [],
                      "style_report": {}, "revision_round": 0, "final_chapter": ""})
        state = graph_mod.run_chapter(app, state)
    assert "废弃水文站" in nodes._memory_context(
        {"chapter_index": 3, "memory": state["memory"]}, {"index": 3})
