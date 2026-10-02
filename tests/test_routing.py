"""流水线路由回归测试。

覆盖：
  ① 入口路由 —— 保证「策划案只生成一次」（续写不该重跑策划 Agent）
  ② 校对回退路由 —— fail 打回重写，超轮数强制放行（防死循环）
  ③ 图结构与三路并行校对的汇合判定
  ④ 场景节拍的解析与渲染
  ⑤ 记忆上下文的分层取舍（必带项 vs 检索项）
"""
import config
import graph as graph_mod
import nodes


# ── ① 入口路由 ────────────────────────────────────────────────
def test_entry_goes_to_planner_when_no_outline():
    assert graph_mod.route_entry({"outline": {}}) == "planner"
    assert graph_mod.route_entry({}) == "planner"


def test_entry_skips_planner_when_outline_exists():
    """已有策划案（续写）必须直接进 writer，否则每章都重跑策划、白烧钱。"""
    assert graph_mod.route_entry({"outline": {"title": "已有"}}) == "writer"


# ── ② 校对回退路由 ────────────────────────────────────────────
def test_fail_under_limit_goes_to_rewrite():
    st = {"review_verdict": "fail", "revision_round": 0}
    assert graph_mod.route_after_review(st) == "rewrite"


def test_fail_at_last_allowed_round_still_rewrites():
    st = {"review_verdict": "fail", "revision_round": config.MAX_REVISION_ROUNDS - 1}
    assert graph_mod.route_after_review(st) == "rewrite"


def test_fail_over_limit_is_forced_through():
    """超轮数必须放行，否则永远出不来（死循环）。"""
    st = {"review_verdict": "fail", "revision_round": config.MAX_REVISION_ROUNDS}
    assert graph_mod.route_after_review(st) == "polish"


def test_pass_goes_straight_to_polish():
    assert graph_mod.route_after_review({"review_verdict": "pass"}) == "polish"
    assert graph_mod.route_after_review({}) == "polish"


def test_bump_revision_round_increments():
    assert graph_mod.bump_revision_round({"revision_round": 2}) == {"revision_round": 3}
    assert graph_mod.bump_revision_round({}) == {"revision_round": 1}


# ── ③ 图结构 ──────────────────────────────────────────────────
def test_graph_compiles_and_has_expected_nodes():
    app = graph_mod.build_graph()
    assert app is not None
    for name in ("planner", "writer", "reviewer_ooc", "reviewer_logic",
                 "reviewer_pacing", "merge_reviews", "bump_round",
                 "polisher", "memory_settler"):
        assert name in app.get_graph().nodes, f"缺少节点 {name}"
    # 拆分后不该再有单一的 reviewer 节点
    assert "reviewer" not in app.get_graph().nodes


def test_writer_fans_out_to_three_reviewers():
    """writer 必须同时指向三路 specialist，且三路都汇入 merge_reviews。

    少了任何一条边，并行校对就退化成"只跑其中一路"——而且不会报错，
    只表现为"校对好像没以前严了"，很难被发现。
    """
    app = graph_mod.build_graph()
    edges = {(e.source, e.target) for e in app.get_graph().edges}
    for rv in ("reviewer_ooc", "reviewer_logic", "reviewer_pacing"):
        assert ("writer", rv) in edges, f"writer 没有指向 {rv}"
        assert (rv, "merge_reviews") in edges, f"{rv} 没有汇入 merge_reviews"


# ── ④ 三路校对的汇合判定 ──────────────────────────────────────
def _merge(extra):
    st = {"chapter_index": 1, "outline": {"chapters": []}}
    st.update(extra)
    return nodes.merge_reviews_node(st)


def test_merge_passes_when_all_routes_clear():
    out = _merge({})
    assert out["review_verdict"] == "pass"
    assert out["review_comments"] == []


def test_merge_fails_when_any_route_reports_critical():
    """只要有一路报出 critical，整章就该打回——这是拆分后必须保持的语义。"""
    out = _merge({
        "review_comments_ooc": [{"severity": "minor", "issue": "小"}],
        "review_comments_logic": [{"severity": "critical", "issue": "硬伤"}],
        "review_comments_pacing": [],
    })
    assert out["review_verdict"] == "fail"


def test_merge_treats_minor_only_as_pass():
    """三路都只有 minor → pass，意见留给润色参考（避免空转重写）。"""
    out = _merge({
        "review_comments_ooc": [{"severity": "minor", "issue": "甲"}],
        "review_comments_logic": [],
        "review_comments_pacing": [{"severity": "minor", "issue": "乙"}],
    })
    assert out["review_verdict"] == "pass"
    assert len(out["review_comments"]) == 2


def test_merge_puts_critical_first_and_tags_its_source():
    out = _merge({
        "review_comments_ooc": [{"severity": "minor", "issue": "甲"}],
        "review_comments_logic": [{"severity": "critical", "issue": "乙"}],
        "review_comments_pacing": [],
    })
    cs = out["review_comments"]
    assert cs[0]["severity"] == "critical", "critical 要排前面，撰稿人先看到要紧的"
    assert cs[0]["from"] == "逻辑/时间线/设定"
    assert cs[1]["from"] == "人物一致性"


# ── ⑤ 场景节拍 ────────────────────────────────────────────────
def test_chapter_plan_carries_beats():
    st = {"chapter_index": 1, "outline": {"chapters": [
        {"index": 1, "title": "甲", "beats": [{"goal": "开场", "emotion": 2}]}]}}
    plan = nodes._chapter_plan(st)
    assert plan["beats"] and plan["beats"][0]["goal"] == "开场"


def test_chapter_plan_beats_default_to_empty_list():
    """旧策划案没有 beats 字段时必须给空列表——P1 之前生成的存档要能继续用。"""
    st = {"chapter_index": 1, "outline": {"chapters": [{"index": 1, "title": "甲"}]}}
    assert nodes._chapter_plan(st)["beats"] == []


def test_chapter_plan_ignores_malformed_beats():
    st = {"chapter_index": 1, "outline": {"chapters": [
        {"index": 1, "beats": ["字符串", None, {"goal": "好的"}]}]}}
    assert nodes._chapter_plan(st)["beats"] == [{"goal": "好的"}]


def test_beats_section_renders_every_field():
    txt = nodes._beats_section({"beats": [
        {"goal": "推进", "conflict": "阻力", "turn": "转折", "words": 600, "emotion": 4}]})
    for kw in ("推进", "阻力", "转折", "600", "4/5"):
        assert kw in txt, f"渲染结果缺少 {kw}"


def test_beats_section_is_empty_without_beats():
    assert nodes._beats_section({"beats": []}) == ""


# ── ⑥ 记忆上下文的分层 ────────────────────────────────────────
def test_memory_context_empty_on_first_chapter():
    assert "暂无故事记忆" in nodes._memory_context({})


def test_memory_context_keeps_must_have_and_drops_resolved():
    st = {"chapter_index": 3, "memory": {
        "chapter_summaries": [{"chapter": 1, "summary": "甲"}],
        "foreshadow_pool": [
            {"id": "F1", "desc": "未回收的伏笔", "chapter": 1, "status": "open"},
            {"id": "F2", "desc": "已回收的伏笔", "chapter": 1, "status": "resolved"}],
        "character_states": [{"name": "林岸", "state": "戒备"}],
    }}
    txt = nodes._memory_context(st)
    assert "F1" in txt and "林岸" in txt and "相关前情摘要" in txt
    assert "已回收的伏笔" not in txt, "已回收的伏笔不该占必带项的位置"


def test_memory_context_caps_summaries(monkeypatch):
    """摘要条数超过上限时必须裁剪——这是长篇不撑爆上下文的关键。"""
    monkeypatch.setattr(config, "MEMORY_TOP_K", 3)
    st = {"chapter_index": 50, "memory": {
        "chapter_summaries": [{"chapter": i, "summary": f"第{i}章发生的事"} for i in range(1, 50)],
        "foreshadow_pool": [], "character_states": [],
    }}
    txt = nodes._memory_context(st)
    assert "此处带 3 章" in txt
    assert "第49章发生的事" in txt, "应当保留最近章节"
    assert "第1章发生的事" not in txt, "最早的章节应被裁掉"


# ── ⑦ 章节计划兜底 ────────────────────────────────────────────
def test_chapter_plan_falls_back_beyond_outline():
    """章节数超出策划大纲时，不能崩，要给出"自然推进"的兜底计划。"""
    st = {"chapter_index": 9, "outline": {"chapters": [{"index": 1, "title": "第一章"}]}}
    plan = nodes._chapter_plan(st)
    assert plan["index"] == 9
    assert "自然推进" in plan["outline"]


def test_chapter_plan_matches_outline_entry():
    st = {"chapter_index": 2, "outline": {"chapters": [
        {"index": 1, "title": "甲"}, {"index": 2, "title": "乙", "outline": "乙的大纲"}]}}
    assert nodes._chapter_plan(st)["title"] == "乙"


def test_chapter_plan_fills_fields_the_model_omitted():
    """模型漏写章节字段时，必须补齐而不是让 writer 端 KeyError。

    这是写测试时发现的真实缺口：_chapter_plan 原来直接返回大纲条目，
    一旦模型没给 "outline" 键，writer_node 取 plan['outline'] 就整条流水线崩。
    """
    st = {"chapter_index": 1, "outline": {"chapters": [{"index": 1}]}}
    plan = nodes._chapter_plan(st)
    for key in ("index", "title", "outline", "turning_point", "notes", "beats"):
        assert key in plan, f"缺少 {key}"
    assert plan["outline"] == "" and plan["title"] == "第1章"


def test_chapter_plan_treats_null_fields_as_empty():
    """字段存在但为 null 也要兜住（模型偶尔会明确输出 null）。"""
    st = {"chapter_index": 1, "outline": {"chapters": [
        {"index": 1, "title": None, "outline": None, "turning_point": None}]}}
    plan = nodes._chapter_plan(st)
    assert plan["outline"] == "" and plan["title"] == "第1章"


def test_chapter_plan_accepts_string_index():
    """旧存档里的 index 可能是字符串，比对时必须归一。"""
    st = {"chapter_index": 2, "outline": {"chapters": [{"index": "2", "title": "乙"}]}}
    assert nodes._chapter_plan(st)["title"] == "乙"


# ── ⑧ meta 材料注入 ──────────────────────────────────────────
def test_meta_sections_injects_forbidden_into_all_three_roles():
    """禁止内容要同时注入撰稿（规避）、校对（核查）、润色。"""
    st = {"meta": {"forbidden_list": ["不得描写血腥细节"]}}
    for role in ("writer", "reviewer", "polisher"):
        joined = "\n".join(nodes._meta_sections(st, role))
        assert "不得描写血腥细节" in joined, f"{role} 没拿到禁止内容"


def test_meta_sections_word_count_only_for_writer():
    st = {"meta": {"word_count": 3000}}
    assert "3000" in "\n".join(nodes._meta_sections(st, "writer"))
    assert "3000" not in "\n".join(nodes._meta_sections(st, "reviewer"))
