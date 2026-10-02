"""流水线路由回归测试。

覆盖三件事：
  ① 入口路由 —— 保证「策划案只生成一次」（续写不该重跑策划 Agent）
  ② 校对回退路由 —— fail 打回重写，超轮数强制放行（防死循环）
  ③ 图能编译 + 节点辅助函数的兜底行为
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


# ── ③ 图与节点辅助函数 ────────────────────────────────────────
def test_graph_compiles_and_has_expected_nodes():
    app = graph_mod.build_graph()
    assert app is not None
    for name in ("planner", "writer", "reviewer", "bump_round", "polisher", "memory_settler"):
        assert name in app.get_graph().nodes, f"缺少节点 {name}"


def test_normalize_comments_fills_missing_fields():
    """模型少写字段是常态，必须补齐——撰稿端是按 c['severity'] 下标取值的，
    少一个字段就会 KeyError 打断整条流水线。"""
    got = nodes._normalize_comments([{"issue": "只写了 issue"}])
    assert got[0]["severity"] == "minor"       # 没写严重度 → 按 minor，不触发重写
    assert got[0]["type"] == "logic"
    assert got[0]["quote"] == ""
    assert got[0]["suggestion"] == ""


def test_normalize_comments_severity_judgement():
    assert nodes._normalize_comments([{"severity": "critical"}])[0]["severity"] == "critical"
    assert nodes._normalize_comments([{"severity": "CRITICAL"}])[0]["severity"] == "critical"
    assert nodes._normalize_comments([{"severity": "minor"}])[0]["severity"] == "minor"
    assert nodes._normalize_comments([{"severity": "严重"}])[0]["severity"] == "minor"


def test_normalize_comments_drops_non_dict_items():
    assert nodes._normalize_comments(["字符串", None, 3]) == []


def test_normalize_comments_handles_none():
    assert nodes._normalize_comments(None) == []


def test_memory_context_empty_on_first_chapter():
    assert "暂无故事记忆" in nodes._memory_context({})


def test_memory_context_renders_three_blocks():
    st = {"memory": {
        "chapter_summaries": [{"chapter": 1, "summary": "甲"}],
        "foreshadow_pool": [{"id": "F1", "desc": "乙", "chapter": 1, "status": "open"}],
        "character_states": [{"name": "林岸", "state": "戒备"}],
    }}
    txt = nodes._memory_context(st)
    assert "前情摘要" in txt and "伏笔池" in txt and "人物状态快照" in txt
    assert "F1" in txt and "林岸" in txt


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
    for key in ("index", "title", "outline", "turning_point", "notes"):
        assert key in plan, f"缺少 {key}"
    assert plan["outline"] == "" and plan["title"] == "第1章"


def test_chapter_plan_treats_null_fields_as_empty():
    """字段存在但为 null 也要兜住（模型偶尔会明确输出 null）。"""
    st = {"chapter_index": 1, "outline": {"chapters": [
        {"index": 1, "title": None, "outline": None, "turning_point": None}]}}
    plan = nodes._chapter_plan(st)
    assert plan["outline"] == "" and plan["title"] == "第1章"


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
