"""续写超出策划案章数时，本章大纲的自动补写（chapter_planner_node）。

背景（2026-10-02 实跑暴露）：需求里写着"共3章"，写完第 3 章之后作者又续写第 4 章，
而 outline.chapters 只有 3 条 → _chapter_plan() 走兜底分支，
第 4 章既没有标题也没有分章大纲，落盘标题成了 "# 第4章"，正文全靠自由发挥。
这个文件守的就是这条链路。
"""
import graph as graph_mod
import llm
import nodes

from conftest import initial_state


def _boom(*args, **kwargs):
    raise AssertionError("这条路径不该调用模型")


def _state(idx, chapters, finals=None):
    st = initial_state()
    st["chapter_index"] = idx
    st["outline"] = {"title": "旧堤", "theme": "守与放", "core_conflict": "真相与安全",
                     "ending_direction": "开放式", "characters": [],
                     "chapters": chapters}
    st["final_chapters"] = finals or []
    return st


CHAPTER_JSON = {
    "title": "芦苇荡里的第二个脚印",
    "outline": "承接上一章，苏晓沿堤往东，发现第二组脚印。",
    "turning_point": "脚印的尺码与老周的靴子一致",
    "beats": [{"goal": "沿堤东行", "conflict": "无", "turn": "发现脚印",
               "words": 600, "emotion": 2},
              {"goal": "比对靴印", "conflict": "内心抗拒", "turn": "确认来源",
               "words": 800, "emotion": 4}],
    "notes": "",
}


def _fake_chat(payload=CHAPTER_JSON, sink=None):
    """替身：注意 llm.chat_json 返回的是**已解析的 dict**，不是 JSON 字符串。"""
    def _chat(system, user, temperature=0.3):
        if sink is not None:
            sink["system"], sink["user"] = system, user
        return dict(payload)
    return _chat


# ── ① 本章已在策划案内 → 必须是空操作 ──────────────────────────
def test_noop_when_chapter_already_in_outline(monkeypatch):
    """已有大纲就什么都不做——否则每章白烧一次模型调用，还可能覆盖原案。"""
    monkeypatch.setattr(llm, "chat_json", _boom)
    st = _state(2, [{"index": 1, "title": "甲"}, {"index": 2, "title": "乙"}])
    assert nodes.chapter_planner_node(st) == {}


def test_noop_when_outline_index_is_string(monkeypatch):
    """旧存档的 index 可能是字符串，比对不归一会重复补写同一章。"""
    monkeypatch.setattr(llm, "chat_json", _boom)
    st = _state(2, [{"index": "2", "title": "乙"}])
    assert nodes.chapter_planner_node(st) == {}


# ── ② 超出策划案 → 补写并写回 ──────────────────────────────────
def test_extends_when_chapter_beyond_outline(monkeypatch):
    monkeypatch.setattr(llm, "chat_json", _fake_chat())
    st = _state(4, [{"index": i, "title": f"第{i}章"} for i in (1, 2, 3)])
    out = nodes.chapter_planner_node(st)

    chs = out["outline"]["chapters"]
    assert len(chs) == 4
    assert chs[-1]["index"] == 4, "章号必须按实际章号钉死（模型常漏写 index）"
    assert chs[-1]["title"] == "芦苇荡里的第二个脚印"
    assert len(chs[-1]["beats"]) == 2
    # 原始 outline 不能被就地改写：state 里的对象别处还在引用
    assert len(st["outline"]["chapters"]) == 3


def test_extended_chapter_is_visible_to_chapter_plan(monkeypatch):
    """补完之后 _chapter_plan() 必须取得到它，否则等于白补——
    撰稿端拿到的仍然是「第4章」兜底，问题一点没解决。
    """
    monkeypatch.setattr(llm, "chat_json", _fake_chat())
    st = _state(4, [{"index": i, "title": f"第{i}章"} for i in (1, 2, 3)])
    st["outline"] = nodes.chapter_planner_node(st)["outline"]

    plan = nodes._chapter_plan(st)
    assert plan["title"] == "芦苇荡里的第二个脚印"
    assert plan["beats"], "补写的场景节拍要能传到撰稿端"


def test_index_pinned_even_if_model_omits_it(monkeypatch):
    bad = {k: v for k, v in CHAPTER_JSON.items() if k != "index"}
    monkeypatch.setattr(llm, "chat_json", _fake_chat(bad))
    st = _state(5, [{"index": 1, "title": "甲"}])
    out = nodes.chapter_planner_node(st)
    assert out["outline"]["chapters"][-1]["index"] == 5


def test_empty_title_falls_back_to_chapter_number(monkeypatch):
    """模型给了大纲但没给标题：仍然写回（梗概/节拍有用），标题退化成章号。"""
    bad = dict(CHAPTER_JSON, title="   ")
    monkeypatch.setattr(llm, "chat_json", _fake_chat(bad))
    st = _state(4, [{"index": 1, "title": "甲"}])
    out = nodes.chapter_planner_node(st)
    assert out["outline"]["chapters"][-1]["title"] == "第4章"


def test_prompt_carries_role_name_anchor_and_previous_ending(monkeypatch):
    """三件事缺一不可：英文角色名（Mock 靠它分流）、章号锚点、上一章结尾。"""
    seen = {}
    monkeypatch.setattr(llm, "chat_json", _fake_chat(sink=seen))
    finals = [{"index": 3, "title": "丙", "text": "雨停在堤的尽头。"}]
    st = _state(4, [{"index": i, "title": f"第{i}章"} for i in (1, 2, 3)], finals)
    nodes.chapter_planner_node(st)

    assert "ChapterExtender" in seen["system"], "丢了英文角色名，Mock 会掉进 Writer 兜底"
    assert "【本章序号】4" in seen["user"], "章号锚点丢了会让模型/mock 取错章号"
    assert "雨停在堤的尽头。" in seen["user"], "没带上一章结尾，续写接不上"


# ── ③ 降级路径：补不出来也不能拦路，但必须告警 ──────────────────
def test_model_failure_degrades_with_warning(monkeypatch):
    notices = []
    llm.set_notice_cb(lambda level, msg: notices.append((level, msg)))

    def boom(*args, **kwargs):
        raise RuntimeError("模型超时")

    monkeypatch.setattr(llm, "chat_json", boom)
    st = _state(4, [{"index": i, "title": f"第{i}章"} for i in (1, 2, 3)])

    assert nodes.chapter_planner_node(st) == {}, "失败时不该改动 outline"
    assert notices, "无声降级就是原来的 bug，必须告警"
    assert notices[-1][0] == "warn"
    assert "第4章" in notices[-1][1]


def test_success_emits_info_notice(monkeypatch):
    notices = []
    llm.set_notice_cb(lambda level, msg: notices.append((level, msg)))
    monkeypatch.setattr(llm, "chat_json", _fake_chat())

    st = _state(4, [{"index": i, "title": f"第{i}章"} for i in (1, 2, 3)])
    nodes.chapter_planner_node(st)
    assert notices and notices[-1][0] == "info"
    assert "芦苇荡里的第二个脚印" in notices[-1][1]


# ── ④ 端到端（Mock）：标题不许再退化成流水号 ────────────────────
def _run(app, state, n):
    for idx in range(1, n + 1):
        state.update({
            "chapter_index": idx, "chapter_draft": "", "review_comments": [],
            "review_verdict": "pass", "review_comments_ooc": [],
            "review_comments_logic": [], "review_comments_pacing": [],
            "style_report": {}, "revision_round": 0, "final_chapter": "",
        })
        state = graph_mod.run_chapter(app, state)
    return state


def test_mock_pipeline_beyond_plan_gets_real_titles(mock_mode):
    """Mock 的策划案只给 1 章，跑到第 2、3 章时必须自动补大纲。"""
    app = graph_mod.build_graph()
    state = _run(app, initial_state(), 3)

    titles = [c["title"] for c in state["final_chapters"]]
    assert titles[0] == "雨夜来客", "第 1 章沿用策划案原标题"
    assert len(state["outline"]["chapters"]) == 3, "第 2、3 章的大纲应已补进策划案"
    for idx in (2, 3):
        assert titles[idx - 1] != f"第{idx}章", f"第 {idx} 章标题退化成了流水号"
