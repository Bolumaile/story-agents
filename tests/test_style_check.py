"""去 AI 味的规则层检测（优化建议第 8 条）。

这一层只做提示、不改稿，所以测试关注的是「能不能把明显的模型腔指出来、
且不会对着正常白描乱报」。
"""
import style_check as sc

# 高密度套话 + 短句均质 + 段末升华，典型的模型腔
AI_TEXT = (
    "雨还在下。然而，他此刻却仿佛听见了什么。只见远处的灯忽然亮起，他不禁心头一紧。"
    "与此同时，门外的脚步声宛如催命的鼓点。瞬间，空气仿佛凝固了。他猛地站起身，犹如困兽。"
    "忽然，门开了。此刻，他终于明白，这一切不过是开始。或许，这就是命运的安排。就这样，他走了出去。"
    "然而雨没有停。仿佛一切都没有发生过。只见他回头看了一眼，不禁又想起那句话。"
    "这一切，终究会过去的。瞬间，他觉得夜色忽然变得很轻。"
) * 2

# 白描：无套话、无明喻、无升华，全靠动作与物象说话
HUMAN_TEXT = (
    "雨下了一夜。他把鞋晾在门槛上，鞋尖朝里。灶上温着半锅粥，结了一层皮。"
    "他拿筷子挑开，又盖上。门外有脚步，走过去，没停。他坐着没动，听那脚步一直走到巷子尽头。"
    "然后他站起来，把粥倒回锅里，添了半碗水。火苗缩了一下，又立起来。"
    "天亮前他出了门，顺手把门带上，没上锁。"
) * 2


# ── 统计口径 ─────────────────────────────────────────────────
def test_stats_are_always_present():
    s = sc.check(HUMAN_TEXT)["stats"]
    for key in ("chars", "sentences", "avg_sentence", "sentence_std",
                "cliche_total", "cliche_per_k", "cliche_words",
                "transition_per_k", "simile_per_k", "summary_ratio"):
        assert key in s, f"缺少统计项 {key}"


def test_chars_count_ignores_whitespace():
    assert sc.analyze("一 二\n三")["chars"] == 3


def test_sentence_counting_splits_on_chinese_punctuation():
    assert sc.analyze("甲。乙！丙？丁")["sentences"] == 4


# ── 检测能力 ─────────────────────────────────────────────────
def test_cliche_density_is_flagged():
    assert any(f["code"] == "cliche" for f in sc.check(AI_TEXT)["findings"])


def test_simile_density_is_flagged():
    assert any(f["code"] == "simile" for f in sc.check(AI_TEXT)["findings"])


def test_transition_density_is_flagged():
    assert any(f["code"] == "transition" for f in sc.check(AI_TEXT)["findings"])


def test_summary_sentence_density_is_flagged():
    assert any(f["code"] == "summary" for f in sc.check(AI_TEXT)["findings"])


def test_clean_prose_is_not_flagged_for_cliche():
    """白描文本不该被扣上"套话"或"比喻"的帽子——否则提示会被用户无视。"""
    codes = {f["code"] for f in sc.check(HUMAN_TEXT)["findings"]}
    assert "cliche" not in codes
    assert "simile" not in codes
    assert "summary" not in codes


def test_cliche_stats_are_accurate():
    st = sc.analyze("然而然而然而")
    assert st["cliche_total"] == 3


# ── 边界 ─────────────────────────────────────────────────────
def test_short_text_skips_judgement():
    """太短的片段统计意义不足，不该下结论。"""
    r = sc.check("雨还在下。")
    assert r["findings"] == []
    assert r["stats"]["chars"] > 0


def test_empty_text_is_safe():
    r = sc.check("")
    assert r["findings"] == [] and r["stats"]["chars"] == 0


def test_none_text_is_safe():
    """style_check 挂在定稿路径上，任何输入都不能抛异常。"""
    assert sc.check(None)["findings"] == []


def test_format_findings_is_single_line():
    line = sc.format_findings(sc.check(AI_TEXT)["findings"])
    assert line and "\n" not in line


def test_format_findings_handles_empty():
    assert sc.format_findings([]) == ""
