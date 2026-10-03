"""产物写入（products.py）的回归：双标题、原子写、每章刷全套文件。

对应代码审查里的三条：

- **P0-1 双标题**：程序会自己拼一个一级标题，而 `final_chapter`（润色输出）
  正文里**本来就带一个**，于是每个产物文件开头都是「# 标题」+「# 第N章 标题」，
  甚至两行一模一样。下面的 `REAL_BODY` 直接取自修复前真实产物
  `outputs_web/run_20261002_225256/chapter_01.md` 的正文，不是编造的样例。
- **P0-4 原子写**：以前就地覆盖写，写一半崩 = 半截 md / 非法 JSON，且下次覆盖
  不自愈。现在统一 tmp + os.replace。
- **P1-3 每章都刷全套**：网页端原先只在"所有章节都成功"之后才写
  outline.json / memory.json，中途失败产物目录里就只剩 chapter_XX.md。
"""
import json
import os

import pytest

import products


# 修复前真实产物的正文：首行是正文自带的一级标题（程序又拼了一个，成了两行标题）
REAL_BODY = (
    "# 第一章 积水线\n"
    "\n"
    "雨落到第二十三天，巷口的积水已经漫过了第三级台阶。\n"
)
# 另一种真实形态：程序拼的标题与正文里的一级标题**完全相同**，看起来像复制粘贴出错
SAME_BODY = "# 天桥下的半个罐头\n\n他把最后半罐推给猫。\n"


# ── 标题归一 ─────────────────────────────────────────────────────

def test_real_double_title_body_is_normalized():
    out = products.compose_chapter_md("积水线", REAL_BODY)
    heads = [ln for ln in out.splitlines() if ln.startswith("# ")]
    assert heads == ["# 积水线"], f"应当只剩程序拼的那一个一级标题，实际 {heads}"
    assert "雨落到第二十三天" in out


def test_identical_titles_do_not_appear_twice():
    out = products.compose_chapter_md("天桥下的半个罐头", SAME_BODY)
    assert out.count("# 天桥下的半个罐头") == 1


def test_strip_leading_h1_only_touches_first_line():
    """二级标题、正文中间的标题都不能被误删。"""
    text = "## 场景一\n\n正文\n\n# 尾声\n\n完"
    assert products.strip_leading_h1(text) == text


def test_strip_leading_h1_leaves_plain_text_alone():
    assert products.strip_leading_h1("直接就是正文\n第二行") == "直接就是正文\n第二行"


def test_strip_leading_h1_tolerates_empty():
    assert products.strip_leading_h1("") == ""
    assert products.strip_leading_h1(None) == ""


def test_chapter_body_headings_are_demoted_one_level():
    """正文里残留的 `# xxx` 要降到 `##`，不能跟章标题抢同一层。"""
    out = products.compose_chapter_md("积水线", "# 积水线\n\n正文\n\n# 尾声\n\n完")
    assert "\n## 尾声\n" in out
    assert "\n# 尾声\n" not in out


def test_final_md_demotes_body_headings_two_levels():
    out = products.compose_final_md("测试小说", [
        {"index": 1, "title": "积水线", "text": "# 第一章 积水线\n\n正文\n\n# 尾声\n\n完"}])
    assert out.startswith("# 测试小说\n")
    assert "\n## 第1章 积水线\n" in out
    assert "\n### 尾声\n" in out          # 章标题是 ##，正文标题必须再低一级
    assert out.count("# 第一章 积水线") == 0


def test_final_md_keeps_all_chapters_in_order():
    out = products.compose_final_md("书名", [
        {"index": 1, "title": "甲", "text": "一"},
        {"index": 2, "title": "乙", "text": "二"}])
    assert out.index("## 第1章 甲") < out.index("## 第2章 乙")
    assert out.rstrip().endswith("二")


# ── 原子写 ───────────────────────────────────────────────────────

def test_atomic_write_text_creates_file_without_leaving_tmp(tmp_path):
    p = tmp_path / "sub" / "a.md"
    products.atomic_write_text(str(p), "内容")
    assert p.read_text(encoding="utf-8") == "内容"
    assert not (tmp_path / "sub" / "a.md.tmp").exists()


def test_atomic_write_json_roundtrips_chinese(tmp_path):
    p = tmp_path / "m.json"
    products.atomic_write_json(str(p), {"items": ["矿泉水"], "标题": "积水线"})
    assert json.loads(p.read_text(encoding="utf-8"))["标题"] == "积水线"


def test_failed_write_keeps_previous_content_intact(tmp_path, monkeypatch):
    """写一半失败时，旧文件必须完好 —— 这正是"就地覆盖"做不到的。"""
    p = tmp_path / "m.json"
    p.write_text('{"old": 1}', encoding="utf-8")

    def boom(src, dst):
        raise OSError("模拟断电/磁盘满")

    monkeypatch.setattr(products.os, "replace", boom)
    with pytest.raises(OSError):
        products.atomic_write_json(str(p), {"new": 2})

    assert p.read_text(encoding="utf-8") == '{"old": 1}'      # 旧内容没被破坏
    assert not (tmp_path / "m.json.tmp").exists()             # 临时文件已清理


# ── 每章刷全套产物 ───────────────────────────────────────────────

def _state(chapters):
    return {
        "outline": {"title": "测试小说"},
        "memory": {"items": ["矿泉水"]},
        "final_chapters": chapters,
    }


def test_one_chapter_writes_all_four_products(tmp_path):
    """只写了 1 章也必须有大纲与记忆库（P1-3：以前它们写在循环之外）。"""
    st = _state([{"index": 1, "title": "积水线", "text": REAL_BODY}])
    products.write_chapter_products(str(tmp_path), st)

    for name in ("outline.json", "memory.json", "chapter_01.md", "final.md"):
        assert (tmp_path / name).exists(), f"缺 {name}"
    assert json.loads((tmp_path / "memory.json").read_text(encoding="utf-8")) == \
        {"items": ["矿泉水"]}
    assert (tmp_path / "chapter_01.md").read_text(encoding="utf-8").startswith("# 积水线")


def test_midway_failure_still_leaves_complete_directory(tmp_path):
    """模拟"第 2 章崩了"：第 1 章的产物必须自洽，不需要 final.md 才补写。"""
    st = _state([{"index": 1, "title": "甲", "text": "正文一"}])
    products.write_chapter_products(str(tmp_path), st)      # 第 1 章后落盘
    # 第 2 章崩了，什么都不写 —— 目录里此时仍应有完整的四件套
    for name in ("outline.json", "memory.json", "chapter_01.md", "final.md"):
        assert (tmp_path / name).exists(), f"第 2 章中途失败后缺 {name}"


def test_write_is_idempotent_and_advances_with_chapters(tmp_path):
    st = _state([{"index": 1, "title": "甲", "text": "正文一"}])
    products.write_chapter_products(str(tmp_path), st)
    products.write_chapter_products(str(tmp_path), st)     # 重跑一次不该出错
    st["final_chapters"].append({"index": 2, "title": "乙", "text": "正文二"})
    products.write_chapter_products(str(tmp_path), st)

    assert (tmp_path / "chapter_01.md").exists()
    assert (tmp_path / "chapter_02.md").exists()
    final = (tmp_path / "final.md").read_text(encoding="utf-8")
    assert "## 第1章 甲" in final and "## 第2章 乙" in final


def test_product_files_have_no_double_title_on_real_shape(tmp_path):
    """把真实形态走完整条链路，产物里每个文件的一级标题都只能有一个。"""
    st = _state([{"index": 1, "title": "积水线", "text": REAL_BODY}])
    products.write_chapter_products(str(tmp_path), st)

    ch = (tmp_path / "chapter_01.md").read_text(encoding="utf-8")
    assert [ln for ln in ch.splitlines() if ln.startswith("# ")] == ["# 积水线"]

    final = (tmp_path / "final.md").read_text(encoding="utf-8")
    assert [ln for ln in final.splitlines() if ln.startswith("# ")] == ["# 测试小说"]


def test_products_dir_name_never_leaks_absolute_path():
    assert products.products_dir_name(os.path.join("a", "b", "run_1")) == "run_1"
    assert products.products_dir_name(None) is None
