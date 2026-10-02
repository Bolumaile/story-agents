"""JSON 修复链回归测试。

llm.parse_json 是本项目最"救火"的一段代码——每个分支都对应一次真实故障，
改它之前先跑这里。所有用例都是纯字符串处理，不联网。
"""
import json

import pytest

from llm import JSONParseError, parse_json, _salvage_objects


# ── ① 正常路径 ────────────────────────────────────────────────
def test_plain_json():
    assert parse_json('{"verdict": "pass"}') == {"verdict": "pass"}


def test_json_wrapped_in_markdown_fence():
    """模型爱把 JSON 包在 ```json 里，必须剥掉围栏。"""
    text = '```json\n{"verdict": "fail", "comments": []}\n```'
    assert parse_json(text) == {"verdict": "fail", "comments": []}


def test_json_with_fence_and_no_language_tag():
    assert parse_json('```\n{"a": 1}\n```') == {"a": 1}


def test_json_after_chitchat_prefix():
    """JSON 前面带寒暄/解释，要能截出花括号之间的内容。"""
    text = '好的，我来校验这一章：\n{"verdict": "pass", "comments": []}\n以上是结果。'
    assert parse_json(text)["verdict"] == "pass"


# ── ② 核心坑：字符串值里未转义的英文双引号 ──────────────────────
# 报错形如 Expecting ',' delimiter —— 用户真实遇到过的那次就是这个。
def test_unescaped_quotes_inside_string_value():
    raw = ('{"verdict": "fail", "comments": [{"type": "logic", "severity": "critical", '
           '"quote": "他把地图叠好，塞回遮阳板", '
           '"issue": "改"删"字用词不当", '
           '"suggestion": "删除"塞回遮阳板"这半句"}]}')
    data = parse_json(raw)
    assert data["verdict"] == "fail"
    c = data["comments"][0]
    assert c["quote"] == "他把地图叠好，塞回遮阳板"
    # 内层引号应被转义保留，而不是把内容吃掉
    assert c["suggestion"] == '删除"塞回遮阳板"这半句'
    assert c["issue"] == '改"删"字用词不当'


def test_unescaped_quote_before_closing_bracket():
    """内层引号后面紧跟收尾括号，最容易被误判成合法收尾。"""
    raw = '{"verdict": "fail", "comments": [{"issue": "不应出现"这句话""}]}'
    # 这里 "这句话" 的收尾引号后被误判的概率最高，只要整体还能解析出 issue 即算通过
    data = parse_json(raw)
    joined = json.dumps(data, ensure_ascii=False)
    assert "不应出现" in joined


def test_existing_valid_escapes_are_preserved():
    """已经正确转义的 \\" 不能被二次破坏。"""
    raw = '{"suggestion": "把 \\"好\\" 改成 \\"很好\\""}'
    assert parse_json(raw)["suggestion"] == '把 "好" 改成 "很好"'


def test_chinese_quotes_are_untouched():
    """中文引号「」是提示词里推荐的写法，不能被改写。"""
    raw = '{"suggestion": "删除「塞回遮阳板」这半句。"}'
    assert parse_json(raw)["suggestion"] == "删除「塞回遮阳板」这半句。"


def test_real_newline_inside_string_is_escaped():
    """字符串里出现真实换行也是非法 JSON，要转成 \\n。"""
    raw = '{"summary": "第一行\n第二行"}'
    assert parse_json(raw)["summary"] == "第一行\n第二行"


# ── ③ 截断：补全括号 / 回退到最后一个完整元素 ──────────────────
def test_truncated_missing_closing_braces():
    raw = '{"verdict": "fail", "comments": [{"type": "logic"'
    data = parse_json(raw)
    assert data["verdict"] == "fail"
    assert isinstance(data["comments"], list)


def test_truncated_mid_value_unclosed_string():
    raw = '{"verdict": "pass", "summary": "话说到一半'
    data = parse_json(raw)
    assert data["verdict"] == "pass"


def test_truncated_dangling_comma_and_key():
    raw = '{"verdict": "pass", "comments": [],'
    assert parse_json(raw)["verdict"] == "pass"


def test_truncated_array_falls_back_to_last_complete_item():
    """截断在元素中间：应丢掉半截元素，保住前面完整的那些。"""
    raw = ('{"verdict": "fail", "comments": ['
           '{"type": "ooc", "severity": "critical", "issue": "甲", "suggestion": "乙"}, '
           '{"type": "logic", "severity": "crit')
    data = parse_json(raw)
    assert data["verdict"] == "fail"
    kept = data.get("comments") or []
    assert any(c.get("issue") == "甲" for c in kept)


# ── ④ 失败路径：必须抛 JSONParseError，且错误信息要能指路 ────────
def test_empty_content_raises():
    with pytest.raises(JSONParseError) as ei:
        parse_json("")
    assert "空" in str(ei.value)
    assert ei.value.raw == ""


def test_pure_prose_raises():
    with pytest.raises(JSONParseError):
        parse_json("这一章写得不错，但我认为时间线有点问题，建议修改。")


def test_unbalanced_braces_are_diagnosed_as_truncation():
    """括号没闭合时，诊断要指向「被长度截断」，而不是笼统的格式错误。

    这条直接影响前端给用户的提示是否准确（曾经把截断误报成引号问题）。
    注：修复链很能打，真实截断绝大多数已被救回——所以这里用最小不平衡输入
    来固定这个诊断分支，避免它被无声改掉。
    """
    with pytest.raises(JSONParseError) as ei:
        parse_json("{")
    assert "截断" in str(ei.value)


def test_truncation_error_helper_message_is_actionable():
    """生产里的截断主要走 _truncation_error（chat 发现 finish_reason=length 时抛），
    文案必须给出长度上限与「关掉深度思考」的建议。"""
    from llm import _truncation_error

    err = _truncation_error("正文" * 100, 8192)
    msg = str(err)
    assert "截断" in msg
    assert "8192" in msg
    assert "深度思考" in msg
    assert err.raw, "必须带上已生成的原文，供上层展示"


def test_error_keeps_raw_output_for_upper_layer():
    """JSONParseError 要带原文——fail-open 告警要靠它写出可读的提示。"""
    raw = "这不是 JSON"
    with pytest.raises(JSONParseError) as ei:
        parse_json(raw)
    assert ei.value.raw == raw
    assert raw[:200] in str(ei.value)


def test_json_array_is_rejected():
    """本项目所有角色都要求输出对象，数组不算成功。"""
    with pytest.raises(JSONParseError):
        parse_json("[1, 2, 3]")


def test_empty_object_is_rejected():
    """空对象没有信息量，当作失败处理（原文 obj 为假值时会继续找候选）。"""
    with pytest.raises(JSONParseError):
        parse_json("{}")


# ── ⑤ 兜底捞取：整份废了也要把完整的那几条评论捞回来 ──────────────
def test_salvage_picks_verdict_and_comments():
    fragment = ('{"verdict":"fail","comments":['
                '{"type":"ooc","issue":"甲","suggestion":"乙"},'
                '{"type":"logic","issue":"丙","suggestion":"丁"}')
    data = _salvage_objects(fragment)
    assert data["verdict"] == "fail"
    issues = {c["issue"] for c in data["comments"]}
    assert {"甲", "丙"} <= issues


def test_salvage_picks_summary_key():
    data = _salvage_objects('{"summary":"林岸收到半页笔记"}')
    assert data["summary"] == "林岸收到半页笔记"


def test_salvage_on_garbage_returns_empty():
    assert _salvage_objects("完全不是 JSON") == {}
