"""去 AI 味：规则层检测（纯代码，不消耗 token）。

对应优化建议第 8 条。这里只做**检测与提示**，不自动改稿——
AI 腔的判定天然带主观性，自动替换容易把作者的个人风格也一起抹平。
提示交给人来决策，模型层的软化则交给润色环节的提示词。

检测四个信号（中文小说语境下的经验阈值，可按自己的口味调）：
1. **高频连接词/套话**：「然而」「只见」「此刻」「仿佛」这类词密度过高时，
   文字会有明显的"模型味"。
2. **句式长度方差**：句子长度过于均匀（标准差小）时，读起来节奏平、缺少起伏。
3. **总结句密度**：叙述中间频繁出现"这一切""从此""终究"式的收束句，
   是模型爱在段末升华的典型习惯。
4. **比喻密度**：每千字比喻数量过高，显得用力过猛。

统计口径：字数按去空白后的字符数计（中文里约等于字数）。
"""
import re
import statistics
from typing import Any, Dict, List

# ── 词表 ─────────────────────────────────────────────────────────

# 套话 / 模型高频用语
CLICHE_WORDS = (
    "然而", "只见", "此刻", "仿佛", "彷佛", "忽然", "猛地", "瞬间",
    "不禁", "宛如", "犹如", "恍若", "竟然", "竟", "与此同时", "不由得",
)

# 转折/推进连接词（密度过高会让行文像说明文）
TRANSITION_WORDS = ("然而", "但是", "不过", "于是", "因此", "所以", "而后")

# 比喻标记词
SIMILE_WORDS = ("像", "仿佛", "彷佛", "如同", "宛如", "好似", "犹如", "恍如", "似")

# 收束/升华句的句首标记
SUMMARY_HEADS = ("这一切", "从此", "就这样", "总之", "或许", "也许", "终究",
                 "最终", "而这", "那一刻")

# ── 阈值（每千字次数 / 绝对量）────────────────────────────────────

TH_CLICHE_TOTAL = 8.0        # 套话合计：每千字超过 8 次
TH_CLICHE_SINGLE = 2.5       # 单个套话词：每千字超过 2.5 次
TH_TRANSITION = 9.0          # 连接词合计：每千字超过 9 次
TH_SIMILE = 6.0              # 比喻：每千字超过 6 处
TH_SUMMARY_RATIO = 0.08      # 总结句占比超过 8%
TH_MIN_SENTENCE_STD = 4.5    # 句长标准差低于 4.5，视为节奏过平
# 句子太少时标准差没有意义（一段短白描本就该全是短句），
# 所以要求"句数足够多却依然高度均一"才算信号。
MIN_SAMPLES = 20

_SENT_SPLIT = re.compile(r"[。！？!?…\n]+")


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if s.strip()]


def analyze(text: str) -> Dict[str, Any]:
    """产出一份风格统计（无论是否超阈值都给，便于前端展示）。"""
    clean = re.sub(r"\s+", "", text or "")
    n_chars = len(clean)
    sents = _sentences(text)
    lengths = [len(re.sub(r"\s+", "", s)) for s in sents]
    per_k = (lambda c: round(c * 1000.0 / n_chars, 1)) if n_chars else (lambda c: 0.0)

    cliche_hits = {w: (text or "").count(w) for w in CLICHE_WORDS}
    cliche_hits = {w: c for w, c in cliche_hits.items() if c}
    transition_total = sum((text or "").count(w) for w in TRANSITION_WORDS)
    simile_total = sum((text or "").count(w) for w in SIMILE_WORDS)
    summary_cnt = sum(1 for s in sents if s.startswith(SUMMARY_HEADS))

    std = round(statistics.pstdev(lengths), 2) if len(lengths) >= 2 else 0.0
    return {
        "chars": n_chars,
        "sentences": len(sents),
        "avg_sentence": round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
        "sentence_std": std,
        "cliche_total": sum(cliche_hits.values()),
        "cliche_per_k": per_k(sum(cliche_hits.values())),
        "cliche_words": dict(sorted(cliche_hits.items(), key=lambda kv: -kv[1])[:6]),
        "transition_per_k": per_k(transition_total),
        "simile_per_k": per_k(simile_total),
        "summary_ratio": round(summary_cnt / len(sents), 3) if sents else 0.0,
    }


def check(text: str) -> Dict[str, Any]:
    """检测并返回 {stats, findings}。findings 为空表示没发现问题。"""
    st = analyze(text)
    findings: List[Dict[str, str]] = []

    if st["chars"] < 200:
        # 太短没统计意义，直接跳过判断（仍返回 stats）
        return {"stats": st, "findings": findings}

    # 1. 套话密度
    if st["cliche_per_k"] > TH_CLICHE_TOTAL or any(
            (c * 1000.0 / st["chars"]) > TH_CLICHE_SINGLE for c in st["cliche_words"].values()):
        top = "、".join(f"「{w}」{c}次" for w, c in list(st["cliche_words"].items())[:4])
        findings.append({
            "code": "cliche",
            "label": "套话偏多",
            "detail": f"模型高频用语合计 {st['cliche_total']} 次"
                      f"（每千字 {st['cliche_per_k']} 次）：{top}",
        })

    # 2. 连接词密度
    if st["transition_per_k"] > TH_TRANSITION:
        findings.append({
            "code": "transition",
            "label": "连接词过密",
            "detail": f"转折/推进词每千字 {st['transition_per_k']} 次，行文像说明文，"
                      f"可删掉一半让动作自己说话。",
        })

    # 3. 句长节奏
    if st["sentences"] >= MIN_SAMPLES and st["sentence_std"] < TH_MIN_SENTENCE_STD:
        findings.append({
            "code": "rhythm",
            "label": "句式节奏偏平",
            "detail": f"句长标准差 {st['sentence_std']}（低于 {TH_MIN_SENTENCE_STD}），"
                      f"平均句长 {st['avg_sentence']} 字——长短句交错会更有呼吸感。",
        })

    # 4. 比喻密度
    if st["simile_per_k"] > TH_SIMILE:
        findings.append({
            "code": "simile",
            "label": "比喻偏密",
            "detail": f"比喻标记每千字 {st['simile_per_k']} 处，"
                      f"留几处最狠的，其余白描即可。",
        })

    # 5. 总结句密度
    if st["summary_ratio"] > TH_SUMMARY_RATIO:
        findings.append({
            "code": "summary",
            "label": "收束句偏多",
            "detail": f"段末升华式收束句占 {st['summary_ratio']:.0%}"
                      f"（阈值 {TH_SUMMARY_RATIO:.0%}），是「急着替读者总结」的习惯。",
        })

    return {"stats": st, "findings": findings}


def format_findings(findings: List[Dict[str, str]]) -> str:
    """把 findings 压成一行告警文案。"""
    return "；".join(f"{f['label']}（{f['detail']}）" for f in findings)
