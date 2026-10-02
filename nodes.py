"""四个 Agent 节点实现。每个节点 = 读 State → 调 LLM → 写回 State。"""
import config
import prompts
import llm
from state import StoryState


def _memory_context(state: StoryState) -> str:
    """故事记忆：前情摘要 + 伏笔池 + 人物状态快照（替代全文塞上下文）。"""
    mem = state.get("memory") or {}
    summaries = mem.get("chapter_summaries") or []
    pool = mem.get("foreshadow_pool") or []
    chars = mem.get("character_states") or []
    if not summaries and not pool and not chars:
        return "（这是第一章，暂无故事记忆）"
    lines = []
    if summaries:
        lines.append("【前情摘要（按章，只含已发生事实）】")
        for s in summaries:
            lines.append(f"- 第{s['chapter']}章：{s['summary']}")
    if pool:
        lines.append("【伏笔池】")
        for f in pool:
            lines.append(f"- [{f['status']}] {f['id']}：{f['desc']}（埋于第{f['chapter']}章）")
    if chars:
        lines.append("【人物状态快照】")
        for cs in chars:
            lines.append(f"- {cs['name']}：{cs['state']}")
    return "\n".join(lines)


def _last_chapter_text(state: StoryState) -> str:
    """只带上一章全文，用于文风衔接（更早的章节走摘要）。"""
    finals = state.get("final_chapters") or []
    if not finals:
        return ""
    ch = finals[-1]
    return f"【上一章全文（衔接用）】\n第{ch['index']}章 {ch['title']}\n{ch['text']}"


def _chapter_plan(state: StoryState) -> dict:
    idx = state["chapter_index"]
    chapters = state["outline"].get("chapters", [])
    for ch in chapters:
        if ch.get("index") == idx:
            # 补齐模型可能漏写的字段（与 _normalize_comments 同理）：
            # 下游撰稿端是按 plan['outline'] 下标取值的，模型少写一个键
            # 就会 KeyError 把整条流水线打断——宁可留空，也不要炸。
            plan = dict(ch)
            plan["index"] = idx
            plan["title"] = ch.get("title") or f"第{idx}章"
            plan["outline"] = ch.get("outline") or ""
            plan["turning_point"] = ch.get("turning_point") or ""
            plan["notes"] = ch.get("notes") or ""
            return plan
    # 章节数超出策划案时，让撰稿人根据既有信息自行续写
    return {"index": idx, "title": f"第{idx}章", "outline": "（按核心冲突自然推进）", "turning_point": "", "notes": ""}


def _meta_sections(state: StoryState, for_role: str) -> list:
    """把表单收集的可选材料编译成 prompt 片段。for_role: writer/reviewer/polisher。"""
    meta = state.get("meta") or {}
    sections = []
    if for_role in ("writer", "polisher") and meta.get("style_sample"):
        sections.append(
            f"\n【文风参考】只模仿其语气与节奏，禁止抄袭原句：\n{meta['style_sample']}")
    if for_role == "writer":
        reqs = []
        if meta.get("word_count"):
            reqs.append(f"本章字数约 {meta['word_count']} 字（±20%）")
        if meta.get("foreshadow"):
            reqs.append(f"本章必须埋入伏笔：{meta['foreshadow']}")
        if meta.get("cliffhanger"):
            reqs.append(f"本章结尾悬念要求：{meta['cliffhanger']}")
        if reqs:
            sections.append("\n【本章硬性要求】\n" + "\n".join(f"- {r}" for r in reqs))
    if meta.get("forbidden_list"):
        items = "\n".join(f"- {x}" for x in meta["forbidden_list"])
        if for_role in ("writer", "polisher"):
            sections.append(f"\n【禁止内容（绝对不可出现）】\n{items}")
        elif for_role == "reviewer":
            sections.append(
                f"\n【禁止内容清单（逐条核对正文是否违规）】\n{items}")
    return sections


# ── 1. 策划 Agent ─────────────────────────────────────────────
def planner_node(state: StoryState) -> dict:
    user_msg = f"用户创作需求：\n{state['user_prompt']}\n\n请输出完整策划案 JSON。"
    outline = llm.chat_json(prompts.PLANNER_SYSTEM, user_msg,
                            temperature=config.TEMPERATURE_PLANNER)
    if not outline.get("chapters"):
        raise ValueError(
            "策划案里没有 chapters 字段，无法继续。"
            "多半是模型输出被截断或没按要求给结构——请重试，或换一个更稳的模型。")

    # 产品规则 1：策划输出写入记忆库——人物快照先入库，供后续撰稿/校对读取
    memory = state.get("memory") or _empty_memory()
    if not memory.get("character_states"):
        memory["character_states"] = [
            {"name": c.get("name", ""),
             "state": (f"性格：{c.get('personality', '')}；动机：{c.get('motivation', '')}"
                       + (f"；禁忌：{c.get('taboo', '')}" if c.get("taboo") else ""))}
            for c in outline.get("characters", []) if c.get("name")
        ]
    return {"outline": outline, "memory": memory}


# ── 2. 撰稿 Agent ─────────────────────────────────────────────
def writer_node(state: StoryState) -> dict:
    plan = _chapter_plan(state)
    round_no = state.get("revision_round", 0)

    sections = [
        f"【故事策划案】\n标题：{state['outline'].get('title')}",
        f"主题：{state['outline'].get('theme')}",
        f"核心冲突：{state['outline'].get('core_conflict')}",
        f"结局方向：{state['outline'].get('ending_direction')}",
        "\n【人物卡】",
    ]
    for c in state["outline"].get("characters", []):
        sections.append(
            f"- {c['name']}｜性格：{c['personality']}｜动机：{c['motivation']}"
            f"｜禁忌：{c['taboo']}｜潜在冲突：{c.get('potential_conflicts', '')}")
    sections.append(f"\n【故事记忆】\n{_memory_context(state)}")
    last = _last_chapter_text(state)
    if last:
        sections.append(f"\n{last}")
    sections.append(
        f"\n【本章任务】第{plan['index']}章《{plan['title']}》\n"
        f"情节大纲：{plan['outline']}\n转折点：{plan.get('turning_point', '')}")

    if round_no > 0 and state.get("review_comments"):
        fb = "\n".join(
            f"- [{c.get('severity')}][{c.get('type')}] 「{c.get('quote')}」 "
            f"{c.get('issue')} → 建议：{c.get('suggestion')}"
            for c in state["review_comments"])
        sections.append(
            f"\n【校对反馈（第{round_no}轮重写，revision_round={round_no}）】\n"
            f"只针对下列问题做定向修改，不要推翻全文：\n{fb}")

    sections += _meta_sections(state, "writer")

    raw = llm.chat(prompts.WRITER_SYSTEM, "\n".join(sections),
                   temperature=config.TEMPERATURE_WRITER)
    # 若为重写轮，轮数 +1 由 reviewer 路由前统一记录；这里仅在首轮写 0
    return {"chapter_draft": raw, "revision_round": round_no}


# ── 3. 校对 Agent ─────────────────────────────────────────────
def _normalize_comments(comments) -> list:
    """补齐模型可能漏写的字段。

    校对结果会直接喂给撰稿 Agent，而撰稿那边是按 c['severity'] 这种下标取值的，
    模型少写一个字段就会 KeyError，把整条流水线打断。所以先在这里统一成规整结构。
    """
    out = []
    for c in comments or []:
        if not isinstance(c, dict):
            continue
        sev = str(c.get("severity") or "").strip().lower()
        out.append({
            "type": str(c.get("type") or "logic"),
            # 判定从严：只有明确写 critical 才算严重问题，其余按 minor 处理，
            # 避免把"没写严重度"误解成严重问题而触发无意义的重写
            "severity": "critical" if sev.startswith("crit") else "minor",
            "quote": str(c.get("quote") or ""),
            "issue": str(c.get("issue") or ""),
            "suggestion": str(c.get("suggestion") or ""),
        })
    return out


def reviewer_node(state: StoryState) -> dict:
    plan = _chapter_plan(state)
    meta = state.get("meta") or {}
    char_cards = "\n".join(
        f"- {c['name']}｜{c['personality']}｜动机：{c['motivation']}｜禁忌：{c['taboo']}"
        for c in state["outline"].get("characters", []))
    reqs = []
    if meta.get("word_count"):
        reqs.append(f"字数约 {meta['word_count']} 字（±20%）")
    if meta.get("foreshadow"):
        reqs.append(f"必须埋入伏笔：{meta['foreshadow']}")
    if meta.get("cliffhanger"):
        reqs.append(f"结尾悬念：{meta['cliffhanger']}")
    req_txt = ("【本章硬性要求（未达标记 critical）】\n" + "\n".join(f"- {r}" for r in reqs)) if reqs else ""
    user_msg = (
        f"【人物卡（校验基准）】\n{char_cards}\n\n"
        f"{req_txt}\n\n"
        f"【故事记忆（查时间线与伏笔矛盾）】\n{_memory_context(state)}\n\n"
        f"{_last_chapter_text(state)}\n\n"
        f"【本章应写内容】第{plan['index']}章：{plan['outline']}\n\n"
        f"【待审正文】\n{state['chapter_draft']}\n\n"
        f"请按检查清单逐项校验，输出 JSON（revision_round={state.get('revision_round', 0)}）。")

    try:
        result = llm.chat_json(prompts.REVIEWER_SYSTEM, user_msg,
                               temperature=config.TEMPERATURE_REVIEWER)
    except Exception as e:                            # noqa: BLE001
        # 校对只是「质检」环节，不该因为它的输出格式问题，把已经写好的整章作废。
        if not config.REVIEW_FAIL_OPEN:
            raise
        llm.emit_notice(
            "warn",
            f"第{plan['index']}章校对结果无法解析（{type(e).__name__}），已自动跳过校对、"
            f"直接进入润色。本章未经逻辑与人设校验，建议您人工过一遍。"
            f"原因：{str(e)[:160]}")
        return {"review_verdict": "pass", "review_comments": []}

    verdict = str(result.get("verdict") or "pass").strip().lower()
    comments = _normalize_comments(result.get("comments"))
    crit = [c for c in comments if c["severity"] == "critical"]
    if verdict == "fail" and not crit:
        # 判了 fail 却一条 critical 都没有 → 没有任何可执行的修改点，
        # 打回重写只会空转（还可能让撰稿人越改越偏），按通过处理。
        if comments:
            llm.emit_notice(
                "warn",
                f"第{plan['index']}章校对判为不通过，但未指出任何严重问题，"
                f"已按通过处理（{len(comments)} 条意见转润色环节参考）。")
        verdict = "pass"
    return {"review_verdict": verdict, "review_comments": comments}


# ── 4. 润色 Agent ─────────────────────────────────────────────
def polisher_node(state: StoryState) -> dict:
    minors = [c for c in (state.get("review_comments") or []) if c.get("severity") == "minor"]
    user_msg = f"正文如下：\n{state['chapter_draft']}"
    if minors:
        sug = "\n".join(f"- {c['issue']} → {c['suggestion']}" for c in minors)
        user_msg = f"【润色时可参考的 minor 意见（仅措辞层面化解）】\n{sug}\n\n{user_msg}"
    meta_secs = _meta_sections(state, "polisher")
    if meta_secs:
        user_msg = "\n".join(meta_secs) + "\n\n" + user_msg
    raw = llm.chat(prompts.POLISHER_SYSTEM, user_msg,
                   temperature=config.TEMPERATURE_POLISHER)
    plan = _chapter_plan(state)
    finalized = {
        "index": plan["index"],
        "title": plan["title"],
        "text": raw,
    }
    return {
        "final_chapter": raw,
        "final_chapters": (state.get("final_chapters") or []) + [finalized],
    }


# ── 5. 记忆结算员（MemorySettler，润色定稿后运行）──────────────
def _empty_memory() -> dict:
    return {"chapter_summaries": [], "foreshadow_pool": [], "character_states": []}


def memory_settler_node(state: StoryState) -> dict:
    plan = _chapter_plan(state)
    mem = state.get("memory") or _empty_memory()
    pool_txt = "\n".join(
        f"- {f['id']}：{f['desc']}" for f in (mem.get("foreshadow_pool") or [])) or "（空）"
    user_msg = (
        f"【本章信息】第{plan['index']}章《{plan['title']}》\n"
        f"【当前伏笔池（resolved_foreshadows 只能从中选）】\n{pool_txt}\n\n"
        f"【本章定稿正文】\n{state['final_chapter']}\n\n"
        f"请输出本章记忆结算 JSON。")
    try:
        delta = llm.chat_json(prompts.MEMORY_SYSTEM, user_msg,
                              temperature=config.TEMPERATURE_REVIEWER)
    except Exception as e:                            # noqa: BLE001
        # 记忆结算失败只影响"后续章节的上下文质量"，本章已经定稿入库了，
        # 没必要因此中断整条流水线——告警即可。
        llm.emit_notice(
            "warn",
            f"第{plan['index']}章的记忆结算失败（{type(e).__name__}），"
            f"本章已正常定稿；但后续章节会缺少这部分前情记忆与伏笔记录。"
            f"原因：{str(e)[:140]}")
        return {}

    # 深拷贝后合并增量（不可原地改，保证 LangGraph 状态更新可追溯）
    mem = {k: [dict(x) if isinstance(x, dict) else x for x in v]
           for k, v in mem.items()}
    mem.setdefault("chapter_summaries", []).append(
        {"chapter": plan["index"], "summary": delta.get("summary", "")})

    resolved = set(delta.get("resolved_foreshadows") or [])
    for f in mem.setdefault("foreshadow_pool", []):
        if f.get("id") in resolved:
            f["status"] = "resolved"
    for nf in delta.get("new_foreshadows") or []:
        mem["foreshadow_pool"].append({
            "id": nf.get("id", f"F{len(mem['foreshadow_pool']) + 1}"),
            "desc": nf.get("desc", ""),
            "chapter": plan["index"],
            "status": "open",
        })

    for cu in delta.get("character_updates") or []:
        for cs in mem.setdefault("character_states", []):
            if cs.get("name") == cu.get("name"):
                cs["state"] = cu.get("state", "")
                break
        else:
            mem["character_states"].append(
                {"name": cu.get("name", ""), "state": cu.get("state", "")})

    return {"memory": mem}
