"""五个 Agent 节点实现。每个节点 = 读 State → 调 LLM → 写回 State。

P1 变更（对应优化建议 4~9 条）：
- 策划产物带「场景节拍」，撰稿按节拍推进（第 4 条）
- 记忆从「全量塞」改为「必带项 + FTS5 检索」（第 5 条，见 memory_index.py）
- 伏笔走状态机，带 schema 校验与超期扫描（第 6 条）
- 校对拆成三路并行 specialist，各写各的 state key（第 7 条）
- 润色环节追加去 AI 腔指令 + 规则层检测（第 8 条）
- 模型输出统一过 Pydantic（第 9 条，见 models.py）
"""
import config
import memory_index
import models
import prompts
import style_check
import llm
from state import StoryState


# ── 记忆上下文：必带项 + 检索项 ──────────────────────────────────

def _retrieval_query(state: StoryState, plan: dict) -> str:
    """用「本章要写什么」当查询词：标题 + 梗概 + 转折点 + 各节拍的目标与冲突。"""
    parts = [plan.get("title") or "", plan.get("outline") or "",
             plan.get("turning_point") or ""]
    for b in (plan.get("beats") or []):
        if isinstance(b, dict):
            parts.append(b.get("goal") or "")
            parts.append(b.get("conflict") or "")
    return " ".join(p for p in parts if p)


def _memory_context(state: StoryState, plan: dict = None) -> str:
    """故事记忆：必带项（未回收伏笔 + 人物快照）+ 检索项（相关前情摘要）。

    为什么分成两类：
    - 未回收伏笔和人物当前快照一旦漏掉，长篇立刻出现设定崩坏 —— 必带，不参与淘汰。
    - 前情摘要会随章节数线性膨胀，正是要按相关度检索的那部分。
    """
    plan = plan or {}
    mem = state.get("memory") or {}
    summaries = [s for s in (mem.get("chapter_summaries") or []) if isinstance(s, dict)]
    pool = [f for f in (mem.get("foreshadow_pool") or []) if isinstance(f, dict)]
    chars = [c for c in (mem.get("character_states") or []) if isinstance(c, dict)]
    if not summaries and not pool and not chars:
        return "（这是第一章，暂无故事记忆）"

    lines = []

    # 必带项 1：未回收伏笔
    open_pool = [f for f in pool if (f.get("status") or "open") != "resolved"]
    if open_pool:
        lines.append("【伏笔池（未回收，必带）】")
        for f in open_pool:
            last = f.get("last_advanced_chapter") or f.get("chapter") or 0
            extra = f"，最后推进于第{last}章" if last and last != f.get("chapter") else ""
            lines.append(f"- [{f.get('status')}] {f.get('id')}：{f.get('desc')}"
                         f"（埋于第{f.get('chapter')}章{extra}）")
    resolved_n = len(pool) - len(open_pool)
    if resolved_n:
        lines.append(f"（另有 {resolved_n} 条伏笔已回收，需要时会被检索出来）")

    # 必带项 2：人物当前快照
    if chars:
        lines.append("【人物状态快照（必带）】")
        for cs in chars:
            lines.append(f"- {cs.get('name')}：{cs.get('state')}")

    # 检索项：相关前情摘要
    idx = plan.get("index") or state.get("chapter_index") or 1
    earlier = [s for s in summaries if (s.get("chapter") or 0) < idx]
    if earlier:
        picked = []
        if len(earlier) > config.MEMORY_TOP_K:
            hits = memory_index.select_relevant(
                mem, _retrieval_query(state, plan), limit=config.MEMORY_TOP_K)
            want = {h["chapter"] for h in hits if h.get("kind") == "summary"}
            picked = [s for s in earlier if (s.get("chapter") or 0) in want]
        if not picked:
            # 章数还少，或检索没命中 → 退化为「最近若干章」。
            # 注意不是"退回全量塞"：即便检索能力不可用，上下文也不会随章数无限膨胀。
            picked = earlier[-config.MEMORY_TOP_K:]
        picked.sort(key=lambda s: s.get("chapter") or 0)
        shown = "、".join(f"第{s.get('chapter')}章" for s in picked)
        lines.append(f"【相关前情摘要（共 {len(earlier)} 章，此处带 {len(picked)} 章：{shown}）】")
        for s in picked:
            lines.append(f"- 第{s.get('chapter')}章：{s.get('summary')}")
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
    chapters = state["outline"].get("chapters", []) or []
    for ch in chapters:
        if not isinstance(ch, dict):
            continue
        # 兼容字符串型 index（旧存档或未经 models 规范化的策划案）
        if str(ch.get("index")).strip() == str(idx):
            # 补齐模型可能漏写的字段（与 models 的哲学一致）：
            # 下游撰稿端按 plan['outline'] 取值，模型少写一个键就会 KeyError
            # 把整条流水线打断——宁可留空，也不要炸。
            beats = ch.get("beats")
            return {
                "index": idx,
                "title": ch.get("title") or f"第{idx}章",
                "outline": ch.get("outline") or "",
                "turning_point": ch.get("turning_point") or "",
                "notes": ch.get("notes") or "",
                "beats": [dict(b) for b in beats if isinstance(b, dict)]
                         if isinstance(beats, list) else [],
            }
    # 章节数超出策划案时，让撰稿人根据既有信息自行续写（无节拍可用）
    return {"index": idx, "title": f"第{idx}章", "outline": "（按核心冲突自然推进）",
            "turning_point": "", "notes": "", "beats": []}


def _beats_section(plan: dict) -> str:
    """把场景节拍渲染成撰稿人看得懂的施工图。"""
    beats = plan.get("beats") or []
    if not beats:
        return ""
    lines = ["\n【场景节拍（按顺序推进，每一拍都要写到）】"]
    for i, b in enumerate(beats, 1):
        lines.append(
            f"{i}. 目标：{b.get('goal') or '（未指定）'}"
            f"｜冲突：{b.get('conflict') or '无'}"
            f"｜转折：{b.get('turn') or '（未指定）'}"
            f"｜建议 {b.get('words') or 0} 字"
            f"｜情绪温度 {b.get('emotion') or 3}/5")
    return "\n".join(lines)


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
    raw = llm.chat_json(prompts.PLANNER_SYSTEM, user_msg,
                        temperature=config.TEMPERATURE_PLANNER)
    outline = models.parse_outline(raw)
    if not outline.chapters:
        raise ValueError(
            "策划案里没有 chapters 字段，无法继续。"
            "多半是模型输出被截断或没按要求给结构——请重试，或换一个更稳的模型。")

    # 产品规则 1：策划输出写入记忆库——人物快照先入库，供后续撰稿/校对读取
    memory = models.parse_memory(state.get("memory") or {})
    if not memory.character_states:
        memory.character_states = [
            models.CharacterState(
                name=c.name,
                state=(f"性格：{c.personality}；动机：{c.motivation}"
                       + (f"；禁忌：{c.taboo}" if c.taboo else "")))
            for c in outline.characters if c.name
        ]
    return {"outline": outline.model_dump(), "memory": memory.model_dump()}


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
            f"- {c.get('name')}｜性格：{c.get('personality')}｜动机：{c.get('motivation')}"
            f"｜禁忌：{c.get('taboo')}｜潜在冲突：{c.get('potential_conflicts', '')}")
    sections.append(f"\n【故事记忆】\n{_memory_context(state, plan)}")
    last = _last_chapter_text(state)
    if last:
        sections.append(f"\n{last}")
    sections.append(
        f"\n【本章任务】第{plan['index']}章《{plan['title']}》\n"
        f"情节大纲：{plan.get('outline', '')}\n"
        f"转折点：{plan.get('turning_point', '')}")
    beats = _beats_section(plan)
    if beats:
        sections.append(beats)

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


# ── 3. 校对 Agent（三路并行 specialist）────────────────────────

def _review_cards(state: StoryState) -> str:
    return "\n".join(
        f"- {c.get('name')}｜{c.get('personality')}｜动机：{c.get('motivation')}"
        f"｜禁忌：{c.get('taboo')}"
        for c in state["outline"].get("characters", []))


def _run_review(state: StoryState, plan: dict, system: str,
                sections: list) -> list:
    """跑一路校对，返回规整后的 comments（失败按 fail-open 策略返回空）。

    三路共用这段：它们只在「问什么」上不同，解析、容错、降级策略完全一致。
    """
    user_msg = "\n\n".join(
        [s for s in sections if s] + [
            f"【待审正文】\n{state['chapter_draft']}",
            f"请按你的检查清单逐项校验，输出 JSON"
            f"（revision_round={state.get('revision_round', 0)}）。",
        ])
    try:
        result = llm.chat_json(system, user_msg,
                               temperature=config.TEMPERATURE_REVIEWER)
    except Exception as e:                            # noqa: BLE001
        # 校对只是「质检」环节，不该因为它的输出格式问题，把已经写好的整章作废。
        if not config.REVIEW_FAIL_OPEN:
            raise
        llm.emit_notice(
            "warn",
            f"第{plan['index']}章有一路校对结果无法解析（{type(e).__name__}），"
            f"该维度本次跳过；其余维度照常工作，章节继续进入润色。"
            f"原因：{str(e)[:140]}")
        return []
    return models.dump(models.parse_comments(result.get("comments")))


def reviewer_ooc_node(state: StoryState) -> dict:
    """只查人物一致性。"""
    plan = _chapter_plan(state)
    sections = [
        f"【人物卡（校验基准）】\n{_review_cards(state)}",
        f"【故事记忆（人物前后的状态变化）】\n{_memory_context(state, plan)}",
        f"{_last_chapter_text(state)}",
        f"【本章应写内容】第{plan.get('index')}章：{plan.get('outline', '')}",
        "\n".join(_meta_sections(state, "reviewer")),
    ]
    return {"review_comments_ooc": _run_review(
        state, plan, prompts.REVIEWER_OOC_SYSTEM, sections)}


def reviewer_logic_node(state: StoryState) -> dict:
    """只查时间线 / 设定 / 剧情逻辑。"""
    plan = _chapter_plan(state)
    sections = [
        f"【故事记忆（查时间线与伏笔矛盾）】\n{_memory_context(state, plan)}",
        f"{_last_chapter_text(state)}",
        f"【本章应写内容】第{plan.get('index')}章：{plan.get('outline', '')}\n"
        f"转折点：{plan.get('turning_point', '')}",
        "\n".join(_meta_sections(state, "reviewer")),
    ]
    return {"review_comments_logic": _run_review(
        state, plan, prompts.REVIEWER_LOGIC_SYSTEM, sections)}


def reviewer_pacing_node(state: StoryState) -> dict:
    """只查节奏与篇幅。"""
    plan = _chapter_plan(state)
    if not config.REVIEW_PACING_ENABLED:
        return {"review_comments_pacing": []}
    meta = state.get("meta") or {}
    sections = [
        _beats_section(plan) or "（本章没有提供场景节拍，按情节大纲判断节奏）",
        (f"【本章目标字数】{meta['word_count']} 字（±20%）" if meta.get("word_count") else ""),
        f"【本章应写内容】第{plan.get('index')}章：{plan.get('outline', '')}",
    ]
    return {"review_comments_pacing": _run_review(
        state, plan, prompts.REVIEWER_PACING_SYSTEM, sections)}


def merge_reviews_node(state: StoryState) -> dict:
    """三路校对汇合点：合并意见、统一判定 verdict。

    路由判定只在合并后做一次——每一路各自判 fail 会让「某一路 fail 但没给出
    critical」这种空转情况被重复计三次。
    """
    plan = _chapter_plan(state)
    groups = [
        ("人物一致性", state.get("review_comments_ooc") or []),
        ("逻辑/时间线/设定", state.get("review_comments_logic") or []),
        ("节奏/篇幅", state.get("review_comments_pacing") or []),
    ]
    merged = []
    for label, items in groups:
        for c in items:
            c = dict(c)
            c.setdefault("from", label)
            merged.append(c)

    # critical 排前面，让撰稿人先看到必须改的
    merged.sort(key=lambda c: 0 if c.get("severity") == "critical" else 1)
    crit = [c for c in merged if c.get("severity") == "critical"]
    verdict = "fail" if crit else "pass"

    if verdict == "fail":
        by_dim = {}
        for c in crit:
            by_dim[c.get("from", "其他")] = by_dim.get(c.get("from", "其他"), 0) + 1
        detail = "、".join(f"{k} {v} 条" for k, v in by_dim.items())
        llm.emit_notice("info",
                        f"第{plan['index']}章校对发现 {len(crit)} 个严重问题（{detail}），"
                        f"已打回重写。")
    return {"review_verdict": verdict, "review_comments": merged}


# ── 4. 润色 Agent ─────────────────────────────────────────────
def polisher_node(state: StoryState) -> dict:
    minors = [c for c in (state.get("review_comments") or [])
              if c.get("severity") == "minor"]
    user_msg = f"正文如下：\n{state['chapter_draft']}"
    if minors:
        sug = "\n".join(f"- {c.get('issue')} → {c.get('suggestion')}" for c in minors)
        user_msg = f"【润色时可参考的 minor 意见（仅措辞层面化解）】\n{sug}\n\n{user_msg}"
    meta_secs = _meta_sections(state, "polisher")
    if meta_secs:
        user_msg = "\n".join(meta_secs) + "\n\n" + user_msg

    system = prompts.POLISHER_SYSTEM
    if config.DESLOP_IN_POLISHER:
        system = system + prompts.POLISHER_DESLOP

    raw = llm.chat(system, user_msg, temperature=config.TEMPERATURE_POLISHER)
    plan = _chapter_plan(state)
    finalized = {
        "index": plan["index"],
        "title": plan["title"],
        "text": raw,
    }
    out = {
        "final_chapter": raw,
        "final_chapters": (state.get("final_chapters") or []) + [finalized],
    }

    # 去 AI 味：规则层检测。只提示，不自动改稿——AI 腔的判定带主观性，
    # 自动替换容易连作者的个人风格一起抹掉。
    if config.STYLE_CHECK_ENABLED:
        report = style_check.check(raw)
        out["style_report"] = report
        if report["findings"]:
            llm.emit_notice(
                "info",
                f"第{plan['index']}章风格体检：{style_check.format_findings(report['findings'])}")
    return out


# ── 5. 记忆结算员（MemorySettler，润色定稿后运行）──────────────
def _empty_memory() -> dict:
    return {"chapter_summaries": [], "foreshadow_pool": [], "character_states": []}


def _stale_foreshadows(pool, idx: int):
    """扫描「埋太久没动」的伏笔：既没回收，也连续 N 章没有推进。"""
    out = []
    for f in pool:
        if f.status == "resolved":
            continue
        last = f.last_advanced_chapter or f.chapter or 0
        if idx - last >= config.FORESHADOW_STALE_CHAPTERS:
            out.append(f)
    return out


def memory_settler_node(state: StoryState) -> dict:
    plan = _chapter_plan(state)
    idx = plan["index"]
    mem = models.parse_memory(state.get("memory") or {})
    pool_txt = "\n".join(
        f"- {f.id}：{f.desc}" for f in mem.foreshadow_pool) or "（空）"
    user_msg = (
        f"【本章信息】第{idx}章《{plan['title']}》\n"
        f"【当前伏笔池（resolved / advanced 的 id 只能从中选）】\n{pool_txt}\n\n"
        f"【本章定稿正文】\n{state['final_chapter']}\n\n"
        f"请输出本章记忆结算 JSON。")
    try:
        raw = llm.chat_json(prompts.MEMORY_SYSTEM, user_msg,
                            temperature=config.TEMPERATURE_REVIEWER)
    except Exception as e:                            # noqa: BLE001
        # 记忆结算失败只影响"后续章节的上下文质量"，本章已经定稿入库了，
        # 没必要因此中断整条流水线——告警即可。
        llm.emit_notice(
            "warn",
            f"第{idx}章的记忆结算失败（{type(e).__name__}），"
            f"本章已正常定稿；但后续章节会缺少这部分前情记忆与伏笔记录。"
            f"原因：{str(e)[:140]}")
        return {}

    delta = models.parse_memory_delta(raw)

    # ── 合并增量：全部走 Pydantic 模型，坏数据在这里被收敛，不往下游传播 ──
    mem.chapter_summaries.append(
        models.ChapterSummary(chapter=idx, summary=delta.summary))

    resolved = set(delta.resolved_foreshadows)
    advanced = set(delta.advanced_foreshadows)
    known_ids = set()
    for f in mem.foreshadow_pool:
        known_ids.add(f.id)
        if f.id in resolved:
            f.status = "resolved"
            f.last_advanced_chapter = idx
        elif f.id in advanced:
            # open → progressing：只有从"埋下"走到"开始推进"才升级状态；
            # 已经是 progressing / deferred 的保持不变（deferred 由作者手工设置，不覆盖）
            if f.status == "open":
                f.status = "progressing"
            f.last_advanced_chapter = idx

    for nf in delta.new_foreshadows:
        fid = nf.id.strip() or f"F{len(mem.foreshadow_pool) + 1}"
        if fid in known_ids:
            # 同一 id 重复埋设：以新描述为准，但不产生重复条目
            for f in mem.foreshadow_pool:
                if f.id == fid and nf.desc and f.desc != nf.desc:
                    f.desc = nf.desc
                    break
            continue
        known_ids.add(fid)
        mem.foreshadow_pool.append(models.Foreshadow(
            id=fid, desc=nf.desc, chapter=idx, status="open",
            last_advanced_chapter=idx, deadline_chapter=None))

    for cu in delta.character_updates:
        if not cu.name:
            continue
        for cs in mem.character_states:
            if cs.name == cu.name:
                cs.state = cu.state
                break
        else:
            mem.character_states.append(
                models.CharacterState(name=cu.name, state=cu.state))

    # ── 伏笔超期巡检：伏笔烂尾是长篇最伤读者的问题，这里主动兜住 ──
    stale = _stale_foreshadows(mem.foreshadow_pool, idx)
    if stale:
        detail = "、".join(
            f"{f.id}（第{f.chapter}章埋下，最后推进于第{f.last_advanced_chapter}章）"
            for f in stale[:5])
        more = f"等 {len(stale)} 条" if len(stale) > 5 else ""
        llm.emit_notice(
            "warn",
            f"第{idx}章结算后，有 {len(stale)} 条伏笔已连续 "
            f"{config.FORESHADOW_STALE_CHAPTERS} 章以上未推进：{detail}{more}。"
            f"建议在后续章节安排推进或回收，避免烂尾。")

    return {"memory": mem.model_dump()}
