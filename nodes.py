"""各 Agent 节点实现。每个节点 = 读 State → 调 LLM → 写回 State。

P1 变更（对应优化建议 4~9 条）：
- 策划产物带「场景节拍」，撰稿按节拍推进（第 4 条）
- 记忆从「全量塞」改为「必带项 + FTS5 检索」（第 5 条，见 memory_index.py）
- 伏笔走状态机，带 schema 校验与超期扫描（第 6 条）
- 校对拆成三路并行 specialist，各写各的 state key（第 7 条）
- 润色环节追加去 AI 腔指令 + 规则层检测（第 8 条）
- 模型输出统一过 Pydantic（第 9 条，见 models.py）

实跑后的补丁（2026-10-02）：
- 物资账本 / 必带项上界 / 角色名归一（见 models.py、memory_index.py）
- 续写超出策划案章数时自动补本章大纲（chapter_planner_node，排在 planner 之后）
"""
import copy

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


def _rank_open_foreshadows(open_pool, idx: int) -> list:
    """必带项里未回收伏笔的排序：deadline 临近的优先，其次最久没推进的。

    截断必然有代价，所以把「最可能马上要兑现」和「最容易被遗忘」的排在最前面。
    这里是纯 dict（还没过 Pydantic），字段可能缺失，一律 get + 兜底。
    """
    def key(f):
        dl = f.get("deadline_chapter")
        last = f.get("last_advanced_chapter") or f.get("chapter") or 0
        return (0 if dl else 1, dl or 9999, -(idx - last), f.get("chapter") or 0)
    return sorted(open_pool, key=key)


# 设定档案的展示标签。放模块级是因为「记忆上下文」与「结算请求」两处都要用同一套，
# 各自维护一份迟早会漂。
_SETTING_KIND_LABEL = {"place": "【地点】", "object": "【设施】",
                       "rule": "【规则】", "relation": "【关系】", "other": ""}


def _setting_line(s) -> str:
    """把设定条目渲染成一行，给记忆结算员看（含类型与废弃标注）。"""
    label = _SETTING_KIND_LABEL.get(s.kind or "other", "")
    detail = f"：{s.detail}" if (s.detail or "").strip() else ""
    tag = "【已废弃】" if s.status != "active" else ""
    return f"- {label}{s.name}{detail}{tag}"


def _qty_display(count, unit, qty) -> str:
    """账本存量的展示口径：有件数就用件数（5 瓶），没有才退回描述。

    为什么优先件数：撰稿人与校对都按这个数写、按这个数核；描述里可能混着
    "约五天份，每天一瓶"这类换算说明，直接丢给模型会各自算出自家的数。
    """
    u = str(unit or "").strip()
    if count is not None:
        return f"{count}{u}" if u else str(count)
    return str(qty or "").strip()


def _memory_context(state: StoryState, plan: dict = None) -> str:
    """故事记忆：必带项（未回收伏笔 + 人物快照 + 物资账本 + 设定档案）+ 检索项（相关前情摘要）。

    为什么分成两类：
    - 未回收伏笔、人物当前快照、物资账本、已确立设定一旦漏掉，长篇立刻出现设定崩坏
      —— 必带，不参与淘汰。
    - 前情摘要会随章节数线性膨胀，正是要按相关度检索的那部分。
    """
    plan = plan or {}
    mem = state.get("memory") or {}
    summaries = [s for s in (mem.get("chapter_summaries") or []) if isinstance(s, dict)]
    pool = [f for f in (mem.get("foreshadow_pool") or []) if isinstance(f, dict)]
    chars = [c for c in (mem.get("character_states") or []) if isinstance(c, dict)]
    items = [i for i in (mem.get("items") or []) if isinstance(i, dict)]
    settings = [s for s in (mem.get("settings") or []) if isinstance(s, dict)]
    if not summaries and not pool and not chars and not items and not settings:
        return "（这是第一章，暂无故事记忆）"

    idx = plan.get("index") or state.get("chapter_index") or 1
    lines = []

    # 必带项 1：未回收伏笔（带条数上限）
    open_pool = [f for f in pool if (f.get("status") or "open") != "resolved"]
    if open_pool:
        ranked = _rank_open_foreshadows(open_pool, idx)
        limit = max(1, config.MEMORY_MUST_HAVE_MAX)
        shown, hidden = ranked[:limit], ranked[limit:]
        lines.append(f"【伏笔池（未回收，必带，共 {len(open_pool)} 条）】")
        for f in shown:
            last = f.get("last_advanced_chapter") or f.get("chapter") or 0
            extra = f"，最后推进于第{last}章" if last and last != f.get("chapter") else ""
            dl_n = f.get("deadline_chapter")
            dl = f"，期望第{dl_n}章前回收" if dl_n else ""
            lines.append(f"- [{f.get('status')}] {f.get('id')}：{f.get('desc')}"
                         f"（埋于第{f.get('chapter')}章{extra}{dl}）")
        if hidden:
            # 截断是必须的（实测必带项按约 6 条/章线性增长），但不能让被截掉的伏笔
            # 就此消失：至少给出 id，这样本章若涉及它们，记忆结算仍能推进/回收。
            ids = "、".join(str(f.get("id") or "?") for f in hidden)
            lines.append(f"- （另有 {len(hidden)} 条较早伏笔未列描述，id：{ids}"
                         f"——本章若涉及，仍须在记忆结算里推进或回收）")
    resolved_n = len(pool) - len(open_pool)
    if resolved_n:
        lines.append(f"（另有 {resolved_n} 条伏笔已回收，需要时会被检索出来）")

    # 必带项 2：人物当前快照
    if chars:
        lines.append("【人物状态快照（必带）】")
        for cs in chars:
            lines.append(f"- {cs.get('name')}：{cs.get('state')}")

    # 必带项 3：物资账本
    # 为什么是"必带"而不是"检索"：物资矛盾是长篇里最容易犯、读者最容易发现的硬伤，
    # 而且它天然是个硬约束（有/没有），不该由检索相关度决定带不带。
    #
    # 展示上分两组：可用的逐条列（要带存量与备注），不可用的压成一行只报名字——
    # 已耗尽的东西不需要排版美感，只需要"别再写出来"这个约束，压一行能省不少字。
    if items:
        avail = [i for i in items if (i.get("status") or "available") == "available"]
        gone = [i for i in items if (i.get("status") or "available") != "available"]
        lines.append("【物资账本（必带）——只能用清单里的东西】")
        for it in avail:
            disp = _qty_display(it.get("count"), it.get("unit"), it.get("qty"))
            seg = f"- {it.get('name')}" + (f"（存量：{disp}）" if disp else "")
            note = (it.get("note") or "").strip()
            if note:
                seg += f"｜{note}"
            lines.append(seg)
        if gone:
            names = "、".join(
                f"{it.get('name')}"
                f"[{'已耗尽' if it.get('status') == 'consumed' else '已丢失'}]"
                for it in gone)
            lines.append(f"- （以下已不可用，本章绝不能再次出现：{names}）")

    # 必带项 4：设定档案（地点 / 固定设施 / 世界规则）
    # 为什么必带：逻辑校对手上原本**没有任何"与前文一致"的依据**——前情摘要只记情节，
    # top-k 又只带上一章。第四次实跑里，第 1 章早已确立的「便利店卷帘门 + 后门铁门」
    # 在第 4 章被连着两轮判成 critical 设定矛盾，正文没按它改是对的：
    # 那是纯粹的误报，代价却是白烧重写轮次。有了这本账，判断基准就落到了纸面上。
    if settings:
        active = [s for s in settings if (s.get("status") or "active") == "active"]
        gone_s = [s for s in settings if (s.get("status") or "active") != "active"]
        lines.append("【已确立设定（必带）——这些是既成事实，与前文一致的描写不算矛盾】")
        # 排序：最久没被提到的排前面。依据是"边际价值"，不是"重要性"——
        # 撰稿与校对每章都能拿到**上一章全文**，越是新近出现过的设定，越可能已经在
        # 那份全文里；反过来，第 1 章立下、此后再没提过的设定，除了这本档案之外
        # 没有任何别的来源，它恰恰是最容易被误判为"与前文矛盾"的那一类。
        ranked = sorted(active, key=lambda s: (s.get("last_mentioned_chapter")
                                               or s.get("chapter") or 0,
                                               s.get("chapter") or 0))
        limit = max(1, config.MEMORY_SETTING_MAX)
        shown, hidden = ranked[:limit], ranked[limit:]
        for s in shown:
            label = _SETTING_KIND_LABEL.get(s.get("kind") or "other", "")
            seg = f"- {label}{s.get('name')}"
            if (s.get("detail") or "").strip():
                seg += f"：{s.get('detail')}"
            ch = s.get("chapter") or 0
            seg += f"（第{ch}章确立）" if ch else "（确立章节不详）"
            if (s.get("note") or "").strip():
                seg += f"｜{s.get('note')}"
            lines.append(seg)
        if hidden:
            names = "、".join(str(s.get("name")) for s in hidden)
            lines.append(f"- （另有 {len(hidden)} 条较早设定未列描述：{names}"
                         f"——它们同样有效，凡正文与前文冲突，一律以设定档案为准，"
                         f"不要判为矛盾）")
        if gone_s:
            names = "、".join(f"{s.get('name')}[已废弃]" for s in gone_s)
            lines.append(f"- （以下设定已废弃/已毁，不得再当作有效设定或场景使用：{names}）")

    # 检索项：相关前情摘要
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
    # 开局物资清单也先入库：这样第 1 章的撰稿人拿到 prompt 时账本就已经在了，
    # 从源头掐掉「第 1 章自己列了 4 样东西、紧接着又摸出别的东西」这类内伤
    # （实测第 1 章 0 轮重写通过，这份矛盾被当正典固化，之后每章都要重新审一遍）。
    if not memory.items:
        memory.items = [
            models.Item(name=it.name.strip(), count=it.count, unit=it.unit.strip(),
                        qty=it.qty.strip(), note=it.note.strip(),
                        status="available", chapter=1, last_changed_chapter=1)
            for it in outline.initial_items if it.name.strip()
        ]
    # 开局设定档案同理：地点 / 固定设施 / 世界规则在策划阶段就立档。
    # 撰稿从第 1 章起就有"既成事实"可依，逻辑校对也有了比对基准——
    # 治的正是「第 1 章立下的那两扇门，第 4 章被当成设定矛盾判 critical」这类误报。
    if not memory.settings:
        memory.settings = [
            models.Setting(name=s.name.strip(), detail=s.detail.strip(),
                           kind=s.kind, status="active", note=s.note.strip(),
                           chapter=1, last_mentioned_chapter=1)
            for s in outline.initial_settings if s.name.strip()
        ]
    # 账本要「能逐章对账」，前提是它带着数字。策划给「约五天份」这类模糊量时，
    # 后续每一章的撰稿与校对都会各自换算一遍、各自算出不同的数 —— 实测这是烧掉
    # 重写轮次的最大一块（7 轮里有 ch2/ch3 各 3 轮，约六成 critical 是账目类）。
    # 这里只提醒、不替它换算：猜出来的数字比"没有数字"更危险（会立刻触发误报）。
    fuzzy = [it.name for it in memory.items
             if it.count is None and models._qty_to_count(it.qty) is None]
    if fuzzy:
        shown = "、".join(fuzzy[:5]) + (" 等" if len(fuzzy) > 5 else "")
        llm.emit_notice(
            "warn",
            f"策划给的开局物资里，{len(fuzzy)} 项没有可计数的件数（{shown}）。"
            f"后续章节要按件数核对消耗，模糊量会让撰稿与校对各算一个数、反复打回重写；"
            f"建议改成具体件数（写「5 瓶」而不是「约五天份」）。")
    return {"outline": outline.model_dump(), "memory": memory.model_dump()}


# ── 1.5 章节大纲补充：本章不在策划案范围内时 ────────────────────
def chapter_planner_node(state: StoryState) -> dict:
    """策划案没写到本章时，为本章补一份分章大纲。

    为什么单独做成一个节点，而不是塞进 _chapter_plan()：
    _chapter_plan() 每章要被调用 7 次（撰稿 1 + 三路校对 3 + 汇合 1 + 润色 1 + 结算 1）。
    在里面发 LLM 请求，同一章会被规划出好几份互不相同的大纲，还白烧 token。
    这里只生成一次、写回 outline["chapters"]，后续那 7 次调用就都命中同一份。

    触发场景（2026-10-02 实跑暴露）：需求写着"共3章"，写完第 3 章之后又续写第 4 章，
    而 outline.chapters 只有 3 条 → _chapter_plan() 走兜底分支，第 4 章
    既没有标题也没有分章大纲，落盘标题成了 "# 第4章"，正文全靠自由发挥。
    """
    outline = state.get("outline") or {}
    chapters = [c for c in (outline.get("chapters") or []) if isinstance(c, dict)]
    idx = state.get("chapter_index") or 1
    # 策划案里已经有这一章 → 什么都不做。
    # 这条同时是重写循环（bump_round → writer）里重复进入时的快速返回。
    if any(str(c.get("index")).strip() == str(idx) for c in chapters):
        return {}

    # 给模型的信息：策划案骨架 + 已定稿章节标题 + 上一章结尾（要接得上）
    done = state.get("final_chapters") or []
    titles = "、".join(f"第{c.get('index')}章《{c.get('title')}》" for c in done) or "（尚无）"
    tail = ""
    if done:
        tail = ("\n\n【上一章结尾（本章开头要接得上）】\n"
                + str(done[-1].get("text") or "")[-600:])

    # 【本章序号】独立成行且不带"第/章"两字，是为了给下面解析用的锚点，
    # 避免模型（或 mock）从"已规划到第 N 章"里取错章号。
    # 措辞用"当前共 N 章"而不是"原案 N 章"：chapters 里可能已经含上一章刚补写的条目，
    # "原本只规划到第 N 章"在补第 3 章时会算成 2，是错的。
    user_msg = (
        f"【本章序号】{idx}（策划案当前共 {len(chapters)} 章，本章不在其中）\n"
        f"【书名】{outline.get('title')}\n"
        f"【主题】{outline.get('theme')}\n"
        f"【核心冲突】{outline.get('core_conflict')}\n"
        f"【结局方向】{outline.get('ending_direction')}\n"
        f"【已定稿章节】{titles}"
        f"\n\n请为第 {idx} 章补一份分章大纲 JSON。{tail}")
    try:
        raw = llm.chat_json(prompts.CHAPTER_EXTENDER_SYSTEM, user_msg,
                            temperature=config.TEMPERATURE_PLANNER)
        chapter = models.Chapter.model_validate(raw)
    except Exception as e:                              # noqa: BLE001
        # 补不出大纲也不能拦路——退回原来的兜底（按核心冲突自然推进），但必须让作者知道，
        # 否则又是"标题悄悄变成《第4章》"那种无声降级。
        llm.emit_notice(
            "warn",
            f"第{idx}章不在策划案范围内（策划案当前共 {len(chapters)} 章），"
            f"自动补写章节大纲失败（{type(e).__name__}），"
            f"本章将按核心冲突自然推进，标题暂用「第{idx}章」。"
            f"原因：{str(e)[:140]}")
        return {}

    chapter.index = idx          # 模型常漏写 index，必须按实际章号钉死，否则匹配不上
    if not chapter.title.strip():
        chapter.title = f"第{idx}章"
    new_outline = copy.deepcopy(outline)
    new_outline["chapters"] = chapters + [chapter.model_dump()]
    llm.emit_notice(
        "info",
        f"第{idx}章不在策划案范围内（策划案当前共 {len(chapters)} 章），"
        f"已自动补写章节大纲《{chapter.title}》"
        f"（含 {len(chapter.beats)} 个场景节拍），并写回策划案供后续复用。")
    return {"outline": new_outline}


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
    return {"chapter_summaries": [], "foreshadow_pool": [],
            "character_states": [], "items": [], "settings": []}


def _item_line(it) -> str:
    """把账本条目渲染成一行，给记忆结算员看（含件数与状态标注）。"""
    disp = _qty_display(it.count, it.unit, it.qty)
    qty = f"（存量：{disp}）" if disp and it.status == "available" else ""
    tag = {"consumed": "【已耗尽】", "lost": "【已丢失】"}.get(it.status, "")
    return f"- {it.name}{qty}{tag}"


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
    # 把人物名单与物资账本一并给模型：
    # - 名单是治「同一角色裂成多条」的源头约束（实测「橘猫」「橘猫（流浪猫）」「猫」并存）；
    # - 账本是让结算员"按账对账"，而不是每章重新自由发挥一遍物资。
    names_txt = "、".join(cs.name for cs in mem.character_states if cs.name) or "（暂无）"
    items_txt = "\n".join(_item_line(it) for it in mem.items) \
        or "（空——本章正文里出现的物资将作为账本起点）"
    settings_txt = "\n".join(_setting_line(s) for s in mem.settings) \
        or "（空——本章正文里确立的地点与规则将作为档案起点）"
    user_msg = (
        f"【本章信息】第{idx}章《{plan['title']}》\n"
        f"【当前伏笔池（resolved / advanced 的 id 只能从中选）】\n{pool_txt}\n\n"
        f"【当前人物名单（character_updates 的 name 只能从中选）】\n{names_txt}\n\n"
        f"【当前物资账本（item_changes 的 name 优先取账本原名）】\n{items_txt}\n\n"
        f"【当前设定档案（setting_updates 的 name 优先取档案原名）】\n{settings_txt}\n\n"
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
        key = models.canon_name(cu.name)
        for cs in mem.character_states:
            # 先精确、再按归一后的名字匹配（「橘猫（流浪猫）」==「橘猫」）。
            # 命中后**保留账本里已有的 name 不动**——名字漂移比合并本身更麻烦：
            # 稳定的名字才能让「人物卡」与「记忆库」始终对得上号。
            if cs.name == cu.name or (key and models.canon_name(cs.name) == key):
                cs.state = cu.state
                break
        else:
            mem.character_states.append(
                models.CharacterState(name=cu.name, state=cu.state))

    # ── 物资账本：新增 or 更新，名字同样走归一匹配；同时逐件对账 ──
    # 为什么必须对账：第二次实跑暴露「结算静默采信本章数字」——ch2 末写 4 块饼干，
    # ch3 正文只剩 2 块且没交代去向，校对连报两轮都没改掉，账本最终落成
    # 「压缩饼干 2块 / status=consumed」：**"已耗尽"却还剩 2 块，字段语义自相矛盾**。
    # 静默采信会把这类硬伤固化成正典，所以这里把差额与矛盾都显式报出来。
    by_key = {models.canon_name(it.name): it for it in mem.items if it.name.strip()}
    recon = []
    for ch in delta.item_changes:
        name = ch.name.strip()
        if not name:
            continue
        key = models.canon_name(name)
        hit = by_key.get(key) if key else None
        if hit is None:
            # 本章首次出现的物资（含捡到、别人给的、买来的）。
            # 状态缺省按可用处理——它此刻确实在手上。
            it = models.Item(name=name, count=ch.count, unit=ch.unit, qty=ch.qty,
                             note=ch.note, status=ch.status or "available",
                             chapter=idx, last_changed_chapter=idx)
            mem.items.append(it)
            by_key[key or name] = it
            continue
        # 已有物资：只覆盖模型明确给出的字段。
        # 特别地，status 为空表示"状态没变"，绝不能当成 available 写回去
        # ——那会让已经吃完的东西复活，正是这本账要防的事。
        prev_count = hit.count
        unit = hit.unit or ch.unit
        if ch.unit and not hit.unit:
            hit.unit = ch.unit
        if ch.qty:
            hit.qty = ch.qty

        gone = ch.status in ("consumed", "lost")
        new_count = ch.count
        if gone:
            # 同一条里既说"没了"又说"还剩几件"：以状态为准，件数归零。
            if new_count not in (None, 0):
                label = "已耗尽" if ch.status == "consumed" else "已丢失"
                recon.append(
                    f"{hit.name}：结算同时给出「{label}」与剩余量"
                    f"「{ch.qty or new_count}」，两者矛盾，件数已按 0 处理")
                hit.qty = ""
            new_count = 0
        if ch.status:
            hit.status = ch.status
        if ch.note:
            hit.note = ch.note
        if new_count is not None:
            hit.count = new_count
        # 件数变了却不说原因 = 数字在账本里凭空消失/出现，正是要抓的静默采信。
        if (not gone and prev_count is not None and new_count is not None
                and new_count != prev_count and not ch.note.strip()):
            recon.append(
                f"{hit.name}：件数由 {prev_count}{unit} 变为 {new_count}{unit}，"
                f"结算未说明原因（正文里可能凭空消耗或新增），请核对")
        hit.last_changed_chapter = idx

    # ── 设定档案：新增 or 更新，名字走同一套 canon_name 归一 ──
    # 与物资的差别：设定不会"消耗"，只会在被毁/废弃时失效，所以不做数量对账；
    # 要守的是另外两条：
    #   ① 空字段不覆盖（"只补一句描写"不能把 status 刷回 active，那是复活 bug 的设定版）；
    #   ② 标为废弃却不说原因 → 告警（某个地点悄悄失效，后文就会莫名把它写没了）。
    s_by_key = {models.canon_name(s.name): s for s in mem.settings if s.name.strip()}
    retired = []                       # [(名字, 是否写了原因)]
    for sc in delta.setting_updates:
        name = sc.name.strip()
        if not name:
            continue
        key = models.canon_name(name)
        hit = s_by_key.get(key) if key else None
        if hit is None:
            # 只报名字、不附带任何信息的条目：那只是"本章提到了某个还不成档的东西"，
            # 没有新信息可记。放它进档会往档案里塞一堆空壳条目，把真设定挤掉。
            if not (sc.detail or sc.kind or sc.status or sc.note):
                continue
            it = models.Setting(name=name, detail=sc.detail,
                                kind=sc.kind or "other",
                                status=sc.status or "active", note=sc.note,
                                chapter=idx, last_mentioned_chapter=idx)
            mem.settings.append(it)
            s_by_key[key or name] = it
            continue
        if sc.detail:
            hit.detail = sc.detail
        if sc.kind:
            hit.kind = sc.kind
        if sc.status:
            hit.status = sc.status
            if sc.status == "retired":
                retired.append((hit.name, bool(sc.note.strip())))
        if sc.note:
            hit.note = sc.note
        # 只要本章结算提到了它，就算"还活着"——排序靠的就是这个时间戳。
        # 没提到也不惩罚：陈旧只会让它更早出现在提示里（安全侧）。
        hit.last_mentioned_chapter = idx

    if retired:
        names = "、".join(n for n, _ in retired)
        silent = [n for n, has_note in retired if not has_note]
        level = "warn" if silent else "info"
        extra = (f"其中 {len(silent)} 条没写废弃原因（{'、'.join(silent)}），"
                 f"请核对正文是否真的交代了它为何失效。" if silent else "")
        llm.emit_notice(
            level,
            f"第{idx}章设定档案：{names} 被标记为已废弃，"
            f"后续章节不会再把它当作有效设定。{extra}")

    if recon:
        shown = "；".join(recon[:4])
        more = f" 等共 {len(recon)} 处" if len(recon) > 4 else ""
        llm.emit_notice(
            "warn",
            f"第{idx}章物资账本对账发现 {len(recon)} 处异常：{shown}{more}。"
            f"账本已按更严格的一侧记录，建议核对正文与账本是否对得上。")

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
