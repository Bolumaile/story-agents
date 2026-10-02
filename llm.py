"""LLM 调用封装：DeepSeek API 与本地 OpenAI 兼容服务统一走一个 client。

USE_MOCK=True 时走离线 MockLLM（不联网），用于验证流水线流转逻辑。
"""
import json
import re
import time

from openai import OpenAI

import config

# mock 开关（由 main.py / web 端设置）
USE_MOCK = False


# 智谱这些模型「强制开启思考」，关不掉：传 thinking:disabled 轻则被忽略、
# 重则直接报参数错误。识别出来就不下发该字段，保持平台默认。
_ZHIPU_FORCE_THINKING = ("glm-5.3-flashx", "glm-5.3-flash", "glm-5.3",
                         "glm-4.7", "glm-4.5v")


# 明确拒绝未知字段的平台。本项目面向国内平台与自建中转，基本碰不到，
# 只作兜底——真碰上了就不下发，免得每次都要先失败一次再重试。
_STRICT_PLATFORMS = ("api.openai.com", "anthropic.com")


def _can_send_thinking(base_url: str) -> bool:
    """是否值得尝试下发 thinking 字段。

    刻意不按「是不是 DeepSeek/智谱官方域名」来判断：大量用户走的是第三方
    中转地址，域名认不出来，一旦按域名判断就会漏发，开关再次变成摆设。
    国内主流平台及常见中转都认这个字段，不认的多半会忽略它；
    真正报错的由 chat() 摘掉参数重试兜住。
    """
    u = (base_url or "").lower()
    return not any(d in u for d in _STRICT_PLATFORMS)


def zhipu_force_thinking(model: str = "") -> bool:
    """该模型是否属于智谱「强制思考」名单（用户关不掉，只能换模型）。"""
    m = (model or config.MODEL_NAME or "").lower()
    return any(m.startswith(p) for p in _ZHIPU_FORCE_THINKING)


def _thinking_kwargs() -> dict:
    """下发「思考开关」。

    这里曾经只对 DeepSeek 官方下发（按域名判断），结果是：用户改用智谱时，
    关掉「深度思考」后请求里根本没带这个字段，而智谱的 thinking.type 默认就是
    enabled —— 模型照旧一路思考，又慢又容易超时，开关形同虚设。
    """
    if not _can_send_thinking(config.DEEPSEEK_BASE_URL):
        return {}
    if config.ENABLE_THINKING:
        return {"thinking": {"type": "enabled"}}
    if zhipu_force_thinking():
        return {}                      # 强制思考模型：传 disabled 会被拒，别传
    return {"thinking": {"type": "disabled"}}


def _is_param_error(exc: Exception) -> bool:
    """是否是「平台不接受请求参数」类错误，用于摘掉 thinking 重试一次。

    不苛刻匹配关键字：各家提示措辞太杂，宁可多重试一次也别让用户卡住。
    重试只做一次，且会把 thinking 摘掉，不会死循环。
    """
    return getattr(exc, "status_code", None) in (400, 422)


def _get_client() -> OpenAI:
    # timeout 在流式请求下 = 相邻两段数据之间的最长等待，见 config 注释
    return OpenAI(
        api_key=config.DEEPSEEK_API_KEY or "local",
        base_url=config.DEEPSEEK_BASE_URL,
        timeout=config.LLM_TIMEOUT,
        max_retries=config.LLM_MAX_RETRIES,
    )


# ── 流式进度回调：web 端设置后，可把「已生成 N 字」实时推给前端 ──
_PROGRESS_CB = None


def set_progress_cb(cb):
    global _PROGRESS_CB
    _PROGRESS_CB = cb


def _emit_progress(kind: str, chars: int):
    if _PROGRESS_CB is None:
        return
    try:
        _PROGRESS_CB(kind, chars)
    except Exception:          # 进度推送失败绝不能影响正事
        pass


class JSONParseError(ValueError):
    """模型输出不是合法 JSON，且自动修复也没救回来。

    带 raw 原文，方便上层决定「降级放行」还是直接报错。
    """

    def __init__(self, message: str, raw: str = ""):
        super().__init__(message)
        self.raw = raw


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _fix_stray_quotes(s: str) -> str:
    """给「字符串值内部未转义的英文双引号」补上转义。

    这是本项目最常踩的坑：模型解释正文时爱直接写 把"塞回遮阳板"改成… ，
    内层引号没转义 → JSON 在第一个内层引号处提前结束，
    报 Expecting ',' delimiter（注意：这与"被截断"报的错完全不同）。

    判定办法：在字符串内部遇到 " 时，往后看第一个非空白字符——
    若是 : , } ] 或已到末尾，说明它是合法的收尾引号；否则就是内层引号，补转义。
    """
    out, i, n, in_str = [], 0, len(s), False
    while i < n:
        c = s[i]
        if not in_str:
            out.append(c)
            if c == '"':
                in_str = True
            i += 1
            continue
        if c == "\\":                      # 已有的转义序列原样保留
            out.append(s[i:i + 2])
            i += 2
            continue
        if c == '"':
            j = i + 1
            while j < n and s[j] in " \t\r\n":
                j += 1
            if j >= n or s[j] in ":,}]":
                out.append('"')
                in_str = False             # 合法收尾
            else:
                out.append('\\"')          # 内层引号 → 转义
            i += 1
            continue
        # 字符串里出现真实换行/制表符也是非法 JSON，一并转义
        if c == "\n":
            out.append("\\n")
        elif c == "\r":
            out.append("\\r")
        elif c == "\t":
            out.append("\\t")
        else:
            out.append(c)
        i += 1
    if in_str:
        out.append('"')                    # 结尾被截断：补上未闭合的引号
    return "".join(out)


def _trim_dangling_tail(s: str) -> str:
    """去掉尾部悬空的 , 或 "键": 这类残缺片段（截断常留下的尾巴）。"""
    while True:
        new = re.sub(r"[,\s]+$", "", s)
        new = re.sub(r'"[^"\\]*"\s*:\s*$', "", new).rstrip()
        if new == s:
            return s
        s = new


def _close_braces(s: str) -> str:
    """扫描一遍，补齐未闭合的括号（先处理未闭合的引号与悬空尾巴）。"""
    stack, in_str, i, n = [], False, 0, len(s)
    last_comma = None
    while i < n:
        c = s[i]
        if in_str:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
        elif c in "{[":
            stack.append("}" if c == "{" else "]")
        elif c in "}]" and stack:
            stack.pop()
        elif c == ",":
            last_comma = i
        i += 1

    out = s
    if in_str:
        out += '"'
    out = _trim_dangling_tail(out)
    while stack:
        out += stack.pop()
    _close_braces.last_comma = last_comma          # 供回退方案使用
    return out


def _salvage_objects(text: str) -> dict:
    """最后一招：从残缺内容里捞出还能用的字段（针对本项目的扁平结构）。

    校对输出是「一个 verdict + 一组扁平 comment 对象」，
    所以即使整份 JSON 废了，也常能逐条捞回完整的那几条评论。
    """
    data: dict = {}
    mv = re.search(r'"verdict"\s*:\s*"(pass|fail)"', text)
    if mv:
        data["verdict"] = mv.group(1)

    # 逐个抓取不含嵌套的对象（{...} 内没有 { }）
    got = []
    for m in re.finditer(r"\{[^{}]*\}", text):
        seg = m.group(0)
        for candidate in (seg, _close_braces(_fix_stray_quotes(seg))):
            try:
                obj = json.loads(candidate)
            except Exception:                       # noqa: BLE001
                continue
            if isinstance(obj, dict) and any(
                    k in obj for k in ("issue", "quote", "suggestion", "summary")):
                got.append(obj)
                break
    if got:
        data["comments"] = got

    # 单层键值兜底（summary / title 之类）
    for key in ("summary", "title", "theme"):
        if key not in data:
            m2 = re.search(rf'"{key}"\s*:\s*"([^"]*)"', _fix_stray_quotes(text))
            if m2:
                data[key] = m2.group(1)
    return data


def parse_json(text: str) -> dict:
    """容错解析 JSON，尽力把模型的输出救回来。

    修复链（依次尝试，任一成功即返回）：
      ① 剥掉 markdown 围栏后直接解析
      ② 截取首尾花括号之间再解析
      ③ 修复字符串内未转义的英文双引号 —— 报 Expecting ',' delimiter 的主因
      ④ 补全被截断的括号/引号 —— 报 Unterminated string 的主因
      ⑤ 截断在元素中间时，回退到最后一个完整元素再补全
      ⑥ 兜底逐条捞取完整对象
    """
    text = (text or "").strip()
    if not text:
        raise JSONParseError("模型返回了空内容，无法解析为 JSON", "")

    m = _JSON_FENCE.search(text)
    if m:
        text = m.group(1).strip()

    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])

    for raw in list(candidates):
        candidates.append(_fix_stray_quotes(raw))
    for raw in list(candidates):
        closed = _close_braces(raw)
        candidates.append(closed)
        # 截断回退：从该候选的最后一个逗号处截断，丢掉半截元素再补全
        cut = _close_braces.last_comma
        if cut:
            candidates.append(_close_braces(raw[:cut]))

    for cand in candidates:
        try:
            obj = json.loads(cand)
        except Exception:                            # noqa: BLE001
            continue
        if isinstance(obj, dict) and obj:
            return obj

    data = _salvage_objects(_fix_stray_quotes(text))
    if data:
        return data

    # 诊断到底是被截断还是引号问题，好让上层给出准确建议
    why = "引号转义错误或格式残缺"
    if text.count("{") > text.count("}"):
        why = "输出被长度上限截断（括号没有闭合）"
    raise JSONParseError(
        f"模型输出的内容不是合法 JSON（{why}）。开头片段：{text[:200]}", text)


# ── 告警通道：节点遇到「可降级」的异常时推给前端，而不是中断流水线 ──
_NOTICE_CB = None


def set_notice_cb(cb):
    global _NOTICE_CB
    _NOTICE_CB = cb


def emit_notice(level: str, message: str):
    if _NOTICE_CB is None:
        return
    try:
        _NOTICE_CB(level, message)
    except Exception:                                # noqa: BLE001
        pass


class MockLLM:
    """离线 mock：返回固定结构，第一轮校对故意打回，验证回退链路。"""

    def chat(self, system: str, user: str, temperature: float = 0.7, json_mode: bool = False) -> str:
        if "StoryPlanner" in system:
            return json.dumps({
                "title": "测试小说", "theme": "一个人在边界处确认自己的位置",
                "characters": [{
                    "name": "林岸", "personality": "沉默但执拗",
                    "motivation": "查清父亲失踪的真相", "taboo": "绝不对陌生人透露家庭住址",
                    "potential_conflicts": "与守堤人老崔对真相的知情权分歧"}],
                "chapters": [{"index": 1, "title": "雨夜来客",
                              "outline": "雨夜，陌生访客敲开林岸的门，带来了半页残缺的笔记。",
                              "turning_point": "林岸认出笔记上的字迹是父亲的",
                              "beats": [
                                  {"goal": "林岸在雨夜独处，交代他的戒备与独居状态",
                                   "conflict": "无", "turn": "敲门声打断平静",
                                   "words": 600, "emotion": 2},
                                  {"goal": "来访者递上半页残缺笔记",
                                   "conflict": "林岸不信任来客，不肯开门",
                                   "turn": "笔记出现，局面改变",
                                   "words": 900, "emotion": 3},
                                  {"goal": "林岸辨认笔记字迹",
                                   "conflict": "内心抗拒承认这个可能",
                                   "turn": "确认是父亲的笔迹",
                                   "words": 700, "emotion": 4},
                              ],
                              "notes": ""}],
                "core_conflict": "真相与安全的取舍",
                "ending_direction": "开放式：堤上留下脚印，人未归",
                # 开局物资清单：让第 1 章的撰稿人一开始就有一本账可比对。
                # 故意给两样"会被消耗"的东西，便于离线验证账本更新与"已耗尽不可再用"。
                "initial_items": [
                    {"name": "半瓶矿泉水", "qty": "半瓶", "note": "主舱侧袋"},
                    {"name": "手电筒", "qty": "1 支", "note": "电量不明"},
                ],
            }, ensure_ascii=False)
        # 校对拆成三路并行后，Mock 也要分流（判断顺序：具体的在前）。
        # 只让 OOC 这一路在第一轮打回——三路同时 fail 会让重写轮数行为难以验证。
        if "ReviewerOOC" in system:
            if "revision_round=0" in user:
                return json.dumps({
                    "verdict": "fail",
                    "comments": [{"type": "ooc", "severity": "critical",
                                  "quote": "林岸笑着说出了家庭住址",
                                  "issue": "违反人物禁忌：绝不对陌生人透露家庭住址",
                                  "suggestion": "改为沉默应对，转移话题"}],
                }, ensure_ascii=False)
            return json.dumps({"verdict": "pass", "comments": []}, ensure_ascii=False)
        if "ReviewerLogic" in system:
            return json.dumps({"verdict": "pass", "comments": []}, ensure_ascii=False)
        if "ReviewerPacing" in system:
            return json.dumps({"verdict": "pass", "comments": []}, ensure_ascii=False)
        if "Reviewer" in system:          # 兜底：未拆分的老式单一校对提示词
            if "revision_round=0" in user:
                return json.dumps({
                    "verdict": "fail",
                    "comments": [{"type": "ooc", "severity": "critical",
                                  "quote": "林岸笑着说出了家庭住址",
                                  "issue": "违反人物禁忌：绝不对陌生人透露家庭住址",
                                  "suggestion": "改为沉默应对，转移话题"}],
                }, ensure_ascii=False)
            return json.dumps({"verdict": "pass", "comments": []}, ensure_ascii=False)
        if "RequirementPolisher" in system:
            return ("短篇软末世题材，基调压抑安静：写物资未断但人心先荒的过渡期城市。"
                    "主角是一个不善言辞的内向少年，靠拾荒独自生活，一只受伤的流浪猫"
                    "闯入他的据点，两个孤独的生命互相试探、慢慢依存。全篇无激烈打斗，"
                    "冲突都压在细节与沉默里，共 3 章，每章结尾留一个安静但不安的悬念。")
        if "Polisher" in system:
            return "[润色后] " + user.split("正文如下：")[-1].strip()
        if "MemorySettler" in system:
            # 按章号产出递增的伏笔 id，且不推进任何旧伏笔。
            # 这样跑够章数就能顺带验证两件事：伏笔状态机去重、以及
            # 「埋太久没推进」的超期巡检会不会真的报警。
            m = re.search(r"第(\d+)章", user)
            idx = int(m.group(1)) if m else 1
            # 人物名在第 2 章故意带括号备注，用来离线验证「角色名归一后不裂条」
            who = "林岸（主角）" if idx == 2 else "林岸"
            # 物资变动按章给：第 1 章减量 → 第 2 章耗尽 + 新捡一件 → 第 3 章起无变动。
            # 三条路径（改存量 / 标耗尽 / 新增）都能在离线测试里被覆盖到。
            if idx == 1:
                items = [{"name": "半瓶矿泉水", "qty": "只剩两口"},
                         {"name": "手电筒", "qty": "1 支"}]
            elif idx == 2:
                items = [{"name": "半瓶矿泉水", "status": "consumed"},
                         {"name": "塑料布", "qty": "1 张", "note": "第2章在路上捡到"}]
            else:
                items = []
            return json.dumps({
                "summary": f"第{idx}章：林岸收到半页笔记，逐步接近父亲失踪的真相。",
                "new_foreshadows": [
                    {"id": f"F{idx}", "desc": f"第{idx}章留下的悬念：折角与缺页"}],
                "resolved_foreshadows": [],
                "advanced_foreshadows": [],
                "character_updates": [
                    {"name": who, "state": "确认父亲失踪另有隐情，进入戒备状态"}],
                "item_changes": items,
            }, ensure_ascii=False)
        if "ReaderAgent" in system:
            return ("读下来像在雨夜隔着一层玻璃看别人生活——安静，但一直有东西在轻轻敲。"
                    "最抓我的是「折角是父亲的」那句，一个动作把悬念立住了，不用喊。"
                    "劝退点：来客动机交待得太省，第一遍读容易困惑。"
                    "追读意愿 8/10：想知道那半页笔记的来历，明天会接着看。")
        if "StyleSample" in system:
            return ("雨落在铁皮棚上，声音铺得很平。他蹲在檐下水洼边，把手心里的碎米摊开，"
                    "等那只猫自己决定来不来。风从街口拐进来，卷走了两层塑料袋。"
                    "他没有抬头。远处有人推着车经过，轮子压过水，吱呀一声，又没了。"
                    "天光一寸寸暗下去，他手里的米还是原来的样子。")
        if "FieldFiller" in system:
            return json.dumps({
                "characters": [
                    {"name": "沈砚", "age": "34", "appearance": "瘦，常年穿洗旧的灰夹克，右手中指有旧茧",
                     "personality": "沉默寡言，习惯先观察再开口", "speech": "话少，句子短，常以沉默作答",
                     "motivation": "查清父亲当年失踪的真正原因", "obsession": "每晚把父亲留下的半页笔记拿出来看一遍",
                     "fear": "怕最终查到的真相会毁掉对父亲的记忆", "weakness": "对旧物心软，舍不得丢弃任何有字的东西",
                     "taboo": "绝不向任何人透露笔记的存在"},
                    {"name": "崔明远", "age": "61", "appearance": "背微驼，晒得发黑，左手缺半截小指",
                     "personality": "固执，认死理，嘴上不饶人", "speech": "语速慢，爱用反问句压人",
                     "motivation": "守住堤上守了一辈子的那点体面", "obsession": "每天清晨沿堤走满七公里",
                     "fear": "怕被人指认成当年那件事的知情者", "weakness": "耳背，雨夜听不清脚步声",
                     "taboo": "绝不承认自己知道那晚发生了什么"},
                ],
                "world_era": "当代南方水乡小镇，多雨，堤外是连片的鱼塘",
                "world_rules": "现实向，无超自然；一切疑点都能用人的动机解释",
                "world_locations": "老堤防与废弃水文站、半页残缺的手写笔记、镇上唯一还在营业的照相馆",
                "goal": "在下一次汛期前查清父亲失踪的真相",
                "conflict": "真相与安全的取舍：越接近答案，越可能牵连还活着的人",
                "style_sample": ("雨落在铁皮棚上，声音铺得很平。他蹲在檐下水洼边，把手心里的碎米摊开，"
                                 "等那只猫自己决定来不来。风从街口拐进来，卷走了两层塑料袋。他没有抬头。"
                                 "远处有人推着车经过，轮子压过水，吱呀一声，又没了。天光一寸寸暗下去。"),
                "word_count": 3000,
                "foreshadow": "父亲的笔记本缺页，但页码连续",
                "cliffhanger": "每章结尾留下一个未完成的动作",
                "forbidden": "不得描写血腥酷刑细节\n主角不得使用枪械",
            }, ensure_ascii=False)
        # Writer
        if "revision_round=" in user and "revision_round=0" not in user:
            return "雨还在下。林岸握着门把手，没有作声，只把那半页笔记推回对方面前。"
        return ("雨夜，敲门声比雷声先到。林岸在门后站了很久，才把门开了一条缝。"
                "陌生人的雨衣往下滴水，手里捏着半页残缺的笔记。"
                "「这东西，不该出现在别人家里。」来人说。"
                "林岸没有接话，目光落在笔记的折角上——那个折法，是父亲的。")


_MOCK = MockLLM()


def _consume_stream(client, kwargs: dict, system: str, user: str,
                    temperature: float) -> tuple:
    """发一次流式请求并拼接完整文本，返回 (文本, finish_reason)。"""
    stream = client.chat.completions.create(
        model=config.MODEL_NAME,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        **kwargs,
    )

    parts, n_chars, n_reason, finish = [], 0, 0, ""
    last_tick = time.time()
    # 兜底心跳检查：正常情况下 httpx 的 read timeout 会先触发，
    # 但经过某些代理时不一定，所以自己再守一道。
    guard = config.LLM_TIMEOUT + 60

    for chunk in stream:
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue                      # 部分平台会补发只带 usage 的空分片
        ch = choices[0]
        fr = getattr(ch, "finish_reason", None)
        if fr:
            finish = fr
        delta = getattr(ch, "delta", None)
        if delta is None:
            continue
        piece = getattr(delta, "content", None)
        if piece:
            parts.append(piece)
            n_chars += len(piece)
            last_tick = time.time()
            _emit_progress("writing", n_chars)
            continue
        reason = getattr(delta, "reasoning_content", None)
        if reason:
            # 思考分片也算「有数据在流动」，不能当卡死
            n_reason += len(reason)
            last_tick = time.time()
            _emit_progress("thinking", n_reason)
            continue
        if time.time() - last_tick > guard:
            raise TimeoutError(
                f"连续 {int(time.time() - last_tick)} 秒没有收到任何数据")

    text = "".join(parts).strip()
    if not text and n_reason:
        # 思考型模型偶尔把全部内容留在思维链里，没产出正文
        raise RuntimeError(
            f"模型只输出了思考过程（{n_reason} 字），没有产出正文。"
            "该模型可能无法真正关闭思考，建议换用可关闭思考的版本。")
    if not text:
        raise RuntimeError(
            f"模型没有返回正文（finish_reason={finish or '未知'}）。"
            + ("输出被长度上限截断，多半是思考模式吃掉了额度——"
               "请关掉「深度思考」或调高上限。"
               if finish == "length"
               else "请确认该模型支持 JSON 输出，或换一个模型／重试。"))
    return text, finish


def _chat_raw(system: str, user: str, temperature: float, json_mode: bool,
              max_tokens: int = None, extra_note: str = "") -> tuple:
    """发一次请求，返回 (文本, finish_reason)。extra_note 追加在 user 末尾。"""
    if extra_note:
        user = f"{user}\n\n{extra_note}"
    client = _get_client()
    limit = max_tokens or (config.MAX_TOKENS * 4 if config.ENABLE_THINKING
                           else config.MAX_TOKENS)
    kwargs = {"max_tokens": limit, "stream": True}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    tk = _thinking_kwargs()
    if tk:
        kwargs["extra_body"] = tk

    try:
        return _consume_stream(client, kwargs, system, user, temperature)
    except Exception as e:                                # noqa: BLE001
        # 个别平台/版本不认 thinking 字段 → 摘掉它重试一次，别让用户卡在这
        if tk and _is_param_error(e):
            kwargs.pop("extra_body", None)
            return _consume_stream(client, kwargs, system, user, temperature)
        raise


def _truncation_error(text: str, limit: int) -> JSONParseError:
    return JSONParseError(
        f"模型输出被长度上限（{limit} token）截断，JSON 只写了一半，无法解析。"
        f"已生成约 {len(text)} 字。"
        "建议：① 换输出更简洁的模型；② 关掉「深度思考」（思考会占用输出额度）。",
        text)


def chat(system: str, user: str, temperature: float = 0.7, json_mode: bool = False,
         max_tokens: int = None) -> str:
    """单轮对话调用（返回纯文本）。json_mode=True 时强制模型输出 JSON。

    三个坑的防御（都踩过）：
    ① 一律走流式（stream=True）。非流式要等服务端生成完整篇才返回，
       写一章三千字动辄两三分钟，客户端"总时长超时"必然误杀 →
       改成流式后超时语义变为「相邻分片间隔」，只要模型在吐字就不会超时。
    ② 思考模式：DeepSeek 与智谱都默认开启且会显式下发关闭，否则每步慢数倍。
    ③ JSON 模式下模型可能输出「无尽空白」直到 token 上限 → 显式设上限。
    """
    if USE_MOCK:
        return _MOCK.chat(system, user, temperature, json_mode)
    text, finish = _chat_raw(system, user, temperature, json_mode, max_tokens)
    limit = max_tokens or (config.MAX_TOKENS * 4 if config.ENABLE_THINKING
                           else config.MAX_TOKENS)
    if finish == "length":
        if json_mode:
            # JSON 被截断一定解析不了，直接给出准确诊断（别报成"格式错误"）
            raise _truncation_error(text, limit)
        emit_notice("warn", f"模型输出达到长度上限被截断，本章可能不完整"
                            f"（已生成约 {len(text)} 字）。")
    return text


def chat_json(system: str, user: str, temperature: float = 0.3,
              max_tokens: int = None) -> dict:
    """调模型并解析 JSON；解析失败自动附纠错提示重试一次。

    为什么值得为它单独写一个入口：中文模型答 JSON 时最爱在字符串值里
    直接写英文双引号去引原文（把"塞回遮阳板"改成…），JSON 会在那里提前
    结束并报 Expecting ',' delimiter。重试时明确禁止英文双引号，命中率很高。
    """
    if USE_MOCK:
        return parse_json(_MOCK.chat(system, user, temperature, True))

    problems, last_raw = [], ""
    for attempt in range(2):
        note = ""
        if attempt:
            note = ("【重要】你上一次的输出不是合法 JSON（" + problems[0] + "）。"
                    "请重新完整输出：字符串值内部禁止出现英文双引号 \"，"
                    "需要引用时一律用中文引号「」；不要输出任何解释或代码块标记。")
        text, finish = _chat_raw(system, user, temperature, True, max_tokens, note)
        last_raw = text
        limit = max_tokens or (config.MAX_TOKENS * 4 if config.ENABLE_THINKING
                               else config.MAX_TOKENS)
        if finish == "length" and attempt == 0:
            problems.append(f"输出被长度上限截断（约 {len(text)} 字）")
            continue
        try:
            data = parse_json(text)
        except JSONParseError as e:
            problems.append(str(e)[:90])
            continue
        if finish == "length":
            # 重试后仍被截断，但残缺内容还能救回一部分：用上，同时明确告知不完整
            emit_notice("warn",
                        f"模型输出被长度上限截断（约 {len(text)} 字），"
                        f"已从残缺内容中尽量恢复可用字段，结果可能不完整。")
        return data

    raise JSONParseError(problems[-1] if problems else "JSON 解析失败", last_raw)
