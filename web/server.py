"""网页端后端：FastAPI + SSE 实时推送流水线进度。

启动：
  cd story-agents
  python -m uvicorn web.server:app --port 8765
  浏览器打开 http://127.0.0.1:8765
"""
import copy
import json
import os
import queue
import threading
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
import llm
import models
import products
import prompts
import nodes
import graph as graph_mod
import state as state_mod
from state import StoryState

app = FastAPI(title="小说多 Agent 创作工坊")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
# 静态资源（字体等）：/static/fonts/xxx.ttf
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 单用户本地工具：锁防并发；会话保存上一次运行结束后的完整 State，支持续写
#
# 这把锁的语义是「一次创作独占」：worker 全程持锁，其它会改动全局模型配置的
# 接口（润色需求 / 读者反馈 / AI 补全 / 只预览大纲 / 新建故事）用**非阻塞**方式
# 取锁，取不到就返回 409 说清楚原因。为什么不排队等：那些接口的调用链会读
# config.DEEPSEEK_API_KEY / MODEL_NAME / ENABLE_THINKING 这些**模块级全局量**，
# 而 worker 正在跑时也在读同一份——排队等上几分钟又没有任何进度反馈，
# 不如立刻告诉用户"正在生成中"。详见 apply_llm_settings 的说明。
_lock = threading.Lock()
_session: Dict[str, Any] = {"state": None, "out_dir": None, "saved_at": None}

# 「停止生成」信号：由 /api/generate/stop 或客户端断开连接置位，
# worker 在每章开头检查它，置位就不再开始下一章（正在跑的那一章会自然跑完）。
_cancel = threading.Event()

MAX_CHAPTERS_PER_RUN = 10   # 单次运行章节数上限（防误填几十章）

# ── 会话落盘 ────────────────────────────────────────────────────
# 原来 _session 只活在进程内存里：关掉黑窗口（或电脑重启）后，策划案、记忆库、
# 已定稿章节全部丢失，「续写下一章」直接报「没有可续写的会话」，用户只能从头再来。
# 现在把 State 存成 outputs_web/_session.json，服务重启时自动读回。
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs_web")
SESSION_PATH = os.path.join(OUTPUTS_DIR, "_session.json")


def _display_dir(path: Optional[str]) -> Optional[str]:
    """把产物目录转成「相对项目根」的展示用路径。

    为什么不在界面上显示绝对路径：它含盘符与 Windows 用户名
    （`E:\\...\\<用户名>\\...`），而这一行最容易出现在截图、issue 与 README 里
    —— 本项目的界面截图就是这么外泄的。程序按约定必须在项目根启动，
    从项目根找 `outputs_web/run_xxx/` 不会有歧义，展示相对路径足够。
    磁盘上真正用的仍是绝对路径，只有下发给前端的这份是相对的。
    """
    if not path:
        return None
    abs_path = os.path.abspath(path)
    try:
        rel = os.path.relpath(abs_path, PROJECT_ROOT)
    except ValueError:                      # 跨盘符时 relpath 会抛
        return os.path.basename(abs_path)
    if rel.startswith(".."):                # 不在项目内：只报最后一级，不外泄完整路径
        return os.path.basename(abs_path) or "."
    return rel.replace("\\", "/") + "/"



def _save_session() -> None:
    """把当前会话原子写入磁盘：先写临时文件再替换，避免写一半断电留下坏档。"""
    if not _session.get("state"):
        return
    try:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        payload = {
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "out_dir": _session.get("out_dir"),
            "state": _session["state"],
        }
        tmp = SESSION_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, SESSION_PATH)          # 同一分区内的原子替换
        _session["saved_at"] = payload["saved_at"]
    except Exception as e:                     # noqa: BLE001
        # 存档失败不该影响创作本身，打印一行日志即可
        print(f"[session] 会话存档写入失败：{type(e).__name__}: {e}")


def _load_session() -> None:
    """服务启动时把上次的会话读回来，让「续写下一章」在重启后依然可用。"""
    try:
        with open(SESSION_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return
    except Exception as e:                     # noqa: BLE001
        print(f"[session] 会话存档读取失败（已忽略）：{type(e).__name__}: {e}")
        return
    state = data.get("state")
    if isinstance(state, dict) and state.get("outline"):
        _session["state"] = state
        old = data.get("out_dir")
        # 早期存档写的是 "web\..\outputs_web\..." 这种带相对段的路径，顺手规范化
        _session["out_dir"] = os.path.abspath(old) if old else None
        _session["saved_at"] = data.get("saved_at")
        n = len(state.get("final_chapters") or [])
        print(f"[session] 已恢复上次进度：{n} 章（存档于 {_session['saved_at']}）")


def _remember(state: Dict[str, Any], out_dir: str) -> None:
    """更新内存会话并立刻落盘（每章定稿后调用，中途出错也不丢已写好的部分）。"""
    _session["state"] = copy.deepcopy(state)
    _session["out_dir"] = out_dir
    _save_session()


def _merge_update(state: Dict[str, Any], delta: Any) -> bool:
    """把 LangGraph 的一次节点更新并入 state；返回是否真的并进了内容。

    为什么需要这一层：LangGraph 对「一个字段都没写」的节点，在
    stream_mode="updates" 下上报的是 {node: None}，而不是空字典（1.2.12 实测）。
    而本项目里「合法的空操作」节点确实存在：

      · chapter_planner_node —— 本章已在策划案范围内时 return {}
      · memory_settler_node  —— 记忆结算失败、降级放行时 return {}

    于是 state.update(None) 会抛 `TypeError: 'NoneType' object is not iterable`，
    整条流水线当场断掉。2026-10-02 的实跑正是死在这里：看板刚打出「策划完成」
    就「出错中断」，run 目录一片空白（一章都没写出来）。

    守卫放在消费端、而不是去改节点的返回值：`return {}` 表示「本次不更新」是
    正确语义，消费端本来就不该假设 delta 一定是 dict。只要以后还有空操作节点，
    这个坑就会再出现，而 CLI 走的是 app.invoke，踩不到——所以必须有测试钉住。
    """
    if not delta:
        return False
    state.update(delta)
    return True


def session_summary() -> Dict[str, Any]:
    """给前端的会话摘要：够恢复界面即可，不带 meta / user_prompt 等大字段。"""
    st = _session.get("state")
    if not st:
        return {"has_session": False}
    chapters = [
        {"index": c.get("index"), "title": c.get("title") or "",
         "text": c.get("text") or "", "rounds": c.get("rounds", 0)}
        for c in (st.get("final_chapters") or [])
    ]
    return {
        "has_session": True,
        "saved_at": _session.get("saved_at"),
        "out_dir": _display_dir(_session.get("out_dir")),
        "outline": st.get("outline") or {},
        "memory": st.get("memory") or {},
        "chapters": chapters,
    }


# 服务启动即尝试恢复上次进度（uvicorn 导入本模块时执行）
_load_session()


# ── 请求模型（表单字段一一对应）────────────────────────────────
class CharacterCard(BaseModel):
    name: str = ""          # 姓名
    age: str = ""           # 年龄
    appearance: str = ""    # 外貌
    personality: str = ""   # 性格
    speech: str = ""        # 说话习惯
    motivation: str = ""    # 动机
    obsession: str = ""     # 内心执念
    fear: str = ""          # 恐惧
    weakness: str = ""      # 弱点
    taboo: str = ""         # 禁忌 / OOC 红线


class GenReq(BaseModel):
    # 核心需求
    requirement: str
    # 人物卡（动态多张）
    characters: List[CharacterCard] = []
    # 世界观
    world_era: str = ""        # 时代与环境
    world_rules: str = ""      # 世界/力量规则
    world_locations: str = ""  # 关键地点与物品
    # 故事目标
    goal: str = ""             # 主角目标
    conflict: str = ""         # 核心冲突
    # 进阶选项
    style_sample: str = ""     # 文风参考片段
    word_count: int = 0        # 每章字数
    foreshadow: str = ""       # 每章必须埋的伏笔
    cliffhanger: str = ""      # 结尾悬念要求
    forbidden: str = ""        # 禁止内容（一行一条）
    # 运行设置
    chapters: int = 1
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    mock: bool = False
    thinking: bool = False   # 深度思考：慢数倍，默认关闭
    # 断点续写：mode=continue 复用服务端会话中的策划案/记忆/已定稿章节
    mode: str = "new"                       # "new" / "continue"
    memory_override: Optional[Dict] = None  # 用户在记忆库面板编辑后的 JSON（续写时生效）


class PolishReq(BaseModel):
    text: str
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    mock: bool = False
    thinking: bool = False


class ReaderReq(BaseModel):
    text: str
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    mock: bool = False
    thinking: bool = False


class TestReq(BaseModel):
    """测试连接：验证 API Key / Base URL / 模型名 三件套是否可用。"""
    api_key: str = ""
    base_url: str = ""
    model: str = ""


class FillReq(BaseModel):
    """表单分段 AI 补全：section 决定要补哪些字段。"""
    section: str                       # characters / world / advanced
    requirement: str = ""              # 核心需求（作为补全依据）
    current: Dict[str, Any] = {}       # 该分段已填内容
    overwrite: bool = False            # False=只填空缺；True=全部用 AI 结果替换
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    mock: bool = False
    thinking: bool = False


# 每个分段允许 AI 回填的字段（白名单，防止模型塞入表单没有的键）
SECTION_KEYS = {
    "characters": ["characters"],
    "world": ["world_era", "world_rules", "world_locations", "goal", "conflict"],
    "advanced": ["style_sample", "word_count", "foreshadow", "cliffhanger", "forbidden"],
}

SECTION_SCHEMA = {
    "characters": """{
  "characters": [
    {"name":"姓名","age":"年龄","appearance":"外貌","personality":"性格",
     "speech":"说话习惯","motivation":"动机","obsession":"内心执念",
     "fear":"恐惧","weakness":"弱点","taboo":"人设禁忌——绝不能做的事"}
  ]
}（2-3 个主要人物；taboo 必须具体，后续校对按它判 OOC）""",
    "world": """{
  "world_era":"时代与环境（一句话）",
  "world_rules":"世界/力量规则（一句话）",
  "world_locations":"关键地点与物品（一句话，逗号分隔）",
  "goal":"主角目标（一句话）",
  "conflict":"核心冲突（一句话）"
}""",
    "advanced": """{
  "style_sample":"文风参考样例，100-150 字的叙事片段，不要出现具体人名",
  "word_count":3000,
  "foreshadow":"每章必须埋的伏笔（一句话）",
  "cliffhanger":"结尾悬念要求（一句话）",
  "forbidden":"禁止内容，一行一条，用 \\n 分隔"
}""",
}


def apply_llm_settings(api_key="", base_url="", model="", mock=False, thinking=None):
    """把本次请求的模型设置写进**模块级全局配置**。

    ⚠ 调用者必须已经持有 `_lock`（或由 worker 在锁内调用）。

    为什么这个约定是硬要求：config.DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL /
    MODEL_NAME / ENABLE_THINKING 与 llm.USE_MOCK 都是进程级全局量，而流水线里
    每一次 LLM 调用都是在读它们。以前这几个函数**全部不持锁**，于是
    「写到第 3 章时点一下『需求润色 / AI 补全 / 读者反馈 / 大纲预览』」就会
    把正在跑的这次创作的模型、Key、思考开关中途换掉，甚至真模型与 mock 混跑；
    又因为这里只在字段非空时才覆盖，还会产生"换了 model 但沿用旧 key"的
    半新半旧配置。两个并发 /api/generate 时更明显：后一个请求先改配置，
    前一个拿到锁后执行的却是后一个的配置。

    彻底的做法是把配置随请求参数化（一路传到 llm.chat），那是较大的重构；
    当前用「所有写全局配置的入口统一持同一把锁」把竞态从根上掐掉，
    代价是并发点这些功能会得到一句明确的 409 而不是静默把稿子写坏。
    """
    if api_key:
        config.DEEPSEEK_API_KEY = api_key
    if base_url:
        config.DEEPSEEK_BASE_URL = base_url
    if model:
        config.MODEL_NAME = model
    if thinking is not None:
        config.ENABLE_THINKING = bool(thinking)
    llm.USE_MOCK = mock


def _acquire_or_409(what: str) -> None:
    """非阻塞取「创作独占锁」；正在生成时直接 409，说清楚发生了什么。

    用非阻塞而不是排队等待：等待期间前端既没有进度、也不知道要等多久，
    体感就是"按钮点了没反应"。这里如实告诉用户"正在生成中"更有用。
    """
    if not _lock.acquire(blocking=False):
        raise HTTPException(
            409,
            f"正在生成中，暂时不能{what}。请等这一章跑完，"
            f"或先点「停止生成」（已定稿的章节会保留）。")


def build_user_prompt(req: GenReq) -> str:
    """把结构化表单编译成策划 Agent 可读的需求文本。"""
    parts = [f"【核心需求】\n{req.requirement.strip()}"]

    if req.characters:
        parts.append("【人物卡】")
        for i, c in enumerate(req.characters, 1):
            if not c.name.strip():
                continue
            lines = [f"{i}. 姓名：{c.name}"]
            if c.age: lines.append(f"   年龄：{c.age}")
            if c.appearance: lines.append(f"   外貌：{c.appearance}")
            if c.personality: lines.append(f"   性格：{c.personality}")
            if c.speech: lines.append(f"   说话习惯：{c.speech}")
            if c.motivation: lines.append(f"   动机：{c.motivation}")
            if c.obsession: lines.append(f"   内心执念：{c.obsession}")
            if c.fear: lines.append(f"   恐惧：{c.fear}")
            if c.weakness: lines.append(f"   弱点：{c.weakness}")
            if c.taboo: lines.append(f"   禁忌/OOC红线：{c.taboo}")
            parts.append("\n".join(lines))

    world = []
    if req.world_era: world.append(f"- 时代与环境：{req.world_era}")
    if req.world_rules: world.append(f"- 世界/力量规则：{req.world_rules}")
    if req.world_locations: world.append(f"- 关键地点与物品：{req.world_locations}")
    if world:
        parts.append("【世界观】\n" + "\n".join(world))

    goal = []
    if req.goal: goal.append(f"- 主角目标：{req.goal}")
    if req.conflict: goal.append(f"- 核心冲突：{req.conflict}")
    if goal:
        parts.append("【故事目标】\n" + "\n".join(goal))

    extra = []
    if req.forbidden.strip():
        items = [x.strip() for x in req.forbidden.splitlines() if x.strip()]
        if items:
            extra.append("- 禁止内容（所有章节绝对不可出现）：" + "；".join(items))
    if extra:
        parts.append("【创作要求】\n计划写 " + str(req.chapters) + " 章\n" + "\n".join(extra))

    return "\n\n".join(parts)


def build_meta(req: GenReq) -> Dict[str, Any]:
    forbidden_list = [x.strip() for x in req.forbidden.splitlines() if x.strip()]
    return {
        "style_sample": req.style_sample.strip(),
        "word_count": req.word_count,
        "foreshadow": req.foreshadow.strip(),
        "cliffhanger": req.cliffhanger.strip(),
        "forbidden_list": forbidden_list,
    }


def _fresh_state(req: GenReq) -> StoryState:
    """表单 → 初始 State。字段清单在 state.new_state（与 CLI 共用一份）。"""
    return state_mod.new_state(build_user_prompt(req), build_meta(req))


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


@app.get("/api/session")
def api_session():
    """查询是否存在可续写的会话。

    页面刷新、浏览器重开、甚至服务重启（start.bat 重新跑）之后，
    前端都用它把上次的大纲 / 已定稿章节 / 记忆库恢复出来。
    """
    return session_summary()


@app.post("/api/session/clear")
def api_session_clear():
    """新建故事：清掉当前会话（已生成的文件仍留在 outputs_web，不做删除）。

    必须在锁内执行：生成任务跑到一半时点「新建故事」，以前会删掉
    `_session.json` 并把内存会话置空，紧接着 worker 的 `_remember` 又把 state
    写回来、`os.replace` 重建文件 —— 于是「新建」之后旧会话又冒出来（看似随机）。
    取不到锁就给 409，而不是让两边交错。
    """
    _acquire_or_409("新建故事")
    try:
        _session["state"] = None
        _session["out_dir"] = None
        _session["saved_at"] = None
        try:
            if os.path.exists(SESSION_PATH):
                os.remove(SESSION_PATH)        # 只删本程序生成的存档，不碰用户产物
        except Exception as e:                 # noqa: BLE001
            print(f"[session] 存档删除失败：{type(e).__name__}: {e}")
        # 告警去重表是进程级的，key 只与"值名"相关（如 item.status.xxx）。
        # 不清空的话，**上一个故事触发过的同类告警会在下一个故事里被永久屏蔽**。
        models.reset_warnings()
    finally:
        _lock.release()
    return {"ok": True}


class MemoryEditReq(BaseModel):
    """记忆库面板的手工编辑结果（整份替换）。"""
    memory: Dict[str, Any] = {}


@app.post("/api/session/memory")
def api_session_memory(req: MemoryEditReq):
    """把用户在记忆库面板里手工改过的 JSON 落盘。

    以前「保存记忆修改」按钮只改了一句提示文字：既不写 localStorage、
    也不回写服务端，刷新页面就被 `_session.json` 里的旧值覆盖回去 ——
    按钮文案与行为不符。这里让它真正落盘。
    """
    _acquire_or_409("保存记忆库")
    try:
        st = _session.get("state")
        if not st:
            raise HTTPException(400, "当前没有可续写的会话，记忆库暂时无处保存"
                                     "（先完整创作至少一章）。")
        st["memory"] = req.memory or {}
        _save_session()
        saved_at = _session.get("saved_at")
    finally:
        _lock.release()
    return {"ok": True, "saved_at": saved_at}


@app.post("/api/polish_requirement")
def api_polish_requirement(req: PolishReq):
    """核心需求 AI 润色：只言片语 → 结构完整的创作需求。"""
    if not req.text.strip():
        raise HTTPException(400, "内容为空，先写点东西再润色")
    _acquire_or_409("润色需求")
    try:
        apply_llm_settings(req.api_key, req.base_url, req.model, req.mock,
                           getattr(req, "thinking", False))
        raw = llm.chat(prompts.REQ_POLISHER_SYSTEM,
                       f"用户的原始需求：\n{req.text.strip()}",
                       temperature=0.6)
    finally:
        _lock.release()
    return {"text": raw}


@app.post("/api/reader_feedback")
def api_reader_feedback(req: ReaderReq):
    """读者反馈 Agent：以挑剔读者视角给阅读感受与追读意愿分。"""
    if not req.text.strip():
        raise HTTPException(400, "正文为空")
    _acquire_or_409("生成读者反馈")
    try:
        apply_llm_settings(req.api_key, req.base_url, req.model, req.mock,
                           getattr(req, "thinking", False))
        raw = llm.chat(prompts.READER_SYSTEM,
                       f"【正文】\n{req.text.strip()}\n\n请输出你的阅读反馈。",
                       temperature=0.7)
    finally:
        _lock.release()
    return {"text": raw}


def _explain_conn_error(exc: Exception) -> str:
    """把 SDK 抛出的异常翻译成用户能看懂的中文提示（按最常见踩坑排序）。"""
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    resp = getattr(exc, "response", None)
    body = ""
    if resp is not None:
        try:
            body = (resp.text or "")[:300]
        except Exception:
            body = ""
    low = (body or "").lower()

    if "APIConnection" in name or "Connect" in name or "connect" in low:
        return ("连不上这个地址。请检查：① Base URL 有没有写错、有没有多写/少写路径；"
                "② 本地模型服务（Ollama/LM Studio）是否已经启动；③ 网络或代理是否正常。")
    if "Timeout" in name or "timeout" in low or "timed out" in low:
        return ("请求超时。常见于「思考型模型 + 输出被截断」，或网络到该平台很慢。"
                "可换该平台的快速版模型（如 glm-5.3-flash / qwen3.8-flash）再试。")
    if status == 401 or "invalid api key" in low or "unauthorized" in low or "authentication" in low:
        return ("认证失败（401）：API Key 无效、复制不完整（漏字符/带空格），"
                "或这个 Key 不属于当前选择的平台。")
    if status == 403:
        return "无权限（403）：Key 有效但没开通该模型，或账号未实名。请到平台控制台确认。"
    if status == 404 or "not found" in low or "does not exist" in low or "model_not_exist" in low:
        return ("模型或地址不存在（404）：最可能是「模型名写错」或「Base URL 漏了 /v1」。"
                "请到控制台复制准确的模型 ID 与接口地址。")
    if status == 429:
        return "被限流或额度用尽（429）：稍后重试，或到控制台查看余额与并发限制。"
    if status in (400, 422):
        return f"请求被平台拒绝（{status}）：模型名或参数不被接受。" + (f"详情：{body}" if body else "")
    if status:
        return f"调用失败（HTTP {status}）。" + (f"详情：{body}" if body else "")
    return f"调用失败：{name} {exc}"


@app.post("/api/test_connection")
def api_test_connection(req: TestReq):
    """测试连接：真实发一次最小请求，验证 Key / Base URL / 模型名。

    ① 顺带尝试 GET {base_url}/models，把平台真实模型清单回传前端当候选；
       拿不到不影响结论（不少平台不开放该接口）。
    ② 再发一条最小对话，确认这个模型真的能调通。
    """
    from openai import OpenAI

    key = req.api_key.strip() or config.DEEPSEEK_API_KEY
    base = req.base_url.strip() or config.DEEPSEEK_BASE_URL
    model = req.model.strip() or config.MODEL_NAME

    out = {"ok": False, "stage": "", "base_url": base, "model": model,
           "latency_ms": None, "reply": "", "models": [], "message": "", "note": ""}

    if not key:
        out.update(stage="key",
                   message="还没填 API Key。本地模型（Ollama / LM Studio）可以不填，"
                           "但地址只要不是本机，就必须填 Key。")
        return out

    # ① 拉模型清单（可选能力，失败静默）
    try:
        c = OpenAI(api_key=key, base_url=base, timeout=20.0, max_retries=0)
        out["models"] = sorted(m.id for m in c.models.list().data)
    except Exception:
        pass

    # ② 真跑一次最小对话
    t0 = time.time()
    try:
        c = OpenAI(api_key=key, base_url=base, timeout=90.0, max_retries=0)
        resp = c.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "请只回复两个字：可用"}],
            max_tokens=256,
        )
        msg = resp.choices[0].message
        text = (msg.content or "").strip()
        reasoning = (getattr(msg, "reasoning_content", None) or "").strip()
        if not text:
            text = "（思考型模型把内容放在思考过程里了，本次未产出正文；连接本身正常）" \
                if reasoning else "（返回内容为空，但连接正常）"
        out.update(ok=True, stage="done",
                   latency_ms=int((time.time() - t0) * 1000), reply=text,
                   message=f"连接成功，模型「{model}」可以正常调用")
        if out["models"] and model not in out["models"]:
            out["note"] = (f"提示：平台返回的模型清单里没列出「{model}」，但调用成功了，"
                           "说明该平台没把这个模型放进 /models 列表。")
        return out
    except Exception as e:
        out.update(stage="chat", latency_ms=int((time.time() - t0) * 1000),
                   message=_explain_conn_error(e))
        return out


def _blank(v) -> bool:
    """判断表单字段是否为空（用于「只填空缺」策略）。"""
    if v is None:
        return True
    if isinstance(v, str):
        return not v.strip()
    if isinstance(v, (list, dict, tuple)):
        return len(v) == 0
    if isinstance(v, (int, float)):
        return v == 0
    return False


@app.post("/api/ai_fill")
def api_ai_fill(req: FillReq):
    """分段 AI 补全：人物卡 / 世界观与目标 / 进阶选项。

    overwrite=False 时只补空着的字段，用户已填内容原样保留；
    overwrite=True 时用 AI 结果整体替换该分段。
    """
    keys = SECTION_KEYS.get(req.section)
    if not keys:
        raise HTTPException(400, f"未知分段：{req.section}")
    if not req.mock and not (req.api_key or config.DEEPSEEK_API_KEY):
        if "api.deepseek.com" in (req.base_url or config.DEEPSEEK_BASE_URL):
            raise HTTPException(400, "未填 API Key。本地模型请填 Base URL，或勾选 mock 先试跑")

    current = req.current or {}
    todo = list(keys) if req.overwrite else [k for k in keys if _blank(current.get(k))]
    if not todo:
        return {"fields": {}, "todo": [],
                "message": "该分段已填写完整；如需让 AI 重写，请点「AI 重写」"}

    _acquire_or_409("AI 补全表单")
    try:
        apply_llm_settings(req.api_key, req.base_url, req.model, req.mock,
                           getattr(req, "thinking", False))

        filled = {k: v for k, v in current.items() if not _blank(v)}
        user_msg = (
            f"【核心需求】\n{req.requirement.strip() or '（用户未填写，请自设一套内在自洽的通用设定）'}\n\n"
            f"【用户已填写的内容（不得改变其原意）】\n"
            f"{json.dumps(filled, ensure_ascii=False, indent=2) if filled else '（暂无）'}\n\n"
            f"【本次需要补全的字段】{'、'.join(todo)}\n\n"
            f"请严格按下面的 JSON 结构输出（字段名一字不差）：\n{SECTION_SCHEMA[req.section]}"
        )
        # 走 chat_json：解析失败会自动附「禁止英文双引号」的提示重试一次，
        # 补全表单时模型也常犯这个毛病（在 issue/suggestion 里直接引原文）
        data = llm.chat_json(prompts.FIELD_FILLER_SYSTEM, user_msg, temperature=0.8)

        result = {}
        for k in todo:
            if k in data and not _blank(data[k]):
                result[k] = data[k]

        # 文风样例用专用提示词单独生成，质感更稳
        if "style_sample" in todo:
            sample = llm.chat(
                prompts.STYLE_SAMPLE_SYSTEM,
                f"【题材与基调】\n{req.requirement.strip() or '（未指定，按经典短篇小说的语言质感写）'}",
                temperature=0.9).strip()
            if sample:
                result["style_sample"] = sample
    finally:
        _lock.release()

    return {"fields": result, "todo": todo}


@app.post("/api/outline_preview")
def api_outline_preview(req: GenReq):
    """风格预览：只跑策划 Agent 出大纲，不写正文。"""
    if not req.requirement.strip():
        raise HTTPException(400, "核心需求不能为空")
    _acquire_or_409("预览大纲")
    try:
        apply_llm_settings(req.api_key, req.base_url, req.model, req.mock,
                           getattr(req, "thinking", False))
        state = _fresh_state(req)
        delta = nodes.planner_node(state)
    finally:
        _lock.release()
    return {"outline": delta["outline"]}


@app.post("/api/generate/stop")
def api_generate_stop():
    """请求停止当前的生成任务。

    刻意**不取锁**：它必须在创作进行中也能立刻响应。置位后 worker 会在
    **下一章开头**退出（正在跑的那一章让它自然跑完，避免留下半截正文），
    已定稿的章节照常落盘。
    """
    _cancel.set()
    return {"ok": True}


@app.post("/api/generate")
def api_generate(req: GenReq):
    if not req.requirement.strip():
        raise HTTPException(400, "核心需求不能为空")
    if not req.mock and not (req.api_key or config.DEEPSEEK_API_KEY):
        if "api.deepseek.com" in (req.base_url or config.DEEPSEEK_BASE_URL):
            raise HTTPException(400, "未填 API Key。本地模型请填 Base URL，或勾选 mock 先试跑")

    chapters = max(1, min(req.chapters, MAX_CHAPTERS_PER_RUN))
    continue_mode = req.mode == "continue"

    if continue_mode:
        if not _session.get("state"):
            raise HTTPException(400, "没有可续写的会话：请先完整创作至少一章")
        state = copy.deepcopy(_session["state"])
        # 产品规则 2：续写模式表单单向只读，修改不回写——
        # 策划案 / 人物卡 / 世界观 / 进阶项全部沿用会话，唯一入口是记忆库面板（memory_override）
        if isinstance(req.memory_override, dict) and req.memory_override:
            state["memory"] = req.memory_override
    else:
        state = _fresh_state(req)

    start_idx = len(state["final_chapters"]) + 1
    app_graph = graph_mod.build_graph()

    out_dir = _session.get("out_dir") if continue_mode and _session.get("out_dir") else None
    if not out_dir:
        out_dir = os.path.join(OUTPUTS_DIR, f"run_{time.strftime('%Y%m%d_%H%M%S')}")
    # 续写时 out_dir 来自会话，而那个目录可能已被用户手工删掉（或换了盘）。
    # 这里必须**在开跑之前**就补建：否则一路 LLM 调用全跑完、到写产物时才抛
    # FileNotFoundError，整章的钱白烧。
    os.makedirs(out_dir, exist_ok=True)

    def sse():
        """SSE 推送。

        流水线跑在独立线程里，主线程只负责把消息实时推给前端。
        原因：模型一次调用要几十秒到几分钟，如果直接在生成器里同步跑，
        这段时间生成器被阻塞，进度和心跳一个都发不出去——用户只看到
        「长时间无响应」，然后被中间的代理/浏览器掐断连接。

        另外整个流程包在 try 里：任何一步抛异常（模型超时、JSON 解析失败…）
        都以 error 事件推给前端，而不是让连接直接断掉，否则前端只会看到
        "network error"，完全不知道后端发生了什么。
        """
        q: "queue.Queue" = queue.Queue()

        def push(payload: dict):
            q.put(payload)

        # 进度节流：模型每吐一个分片就推一次太吵，攒够字数或间隔才推
        throttle = {"t": 0.0, "n": 0}

        def on_progress(kind: str, chars: int):
            now = time.time()
            if kind == "writing" and now - throttle["t"] < 0.4 \
                    and chars - throttle["n"] < 60:
                return
            throttle["t"], throttle["n"] = now, chars
            push({"type": "writing", "kind": kind, "chars": chars})

        # 可降级的异常（校对输出格式坏了、记忆结算失败…）走这里：
        # 前端显示一条醒目告警，但流水线继续跑，不白费已经写好的正文
        def on_notice(level: str, message: str):
            push({"type": "notice", "level": level, "message": message})

        def worker():
            try:
                with _lock:
                    # 配置在**锁内**应用：api_generate 是在启动线程前就返回的，
                    # 若在端点里改配置，趁 worker 还没拿到锁的窗口，另一个请求
                    # 就能把配置改掉（见 apply_llm_settings 的说明）。
                    apply_llm_settings(req.api_key, req.base_url, req.model,
                                       req.mock, getattr(req, "thinking", False))
                    _cancel.clear()          # 新的一次运行，先清掉上一轮可能留下的停止信号
                    llm.set_progress_cb(on_progress)
                    llm.set_notice_cb(on_notice)
                    push({"type": "start", "chapters": chapters, "mock": req.mock,
                          "mode": req.mode, "start_index": start_idx})

                    stopped = False
                    for idx in range(start_idx, start_idx + chapters):
                        # 「停止生成」只在章与章之间生效：正在跑的那一章让它自然
                        # 收尾（否则会留下半截正文），已定稿的章节照常落盘。
                        if _cancel.is_set():
                            stopped = True
                            break
                        # 每章重置字段清单与 CLI 共用（state.reset_chapter_fields），
                        # 漏清三路校对 key 会把上一章的意见串进这一章。
                        state.update(state_mod.reset_chapter_fields(idx))
                        push({"type": "chapter_start", "index": idx,
                              "is_first": not state["outline"], "thinking": req.thinking})

                        for update in app_graph.stream(state, stream_mode="updates"):
                            for node, delta in update.items():
                                # 空操作节点会上报 {node: None}，不能直接 state.update
                                # —— 详见 _merge_update 的说明。
                                if not _merge_update(state, delta):
                                    continue
                                if node == "planner":
                                    push({"type": "planner_done", "outline": state["outline"]})
                                    # 策划案一出来就先存盘：后面写正文万一失败，
                                    # 重开时策划案还在，续写只需重跑撰写而不用重跑策划。
                                    _remember(state, out_dir)
                                elif node == "writer":
                                    push({"type": "writer_done", "index": idx,
                                          "round": state.get("revision_round", 0)})
                                elif node == "merge_reviews":
                                    # 三路 specialist 的汇合点：一次性把合并后的
                                    # 结果推给前端（各路的中间态没有单独推送价值）
                                    push({"type": "review", "index": idx,
                                          "verdict": state["review_verdict"],
                                          "comments": state["review_comments"]})
                                elif node == "bump_round":
                                    push({"type": "rewrite", "index": idx,
                                          "round": state["revision_round"]})

                        final = state["final_chapters"][-1]
                        # 产物统一走 products 模块（原子写 + 标题归一）。
                        # 这里同时把 outline.json / memory.json / final.md 一起刷掉：
                        # 原先它们写在逐章循环**之外**，中途任何一章失败，产物目录里
                        # 就只剩 chapter_XX.md，用户按目录找文件会以为内容丢了。
                        products.write_chapter_products(out_dir, state)
                        push({
                            "type": "chapter_done", "index": idx,
                            "title": final["title"], "text": final.get("text") or "",
                            "rounds": state.get("revision_round", 0),
                            "mock": req.mock,
                            "style_report": state.get("style_report") or {},
                            "minor_comments": [c for c in (state.get("review_comments") or [])
                                               if c.get("severity") == "minor"],
                        })
                        # 每章定稿立刻落盘：万一后面某章失败、或用户直接关掉窗口，
                        # 已写好的章节与记忆库都还在，重开就能接着续写。
                        _remember(state, out_dir)

                    # 保存会话供续写（包含策划案 / 记忆 / 已定稿章节）
                    _remember(state, out_dir)

                    if stopped:
                        push({"type": "stopped",
                              "out_dir": _display_dir(out_dir),
                              "memory": state.get("memory") or {},
                              "outline": state["outline"],
                              "total_chapters": len(state["final_chapters"])})
                    else:
                        push({"type": "all_done", "out_dir": _display_dir(out_dir),
                              "memory": state.get("memory") or {},
                              "outline": state["outline"],
                              "mock": req.mock,
                              "total_chapters": len(state["final_chapters"])})
            except Exception as e:                      # noqa: BLE001
                # 已定稿的章节仍然入库，便于用户续写已写好的部分
                if state.get("final_chapters"):
                    products.write_chapter_products(out_dir, state)
                    _remember(state, out_dir)
                push({"type": "error",
                      "message": _explain_run_error(e),
                      "detail": f"{type(e).__name__}: {e}"[:500],
                      "done_chapters": len(state.get("final_chapters") or [])})
            finally:
                llm.set_progress_cb(None)
                llm.set_notice_cb(None)
                q.put(None)                             # 结束哨兵

        threading.Thread(target=worker, daemon=True).start()

        try:
            while True:
                try:
                    item = q.get(timeout=15)
                except queue.Empty:
                    yield _evt({"type": "ping"})        # 心跳：保持连接不被中间层掐断
                    continue
                if item is None:
                    break
                yield _evt(item)
        except GeneratorExit:
            # 客户端断开了（关页面 / 点了「停止生成」/ 网络断）。
            # 置位停止信号：worker 是 daemon 线程，不通知它的话，浏览器已经走了，
            # 它还会把剩下的章节一章一章跑完——继续烧 token。
            _cancel.set()
            raise

    return StreamingResponse(sse(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


def _evt(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _explain_run_error(exc: Exception) -> str:
    """把流水线运行中抛出的异常翻译成用户能看懂的中文说明。"""
    name = type(exc).__name__
    msg = str(exc) or name
    low = msg.lower()
    if "Timeout" in name or "timed out" in low or "超时" in msg or "没有收到任何数据" in msg:
        model = config.MODEL_NAME or "当前模型"
        tips = []
        if llm.zhipu_force_thinking():
            tips.append(f"你用的「{model}」是智谱的强制思考模型，思考过程关不掉，"
                        "写长文天然就慢。换成 glm-5.2 / glm-5.1 / glm-4.6 会明显更快。")
        else:
            tips.append("先确认「深度思考」是关闭状态——开着会让每一步都先跑一段长思维链。")
        tips.append("换该平台的快速版模型（如 glm-5.2 / qwen3.8-flash）通常立刻见效。")
        tips.append("检查网络或代理：到该平台的连接是否稳定。")
        return ("调用模型超时了：已连通，但长时间没有收到模型返回的任何内容。\n"
                "（注意：这跟「测试连接」通过并不矛盾——那条请求短，很容易过；"
                "写整章正文是长任务，慢模型就容易顶不住。）\n"
                + "\n".join(f"{i}. {t}" for i, t in enumerate(tips, 1))
                + f"\n（原始信息：{msg}）")
    if "APIConnection" in name or "Connect" in name:
        return (f"和模型服务断开了连接。检查网络、代理，"
                f"或本地模型服务是否还在运行。\n（原始信息：{msg}）")
    if "429" in msg:
        return f"被平台限流或余额不足。稍后重试，或到平台控制台查看余额。\n（原始信息：{msg}）"
    if "401" in msg:
        return f"认证失败，API Key 可能已失效或被停用。\n（原始信息：{msg}）"
    if "没有返回正文" in msg or "只输出了思考过程" in msg:
        return (f"{msg}\n\n建议：① 关掉「深度思考」再试；② 换该平台的快速版模型；"
                "③ 若所用模型强制思考（如 glm-5.3），换 glm-5.2 即可关闭思考。")
    if isinstance(exc, llm.JSONParseError) or "不是合法 JSON" in msg \
            or "没有按要求输出 JSON" in msg or "空内容" in msg:
        if "截断" in msg or "length" in low:
            return (f"{msg}\n\n"
                    "说明：模型输出在中途被长度上限切断了，JSON 只写了一半，所以解析不了。\n"
                    "建议：1) 换表达更简洁的模型（glm-5.2 / qwen3.8-plus）；"
                    "2) 关掉「深度思考」——思考过程会占用同一份输出额度；"
                    "3) 减少同时提供的人物卡与章节信息量后重试。")
        return (f"{msg}\n\n"
                '说明：最常见的原因是模型在 JSON 字符串里直接用了英文双引号去引原文'
                '（例如 把"塞回遮阳板"改成…），未转义的引号会让 JSON 在那里提前结束。'
                '程序已自动修复并重试过一次。若仍失败，建议：\n'
                '1) 换一个更稳的模型（glm-5.2 / qwen3.8-plus / deepseek-v4-flash）；'
                '2) 把「核心需求」里的英文引号改成中文引号「」；'
                '3) 若反复出现，请把这段原始信息发给我。')
    if isinstance(exc, ValueError):
        return f"{msg}\n（出错环节：{name}）"
    return (f"{msg}\n（出错环节：{name}。可换一个模型重试。）")
