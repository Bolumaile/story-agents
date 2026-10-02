# 小说多 Agent 创作工坊

分工式多 Agent 流水线 + 评审回退：策划 → 撰稿 → 校对（可打回重写）→ 润色 → 记忆结算。
模型：DeepSeek API / 任意 OpenAI 兼容本地服务（LM Studio、Ollama 等）。框架：LangGraph。

## 快速开始

```bash
# 1. 装依赖（隔离 venv 已装好可跳过）
pip install langgraph openai fastapi "uvicorn[standard]"

# 2a. 网页端（推荐）
python -m uvicorn web.server:app --port 8765
# 浏览器打开 http://127.0.0.1:8765，表单里填 Key 或勾选 Mock 试跑

# 2b. 命令行
export DEEPSEEK_API_KEY=sk-xxx          # 或 DEEPSEEK_BASE_URL 指向本地服务
python main.py --prompt "一个关于旧堤防与家族秘密的悬疑故事" --chapters 3

# 无 Key 离线验证流转
python main.py --prompt "测试" --chapters 2 --mock
```

## 网页端功能

- **预制选项**：性格/说话习惯/动机/恐惧/弱点/世界观等字段带下拉预制项（datalist），可点选也可手填；禁忌红线与禁止内容为可点选 chip 追加
- **AI 润色需求**：核心需求只言片语时一键扩写成完整需求（忠实保留用户信息，缺失维度保守补全），可一键还原
- **表单减负**：人物卡可勾选"由策划 Agent 自动设计"；一键填充示例；项目可保存/加载 JSON（不含 API Key）
- **实时看板**：当前 Agent 状态、重写计数（x/3）、打回原因逐条展示
- **断点续写 + 本机记忆**：填过的东西不用再填一遍，详见下方「数据存在哪」
- **结果面板**：章节折叠展示、单章复制/下载、导出全书 Markdown、查看大纲 JSON
- **加分项**：读者反馈 Agent（追读意愿评分）、只预览大纲（不写正文）、单次生成章节数上限 5

## 数据存在哪（关掉再打开还在吗）

| 你的东西 | 存在哪 | 重开浏览器 | 重启服务 | 清了浏览器缓存 |
|---|---|---|---|---|
| API 平台 / 地址 / 模型 / Key | 浏览器 localStorage | 还在 | 还在 | 丢失（Key 要重填） |
| 创作表单（需求、人物卡、世界观、进阶项） | 浏览器 localStorage | 还在 | 还在 | 丢失 |
| 策划案 + 已定稿章节 + 记忆库 | 服务端 `outputs_web/_session.json` | 还在 | 还在 | 还在 |

- 打开页面会自动恢复：顶部出现「已恢复上次创作进度」提示条，直接点「续写下一章」接着写
  （复用策划案与记忆库，不会重跑策划 Agent）。
- 想写全新一篇 → 点「新建故事」：只清会话，`outputs_web/run_*/` 里的 md / json 不会被删除。
- 草稿落盘是防抖的（输入停止 0.7 秒后写一次），关页面前也会补存一次。
- API Key 默认记住在本机；公用电脑请把 Key 下方的「记住在本机」取消勾选，Key 就不落盘了
  （平台与模型名仍会记住）。

## 架构

```
START ─(无策划案)→ planner → writer → reviewer ─(fail 且未超轮数)→ bump_round → writer
      └(已有策划案)→ writer                └─(pass 或超轮数)→ polisher → memory_settler → END
```

| Agent | 职责 | 输出 |
|---|---|---|
| 策划 Planner | 需求 → 梗概/人物卡/分章大纲/转折点 | 严格 JSON |
| 撰稿 Writer | 按大纲+人设写章节正文，收到反馈定向重写 | 正文 |
| 校对 Reviewer | OOC / 时间线 / 设定 / 逻辑四维校验 | JSON 清单（critical→打回，minor→转润色） |
| 润色 Polisher | 只改措辞节奏，不碰剧情与动机 | 正文 |
| 记忆结算 MemorySettler | 章节摘要 + 伏笔池 + 人物状态快照 | JSON 增量 |

- **全局记忆**：策划案 + 故事记忆（State 持久跨章）；写新章时只带「前情摘要 + 伏笔池 + 人物快照 + 上一章全文」，长篇上下文不膨胀。
- **回退保护**：`MAX_REVISION_ROUNDS`（默认 3）防无限循环，超轮强制放行并保留意见。
- **禁止内容**：表单填的禁令同时注入撰稿（规避）、校对（核查）、润色三处。

## 目录

```
story-agents/
├── config.py          # API 配置、重写轮数上限
├── state.py           # StoryState 全局共享状态
├── prompts.py         # 5 个角色的 System Prompt
├── llm.py             # LLM 封装 + MockLLM 离线测试
├── nodes.py           # 节点实现（含 meta 材料注入）
├── graph.py           # LangGraph 调度器 + 回退路由
├── main.py            # CLI 入口
├── web/
│   ├── server.py      # FastAPI + SSE 进度推送 + 会话落盘/恢复
│   └── static/index.html  # 表单页（人物卡动态增删、本机记忆）
└── outputs_web/       # 网页端产物（每 run 一个目录）
    └── _session.json  # 上次会话存档（策划案+章节+记忆），「新建故事」时清掉
```

## 参考项目（已核实）

- **Narcooo/inkos**：全局真相文件 + 伏笔状态机，本工程的 memory_settler 即其简化版
- **voocel/ainovel-cli**：卷弧滚动规划（长篇远期大纲不空洞），可作下一步演进方向
- **wanqili857-byte/fictionforge**：质量门禁（禁词/AI 腔）、修订回流
- **YILING0013/AI_NovelGenerator**：向量检索 + 逐步生成 GUI

## 已知边界（下一步可做）

1. 章节数超出策划大纲时按核心冲突自然续写；后续可加「滚动展开下一卷」（参考 ainovel-cli）
2. 人物是提示词级约束（人设卡+禁忌+快照），非独立 agent；角色级持久记忆可参考 fictionforge
3. config 为进程级单例，网页端同一时间只支持一个生成任务（已加锁）
