# 小说多 Agent 创作工坊

分工式多 Agent 流水线 + 评审回退：策划 → 撰稿 → 校对（可打回重写）→ 润色 → 记忆结算。
模型：DeepSeek API / 任意 OpenAI 兼容本地服务（LM Studio、Ollama 等）。框架：LangGraph。

![网页端表单页](docs/images/ui-form.png)

## 快速开始

```bash
# 1. 装依赖（隔离 venv 已装好可跳过）
pip install -r requirements.txt
# 想完全复现已验证过的版本：pip install -r requirements.lock.txt

# 2. 配置（可选）——网页端也能填，完全不配也能用 Mock 试跑
cp .env.example .env        # 填 DEEPSEEK_API_KEY；本地模型则改 DEEPSEEK_BASE_URL

# 3a. 网页端（推荐）
python -m uvicorn web.server:app --port 8765
# 浏览器打开 http://127.0.0.1:8765，表单里填 Key 或勾选 Mock 试跑

# 3b. 命令行
python main.py --prompt "一个关于旧堤防与家族秘密的悬疑故事" --chapters 3

# 无 Key 离线验证流转
python main.py --prompt "测试" --chapters 2 --mock
```

> 环境变量优先级：**真实环境变量 > `.env` > `config.py` 默认值**。
> 全部可用变量与取舍说明见 `.env.example`。
> 注意：网页端下拉里选好平台与模型后，`DEEPSEEK_MODEL` 就不再生效。

## 开发与测试

```bash
python -m pytest      # 54 项，全部离线，约 1.6 秒
```

测试不联网、不烧 token：`MockLLM` 能跑通整条流水线，JSON 修复链是纯字符串处理。
覆盖四块：JSON 修复链（逐个对应踩过的坑）、图路由与回退、节点兜底行为、
三条降级路径（校对 fail-open / 无 critical 转 pass / 记忆结算失败不中断）。
**改 `llm.py`、`nodes.py` 或 `graph.py` 之前先跑一遍。**

## 网页端功能

- **预制选项**：性格/说话习惯/动机/恐惧/弱点/世界观等字段带下拉预制项（datalist），可点选也可手填；禁忌红线与禁止内容为可点选 chip 追加
- **AI 润色需求**：核心需求只言片语时一键扩写成完整需求（忠实保留用户信息，缺失维度保守补全），可一键还原
- **表单减负**：人物卡可勾选"由策划 Agent 自动设计"；一键填充示例；项目可保存/加载 JSON（不含 API Key）
- **实时看板**：当前 Agent 状态、重写计数（x/3）、打回原因逐条展示
- **断点续写 + 本机记忆**：填过的东西不用再填一遍，详见下方「数据存在哪」
- **结果面板**：章节折叠展示、单章复制/下载、导出全书 Markdown、查看大纲 JSON
- **加分项**：读者反馈 Agent（追读意愿评分）、只预览大纲（不写正文）、单次生成章节数上限 5

![记忆库与创作结果](docs/images/ui-result.png)

<sub>上面是跑完 2 章后的下半页：全局记忆库（可直接编辑，续写时生效）+ 创作结果（按章折叠、可单独复制/下载）。整页长图见 `docs/images/ui-full.png`。</sub>

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

- **全局记忆**：策划案 + 故事记忆（State 持久跨章）；写新章时不带全书正文，只带「前情摘要 + 伏笔池 + 人物快照 + 上一章全文」。
  注意：这几块目前是**全量累积**后塞进 prompt，写到几十章仍会线性膨胀——这正是「已知边界」第 2 条。
- **回退保护**：`MAX_REVISION_ROUNDS`（默认 3）防无限循环，超轮强制放行并保留意见。
- **禁止内容**：表单填的禁令同时注入撰稿（规避）、校对（核查）、润色三处。

### 数据流与持久化

```
用户表单 ──build_user_prompt / build_meta──▶ StoryState ──▶ 各节点读改写 ──▶ 定稿章节
                                                │
             ┌──────────────────────────────────┴───────────────────────────────┐
             ▼                                                                  ▼
   浏览器 localStorage                                          服务端 outputs_web/
   · storyagents.llm.v1   平台 / 地址 / 模型 / Key              · _session.json  策划案+章节+记忆
   · storyagents.form.v1  表单草稿（不含 Key）                  · run_*/         章节 md + final.md
```

**状态只有一份**（StoryState 由 LangGraph 在节点间传递），不存在"两套状态互相打架"。

进度与告警是**单向通道**，只推给前端、不写状态：

| 通道 | 触发时机 | 前端表现 |
|---|---|---|
| `llm.set_progress_cb` | 每收到一段流式输出 | 看板实时显示「已生成 N 字」 |
| `llm.emit_notice` | 校对被跳过 / 记忆结算失败等降级 | 黄色告警条 |

服务端存档落在四个时机：**planner 完成、每章定稿、全部完成、异常分支**。
写 `.tmp` 后 `os.replace` 原子替换——断电也不会留下半个文件。

## 目录

```
story-agents/
├── config.py          # API 配置、重写轮数上限、.env 读取
├── state.py           # StoryState 全局共享状态
├── prompts.py         # 各角色的 System Prompt
├── llm.py             # LLM 封装（流式/超时/JSON 修复链）+ MockLLM 离线测试
├── nodes.py           # 节点实现（含 meta 材料注入）
├── graph.py           # LangGraph 调度器 + 回退路由
├── main.py            # CLI 入口
├── web/
│   ├── server.py      # FastAPI + SSE 进度推送 + 会话落盘/恢复
│   └── static/index.html  # 表单页（人物卡动态增删、本机记忆）
├── tests/             # pytest 最小测试集（离线，54 项）
├── docs/
│   ├── 优化建议-对比同类项目.md   # 横向调研：能力矩阵 + P0–P3 清单
│   └── images/        # README 截图
├── _backup/           # 历史版本备份（不进 git），来源见其 README.txt
├── .env.example       # 环境变量样例与取舍说明
├── requirements.txt   # 依赖清单（顶层）
├── requirements.lock.txt  # pip freeze 精确版本
├── pyproject.toml     # 仅放 pytest 配置
└── outputs_web/       # 网页端产物（不进 git）
    ├── _session.json  # 上次会话存档（策划案+章节+记忆），「新建故事」时清掉
    └── run_*/         # 每次运行的章节 md 与合稿
```

## 参考项目

**本轮已核实**（2026-10-02，含 README 与功能矩阵比对）：

- **Narcooo/inkos**：记忆分「权威 JSON + 可重建的检索投影」两层，伏笔状态机带 schema 校验。
  本工程的 memory_settler 是其简化版；检索投影（SQLite FTS5）是值得借鉴的下一步。
- **HuangLeijiana/novel-agent**：12 Agent；阶段级人类确认（`interrupt()`）；按 Agent 分级选模型。
- **14790897/Novel-Factory-Multi-Agent**：场景节拍（Scene Beats）把粗纲扩成场景；联网检索文风并提炼 brief。
- **bodinggg/LangGraph-based-Novel-by-Agents**：Supervisor 编排 4 个 specialist 并行检查；检查点断点恢复。
- **MaoXiaoYuZ/Long-Novel-GPT**：大纲→章节→正文三段扩写控篇幅；实时显示调用费用。
- **YILING0013/AI_NovelGenerator**：语义检索注入历史细节 + 一致性检查器。

**尚未复核**（引用自其他项目 README 或社区横评，未亲自验证）：

- `voocel/ainovel-cli` —— 卷弧滚动规划（长篇远期大纲不空洞）
- `wanqili857-byte/fictionforge` —— 质量门禁（禁词 / AI 腔）、修订回流

## 已知边界（下一步可做）

完整调研、能力矩阵与按优先级排序的优化清单见
**[`docs/优化建议-对比同类项目.md`](docs/优化建议-对比同类项目.md)**（25 条建议，每条含落点文件）。
这里只列最关键的三条：

1. **章内缺「场景节拍」层**：策划目前只到「章」级，撰稿一次写整章，长章容易前松后紧
2. **记忆是全量塞进 prompt，不是检索**：写到几十章会膨胀且稀释注意力（建议 SQLite FTS5，标准库零依赖）
3. **工程化刚起步**：已有测试与 git 基线；还缺 Docker、按 Agent 分级模型、token/费用统计、
   中途人工干预、书库（当前同一时间只能写一本）

另有两个功能层面的已知边界：

4. 人物是提示词级约束（人设卡+禁忌+快照），非独立 agent
5. `config` 为进程级单例，网页端同一时间只支持一个生成任务（已加锁）

## 许可与致谢

**本项目代码采用 MIT License**，版权 © 2026 Bolumaile。你可以自由使用、修改、
再分发、闭源商用，只需在副本中保留版权声明与许可原文。详见 [`LICENSE`](LICENSE)。

### 第三方资源

- **字体**：`web/static/fonts/` 下的三款字体（思源黑体 / 思源宋体 / 拉丁 Noto Serif，
  共 210 个 woff2 分片）采用 **SIL Open Font License 1.1**，**不适用**上述 MIT 许可。
  版权分属 Adobe 与 The Noto Project Authors；完整许可原文、各字体版权行与版本号见
  [`web/static/fonts/LICENSE-OFL.txt`](web/static/fonts/LICENSE-OFL.txt)。

  OFL 允许商用与再分发，但要求随字体保留版权声明与许可文本。该文件**必须随
  `fonts/` 目录一同分发**，请勿删除或单独摘出字体文件使用。

- **运行依赖**：LangGraph、FastAPI、OpenAI SDK 等，均为宽松许可
  （MIT / Apache-2.0 / BSD），精确版本见 `requirements.lock.txt`。

> ⚠️ `web/static/fonts/yinpin-hongmengti.ttf`（印品鸿蒙体）为**非商用授权**字体，
> 已列入 `.gitignore`、不随仓库分发，代码与 CSS 中也不再引用。请勿将其用于商业用途，
> 发布前建议从本地直接删除。
