# 小说多 Agent 创作工坊

分工式多 Agent 流水线 + 评审回退：**策划（含场景节拍）→ 撰稿 → 三路并行校对（可打回重写）→ 润色（含去 AI 腔）→ 记忆结算**。
模型：DeepSeek API / 任意 OpenAI 兼容本地服务（LM Studio、Ollama 等）。框架：LangGraph。

## 核心特性

| 特性 | 一句话说明 |
|---|---|
| **场景节拍** | 策划不只给章节梗概，还拆出 3–5 个节拍（目标/冲突/转折/字数/情绪温度），撰稿按拍推进，长章不再前松后紧 |
| **三路并行校对** | 人物一致性、逻辑时间线设定、节奏篇幅各由一个 specialist 独立检查，额度专款专用，不再互相挤占 |
| **记忆按需检索** | 写第 N 章时不再带全书正文：**必带项**（未回收伏笔 + 人物快照 + 物资账本）按优先级截断、带上界，**检索项**用本章标题 + 梗概 + 节拍做 SQLite FTS5 检索取 top-k |
| **物资账本** | 策划给出开局清单，逐章追踪消耗。撰稿只能用账本内的东西，已耗尽/已丢失的明确禁止再出现——治「吃掉的压缩饼干下一章又摸出来」这类长篇头号硬伤 |
| **伏笔状态机** | 伏笔有 open / progressing / deferred / resolved 四态，记录最后推进章节；连续多章没动静会自动提醒，防烂尾 |
| **去 AI 味** | 润色环节带去 AI 腔指令；另有纯代码的规则层体检（套话密度、句长节奏、比喻密度、收束句），只提示不擅自改稿 |
| **强类型约束** | 所有模型输出过 Pydantic，缺字段落默认值、中文枚举自动归一，脏数据不会打断流水线也不往下游传播 |

**稳定性工程**（同类项目的高频短板，这里都兜住了）：流式空闲超时、JSON 自动修复链 +
纠错重试、校对降级放行、Mock 全离线验证。

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
python -m pytest      # 156 项，全部离线，约 1.8 秒
```

测试不联网、不烧 token：`MockLLM` 能跑通整条流水线，JSON 修复链与规则层检测都是纯字符串处理。

| 测试文件 | 覆盖内容 |
|---|---|
| `test_json_repair.py` | JSON 修复链（逐个对应踩过的坑）、流式空闲超时 |
| `test_routing.py` | 图路由与回退、三路并行/汇合的图结构、节拍透传、分层记忆组装与必带项截断、物资账本渲染、章级计划兜底 |
| `test_pipeline_mock.py` | 端到端冒烟、三路校对各自 fail-open、伏笔状态机与超期告警、物资账本三态（改存量/标耗尽/新增）、角色名归一不裂条、风格体检落盘 |
| `test_models.py` | Pydantic 归一：缺字段、类型强转、中文枚举收敛、物资状态与角色名归一、告警去重 |
| `test_memory_index.py` | 中文 bigram 分词、FTS5 检索命中、单字虚词过滤、无索引时降级 |
| `test_style_check.py` | 套话/句长/比喻/收束句各自判据与样本量门槛 |

降级路径均已单独构造用例：校对单路超时只跳过该路、三路都无 critical 转 pass、
记忆结算失败不中断、检索不可用时退回「必带项 + 最近 k 章」。

**改 `llm.py`、`nodes.py`、`graph.py` 或 `models.py` 之前先跑一遍。**

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
START ─(无策划案)→ planner ─┐
      └(已有策划案)─────────┴→ writer ─┬→ reviewer_ooc   ─┐
                                      ├→ reviewer_logic ─┼→ merge_reviews
                                      └→ reviewer_pacing ─┘        │
                                                                   ├─(fail 且未超轮)→ bump_round → writer
                                                                   └─(pass 或超轮)──→ polisher → memory_settler → END
```

> 三路校对并行写**各自独立的 state key**（`review_comments_ooc` / `_logic` / `_pacing`），
> 避免并行覆盖，最后由 `merge_reviews` 汇合判 verdict。

| Agent | 职责 | 输出 |
|---|---|---|
| 策划 Planner | 需求 → 梗概/人物卡/分章大纲/转折点 + **每章场景节拍** | 严格 JSON |
| 撰稿 Writer | 按大纲与节拍写正文，收到反馈定向重写 | 正文 |
| 校对 `reviewer_ooc` | 只看人物一致性（性格/动机/禁忌） | JSON 清单 |
| 校对 `reviewer_logic` | 只看时间线、设定一致性、剧情逻辑 | JSON 清单 |
| 校对 `reviewer_pacing` | 只看节奏、注水、篇幅、节拍落实 | JSON 清单 |
| 汇合 `merge_reviews` | 合并三路意见（critical 排前），统一判定 verdict | 写入 `review_comments` |
| 润色 Polisher | 只改措辞节奏，不碰剧情与动机；附带**去 AI 腔** | 正文 |
| 记忆结算 MemorySettler | 章节摘要 + 伏笔状态 + 人物状态快照 + **物资账本变动** | JSON 增量（过 schema 校验） |

- **全局记忆**：策划案 + 故事记忆（State 持久跨章）。写新章时**不带全书正文**，只带：
  - **必带项**（漏掉会直接崩设定，不参与检索淘汰）：
    - **未回收伏笔**——按「有 deadline 的优先 → 最久没推进的优先」排序，
      超过 `MEMORY_MUST_HAVE_MAX`（默认 12）后截断；被截掉的**至少留下 id**，
      这样本章若涉及它们，记忆结算仍能推进 / 回收
    - **人物当前快照**
    - **物资账本**——可用的逐条列（带存量），已耗尽 / 已丢失的压成一行只报名字，
      起硬约束作用：「这些不能再用」
  - **检索项**：用本章标题 + 梗概 + 节拍当查询词，从历史摘要中取 top-k（`MEMORY_TOP_K`，默认 8）

  > **上界说明（实测修正后已收敛）**：检索项一直是硬上界（top-k）；
  > 必带项原先**没有**上限（实测 3 章攒 18 条伏笔，写第 4 章时必带项约 800 字、
  > 整段记忆上下文 1737 字），现已加截断。相对"全书正文进 prompt"是数量级改善，
  > 但要说准：必带项是**有上界的线性项**，不是常数项。

  检索走 **SQLite FTS5**（Python 标准库自带，零新依赖），中文用 bigram 分词规避
  FTS5 对中文不友好的问题；查询侧会滤掉「第」「章」这类单字虚词，
  否则一次命中一整片、bm25 区分度归零（实测该 bug 会让检索退化成"取最早的几章"）。
  已知边界：单字中文查询命中不了，详见 `memory_index.py` 的文档。
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
| `llm.emit_notice` | 见下 | 黄色告警条（`warn`）/ 蓝色提示条（`info`） |

`notice` 的触发点共 6 处，每处都对应一条「不中断流水线」的降级或提醒：

| 级别 | 时机 |
|---|---|
| `warn` | 某路校对结果无法解析 → 该维度跳过，章节继续 |
| `info` | 校对判定 fail → 列明各维度 critical 条数并打回 |
| `info` | 风格体检有提示 → 附套话/句长/比喻/收束句的各项读数 |
| `warn` | 记忆结算失败 → 本章照常定稿，后续章节缺这部分前情 |
| `warn` | 伏笔连续 ≥ `FORESHADOW_STALE_CHAPTERS` 章未推进 → 提醒防烂尾 |
| `warn` | 模型输出撞长度上限被截断 |

模型输出里的脏数据（枚举写了中文、字段缺失、类型不对）由 `models.py` 归一并经
`emit_notice` 汇总上报，**去重后**只提醒一次，不会 50 条一起刷屏。

服务端存档落在四个时机：**planner 完成、每章定稿、全部完成、异常分支**。
写 `.tmp` 后 `os.replace` 原子替换——断电也不会留下半个文件。

## 目录

```
story-agents/
├── config.py          # API 配置、重写轮数、记忆/伏笔/去 AI 味各开关、.env 读取
├── state.py           # StoryState 全局共享状态
├── models.py          # Pydantic 模型层：策划案/校对意见/伏笔/记忆的 schema 与归一
├── memory_index.py    # 记忆检索（SQLite FTS5 + 中文 bigram 分词，零新依赖）
├── style_check.py     # 去 AI 味的规则层检测（套话/句长/比喻/收束句密度）
├── prompts.py         # 各角色的 System Prompt
├── llm.py             # LLM 封装（流式/超时/JSON 修复链）+ MockLLM 离线测试
├── nodes.py           # 节点实现（含 meta 材料注入、三路校对、汇合判定）
├── graph.py           # LangGraph 调度器 + 三路并行 + 回退路由
├── main.py            # CLI 入口
├── web/
│   ├── server.py      # FastAPI + SSE 进度推送 + 会话落盘/恢复
│   └── static/index.html  # 表单页（人物卡动态增删、本机记忆）
├── tests/             # pytest 测试集（离线，156 项，约 1.8 秒）
├── docs/
│   ├── 优化建议-对比同类项目.md   # 横向调研：能力矩阵 + P0–P3 清单
│   └── images/        # README 截图
├── _backup/           # 历史版本备份（不进 git），来源见其 README.txt
├── .env.example       # 环境变量样例与取舍说明
├── requirements.txt   # 依赖清单（顶层）
├── requirements.lock.txt  # pip freeze 精确版本
├── pyproject.toml     # 项目元信息（名称/许可/作者）+ pytest 配置
└── outputs_web/       # 网页端产物（不进 git）
    ├── _session.json  # 上次会话存档（策划案+章节+记忆），「新建故事」时清掉
    └── run_*/         # 每次运行的章节 md 与合稿
```

## 参考项目

**本轮已核实**（2026-10-02，含 README 与功能矩阵比对）；方括号内为本工程的采纳情况：

- **Narcooo/inkos**：记忆分「权威 JSON + 可重建的检索投影」两层，伏笔状态机带 schema 校验。
  memory_settler 是其简化版；检索投影〔已落地：`memory_index.py`，含中文 bigram 分词〕。
  未跟进的一点——inkos 对脏数据是**拒绝**，本工程改**归一 + 告警**（`models.py`）。
- **HuangLeijiana/novel-agent**：12 Agent；阶段级人类确认（`interrupt()`）；按 Agent 分级选模型。
  〔均未落地，见「已知边界」〕
- **14790897/Novel-Factory-Multi-Agent**：场景节拍（Scene Beats）把粗纲扩成场景；联网检索文风并提炼 brief。
  〔场景节拍已落地；联网检索文风未做〕
- **bodinggg/LangGraph-based-Novel-by-Agents**：Supervisor 编排 4 个 specialist 并行检查；检查点断点恢复。
  〔多路并行已落地：三路 specialist + merge 汇合；检查点恢复未做〕
- **MaoXiaoYuZ/Long-Novel-GPT**：大纲→章节→正文三段扩写控篇幅；实时显示调用费用。
  〔费用统计未做〕
- **YILING0013/AI_NovelGenerator**：语义检索注入历史细节 + 一致性检查器。
  〔已落地：FTS5 全文检索版（非向量）+ 三路一致性校对〕

**尚未复核**（引用自其他项目 README 或社区横评，未亲自验证）：

- `voocel/ainovel-cli` —— 卷弧滚动规划（长篇远期大纲不空洞）
- `wanqili857-byte/fictionforge` —— 质量门禁（禁词 / AI 腔）、修订回流

## 已知边界（下一步可做）

完整调研、能力矩阵与按优先级排序的优化清单见
**[`docs/优化建议-对比同类项目.md`](docs/优化建议-对比同类项目.md)**（25 条建议，每条含落点文件）。

**已完成**（P0 + P1，2026-10-02）：

- ✅ 章内「场景节拍」层——策划拆到 3–5 拍，撰稿按拍推进（原建议第 4 条）
- ✅ 记忆检索化——SQLite FTS5 + 中文 bigram 分词，必带项与检索项分层（第 5 条）
- ✅ 伏笔状态机——四态 + 最后推进章节 + 超期巡检（第 6 条）
- ✅ 校对拆分——三路 specialist 并行 + 汇合判定（第 7 条）
- ✅ 去 AI 腔——润色指令 + 规则层体检（第 8 条）
- ✅ Pydantic 强类型约束——`models.py` 统一归一（第 9 条）

**首次真模型三章实跑暴露的三个问题**（2026-10-02，产物见 `outputs_web/run_20261002_204203/`）
——**均已修复**。留在这里是为了说明这些坑真实存在过、以及修的是什么：

1. **没有物资/道具账本** → 已补（`models.Item` / `MemoryDelta.item_changes`）。
   原症状：3 章里 2 章因物资矛盾被打回，烧掉 3 轮重写；更糟的是第 1 章的背包清单
   自身就与后文矛盾（列了 4 件却没列罐头，紧接着"手摸到罐头"），而第 1 章 0 轮重写通过，
   内伤被当成正典固化。现在：策划先给 `initial_items` 立账 → 撰稿 prompt 必带账本
   （已耗尽 / 已丢失单列一行、明令禁止再用）→ 逻辑校对专项核对物资 →
   结算合并 `item_changes`。其中「只改存量的条目不得把状态重置回 available」
   有专门用例守着——那是账本最容易失守、也最要命的一处。
2. **必带项无上限** → 已加上界（`MEMORY_MUST_HAVE_MAX`，默认 12）。
   截断按「deadline 近 > 最久没推进」排序，被省略的**至少留下 id**。
   同时收紧 `MEMORY_SYSTEM`：伏笔必须"后文有账可算"，
   明确排除「猫蹭手背」「雨停」这类不会有下文的动作与氛围，且每章新增 ≤ 2 条。
3. **同一角色在人物快照里裂成多条** → 已归一（`models.canon_name`）。
   精确匹配失败后按「去括号备注 → 去空白标点 → 转小写」再匹配，
   命中则保留账本里的原名（避免名字漂移）；结算请求里附上当前人物名单，
   要求 `character_updates` 的 name 从名单中取——从源头少犯，而不是只靠事后收拾。

**顺带修掉的检索退化**：`memory_index` 原先会把「第」「章」这类单字虚词当关键词提交，
可它们几乎出现在每条摘要里，一命中就是一整片，bm25 区分度归零。
实测症状：查「第16章 …」时 8 条命中全是第 1–8 章——等于把"按相关度检索"
悄悄换成了"取最早的几章"。现已过滤；检索未命中时退回「必带项 + 最近 k 章」。

**尚未做**：

1. **工程化**：缺 Docker / 容器化、按 Agent 分级选模型、token 与费用统计
2. **交互**：无中途人工干预（人工确认/改稿后再续），当前是「一键跑完」
3. **多书并行**：`config` 为进程级单例，网页端同一时间只能写一本（已加锁）
4. **人物仍是提示词级约束**（人设卡 + 禁忌 + 快照），非独立 agent
5. **人物禁忌压不住**：三章连续被三路校对报同一件事（内心独白过多 vs「惜字如金」人设），
   第3章已升级为 critical。撰稿提示词对角色禁忌的强调力度需要加强

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
