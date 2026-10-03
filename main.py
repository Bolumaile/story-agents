"""入口：命令行驱动多章连载创作。

用法示例：
  # DeepSeek 官方 API（先 export DEEPSEEK_API_KEY=sk-xxx）
  python main.py --prompt "一个关于旧堤防与家族秘密的悬疑故事，5章" --chapters 3

  # 本地模型（先 export DEEPSEEK_BASE_URL=http://localhost:1234/v1）
  python main.py --prompt-file idea.txt --chapters 5 --out outputs

  # 无 Key 离线验证流水线流转（mock，不调用任何模型）
  python main.py --prompt "测试" --chapters 1 --mock

输出：
  outputs/outline.json                    策划案
  outputs/memory.json                     故事记忆库（伏笔/人物/物资/设定）
  outputs/chapter_01.md ...               各章定稿
  outputs/final.md                        全书合稿
"""
import argparse
import os
import sys

import config
import llm
import graph as graph_mod
import products
import state as state_mod
from state import StoryState


def build_initial_state(args) -> StoryState:
    """命令行参数 → 初始 State。

    真正的字段清单在 `state.new_state` 里（与网页端共用一份），这里只负责
    解析 --prompt / --prompt-file。
    """
    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            prompt = f.read().strip()
    if not prompt:
        sys.exit("错误：--prompt 与 --prompt-file 至少提供一个")
    return state_mod.new_state(prompt)


def main():
    ap = argparse.ArgumentParser(description="小说剧本多 Agent 创作流水线（策划→撰稿→校对→润色，校对可打回重写）")
    ap.add_argument("--prompt", help="用户创作需求")
    ap.add_argument("--prompt-file", help="从文件读取创作需求")
    ap.add_argument("--chapters", type=int, default=1, help="要写的章节数（默认 1）")
    ap.add_argument("--out", default="outputs", help="输出目录（默认 outputs）")
    ap.add_argument("--mock", action="store_true", help="离线 mock 模式，验证流转逻辑")
    args = ap.parse_args()

    if args.mock:
        llm.USE_MOCK = True
        print("== MOCK 模式：不调用真实模型，仅验证流水线流转 ==")
    elif not config.DEEPSEEK_API_KEY and "api.deepseek.com" in config.DEEPSEEK_BASE_URL:
        sys.exit("错误：未设置 DEEPSEEK_API_KEY。本地模型请设置 DEEPSEEK_BASE_URL，"
                 "或先用 --mock 验证流转。")

    os.makedirs(args.out, exist_ok=True)
    state = build_initial_state(args)
    app = graph_mod.build_graph()

    # 把「可降级」的告警也打到控制台。伏笔超期提醒、去 AI 味体检、校对降级
    # 这些在网页端走 SSE 推给前端；命令行下不接出来就完全看不到了。
    llm.set_notice_cb(lambda level, msg: print(f"  [{level}] {msg}"))

    # ── 逐章流水线：入口路由保证策划只在第一章前跑一次 ──
    for idx in range(1, args.chapters + 1):
        # 每章重置的字段清单与网页端共用一份（state.reset_chapter_fields），
        # 免得两边各写一遍、漏改一处就把上一章的校对意见串进下一章。
        state.update(state_mod.reset_chapter_fields(idx))
        state = graph_mod.run_chapter(app, state)

        outline = state["outline"]
        title = state["final_chapters"][-1]["title"]

        # outline / memory / chapter_XX.md / final.md 全部走同一个原子写入入口。
        # 顺带治掉「双标题」：程序拼的标题与正文自带的一级标题只留一个。
        products.write_chapter_products(args.out, state)

        rounds = state.get("revision_round", 0)
        n_comments = len(state.get("review_comments") or [])
        path = os.path.join(args.out, f"chapter_{idx:02d}.md")
        print(f"[第{idx}章] 《{title}》 定稿｜重写 {rounds} 轮｜"
              f"遗留 minor 意见 {n_comments} 条（策划案：《{outline.get('title')}》）→ {path}")

    print(f"[完成] 全书合稿 → {os.path.join(args.out, 'final.md')}")


if __name__ == "__main__":
    main()
