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
  outputs/chapter_01.md ...               各章定稿
  outputs/final.md                        全书合稿
"""
import argparse
import json
import os
import sys

import config
import llm
import graph as graph_mod
from state import StoryState


def build_initial_state(args) -> StoryState:
    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            prompt = f.read().strip()
    if not prompt:
        sys.exit("错误：--prompt 与 --prompt-file 至少提供一个")
    return StoryState(
        user_prompt=prompt,
        outline={},
        chapter_index=1,
        chapter_draft="",
        review_comments=[],
        review_verdict="pass",
        # 三路并行校对：每路各写各的 key
        review_comments_ooc=[],
        review_comments_logic=[],
        review_comments_pacing=[],
        style_report={},
        revision_round=0,
        final_chapter="",
        meta={},
        memory={},
        final_chapters=[],
    )


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
        state.update({
            "chapter_index": idx,
            "chapter_draft": "",
            "review_comments": [],
            "review_verdict": "pass",
            # 三路 specialist 各写各的 key，必须随章清空，
            # 否则上一章的意见会漏进下一章的合并结果里
            "review_comments_ooc": [],
            "review_comments_logic": [],
            "review_comments_pacing": [],
            "style_report": {},
            "revision_round": 0,
            "final_chapter": "",
        })
        state = graph_mod.run_chapter(app, state)

        outline = state["outline"]
        with open(os.path.join(args.out, "outline.json"), "w", encoding="utf-8") as f:
            json.dump(outline, f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.out, "memory.json"), "w", encoding="utf-8") as f:
            json.dump(state.get("memory") or {}, f, ensure_ascii=False, indent=2)

        rounds = state.get("revision_round", 0)
        n_comments = len(state.get("review_comments") or [])
        title = state["final_chapters"][-1]["title"]
        path = os.path.join(args.out, f"chapter_{idx:02d}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n{state['final_chapter']}\n")
        print(f"[第{idx}章] 《{title}》 定稿｜重写 {rounds} 轮｜"
              f"遗留 minor 意见 {n_comments} 条 → {path}")

    # ── 合稿 ──
    with open(os.path.join(args.out, "final.md"), "w", encoding="utf-8") as f:
        f.write(f"# {state['outline'].get('title')}\n\n")
        for ch in state["final_chapters"]:
            f.write(f"## 第{ch['index']}章 {ch['title']}\n\n{ch['text']}\n\n")
    print(f"[完成] 全书合稿 → {os.path.join(args.out, 'final.md')}")


if __name__ == "__main__":
    main()
