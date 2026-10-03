"""产物写入：唯一负责把 State 落成磁盘文件的地方（CLI 与网页端共用）。

为什么单独成一个模块（对应审查里的 P0-1 / P0-4 / P1-5，三件事一处解决）：

1. **双标题**：程序会自己拼一个一级标题，而 `final_chapter`（润色输出）正文
   本来**也带一个**，于是每个产物文件开头都是「# 标题」+「# 第N章 标题」，
   甚至两行一模一样。归一逻辑只写一份，就不会有哪个写盘点漏掉。
2. **原子写**：以前全部是就地覆盖写，写一半崩 / 断电 / 磁盘满会留下半截 md、
   非法 JSON，且下次运行再覆盖一次、**无法自愈**。讽刺的是同项目的
   `web.server._save_session` 早就是 tmp + os.replace，关键产物反而没保护。
3. **去重复代码**：写产物原先在 CLI（main.py）与网页端（web/server.py）各写
   一遍，连"写哪几个文件、每章写还是最后写"都不一致（网页端只在全部章节都
   成功后才写 outline/memory，中途失败就什么都不剩）。

调用方只需 `write_chapter_products(out_dir, state)`，每章定稿后调一次。
"""
import json
import os
import re
from typing import Any, Dict, List, Optional


# 行首的一级标题（只认 `# `，`## ` 不会被误删）
_H1_LINE = re.compile(r"^[ \t]*#[ \t]+[^\n]*")
# ATX 标题：最多 3 个前导空格 + 1~6 个 #，且 # 后面必须是空白或行尾
_ATX_HEADING = re.compile(r"^([ \t]{0,3})(#{1,6})(?=[ \t]|$)", re.MULTILINE)


def strip_leading_h1(text: str) -> str:
    """剥掉正文**开头**的一级标题，并把前后的空白清掉。

    只在首行确实是 `# xxx` 时才动手：模型偶尔会把章节标题写进正文首行，
    那就是产物里第二个标题的来源。
    """
    s = text or ""
    m = _H1_LINE.match(s)
    if not m:
        return s.lstrip()
    return s[m.end():].lstrip()


def demote_headings(text: str, levels: int = 1) -> str:
    """把正文里的 ATX 标题整体降级，避免与程序拼的标题抢层级。

    如正文里出现 `# 尾声`，在单章文件里会被降成 `## 尾声`（挂在章标题下面），
    在合稿里降成 `### 尾声`（挂在 `## 第N章` 下面）。
    """
    if levels <= 0 or not text:
        return text or ""

    def _bump(m: "re.Match") -> str:
        n = min(6, len(m.group(2)) + levels)
        return m.group(1) + "#" * n

    return _ATX_HEADING.sub(_bump, text)


def compose_chapter_md(title: str, text: str) -> str:
    """单章定稿：程序拼唯一的一个一级标题，正文自带的一级标题先剥掉。"""
    body = demote_headings(strip_leading_h1(text), 1)
    head = f"# {title}".rstrip()
    return f"{head}\n\n{body}\n" if body else f"{head}\n"


def compose_final_md(book_title: str, chapters: List[Dict[str, Any]]) -> str:
    """全书合稿：书名做一级标题、每章做二级标题，正文标题再降两级。

    合稿里各章标题用 `##`，所以正文里残留的 `#` 必须降到 `###` 以下，
    否则层级打架——把 final.md 导进别的排版工具时结构会错乱。
    """
    out: List[str] = [f"# {book_title or '未命名'}"]
    for ch in chapters or []:
        body = demote_headings(strip_leading_h1(str(ch.get("text") or "")), 2)
        out.append("")
        out.append(f"## 第{ch.get('index')}章 {ch.get('title') or ''}".rstrip())
        if body:
            out.append("")
            out.append(body)
    return "\n".join(out) + "\n"


# ── 原子写 ────────────────────────────────────────────────────────

def atomic_write_text(path: str, text: str) -> None:
    """原子写文本：先写同目录临时文件，再 `os.replace` 覆盖目标。

    `os.replace` 在同一分区内是原子操作，所以任何时刻读到的都是
    "上一个完整版本"或"这一个完整版本"，不存在半截文件。
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            try:
                os.fsync(f.fileno())      # 断电也不留半截；个别文件系统不支持，忽略
            except OSError:
                pass
        os.replace(tmp, path)
    except BaseException:
        # 失败时别把 .tmp 留在产物目录里当垃圾
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(path: str, obj: Any) -> None:
    """原子写 JSON。产物是给人看、也给程序读的，所以带缩进且不转义中文。"""
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


# ── 对外主入口 ────────────────────────────────────────────────────

def _chapter_no(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def write_chapter_products(out_dir: str, state: Dict[str, Any]) -> None:
    """把当前 State 的全部产物刷一遍：outline / memory / 最新一章 / 合稿。

    每章定稿后调用一次即可，重复调用是幂等的。

    为什么 outline.json 与 memory.json 也要**每章**写：网页端原先在逐章循环
    **之外**才写这两个文件，中途任何一章抛异常，产物目录里就只剩
    `chapter_XX.md`，没有大纲也没有记忆库，用户按目录找文件会以为内容丢了。
    CLI 一直是每章写——这次把两边的行为统一到这里。
    """
    outline: Dict[str, Any] = state.get("outline") or {}
    atomic_write_json(os.path.join(out_dir, "outline.json"), outline)
    atomic_write_json(os.path.join(out_dir, "memory.json"), state.get("memory") or {})

    chapters: List[Dict[str, Any]] = state.get("final_chapters") or []
    if chapters:
        last = chapters[-1]
        idx = _chapter_no(last.get("index"), len(chapters))
        atomic_write_text(
            os.path.join(out_dir, f"chapter_{idx:02d}.md"),
            compose_chapter_md(last.get("title") or f"第{idx}章",
                               str(last.get("text") or "")))

    atomic_write_text(
        os.path.join(out_dir, "final.md"),
        compose_final_md(str(outline.get("title") or ""), chapters))


def products_dir_name(out_dir: Optional[str]) -> Optional[str]:
    """产物目录的最后一级名字（测试与日志用，避免打印绝对路径）。"""
    if not out_dir:
        return None
    return os.path.basename(os.path.abspath(out_dir))
