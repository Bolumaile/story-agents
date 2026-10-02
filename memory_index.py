"""记忆检索：把「全量塞进 prompt」换成「按需检索 top-k 相关项」。

对应优化建议第 5 条。用 Python 标准库 `sqlite3` 的 FTS5，**零新依赖**，
不引入向量库（对比同类项目动辄带上 ChromaDB 几百 MB 依赖）。

两个关键设计
------------------------------------------------
1. **中文分词自己做。** FTS5 自带的 `unicode61` 分词器把连续汉字当成一个 token
   （中文没有空格），检索基本失效；`trigram` 分词器又要求查询词至少 3 个字符，
   「笔记」这类 2 字词查不到。所以这里自己做 CJK bigram：
   把「林岸收到笔记」切成 `林岸 岸收 收到 到笔 笔记`，查询侧用同一函数处理，
   2 字词也能命中。

2. **索引即时重建，不落盘。** 数据源就是 `state['memory']`，规模在几百条以内；
   每次调用重建索引是毫秒级，这样就不必维护索引文件的生命周期，也不可能出现
   「索引与权威数据不同步」——权威数据始终只有 memory 一份。
   （同类做法是持久化索引 + 增量写入，那需要处理失效与重建，对单体本地工具过重。）

3. **检索侧滤掉单字虚词。** 「第」「章」「的」这类字几乎每条摘要都有，
   不过滤的话一次命中一整片，bm25 的区分度归零，检索退化成"取最早的几章"。
   只在查询侧过滤，索引侧仍保留（否则查询「第」会彻底失效）。

已知边界：**单字中文查询命中不了**——索引里存的是「瘦猫」这样的二元组，
FTS5 做的是 token 精确匹配，单字「猫」对不上。要修得换分词器或加前缀通配，
代价是把单字噪音重新放回索引，不划算。查询词按设计是「标题 + 梗概 + 节拍」，
本来就不会只有单字。

检索的是"补充材料"，不是全部事实。**未回收伏笔 / 人物当前快照 / 物资账本属于必带项**，
不参与检索淘汰——这几类一旦漏掉，长篇立刻崩设定。
"""
import re
import sqlite3
from typing import Any, Dict, List

import config

# ── 中文分词 ─────────────────────────────────────────────────────

_CJK_RANGES = r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_TOKEN_RE = re.compile(rf"[A-Za-z0-9]+|[{_CJK_RANGES}]+")


def tokenize(text: str) -> List[str]:
    """切成检索用 token：英文/数字整词保留，中文按 bigram 滑窗。"""
    if not text:
        return []
    out: List[str] = []
    for m in _TOKEN_RE.finditer(text):
        seg = m.group(0)
        if seg.isascii():
            out.append(seg.lower())
        elif len(seg) == 1:
            out.append(seg)
        else:
            out.extend(seg[i:i + 2] for i in range(len(seg) - 1))
    return out


def index_text(text: str) -> str:
    """写入 FTS5 的文本形态：token 之间用空格隔开，交给 unicode61 建索引。"""
    return " ".join(tokenize(text))


# 单字 CJK token 的噪音：像「第」「章」「的」这样的字几乎出现在每一条摘要里，
# 一命中就是一整片，bm25 的相关度差异被抹平，检索会退化成"按写入顺序取前 N 条"。
# 实测症状（2026-10-02）：查询「第16章 …」时 8 条命中全是第 1–8 章，
# 等于把"按相关度检索"悄悄换成了"取最早的几章"。
#
# 注意只在检索侧过滤，tokenize() 本身保持纯粹——索引侧仍要保留这些字，
# 否则「第」「章」这类查询会彻底失效。
_NOISE_TOKENS = set("的了是在有和与及于为把被让使向从对就都而也很还只才便又没不之其中第章回节")


# ── 索引 ─────────────────────────────────────────────────────────
# raw 列存原文，检索时一并取出，下游不必再回查 memory
_SCHEMA = ("CREATE VIRTUAL TABLE mem USING fts5("
           "text, kind UNINDEXED, chapter UNINDEXED, ref_id UNINDEXED, raw UNINDEXED)")
_COLUMNS = "(text, kind, chapter, ref_id, raw) VALUES (?, ?, ?, ?, ?)"


def available() -> bool:
    """当前 Python 的 SQLite 是否带 FTS5（绝大多数发行版都带，仍留降级路径）。"""
    try:
        con = sqlite3.connect(":memory:")
        con.execute(_SCHEMA)
        con.close()
        return True
    except Exception:                                    # noqa: BLE001
        return False


def _iter_rows(memory: Dict[str, Any]):
    """把 memory 拆成待索引的行：(可检索文本, 类型, 章节, 引用 id, 原文)。"""
    for s in (memory.get("chapter_summaries") or []):
        if isinstance(s, dict):
            summary = s.get("summary") or ""
            yield (summary, "summary", s.get("chapter") or 0,
                   f"ch{s.get('chapter')}", summary)
    for f in (memory.get("foreshadow_pool") or []):
        if isinstance(f, dict):
            desc = f.get("desc") or ""
            # 把 id 也编进索引：查询里出现 F3 时能命中对应伏笔
            yield (f"{desc} {f.get('id') or ''}", "foreshadow", f.get("chapter") or 0,
                   f.get("id") or "", desc)
    for c in (memory.get("character_states") or []):
        if isinstance(c, dict):
            yield (f"{c.get('name') or ''} {c.get('state') or ''}", "character", 0,
                   c.get("name") or "", c.get("state") or "")


def build(memory: Dict[str, Any]):
    """从 memory 即时构建内存索引，返回 sqlite3 连接。调用方负责 close()。"""
    con = sqlite3.connect(":memory:")
    con.execute(_SCHEMA)
    rows = [(index_text(txt), kind, ch, ref, raw)
            for txt, kind, ch, ref, raw in _iter_rows(memory)]
    if rows:
        con.executemany(f"INSERT INTO mem {_COLUMNS}", rows)
    return con


def search(con, query: str, limit: int = 8) -> List[Dict[str, Any]]:
    """按相关度取前 limit 条。FTS5 的 bm25() 越小越相关，升序即按相关度排。"""
    toks: List[str] = []
    for t in tokenize(query):
        if t not in toks and t not in _NOISE_TOKENS:
            toks.append(t)
    if not toks:
        # 查询本身就是「的」「第」这类单字时，退回未过滤版本——
        # 宁可带回一点噪音，也不要一条都查不到。
        toks = list(dict.fromkeys(tokenize(query)))
    if not toks:
        return []
    # 每个 token 用双引号包起来，避免 FTS5 语法字符（- * : " 等）导致解析报错
    match_expr = " OR ".join('"' + t.replace('"', '') + '"' for t in toks[:24])
    try:
        rows = con.execute(
            "SELECT kind, chapter, ref_id, raw, bm25(mem) AS score "
            "FROM mem WHERE mem MATCH ? ORDER BY score LIMIT ?",
            (match_expr, limit)).fetchall()
    except Exception:                                    # noqa: BLE001
        return []
    return [{"kind": r[0], "chapter": r[1], "ref_id": r[2], "raw": r[3], "score": r[4]}
            for r in rows]


def select_relevant(memory: Dict[str, Any], query: str,
                    limit: int = None) -> List[Dict[str, Any]]:
    """一次性接口：建索引 → 检索 → 关连接。

    检索失败（FTS5 不可用、查询为空等）一律返回空列表，
    由调用方回退到"全量塞"——记忆是上下文增强，不该让主流程失败。
    """
    if not available():
        return []
    con = None
    try:
        con = build(memory)
        return search(con, query, limit or config.MEMORY_TOP_K)
    except Exception:                                    # noqa: BLE001
        return []
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:                            # noqa: BLE001
                pass
