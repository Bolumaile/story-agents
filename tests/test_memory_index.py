"""记忆检索测试（优化建议第 5 条）。

重点验证两件事：
  ① 中文能被真正检索到（bigram 分词方案有效，2 字词也命中）
  ② 任何异常输入都不会让检索炸掉主流程（记忆是增强，不是必需品）
"""
import memory_index as mi


# ── 分词 ─────────────────────────────────────────────────────
def test_tokenize_splits_chinese_into_bigrams():
    assert mi.tokenize("林岸收到") == ["林岸", "岸收", "收到"]


def test_tokenize_keeps_latin_words_whole():
    assert mi.tokenize("old dam 2026") == ["old", "dam", "2026"]


def test_tokenize_handles_mixed_text():
    toks = mi.tokenize("F3 堤防的裂痕")
    assert "f3" in toks and "堤防" in toks


def test_single_chinese_char_is_kept():
    assert mi.tokenize("雨") == ["雨"]


def test_tokenize_handles_empty_and_none():
    assert mi.tokenize("") == []
    assert mi.tokenize(None) == []


def test_index_text_joins_tokens_with_spaces():
    assert mi.index_text("林岸") == "林岸"


# ── 检索 ─────────────────────────────────────────────────────
def test_fts5_is_available_in_this_environment():
    """FTS5 是标准库自带的；若某天不可用，检索会静默退化成"不检索"，
    这条用例让退化变得可见。"""
    assert mi.available() is True


def _mem():
    return {
        "chapter_summaries": [
            {"chapter": 1, "summary": "林岸收到半页残缺笔记，认出字迹是父亲的"},
            {"chapter": 2, "summary": "老崔在堤上巡了七公里"},
        ] + [{"chapter": i, "summary": f"第{i}章：镇上照相馆与鱼塘承包纠纷"}
             for i in range(3, 13)],
        "foreshadow_pool": [
            {"id": "F1", "desc": "笔记折角折法是父亲独有的习惯", "chapter": 1, "status": "open"},
            {"id": "F2", "desc": "老崔左手缺半截小指", "chapter": 2, "status": "open"},
        ],
        "character_states": [{"name": "林岸", "state": "进入戒备状态"}],
    }


def test_relevant_foreshadow_ranks_first():
    hits = mi.select_relevant(_mem(), "笔记的折角有什么讲究", limit=5)
    assert hits and hits[0]["ref_id"] == "F1"


def test_two_char_query_still_matches():
    """「鱼塘」只有两个字——trigram 分词器在这种查询上会失效，bigram 方案能命中。"""
    assert mi.select_relevant(_mem(), "鱼塘", limit=3)


def test_character_can_be_found_by_name():
    hits = mi.select_relevant(_mem(), "林岸现在什么状态", limit=5)
    assert any(h["kind"] == "character" for h in hits)


def test_limit_is_respected():
    hits = mi.select_relevant(_mem(), "笔记 老崔 鱼塘 照相馆 堤", limit=3)
    assert 0 < len(hits) <= 3


def test_hits_carry_raw_text_for_rendering():
    hits = mi.select_relevant(_mem(), "折角", limit=1)
    assert "折角" in hits[0]["raw"]


def test_unrelated_query_matches_nothing():
    """完全不相关的查询应返回空，而不是硬凑几条塞进 prompt。"""
    assert mi.select_relevant(_mem(), "量子计算机的散热方案", limit=5) == []


# ── 异常与边界 ───────────────────────────────────────────────
def test_empty_query_returns_nothing():
    assert mi.select_relevant(_mem(), "", limit=5) == []
    assert mi.select_relevant(_mem(), None, limit=5) == []


def test_empty_memory_returns_nothing():
    assert mi.select_relevant({}, "笔记", limit=5) == []


def test_fts5_special_chars_do_not_raise():
    """查询词里混入 AND / 引号 / 星号时不能把整个流程炸掉。"""
    assert isinstance(mi.select_relevant(_mem(), '笔记 AND "折角" * -x', limit=5), list)


def test_malformed_memory_entries_are_skipped():
    bad = {"chapter_summaries": [None, "字符串", {"chapter": 1, "summary": "正常内容"}],
           "foreshadow_pool": [42], "character_states": [None]}
    assert isinstance(mi.select_relevant(bad, "正常内容", limit=5), list)


def test_index_reflects_current_memory_not_a_stale_snapshot():
    """索引是即时重建的投影：memory 一变，结果立刻跟着变，不存在"索引过期"。"""
    mem = _mem()
    assert mi.select_relevant(mem, "折角", limit=3)
    del mem["foreshadow_pool"][0]
    hits = mi.select_relevant(mem, "折角", limit=3)
    assert not any(h["ref_id"] == "F1" for h in hits)


# ── 单字虚词的噪音（实测退化案例）─────────────────────────────
def test_single_char_noise_tokens_do_not_match_everything():
    """「第」「章」这类字几乎每条摘要都有，必须挡在检索之外。

    不过滤的话一次命中一整片，bm25 的相关度差异归零，
    检索会悄悄退化成"按写入顺序取前 N 条"——实测症状：
    查「第16章 x」时返回的 8 条全是第 1–8 章，而第 16 章根本不该由别人顶替。
    """
    mem = {"chapter_summaries": [{"chapter": i, "summary": f"第{i}章梗概"}
                                 for i in range(1, 13)]}
    assert mi.select_relevant(mem, "第16章 x", limit=8) == []


def test_single_char_query_is_a_known_limitation_not_a_crash():
    """单字中文查询是 bigram 方案的已知边界，这里把边界写清楚而不是假装能命中。

    索引里存的是「瘦猫」「猫蜷」这样的二元组，FTS5 是 token 精确匹配，
    所以单字「猫」查不到——要支持得改成分词器或加前缀通配，
    但那会让单字噪音回到索引里，得不偿失。用例锁住行为：返回空列表，不抛异常。
    """
    mem = {"chapter_summaries": [{"chapter": 3, "summary": "便利店的瘦猫蜷在货架下"}]}
    assert mi.select_relevant(mem, "猫", limit=3) == []
    # 换成两字词就能命中——这才是检索的真实可用形态
    assert [h["chapter"] for h in mi.select_relevant(mem, "瘦猫", limit=3)] == [3]


def test_noise_token_alone_query_falls_back_instead_of_failing():
    """整条查询都是虚词时，宁可带回噪音也不要空手而归。"""
    mem = {"chapter_summaries": [{"chapter": 1, "summary": "第1章：开工"}]}
    assert isinstance(mi.select_relevant(mem, "的", limit=3), list)
