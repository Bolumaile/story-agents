"""pytest 公共配置：把项目根目录加入 sys.path。

必须在任何测试 import 业务模块之前生效——本项目是平铺结构
（config.py / llm.py / nodes.py 都在根目录），从 tests/ 里 import 它们
依赖根目录在 sys.path 上。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_llm_globals():
    """每个用例前后复位模块级全局状态。

    llm.USE_MOCK / 进度回调 / 告警回调 / models 的告警去重表都是模块级单例，
    用例之间会互相污染（典型症状：某个用例设了 USE_MOCK=True，
    后面的用例莫名其妙全走 mock；或前一个用例吃掉了告警、后一个断言不到）。
    """
    import llm
    import models

    old_mock = llm.USE_MOCK
    models.reset_warnings()
    yield
    llm.USE_MOCK = old_mock
    llm.set_progress_cb(None)
    llm.set_notice_cb(None)
    models.reset_warnings()


@pytest.fixture
def mock_mode():
    """把 llm 切到离线 Mock，不联网、不烧 token。"""
    import llm

    llm.USE_MOCK = True
    return llm


def initial_state(prompt: str = "测试需求") -> dict:
    """与 main.build_initial_state 等价的初始状态（测试内联，避免引 CLI 依赖）。"""
    return {
        "user_prompt": prompt,
        "outline": {},
        "chapter_index": 1,
        "chapter_draft": "",
        "review_comments": [],
        "review_verdict": "pass",
        # 三路并行校对：每路各写各的 key
        "review_comments_ooc": [],
        "review_comments_logic": [],
        "review_comments_pacing": [],
        "style_report": {},
        "revision_round": 0,
        "final_chapter": "",
        "meta": {},
        "memory": {},
        "final_chapters": [],
    }
