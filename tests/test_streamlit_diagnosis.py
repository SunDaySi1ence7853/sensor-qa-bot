"""Streamlit 路由可视化页的冒烟测试。

用 streamlit 官方的 AppTest 真跑一遍脚本：页面炸了、指标算错了都会红，
不需要起服务器，也不需要浏览器。
"""
from pathlib import Path

import pytest

pytest.importorskip("streamlit", reason="未装 streamlit 的最小安装环境跳过 UI 测试")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parent.parent / "streamlit_diagnosis.py"


def _run(choice: str | None = None) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    if choice is not None:
        at.radio[0].set_value(choice).run()
    assert not list(at.exception), f"页面抛异常：{list(at.exception)}"
    return at


def _metrics(at: AppTest) -> dict[str, str]:
    return {m.label: m.value for m in at.metric}


def test_path_a_renders_complete():
    at = _run("路径A：degraded → 补采 → 告警 → 完整AI")
    got = _metrics(at)
    assert got["最终状态"] == "complete"
    assert got["规则触发"] == "1 条"
    assert got["LLM 调用"] == "1"


def test_path_b_renders_lite_with_zero_llm():
    at = _run("路径B：正常 → 全绿 → AI 简化（跳 RAG）")
    got = _metrics(at)
    assert got["规则触发"] == "0 条"
    assert got["LLM 调用"] == "0", "全绿路径在页面上也必须显示 0 次调用"


def test_path_c_renders_failed_offline():
    at = _run("路径C：offline → 直达 supervisor → failed")
    got = _metrics(at)
    assert got["最终状态"] == "failed"
    assert got["数据质量"] == "offline"
    assert got["有效读数"] == "0 帧"
    assert got["规则触发"] == "未运行"
    assert got["LLM 调用"] == "0"
