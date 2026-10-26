"""Streamlit 诊断页的冒烟测试。

用 streamlit 官方的 AppTest 真跑一遍脚本：页面炸了、指标算错了都会红，
不需要起服务器，也不需要浏览器。
"""
from pathlib import Path

import pytest

pytest.importorskip("streamlit", reason="未装 streamlit 的最小安装环境跳过 UI 测试")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parent.parent / "streamlit_diagnosis.py"

MODE_DEMO = "演示：三条构造路径"
MODE_LIVE = "实测：当前采样缓冲区"


def _run(mode: str | None = None, choice: str | None = None) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    if mode is not None:
        at.radio[0].set_value(mode).run()
    if choice is not None:
        at.radio[1].set_value(choice).run()
    assert not list(at.exception), f"页面抛异常：{list(at.exception)}"
    return at


def _metrics(at: AppTest) -> dict[str, str]:
    return {m.label: m.value for m in at.metric}


def test_path_a_renders_complete():
    at = _run(choice="路径A：degraded → 补采 → 告警 → 完整AI")
    got = _metrics(at)
    assert got["最终状态"] == "complete"
    assert got["规则触发"] == "1 条"
    assert got["LLM 调用"] == "1"


def test_path_b_renders_lite_with_zero_llm():
    at = _run(choice="路径B：正常 → 全绿 → AI 简化（跳 RAG）")
    got = _metrics(at)
    assert got["规则触发"] == "0 条"
    assert got["LLM 调用"] == "0", "全绿路径在页面上也必须显示 0 次调用"


def test_path_c_renders_failed_offline():
    at = _run(choice="路径C：offline → 直达 supervisor → failed")
    got = _metrics(at)
    assert got["最终状态"] == "failed"
    assert got["数据质量"] == "offline"
    assert got["有效读数"] == "0 帧"
    assert got["规则触发"] == "未运行"
    assert got["LLM 调用"] == "0"


def _click(at: AppTest, needle: str) -> AppTest:
    """按标签点按钮：页面里按钮顺序会变（比如实测模式多了"启动后台采样"）。"""
    target = next((b for b in at.button if needle in b.label), None)
    assert target is not None, f"找不到按钮 {needle!r}，现有：{[b.label for b in at.button]}"
    return target.click().run()


def test_live_mode_with_empty_buffer_is_offline():
    """实测模式：测试环境 buffer 为空，点"运行诊断"应判 offline + failed，且不调模型。"""
    at = _run(mode=MODE_LIVE)
    at = _click(at, "运行诊断")

    assert not list(at.exception), f"页面抛异常：{list(at.exception)}"
    got = _metrics(at)
    assert got["最终状态"] == "failed"
    assert got["数据质量"] == "offline"
    assert got["有效读数"] == "0 帧"
    assert got["LLM 调用"] == "0"


def test_live_mode_with_recent_green_data_goes_lite():
    """实测模式 happy path：buffer 里有刚采的正常读数 → complete + lite + 零模型调用。"""
    from datetime import datetime, timedelta

    from src.tools import sensor_tools as st

    now = datetime.now()
    values = [23.6] * 10
    st._history_buffer["temperature"].extend(
        {
            "timestamp": (now - timedelta(seconds=30 * (len(values) - 1 - i))).isoformat(timespec="seconds"),
            "value": v,
        }
        for i, v in enumerate(values)
    )

    at = _run(mode=MODE_LIVE)
    at = _click(at, "运行诊断")

    assert not list(at.exception), f"页面抛异常：{list(at.exception)}"
    got = _metrics(at)
    assert got["最终状态"] == "complete"
    assert got["数据质量"] == "ok"
    assert got["有效读数"] == "10 帧"
    assert got["规则触发"] == "0 条"
    assert got["LLM 调用"] == "0"


def test_export_buttons_present():
    """演示模式应给出下载入口（报告 + 原始 state）。"""
    at = _run()
    labels = [b.label for b in at.download_button]
    assert any("Markdown" in x for x in labels), labels
    assert any("JSON" in x for x in labels), labels
