"""实测模式（live_diagnose）单测：三条路由 + 导出。

用注入的 reader/clock 造场景，不碰真实 buffer 与真实 LLM。
"""
import json

from src.agents.fakes import FIXED_NOW, make_readings
from src.agents.live_diagnose import diagnose_live, to_json, to_markdown


def test_offline_when_buffer_empty():
    result, stats = diagnose_live(
        "temperature", 1.0, reader=lambda s: [], clock=lambda: FIXED_NOW,
    )

    assert result["collector"]["data_quality"] == "offline"
    assert result["routing_path"] == ["collector", "supervisor"]
    assert result["status"] == "failed"
    assert stats["llm"].calls == []
    assert stats["ai_enabled"] is False


def test_all_green_goes_lite_without_llm():
    result, stats = diagnose_live(
        "temperature", 1.0,
        reader=lambda s: make_readings([23.6] * 10),
        clock=lambda: FIXED_NOW,
    )

    assert result["status"] == "complete"
    assert "ai_analysis_lite" in result["routing_path"]
    assert stats["llm"].calls == [], "全绿路径不应调模型"
    assert stats["reader_calls"] == 1


def test_triggered_without_ai_is_partial_and_explains_why():
    # 单调上升 2.5℃ → R03 告警；关闭 AI 时不应静默出结论
    result, stats = diagnose_live(
        "temperature", 1.0,
        reader=lambda s: make_readings([23.0, 23.5, 24.0, 24.8, 25.5]),
        clock=lambda: FIXED_NOW,
    )

    assert result["status"] == "partial"
    assert result["report"] is None
    assert stats["llm"].calls == []
    reasons = " ".join(e["reason"] for e in result["errors"])
    assert "无相关片段" in reasons or "未启用" in reasons, reasons


def test_ai_enabled_flag_when_adapters_injected():
    # 注入假 LLM/检索，验证开关本身：规则触发后会真的走 AI
    def fake_llm(system: str, user: str) -> str:
        return json.dumps({"root_cause": "散热不良", "recommendation": "检查风扇", "cited": [1]},
                          ensure_ascii=False)

    result, stats = diagnose_live(
        "temperature", 1.0,
        reader=lambda s: make_readings([23.0, 23.5, 24.0, 24.8, 25.5]),
        clock=lambda: FIXED_NOW,
        use_ai=True,
        llm=fake_llm,
        retriever=lambda q: ["手册：温度持续上升常见原因是散热风扇停转。"],
    )

    assert stats["ai_enabled"] is True
    assert len(stats["llm"].calls) == 1
    assert len(stats["retriever"].queries) == 1
    assert result["ai_analysis"]["root_cause"] == "散热不良"
    assert result["status"] == "complete"


def test_markdown_export_contains_facts():
    result, stats = diagnose_live(
        "temperature", 1.0,
        reader=lambda s: make_readings([23.6] * 10),
        clock=lambda: FIXED_NOW,
    )
    md = to_markdown(result, stats, sensor_type="temperature", window_hours=1.0)

    for needle in ("传感器诊断报告", "路由路径", "最终状态", "数据质量", "规则诊断", "置信度"):
        assert needle in md, f"导出的报告缺少 {needle!r}\n---\n{md}"
    assert "complete" in md


def test_json_export_round_trips():
    result, stats = diagnose_live(
        "temperature", 1.0, reader=lambda s: [], clock=lambda: FIXED_NOW,
    )
    payload = json.loads(to_json(result, stats, source="test"))

    assert payload["state"]["status"] == "failed"
    assert payload["calls"]["llm"] == 0
    assert payload["meta"]["source"] == "test"
