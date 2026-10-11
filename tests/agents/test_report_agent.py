import pytest

from src.agents import AgentFailure, ReportAgent
from src.agents.fakes import FAKE_MANUAL, FIXED_NOW, make_readings

agent = ReportAgent(clock=lambda: FIXED_NOW)


def _inp(quality="ok"):
    return {
        "collector": {"readings": make_readings([23.6] * 5), "data_quality": quality,
                      "missing_count": 0, "bad_frame_count": 0},
        "rule": {"items": [
            {"rule_id": "R01_LATEST_THRESHOLD", "level": "normal", "evidence": "最新值 23.6"},
            {"rule_id": "R03_RISING_TREND", "level": "warn", "evidence": "累计 +2.5"},
        ], "triggered_count": 1},
        "ai_analysis": {"root_cause": "散热不良", "recommendation": "检查风扇",
                        "references": [FAKE_MANUAL[1]]},
    }


def test_level_comes_from_rules():
    assert "诊断结论：警告" in agent.run(_inp())["summary"]


def test_evidence_chain_is_traceable():
    chain = agent.run(_inp())["evidence_chain"]
    assert chain[0].startswith("[采集]")
    assert any("R03_RISING_TREND" in line for line in chain)
    assert any(line.startswith("[知识库 1]") for line in chain)


def test_confidence_drops_with_data_quality():
    ok, degraded = agent.run(_inp("ok")), agent.run(_inp("degraded"))
    assert 0 < degraded["confidence"] < ok["confidence"] <= 1
    assert "置信度已下调" in degraded["summary"]


def test_timestamp_from_injected_clock():
    assert agent.run(_inp())["timestamp"] == FIXED_NOW.isoformat(timespec="seconds")


def test_missing_upstream_raises():
    inp = _inp()
    inp.pop("ai_analysis")
    with pytest.raises(AgentFailure):
        agent.run(inp)