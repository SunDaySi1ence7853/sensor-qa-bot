import pytest

from src.agents import AgentFailure, ReportAgent
from src.agents.fakes import FAKE_MANUAL, FIXED_NOW, make_readings

agent = ReportAgent(clock=lambda: FIXED_NOW)

# QUALITY_WEIGHT = {"ok": 1.0, "degraded": 0.7, "offline": 0.3}
# confidence = round(QUALITY_WEIGHT[quality] * (0.5 + 0.5 * min(len(refs), 3) / 3), 2)


def _inp(quality="ok", refs=None):
    if refs is None:
        refs = [FAKE_MANUAL[1]]
    return {
        "collector": {
            "readings": make_readings([23.6] * 5),
            "data_quality": quality,
            "missing_count": 0,
            "bad_frame_count": 0,
        },
        "rule": {
            "items": [
                {"rule_id": "R01_LATEST_THRESHOLD", "level": "normal", "evidence": "最新值 23.6"},
                {"rule_id": "R03_RISING_TREND", "level": "warn", "evidence": "累计 +2.5"},
            ],
            "triggered_count": 1,
        },
        "ai_analysis": {
            "root_cause": "散热不良",
            "recommendation": "检查风扇",
            "references": refs,
        },
    }


def test_happy_path_structure():
    out = agent.run(_inp())
    assert out["summary"] and out["confidence"] and out["evidence_chain"] and out["timestamp"]


def test_level_comes_from_rules_not_rewritten():
    assert "诊断结论：警告" in agent.run(_inp())["summary"]


def test_evidence_chain_is_fully_traceable():
    chain = agent.run(_inp())["evidence_chain"]
    assert chain[0].startswith("[采集]")
    assert any("R01_LATEST_THRESHOLD" in line for line in chain)
    assert any("R03_RISING_TREND" in line for line in chain)
    assert any(line.startswith("[AI]") for line in chain)
    assert any(line.startswith("[知识库 1]") for line in chain)


def test_confidence_formula_ok_one_ref():
    # 1.0 * (0.5 + 0.5 * 1/3) = 0.67
    assert agent.run(_inp("ok", [FAKE_MANUAL[1]]))["confidence"] == 0.67


def test_confidence_formula_ok_three_refs():
    # 1.0 * (0.5 + 0.5 * 3/3) = 1.0
    assert agent.run(_inp("ok", FAKE_MANUAL * 3))["confidence"] == 1.0


def test_confidence_formula_degraded_one_ref():
    # 0.7 * (0.5 + 0.5 * 1/3) = round(0.467, 2) = 0.47
    assert agent.run(_inp("degraded", [FAKE_MANUAL[1]]))["confidence"] == 0.47


def test_degraded_summary_flags_data_quality():
    out = agent.run(_inp("degraded"))
    assert "置信度已下调" in out["summary"]
    assert out["confidence"] < agent.run(_inp("ok"))["confidence"]


def test_timestamp_from_injected_clock():
    assert agent.run(_inp())["timestamp"] == FIXED_NOW.isoformat(timespec="seconds")


@pytest.mark.parametrize("missing_key", ["collector", "rule", "ai_analysis"])
def test_missing_any_upstream_raises(missing_key):
    inp = _inp()
    inp.pop(missing_key)
    with pytest.raises(AgentFailure):
        agent.run(inp)