import pytest

from src.agents import AgentFailure, RuleAgent
from src.agents.fakes import make_readings


def _run(values, sensor="temperature"):
    return RuleAgent().run({"readings": make_readings(values), "sensor_type": sensor})


def _level(out, rule_id):
    return next(i["level"] for i in out["items"] if i["rule_id"] == rule_id)


def test_steady_data_all_normal():
    out = _run([23.6] * 8 + [23.7] * 2)
    assert out["triggered_count"] == 0
    assert all(i["level"] == "normal" and i["evidence"] for i in out["items"])


def test_latest_value_fault():
    assert _level(_run([23.6] * 4 + [50.0]), "R01_LATEST_THRESHOLD") == "fault"


def test_historical_excursion_caught_even_if_latest_normal():
    out = _run([23.6, 23.6, 36.0, 23.6, 23.6])
    assert _level(out, "R01_LATEST_THRESHOLD") == "normal"
    assert _level(out, "R02_WINDOW_EXTREME") == "warn"


def test_rising_trend_warns():
    out = _run([23.0, 23.5, 24.0, 24.8, 25.5])
    assert _level(out, "R03_RISING_TREND") == "warn"
    assert out["triggered_count"] == 1


def test_small_rise_is_not_trend():
    assert _level(_run([23.6, 23.7, 23.8, 23.9, 24.0]), "R03_RISING_TREND") == "normal"


def test_jump_warns():
    assert _level(_run([23.6, 30.0, 23.6]), "R04_JUMP") == "warn"


def test_humidity_uses_its_own_thresholds():
    # 85 对湿度是警告，对温度会是故障，证明阈值按传感器区分
    assert _level(_run([85.0], "humidity"), "R01_LATEST_THRESHOLD") == "warn"


def test_no_readings_refuses_to_conclude():
    with pytest.raises(AgentFailure):
        RuleAgent().run({"readings": [], "sensor_type": "temperature"})