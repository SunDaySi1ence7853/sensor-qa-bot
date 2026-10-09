from datetime import timedelta

import pytest

from src.agents import AgentFailure, CollectorAgent
from src.agents.fakes import FIXED_NOW, make_readings

INP = {"sensor_type": "temperature", "window_hours": 1.0}


def _agent(rows):
    return CollectorAgent(reader=lambda s: rows, clock=lambda: FIXED_NOW)


def test_clean_window_is_ok():
    out = _agent(make_readings([23.6] * 8 + [23.7] * 2)).run(INP)
    assert len(out["readings"]) == 10
    assert out["data_quality"] == "ok"
    assert out["missing_count"] == 0 and out["bad_frame_count"] == 0


def test_bad_frames_reported_not_silently_dropped():
    out = _agent(make_readings([23.6, None, "abc", 200.0, 23.7])).run(INP)
    assert [r["value"] for r in out["readings"]] == [23.6, 23.7]
    assert out["bad_frame_count"] == 3
    assert out["missing_count"] == 0  # 坏帧有时间戳，不重复计为缺口
    assert out["data_quality"] == "degraded"


def test_gap_is_counted():
    rows = make_readings([23.6] * 6)
    del rows[2:4]
    out = _agent(rows).run(INP)
    assert out["missing_count"] == 2
    assert out["data_quality"] == "degraded"


def test_empty_buffer_is_offline():
    out = _agent([]).run(INP)
    assert out["data_quality"] == "offline"
    assert out["readings"] == []


def test_stale_data_is_offline_but_still_returned():
    out = _agent(make_readings([23.6] * 5, end=FIXED_NOW - timedelta(minutes=10))).run(INP)
    assert out["data_quality"] == "offline"
    assert len(out["readings"]) == 5


def test_window_filters_old_readings():
    old = make_readings([20.0] * 3, end=FIXED_NOW - timedelta(hours=2))
    new = make_readings([23.6] * 3)
    out = _agent(old + new).run(INP)
    assert [r["value"] for r in out["readings"]] == [23.6] * 3


def test_unparseable_timestamp_is_bad_frame():
    rows = make_readings([23.6] * 3) + [{"timestamp": "not-a-time", "value": 23.6}]
    assert _agent(rows).run(INP)["bad_frame_count"] == 1


@pytest.mark.parametrize("inp", [
    {"sensor_type": "pressure", "window_hours": 1.0},
    {"sensor_type": "temperature", "window_hours": 0},
])
def test_invalid_input_raises(inp):
    with pytest.raises(AgentFailure):
        _agent([]).run(inp)