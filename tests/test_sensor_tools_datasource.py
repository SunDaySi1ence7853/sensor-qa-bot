"""
测试 src/tools/sensor_tools.py 数据源接入 + 后台采样(第八阶段)
"""
import logging
import threading
from datetime import datetime, timedelta

import pytest
import src.datasource as ds
import src.tools.sensor_tools as tools
from src.tools.sensor_tools import (
    _history_buffer, get_sensor_data, query_history, reset_sensor_hub,
    sample_once, start_sampler, stop_sampler,
    SAMPLE_INTERVAL_SEC, HISTORY_RETENTION_SEC,
)

@pytest.fixture(autouse=True)
def _fresh_hub():
    def _clean():
        stop_sampler()
        reset_sensor_hub()
        for buf in _history_buffer.values(): buf.clear()
        for k in tools._fail_streak: tools._fail_streak[k] = 0
    _clean()
    yield
    _clean()

def _use_serial_with(monkeypatch, frames):
    fake = ds.FakeSerial(script=list(frames))
    monkeypatch.setattr(ds, "_open_serial_port", lambda: fake)
    monkeypatch.setenv("SENSOR_DATA_SOURCE", "serial")
    reset_sensor_hub()

def _use_simulated(monkeypatch):
    monkeypatch.setenv("SENSOR_DATA_SOURCE", "simulated")
    reset_sensor_hub()

class TestSimulatedMode:
    def test_get_sensor_data_fields(self, monkeypatch):
        _use_simulated(monkeypatch)
        r = get_sensor_data.invoke({"sensor_type": "temperature"})
        assert r["data_source"] == "simulated"
        assert 19.5 <= r["value"] <= 28.5
    def test_get_sensor_data_invalid_type(self, monkeypatch):
        _use_simulated(monkeypatch)
        assert "error" in get_sensor_data.invoke({"sensor_type": "pressure"})
    def test_query_history_fields(self, monkeypatch):
        _use_simulated(monkeypatch)
        r = query_history.invoke({"sensor_type": "temperature", "hours": 3})
        assert r["data_points"] == 12
        assert "12 条 × 15 分钟 = 覆盖 3 小时" in r["note"]
    def test_query_history_invalid_hours(self, monkeypatch):
        _use_simulated(monkeypatch)
        assert "error" in query_history.invoke({"sensor_type": "temperature", "hours": 0})

class TestSerialMode:
    def test_routes_temp_frame(self, monkeypatch):
        _use_serial_with(monkeypatch, [b"temp,25.30,C\r\n"])
        r = get_sensor_data.invoke({"sensor_type": "temperature"})
        assert r["value"] == pytest.approx(25.30)
    def test_routes_humi_frame(self, monkeypatch):
        _use_serial_with(monkeypatch, [b"humi,55.20,RH\r\n"])
        r = get_sensor_data.invoke({"sensor_type": "humidity"})
        assert r["value"] == pytest.approx(55.20)
    def test_missing_sensor_reports_error(self, monkeypatch):
        _use_serial_with(monkeypatch, [b"temp,25.30,C\r\n"] * 20)
        r = get_sensor_data.invoke({"sensor_type": "humidity"})
        assert "error" in r
    def test_get_sensor_data_does_not_write_buffer(self, monkeypatch):
        _use_serial_with(monkeypatch, [b"temp,25.30,C\r\n"])
        get_sensor_data.invoke({"sensor_type": "temperature"})
        assert len(_history_buffer["temperature"]) == 0
    def test_history_from_sampled_buffer(self, monkeypatch):
        values = [25.10, 25.50, 24.80]
        _use_serial_with(monkeypatch, [f"temp,{v:.2f},C\r\n".encode() for v in values])
        for _ in values: sample_once(["temperature"])
        r = query_history.invoke({"sensor_type": "temperature", "hours": 1})
        assert r["data_points"] == 3
        assert r["mean"] == pytest.approx(round(sum(values) / 3, 2))
    def test_history_note_reconciles(self, monkeypatch):
        # 4 条落地，中间缺 1 个采样点：跨度 120s = 4 个间隔 → 应有 5 条，缺 1 条
        _use_serial_with(monkeypatch, [])
        now = datetime.now()
        for back in (150, 120, 90, 30):
            _history_buffer["temperature"].append((now - timedelta(seconds=back), 25.0))
        r = query_history.invoke({"sensor_type": "temperature", "hours": 1})
        assert r["data_points"] == 4
        assert "4 条读数" in r["note"] and "跳过 1 个" in r["note"]

class TestRetentionContract:
    def test_maxlen_times_interval_is_24h(self):
        for st, buf in _history_buffer.items():
            assert buf.maxlen * SAMPLE_INTERVAL_SEC == HISTORY_RETENTION_SEC == 86400, st

class TestSampleOnce:
    def test_valid_reading_lands(self, monkeypatch):
        _use_serial_with(monkeypatch, [b"temp,25.30,C\r\n"])
        assert sample_once(["temperature"]) == {"temperature": True}
        assert _history_buffer["temperature"][-1][1] == pytest.approx(25.30)
    def test_invalid_reading_skipped(self, monkeypatch):
        _use_serial_with(monkeypatch, [b"garbage without comma\r\n"] * 20)
        assert sample_once(["temperature"]) == {"temperature": False}
        assert len(_history_buffer["temperature"]) == 0
    def test_read_exception_not_raised(self, monkeypatch):
        _use_simulated(monkeypatch)
        def boom(self, st): raise RuntimeError("串口被拔")
        monkeypatch.setattr(tools._SensorHub, "read", boom)
        assert sample_once(["temperature"]) == {"temperature": False}
    def test_hub_init_exception_not_raised(self, monkeypatch):
        def boom(): raise RuntimeError("配置损坏")
        monkeypatch.setattr(tools, "_get_hub", boom)
        assert sample_once(["temperature"]) == {"temperature": False}
    def test_consecutive_failures_warn(self, monkeypatch, caplog):
        _use_serial_with(monkeypatch, [])  # 脚本为空 → 每次都无效
        caplog.set_level(logging.WARNING, logger=tools.__name__)
        for _ in range(tools._FAIL_WARN_EVERY): sample_once(["temperature"])
        assert any("连续" in r.message and "链路可能已断" in r.message for r in caplog.records)

class TestSamplerThread:
    def test_smoke_first_tick(self, monkeypatch):
        _use_simulated(monkeypatch)
        s = start_sampler(interval_sec=3600)  # 启动即采第一轮，不等间隔
        assert s.ticked.wait(timeout=5)
        assert len(_history_buffer["temperature"]) >= 1
    def test_idempotent_and_daemon(self, monkeypatch):
        _use_simulated(monkeypatch)
        s1 = start_sampler(interval_sec=3600)
        s2 = start_sampler(interval_sec=3600)
        assert s1 is s2 and s1.thread.daemon
        alive = [t for t in threading.enumerate() if t.name == "sensor-sampler" and t.is_alive()]
        assert len(alive) == 1
    def test_thread_survives_exception(self, monkeypatch):
        _use_simulated(monkeypatch)
        def boom(self, st): raise RuntimeError("串口被拔")
        monkeypatch.setattr(tools._SensorHub, "read", boom)
        s = start_sampler(interval_sec=3600)
        assert s.ticked.wait(timeout=5)
        assert s.is_alive()
    def test_stop_is_prompt(self, monkeypatch):
        _use_simulated(monkeypatch)
        s = start_sampler(interval_sec=3600)
        s.ticked.wait(timeout=5)
        stop_sampler(timeout=2)
        assert not s.is_alive()  # Event.wait 可被立即唤醒，不用等满 3600s