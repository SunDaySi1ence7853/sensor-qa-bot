"""
测试 src/tools/sensor_tools.py 数据源接入(第八阶段任务1)
"""
import pytest
import src.datasource as ds
from src.tools.sensor_tools import _history_buffer, get_sensor_data, query_history, reset_sensor_hub

@pytest.fixture(autouse=True)
def _fresh_hub():
    reset_sensor_hub()
    for buf in _history_buffer.values(): buf.clear()
    yield
    reset_sensor_hub()
    for buf in _history_buffer.values(): buf.clear()

def _use_serial_with(monkeypatch, frames):
    fake = ds.FakeSerial(script=list(frames))
    monkeypatch.setattr(ds, "_open_serial_port", lambda: fake)
    monkeypatch.setenv("SENSOR_DATA_SOURCE", "serial")
    reset_sensor_hub()

class TestSimulatedMode:
    def test_get_sensor_data_fields(self, monkeypatch):
        monkeypatch.setenv("SENSOR_DATA_SOURCE", "simulated"); reset_sensor_hub()
        r = get_sensor_data.invoke({"sensor_type": "temperature"})
        assert r["data_source"] == "simulated"
        assert 19.5 <= r["value"] <= 28.5
    def test_get_sensor_data_invalid_type(self, monkeypatch):
        monkeypatch.setenv("SENSOR_DATA_SOURCE", "simulated"); reset_sensor_hub()
        assert "error" in get_sensor_data.invoke({"sensor_type": "pressure"})
    def test_query_history_fields(self, monkeypatch):
        monkeypatch.setenv("SENSOR_DATA_SOURCE", "simulated"); reset_sensor_hub()
        r = query_history.invoke({"sensor_type": "temperature", "hours": 3})
        assert r["data_points"] == 12
    def test_query_history_invalid_hours(self, monkeypatch):
        monkeypatch.setenv("SENSOR_DATA_SOURCE", "simulated"); reset_sensor_hub()
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
    def test_history_from_recorded_buffer(self, monkeypatch):
        values = [25.10, 25.50, 24.80]
        frames = [f"temp,{v:.2f},C\r\n".encode() for v in values]
        _use_serial_with(monkeypatch, frames)
        for _ in values: get_sensor_data.invoke({"sensor_type": "temperature"})
        r = query_history.invoke({"sensor_type": "temperature", "hours": 1})
        assert r["data_points"] == 3
        assert r["mean"] == pytest.approx(round(sum(values) / 3, 2))