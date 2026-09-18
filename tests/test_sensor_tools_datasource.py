"""
测试 src/tools/sensor_tools.py 与数据源层的接通(第八阶段任务1)。

覆盖 10 条:
- 模拟模式:get_sensor_data 字段与来源标注 / 非法类型报错 /
  query_history 字段与采样密度 / 非法 hours 报错 / 非法类型报错
- 串口模式(注入脚本化 FakeSerial，确定性):按 sensor_id 路由 temp/humi /
  查无该传感器时如实报错 / 历史统计来自已录得缓冲、数字可对账 / 空缓冲如实报错
"""

import pytest

import src.datasource as ds
from src.tools.sensor_tools import (
    _history_buffer, get_sensor_data, query_history, reset_sensor_hub,
)


@pytest.fixture(autouse=True)
def _fresh_hub():
    """每个用例前后重置数据源网关与历史缓冲，避免用例间串扰。"""
    reset_sensor_hub()
    for buf in _history_buffer.values():
        buf.clear()
    yield
    reset_sensor_hub()
    for buf in _history_buffer.values():
        buf.clear()


def _use_serial_with(monkeypatch, frames):
    """注入脚本化 FakeSerial 作为串口，并切到 serial 模式。"""
    fake = ds.FakeSerial(script=list(frames))
    monkeypatch.setattr(ds, "_open_serial_port", lambda: fake)
    monkeypatch.setenv("SENSOR_DATA_SOURCE", "serial")
    reset_sensor_hub()


class TestSimulatedMode:
    def test_get_sensor_data_fields_and_source(self, monkeypatch):
        """1. 模拟模式:返回字段齐全并如实标注来源"""
        monkeypatch.setenv("SENSOR_DATA_SOURCE", "simulated")
        reset_sensor_hub()
        r = get_sensor_data.invoke({"sensor_type": "temperature"})
        assert r["sensor_type"] == "temperature"
        assert r["unit"] == "°C"
        assert 19.5 <= r["value"] <= 28.5   # 24 ± 4 ± 0.5，与第六阶段曲线一致
        assert r["data_source"] == "simulated"
        assert "timestamp" in r

    def test_get_sensor_data_invalid_type(self, monkeypatch):
        """2. 非法传感器类型返回 error"""
        monkeypatch.setenv("SENSOR_DATA_SOURCE", "simulated")
        reset_sensor_hub()
        r = get_sensor_data.invoke({"sensor_type": "pressure"})
        assert "error" in r
        assert "temperature" in r["error"]

    def test_query_history_fields(self, monkeypatch):
        """3. 模拟模式历史:字段齐全，采样密度 4 点/小时，均值围绕基线"""
        monkeypatch.setenv("SENSOR_DATA_SOURCE", "simulated")
        reset_sensor_hub()
        r = query_history.invoke({"sensor_type": "temperature", "hours": 3})
        assert r["sensor_type"] == "temperature"
        assert r["hours"] == 3
        assert r["data_points"] == 12          # 4 点/小时 × 3
        assert 19.5 <= r["mean"] <= 28.5
        assert r["min"] <= r["mean"] <= r["max"]
        assert isinstance(r["is_alert"], bool)
        assert r["data_source"] == "simulated"

    def test_query_history_invalid_hours(self, monkeypatch):
        """4. hours 越界返回 error"""
        monkeypatch.setenv("SENSOR_DATA_SOURCE", "simulated")
        reset_sensor_hub()
        for bad in (0, 25):
            r = query_history.invoke({"sensor_type": "temperature", "hours": bad})
            assert "error" in r

    def test_query_history_invalid_type(self, monkeypatch):
        """5. 非法传感器类型返回 error"""
        monkeypatch.setenv("SENSOR_DATA_SOURCE", "simulated")
        reset_sensor_hub()
        r = query_history.invoke({"sensor_type": "pressure", "hours": 3})
        assert "error" in r


class TestSerialMode:
    def test_routes_temp_frame(self, monkeypatch):
        """6. 串口模式:按帧内 sensor_id 路由到 temperature"""
        _use_serial_with(monkeypatch, [b"temp,25.30,C\r\n"])
        r = get_sensor_data.invoke({"sensor_type": "temperature"})
        assert r["value"] == pytest.approx(25.30)
        assert r["data_source"].startswith("serial")

    def test_routes_humi_frame(self, monkeypatch):
        """7. 串口模式:humi 帧路由到 humidity，量程校验通过"""
        _use_serial_with(monkeypatch, [b"humi,55.20,RH\r\n"])
        r = get_sensor_data.invoke({"sensor_type": "humidity"})
        assert r["value"] == pytest.approx(55.20)
        assert r["unit"] == "%"

    def test_missing_sensor_reports_error_honestly(self, monkeypatch):
        """8. 串口流中只有 temp 帧:查 humidity 如实报错，不编造"""
        _use_serial_with(monkeypatch, [b"temp,25.30,C\r\n"] * 20)
        r = get_sensor_data.invoke({"sensor_type": "humidity"})
        assert "error" in r
        assert "humi" in r["error"]

    def test_history_from_recorded_buffer(self, monkeypatch):
        """9. 串口模式历史:统计来自真实录得的缓冲数据，数字可对账"""
        values = [25.10, 25.50, 24.80, 26.00, 25.30]
        frames = [f"temp,{v:.2f},C\r\n".encode() for v in values]
        _use_serial_with(monkeypatch, frames)
        for _ in values:
            assert "error" not in get_sensor_data.invoke({"sensor_type": "temperature"})
        r = query_history.invoke({"sensor_type": "temperature", "hours": 1})
        assert r["data_points"] == 5
        assert r["mean"] == pytest.approx(round(sum(values) / 5, 2))
        assert r["max"] == pytest.approx(26.00)
        assert r["min"] == pytest.approx(24.80)
        assert "5 条" in r["note"]

    def test_history_empty_buffer_errors_honestly(self, monkeypatch):
        """10. 串口模式无落地数据:如实报错并提示先采样，不编造历史"""
        _use_serial_with(monkeypatch, [b"temp,25.30,C\r\n"] * 50)
        r = query_history.invoke({"sensor_type": "humidity", "hours": 2})
        assert "error" in r