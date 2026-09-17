"""
测试 src/datasource.py(第八阶段任务1)

覆盖 15 条:
- SimulatedSource:量程内、相位回溯确定性、噪声种子化可复现、read 有效
- SerialSource+FakeSerial:正常帧解析、坏帧(乱码/NaN/越界)剔除+重试、
  全坏帧不崩溃、超时防御、设备断连防御、坏帧告警日志
- 工厂:配置切换 simulated / serial 降级兜底 / 未知配置回退
- FakeSerial:混合流确实含好帧与坏帧(B线演示材料真实性)
"""

import logging
import math
import sys

import pytest

from src.datasource import (
    DataSource, FakeSerial, SerialSource, SimulatedSource, get_datasource,
)


def _scripted(frames):
    """预置帧序列的 FakeSerial,精确控制每次 readline 输出。"""
    return FakeSerial(script=frames)


class TestSimulatedSource:
    def test_reading_within_range(self):
        """1. 模拟读数落在量程内且围绕基线波动"""
        sim = SimulatedSource(base=25.0, amplitude=5.0, noise=0.2,
                              value_range=(-40.0, 85.0))
        for _ in range(10):
            r = sim.read()
            assert r.valid is True
            assert -40.0 <= r.value <= 85.0
            assert 19.0 <= r.value <= 31.0  # base ± amplitude ± noise

    def test_phase_backtrack_deterministic(self):
        """2. 相位回溯:同一时刻理论值可复现,且相位对齐点取值正确"""
        sim = SimulatedSource(base=25.0, amplitude=5.0, period_sec=600.0)
        assert sim.value_at(12345.6) == sim.value_at(12345.6)
        assert sim.value_at(0.0) == pytest.approx(25.0)    # 相位 0 → 基线
        assert sim.value_at(150.0) == pytest.approx(30.0)  # 相位 π/2 → 峰值

    def test_noise_seed_reproducible(self):
        """3. 噪声按时间戳种子化:同一时刻两次读数完全一致(历史对账用)"""
        sim = SimulatedSource(noise=0.5)
        r1 = sim.reading_at(1000.0)
        r2 = sim.reading_at(1000.0)
        assert r1.value == r2.value

    def test_read_returns_valid_reading(self):
        """4. 正常读取返回有效读数,字段齐全"""
        r = SimulatedSource().read()
        assert r.valid is True
        assert r.sensor_id == "temp"
        assert r.unit == "°C"


class TestSerialSource:
    def test_parses_valid_frame(self):
        """5. 正常协议帧被正确解析"""
        ser = SerialSource(_scripted([b"temp,25.3,C\r\n"]))
        r = ser.read()
        assert r.valid is True
        assert r.value == pytest.approx(25.3)
        assert r.sensor_id == "temp"
        assert r.unit == "C"

    def test_skips_garbage_then_gets_good(self, caplog):
        """6. 乱码坏帧被剔除并重试拿到好帧,且有告警日志"""
        ser = SerialSource(_scripted([
            b"garbage\r\n", b"\x00\xff junk\r\n", b"temp,25.0,C\r\n"
        ]))
        with caplog.at_level(logging.WARNING, logger="src.datasource"):
            r = ser.read()
        assert r.valid is True
        assert r.value == pytest.approx(25.0)
        assert any("坏帧" in rec.message for rec in caplog.records)

    def test_skips_nan_frame(self):
        """7. NaN 坏点被剔除,不影响后续好帧"""
        ser = SerialSource(_scripted([b"temp,nan,C\r\n", b"temp,25.0,C\r\n"]))
        r = ser.read()
        assert r.valid is True
        assert r.value == pytest.approx(25.0)

    def test_skips_out_of_range(self):
        """8. 量程越界读数被剔除"""
        ser = SerialSource(_scripted([b"temp,999,C\r\n", b"temp,25.0,C\r\n"]),
                           value_range=(-40.0, 85.0))
        r = ser.read()
        assert r.valid is True
        assert r.value == pytest.approx(25.0)

    def test_all_bad_frames_returns_invalid(self):
        """9. 全部是坏帧时返回无效读数,不崩溃"""
        ser = SerialSource(_scripted([b"bad1\r\n", b"bad2\r\n", b"bad3\r\n"]))
        r = ser.read()
        assert r.valid is False
        assert math.isnan(r.value)
        assert "坏帧" in r.note

    def test_timeout_returns_invalid(self):
        """10. 串口静默(超时)时返回无效读数"""
        ser = SerialSource(_scripted([]))
        r = ser.read()
        assert r.valid is False
        assert "无数据" in r.note

    def test_unplugged_device_returns_invalid(self):
        """11. 设备断连(readline 抛异常)时不崩溃,返回无效读数"""

        class ExplodingSerial:
            def readline(self):
                raise IOError("设备断开")

        r = SerialSource(ExplodingSerial()).read()
        assert r.valid is False
        assert "读取异常" in r.note


class TestFactory:
    def test_factory_simulated(self):
        """12. 配置 simulated 返回 SimulatedSource"""
        assert isinstance(get_datasource("simulated"), SimulatedSource)

    def test_factory_serial_degrades_without_hardware(self, monkeypatch):
        """13. 无 pyserial/无硬件时,serial 配置自动降级 FakeSerial(B线兜底)"""
        monkeypatch.setitem(sys.modules, "serial", None)  # 令 import serial 失败
        src = get_datasource("serial")
        assert isinstance(src, SerialSource)
        assert isinstance(src.serial, FakeSerial)

    def test_factory_unknown_falls_back(self):
        """14. 未知配置回退 simulated,不崩"""
        assert isinstance(get_datasource("nonsense"), SimulatedSource)


class TestFakeSerial:
    def test_mixed_stream_contains_good_and_bad(self):
        """15. FakeSerial 混合流确实同时含好帧与坏帧(B线演示材料真实)"""
        fake = FakeSerial(bad_ratio=0.5, seed=2026)
        lines = [fake.readline().decode("utf-8", errors="replace").strip()
                 for _ in range(300)]
        good = [l for l in lines if SerialSource.FRAME_RE.match(l)]
        bad = [l for l in lines if l and not SerialSource.FRAME_RE.match(l)]
        assert len(good) > 50   # 好帧占多数
        assert len(bad) > 20    # 坏帧确实存在且被生成