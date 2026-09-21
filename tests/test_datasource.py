"""
测试 src/datasource.py(第八阶段任务1)
"""
import logging, math, sys, pytest
from src.datasource import DataSource, FakeSerial, SerialSource, SimulatedSource, get_datasource

def _scripted(frames): return FakeSerial(script=frames)

class TestSimulatedSource:
    def test_reading_within_range(self):
        sim = SimulatedSource(base=25.0, amplitude=5.0, noise=0.2, value_range=(-40.0, 85.0))
        for _ in range(10):
            r = sim.read()
            assert r.valid is True
            assert 19.0 <= r.value <= 31.0
    def test_phase_backtrack_deterministic(self):
        sim = SimulatedSource(base=25.0, amplitude=5.0, period_sec=600.0)
        assert sim.value_at(12345.6) == sim.value_at(12345.6)
        assert sim.value_at(0.0) == pytest.approx(25.0)
        assert sim.value_at(150.0) == pytest.approx(30.0)
    def test_noise_seed_reproducible(self):
        sim = SimulatedSource(noise=0.5)
        assert sim.reading_at(1000.0).value == sim.reading_at(1000.0).value
    def test_read_returns_valid_reading(self):
        r = SimulatedSource().read()
        assert r.valid is True
        assert r.sensor_id == "temp"

class TestSerialSource:
    def test_parses_valid_frame(self):
        r = SerialSource(_scripted([b"temp,25.3,C\r\n"])).read()
        assert r.valid is True
        assert r.value == pytest.approx(25.3)
    def test_skips_garbage_then_gets_good(self, caplog):
        ser = SerialSource(_scripted([b"garbage\r\n", b"\x00\xff junk\r\n", b"temp,25.0,C\r\n"]))
        with caplog.at_level(logging.WARNING, logger="src.datasource"):
            r = ser.read()
        assert r.valid is True
        assert any("坏帧" in rec.message for rec in caplog.records)
    def test_skips_nan_frame(self):
        r = SerialSource(_scripted([b"temp,nan,C\r\n", b"temp,25.0,C\r\n"])).read()
        assert r.valid is True
    def test_skips_out_of_range(self):
        ser = SerialSource(_scripted([b"temp,999,C\r\n", b"temp,25.0,C\r\n"]), value_range=(-40.0, 85.0))
        assert ser.read().valid is True
    def test_all_bad_frames_returns_invalid(self):
        r = SerialSource(_scripted([b"bad1\r\n", b"bad2\r\n", b"bad3\r\n"])).read()
        assert r.valid is False
    def test_timeout_returns_invalid(self):
        r = SerialSource(_scripted([])).read()
        assert r.valid is False
    def test_unplugged_device_returns_invalid(self):
        class ExpSerial:
            def readline(self): raise IOError("断开")
        r = SerialSource(ExpSerial()).read()
        assert r.valid is False

class TestFactory:
    def test_factory_simulated(self):
        assert isinstance(get_datasource("simulated"), SimulatedSource)
    def test_factory_serial_degrades_without_hardware(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "serial", None)
        src = get_datasource("serial")
        assert isinstance(src, SerialSource)
        assert isinstance(src.serial, FakeSerial)
    def test_factory_unknown_falls_back(self):
        assert isinstance(get_datasource("nonsense"), SimulatedSource)

class TestFakeSerial:
    def test_mixed_stream_contains_good_and_bad(self):
        fake = FakeSerial(bad_ratio=0.5, seed=2026)
        lines = [fake.readline().decode("utf-8", errors="replace").strip() for _ in range(300)]
        good = [l for l in lines if SerialSource.FRAME_RE.match(l)]
        bad = [l for l in lines if l and not SerialSource.FRAME_RE.match(l)]
        assert len(good) > 50
        assert len(bad) > 20