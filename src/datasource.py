"""
数据源抽象层(第八阶段·任务1)。

设计:
- DataSource 抽象基类:统一 read() 接口，内置防御三件套(校验+重试+跳过告警)
- SimulatedSource:正弦+噪声模拟，相位回溯(同一时刻读数可复现，历史查询可对账)
- SerialSource:串口行协议解析，A线接真实 pyserial，B线接 FakeSerial
- FakeSerial:模拟串口字节流(含偶发坏帧)；正常帧在 temp/humi/vib 间轮转
- 类型解析优先级:环境变量 SENSOR_DATA_SOURCE > config.yaml data_source > simulated

串口行协议(文本，便于调试):
    temp,25.30,C\r\n
    humi,55.20,RH\r\n
"""

from __future__ import annotations

import logging
import math
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SensorReading:
    sensor_id: str
    value: float
    unit: str
    timestamp: datetime
    valid: bool = True
    note: str = ""

    def __str__(self) -> str:
        status = "OK" if self.valid else f"INVALID({self.note})"
        return f"[{self.timestamp:%H:%M:%S}] {self.sensor_id} = {self.value:.2f} {self.unit} ({status})"


class DataSource(ABC):
    name: str = "base"

    def __init__(self, sensor_id: str, unit: str,
                 value_range: Tuple[Optional[float], Optional[float]] = (None, None),
                 max_retries: int = 2):
        self.sensor_id = sensor_id
        self.unit = unit
        self.value_range = value_range
        self.max_retries = max_retries

    @abstractmethod
    def _read_raw(self) -> Optional[SensorReading]:
        pass

    def read(self) -> SensorReading:
        last_note = "无数据"
        for attempt in range(self.max_retries + 1):
            try:
                reading = self._read_raw()
            except Exception as exc:
                last_note = f"读取异常:{exc}"
                logger.warning("%s 读取异常(第 %d 次): %s", self.name, attempt + 1, exc)
                continue

            if reading is None:
                last_note = "本轮无数据"
                continue

            ok, note = self._validate(reading)
            if ok:
                return reading
            last_note = reading.note or note
            logger.warning("%s 坏点已跳过: %s", self.name, last_note)

        return SensorReading(
            sensor_id=self.sensor_id, value=float("nan"), unit=self.unit,
            timestamp=datetime.now(), valid=False, note=last_note,
        )

    def _validate(self, reading: SensorReading) -> Tuple[bool, str]:
        lo, hi = self.value_range
        v = reading.value
        if math.isnan(v) or math.isinf(v):
            return False, f"非有限数值:{v}"
        if lo is not None and v < lo:
            return False, f"低于量程下限:{v} < {lo}"
        if hi is not None and v > hi:
            return False, f"超出量程上限:{v} > {hi}"
        return True, ""


class SimulatedSource(DataSource):
    name = "simulated"

    def __init__(self, sensor_id: str = "temp", unit: str = "°C",
                 base: float = 25.0, amplitude: float = 5.0, period_sec: float = 600.0,
                 noise: float = 0.2, value_range=(-40.0, 85.0), **kwargs):
        super().__init__(sensor_id, unit, value_range, **kwargs)
        self.base = base
        self.amplitude = amplitude
        self.period_sec = period_sec
        self.noise = noise

    def value_at(self, ts: float) -> float:
        phase = 2 * math.pi * ((ts % self.period_sec) / self.period_sec)
        return self.base + self.amplitude * math.sin(phase)

    def reading_at(self, ts: float) -> SensorReading:
        rng = random.Random(int(ts))
        value = self.value_at(ts) + rng.uniform(-self.noise, self.noise)
        return SensorReading(self.sensor_id, round(value, 2),
                             self.unit, datetime.fromtimestamp(ts))

    def _read_raw(self) -> SensorReading:
        return self.reading_at(time.time())


class SerialSource(DataSource):
    name = "serial"
    FRAME_RE = re.compile(r"^\s*([A-Za-z0-9_\-]+),(-?\d+(?:\.\d+)?),([A-Za-z°/%]+)\s*$")

    def __init__(self, serial_port, sensor_id: str = "temp", unit: str = "°C",
                 value_range=(-40.0, 85.0), **kwargs):
        super().__init__(sensor_id, unit, value_range, **kwargs)
        self.serial = serial_port

    def _read_raw(self) -> Optional[SensorReading]:
        line = self.serial.readline()
        if not line:
            return None
        text = line.decode("utf-8", errors="replace").strip()
        m = self.FRAME_RE.match(text)
        if not m:
            return SensorReading(self.sensor_id, float("nan"), self.unit,
                                 datetime.now(), valid=False, note=f"坏帧:{text!r}")
        return SensorReading(m.group(1), float(m.group(2)), m.group(3), datetime.now())


class FakeSerial:
    _BAD_FRAMES = (
        b"garbage without comma\r\n",
        b"\x00\xff\xfe junk bytes\r\n",
        b"temp,nan,C\r\n",
        b"temp,999,C\r\n",
        b"temp,25.3\r\n",
    )
    _GOOD_SPECS = {
        "temp": ((20.0, 30.0), "C"),
        "humi": ((40.0, 60.0), "RH"),
        "vib": ((0.5, 2.0), "g"),
    }

    def __init__(self, script: Optional[List[bytes]] = None,
                 bad_ratio: float = 0.3, seed: int = 42,
                 sensor_ids: Tuple[str, ...] = ("temp", "humi", "vib")):
        self._script = list(script) if script is not None else None
        self._cursor = 0
        self._good_cursor = 0
        self._rng = random.Random(seed)
        self.bad_ratio = bad_ratio
        self.sensor_ids = tuple(sensor_ids)

    def _good_frame(self) -> bytes:
        sid = self.sensor_ids[self._good_cursor % len(self.sensor_ids)]
        self._good_cursor += 1
        (lo, hi), unit = self._GOOD_SPECS.get(sid, ((20.0, 30.0), "C"))
        value = self._rng.uniform(lo, hi)
        return f"{sid},{value:.2f},{unit}\r\n".encode()

    def readline(self) -> bytes:
        if self._script is not None:
            if self._cursor >= len(self._script):
                return b""
            frame = self._script[self._cursor]
            self._cursor += 1
            return frame
        if self._rng.random() < self.bad_ratio:
            return self._rng.choice(self._BAD_FRAMES)
        return self._good_frame()

    def write(self, data): pass
    def open(self): pass
    def close(self): pass

    @property
    def is_open(self) -> bool:
        return True


def _config_source_type() -> str:
    import os
    env = os.environ.get("SENSOR_DATA_SOURCE", "").strip().lower()
    if env:
        return env
    try:
        from src.config import get_config
        return str(getattr(get_config(), "data_source", "simulated") or "simulated")
    except Exception:
        return "simulated"

def resolve_source_type() -> str:
    return _config_source_type()

def _open_serial_port():
    port, baud, timeout = "COM3", 9600, 1.0
    try:
        from src.config import get_config
        cfg = get_config()
        port = getattr(cfg, "serial_port", port)
        baud = getattr(cfg, "serial_baudrate", baud)
        timeout = getattr(cfg, "serial_timeout", timeout)
    except Exception:
        pass
    try:
        import serial
        ser = serial.Serial(port=port, baudrate=baud, timeout=timeout)
        try:
            ser.reset_input_buffer()
        except Exception:
            pass
        return ser
    except Exception as exc:
        logger.info("真实串口 %s 打开失败: %s", port, exc)
        return None

def get_datasource(source_type: Optional[str] = None, **kwargs) -> DataSource:
    if source_type is None:
        source_type = _config_source_type()

    if source_type == "serial":
        port = _open_serial_port()
        if port is None:
            logger.warning("真实串口不可用，降级为 FakeSerial 模拟数据源(B线兜底)")
            port = FakeSerial()
        return SerialSource(port, **kwargs)

    if source_type != "simulated":
        logger.warning("未知数据源配置 %r，回退 simulated", source_type)
    return SimulatedSource(**kwargs)

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    print("=== 演示 1: SimulatedSource ===")
    sim = SimulatedSource()
    for _ in range(3): print(sim.read()); time.sleep(0.05)

    print("\n=== 演示 2: SerialSource + FakeSerial ===")
    ser = SerialSource(FakeSerial(bad_ratio=0.4, seed=2026))
    for _ in range(8): print(ser.read()); time.sleep(0.05)