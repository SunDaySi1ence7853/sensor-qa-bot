from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Callable, Optional

from .base import AgentFailure
from .schemas import CollectorInput, CollectorOutput

ROLE = """数据采集 Agent
职责：从第八阶段采样 buffer（唯一事实源）取诊断窗口内读数，如实报告数据质量。
不做：任何诊断判断、阈值比较、补值或插值。
失败信号：
- sensor_type 不支持 / window_hours <= 0 → 抛 AgentFailure
- buffer 为空或最新数据过旧 → data_quality=offline（不抛异常，交给 Supervisor 决定补采还是降级）
"""

# DHT22 物理量程，超出即视为坏帧
VALID_RANGE = {"temperature": (-40.0, 80.0), "humidity": (0.0, 100.0)}

Reader = Callable[[str], list]


def _default_reader(sensor_type: str) -> list:
    # 复用现有采样 buffer，不另起数据通路；list() 拷贝一份，避免遍历时被采样线程改动
    from src.tools.sensor_tools import _history_buffer
    return list(_history_buffer.get(sensor_type, []))


def parse_ts(ts) -> Optional[datetime]:
    if isinstance(ts, datetime):
        dt = ts
    elif isinstance(ts, (int, float)) and not isinstance(ts, bool):
        dt = datetime.fromtimestamp(ts)
    elif isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts)
        except ValueError:
            return None
    else:
        return None
    # 统一成本地 naive 时间，避免 aware/naive 混比抛 TypeError
    if dt.tzinfo is not None:
        dt = dt.astimezone().replace(tzinfo=None)
    return dt


def _valid_value(v, lo: float, hi: float) -> bool:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    return not math.isnan(v) and lo <= v <= hi


class CollectorAgent:
    name = "collector"

    def __init__(
        self,
        reader: Optional[Reader] = None,
        clock: Optional[Callable[[], datetime]] = None,
        interval_sec: float = 30.0,
        stale_factor: float = 3.0,
    ):
        self._reader = reader or _default_reader
        self._clock = clock or datetime.now
        self._interval = interval_sec
        self._stale_sec = stale_factor * interval_sec

    def run(self, inp: CollectorInput) -> CollectorOutput:
        sensor = inp.get("sensor_type")
        hours = inp.get("window_hours")
        if sensor not in VALID_RANGE:
            raise AgentFailure(self.name, f"不支持的传感器类型: {sensor!r}")
        if isinstance(hours, bool) or not isinstance(hours, (int, float)) or hours <= 0:
            raise AgentFailure(self.name, f"window_hours 必须 > 0，收到 {hours!r}")

        now = self._clock()
        start = now - timedelta(hours=hours)
        lo, hi = VALID_RANGE[sensor]

        frame_times: list[datetime] = []      # 窗口内所有时间戳合法的帧（含数值坏帧），用于算缺口
        valid: list[tuple[datetime, float]] = []
        bad = 0

        for raw in self._reader(sensor):
            ts = parse_ts(raw.get("timestamp")) if isinstance(raw, dict) else None
            if ts is None:
                bad += 1                       # 时间戳坏，无法归窗，按坏帧计
                continue
            if not (start <= ts <= now):
                continue
            frame_times.append(ts)
            value = raw.get("value")
            if not _valid_value(value, lo, hi):
                bad += 1
                continue
            valid.append((ts, float(value)))

        valid.sort(key=lambda r: r[0])
        missing = self._count_missing(sorted(frame_times))

        return {
            "readings": [{"timestamp": ts.isoformat(timespec="seconds"), "value": v} for ts, v in valid],
            "data_quality": self._judge(valid, missing, bad, now),
            "missing_count": missing,
            "bad_frame_count": bad,
        }

    def _count_missing(self, times: list[datetime]) -> int:
        # 相邻帧间隔折算成采样槽数，多出来的槽就是缺失点；坏帧有时间戳，不会被重复计为缺口
        missing = 0
        for a, b in zip(times, times[1:]):
            slots = round((b - a).total_seconds() / self._interval)
            if slots > 1:
                missing += slots - 1
        return missing

    def _judge(self, valid, missing: int, bad: int, now: datetime) -> str:
        if not valid:
            return "offline"
        if (now - valid[-1][0]).total_seconds() > self._stale_sec:
            return "offline"
        if missing or bad:
            return "degraded"
        return "ok"


if __name__ == "__main__":
    import json

    from .fakes import FIXED_NOW, make_readings

    clean = make_readings([23.6] * 8 + [23.7] * 2)
    dirty = make_readings([23.6, 23.7, None, 23.6, 999.0, 23.7, 23.6])
    del dirty[1]  # 人为制造一个缺口

    for label, rows in [("干净数据", clean), ("脏数据（缺口+坏帧）", dirty)]:
        agent = CollectorAgent(reader=lambda s, r=rows: r, clock=lambda: FIXED_NOW)
        out = agent.run({"sensor_type": "temperature", "window_hours": 1.0})
        print(f"===== {label} =====")
        print(json.dumps(out, ensure_ascii=False, indent=2))