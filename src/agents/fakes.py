"""fake 依赖：演示（python -m）和单测共用，不碰真串口、真 LLM。"""
from __future__ import annotations

from datetime import datetime, timedelta

FIXED_NOW = datetime(2026, 10, 8, 20, 0, 0)

# 演示用的假手册片段，不是真实知识库内容
FAKE_MANUAL = [
    "DHT22 测温范围 -40~80°C，精度 ±0.5°C；读数相邻两点跳变超过 5°C 通常指示接线松动或传感器故障。",
    "机柜温度持续上升的常见原因：散热风扇停转、通风口堵塞、负载升高。处置：检查风扇运转与通风口，必要时降载。",
]


def make_readings(values, end: datetime = FIXED_NOW, interval: float = 30.0) -> list[dict]:
    """按固定间隔生成 buffer 格式的读数，最后一条时间为 end。"""
    n = len(values)
    return [
        {
            "timestamp": (end - timedelta(seconds=interval * (n - 1 - i))).isoformat(timespec="seconds"),
            "value": v,
        }
        for i, v in enumerate(values)
    ]


class FakeLLM:
    """记录每次调用的 (system, user)，返回预设文本。"""

    def __init__(self, response: str):
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.response


class FakeRetriever:
    """记录查询词，返回预设文档。"""

    def __init__(self, docs):
        self.docs = list(docs)
        self.queries: list[str] = []

    def __call__(self, query: str) -> list[str]:
        self.queries.append(query)
        return list(self.docs)
# ── 带时间洞的读数（degraded 场景专用）──────────────────────────────
from datetime import datetime as _dt, timedelta as _td


def _shift_ts(ts, sec: float):
    if isinstance(ts, _dt):
        return ts - _td(seconds=sec)
    if isinstance(ts, (int, float)):
        return ts - sec
    if isinstance(ts, str):
        shifted = _dt.fromisoformat(ts) - _td(seconds=sec)
        return shifted.isoformat(timespec="seconds") if "." not in ts else shifted.isoformat()
    raise TypeError(f"不认识的时间戳类型：{type(ts)}")


def make_gapped_readings(values, gap_after: int = 5, gap_sec: float = 300.0):
    """构造帧间有洞的读数：前 gap_after 帧整体往前平移 gap_sec 秒。

    degraded 的判定依据是"帧间有洞"，不是"帧少"（见 test_gap_is_counted）。
    往前平移而不是往后，是为了保证最后一帧仍贴着 FIXED_NOW，不会被窗口裁掉。
    """
    rows = [dict(r) for r in make_readings(values)]
    key = next(k for k in ("timestamp", "ts", "time") if k in rows[0])
    for r in rows[:gap_after]:
        r[key] = _shift_ts(r[key], gap_sec)
    return rows