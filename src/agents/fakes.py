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