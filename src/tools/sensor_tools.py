"""
传感器工具集(第八阶段任务1)。
工具函数不再现场造数，统一走 src/datasource.py 数据源层。
"""

import logging
import math   # noqa: F401
import random  # noqa: F401
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from pydantic import BaseModel, Field
from langchain_core.tools import tool

from src.config import get_config
from src.datasource import (
    FakeSerial, SensorReading, SerialSource, SimulatedSource,
    get_datasource, resolve_source_type,
)
from src.vectorstore import load_vectorstore

logger = logging.getLogger(__name__)

class FindingItem(BaseModel):
    sensor_type: str = Field(description="传感器类型，如 temperature/humidity")
    value: float = Field(description="读数")
    status: str = Field(description="状态：正常/异常")

class ReportInput(BaseModel):
    findings: List[FindingItem] = Field(description="巡检发现项列表")

SENSOR_SPECS = {
    "temperature": {"sensor_id": "temp", "unit": "°C", "base": 24.0, "amplitude": 4.0, "noise": 0.5, "period_sec": 3600.0, "value_range": (-40.0, 85.0)},
    "humidity": {"sensor_id": "humi", "unit": "%", "base": 50.0, "amplitude": 5.0, "noise": 0.5, "period_sec": 3600.0, "value_range": (0.0, 100.0)},
    "vibration": {"sensor_id": "vib", "unit": "g", "base": 1.25, "amplitude": 0.75, "noise": 0.5, "period_sec": 3600.0, "value_range": (0.0, 16.0)},
}
_VALID_TYPES = list(SENSOR_SPECS)
_HISTORY_MAX_POINTS = 2880
_history_buffer = {st: deque(maxlen=_HISTORY_MAX_POINTS) for st in SENSOR_SPECS}
_HISTORY_SAMPLES_PER_HOUR = 4
_HISTORY_INTERVAL_SEC = 3600.0 / _HISTORY_SAMPLES_PER_HOUR

class _SensorHub:
    _ROUTE_TRIES = 4
    def __init__(self):
        self.source_type = resolve_source_type()
        self._serial: Optional[SerialSource] = None
        self._sims: Dict[str, SimulatedSource] = {}

        if self.source_type == "serial":
            shared = get_datasource("serial", value_range=(None, None), max_retries=1)
            self._serial = shared
            if isinstance(shared.serial, FakeSerial):
                self.label = "serial(降级模拟)"
                logger.warning("真实串口不可用，降级为 FakeSerial")
            else:
                self.label = "serial"
        else:
            self.label = "simulated"

        for st, spec in SENSOR_SPECS.items():
            self._sims[st] = SimulatedSource(
                sensor_id=spec["sensor_id"], unit=spec["unit"], base=spec["base"],
                amplitude=spec["amplitude"], noise=spec["noise"], period_sec=spec["period_sec"],
                value_range=spec["value_range"],
            )

    @property
    def is_serial(self) -> bool: return self._serial is not None

    def read(self, sensor_type: str) -> SensorReading:
        spec = SENSOR_SPECS[sensor_type]
        if self._serial is None: return self._sims[sensor_type].read()

        last_note = "串口无数据"
        for _ in range(self._ROUTE_TRIES):
            r = self._serial.read()
            if not r.valid: last_note = r.note or "无数据"; continue
            if r.sensor_id != spec["sensor_id"]:
                last_note = f"串口帧来自 {r.sensor_id}，非目标 {spec['sensor_id']}"; continue
            lo, hi = spec["value_range"]
            if not (lo <= r.value <= hi): last_note = f"读数 {r.value} 超出量程"; continue
            return r
        return SensorReading(
            sensor_id=spec["sensor_id"], value=float("nan"), unit=spec["unit"],
            timestamp=datetime.now(), valid=False, note=f"串口未见 {spec['sensor_id']} 帧({last_note})"
        )

_hub: Optional[_SensorHub] = None
def _get_hub() -> _SensorHub:
    global _hub
    if _hub is None: _hub = _SensorHub()
    return _hub

def reset_sensor_hub() -> None:
    global _hub; _hub = None

def _record(sensor_type: str, reading: SensorReading) -> None:
    if reading.valid: _history_buffer[sensor_type].append((reading.timestamp, reading.value))

@tool("get_sensor_data")
def get_sensor_data(sensor_type: str) -> dict:
    """获取指定传感器当前实时读数。当用户询问"当前温度/湿度是多少"时使用此工具。"""
    st = sensor_type.lower()
    if st not in SENSOR_SPECS: return {"error": f"不支持的传感器类型: {sensor_type}"}
    hub = _get_hub()
    reading = hub.read(st)
    _record(st, reading)
    if not reading.valid: return {"error": f"读取失败: {reading.note}", "sensor_type": st}
    return {"sensor_type": st, "value": reading.value, "unit": SENSOR_SPECS[st]["unit"], "timestamp": reading.timestamp.isoformat(), "data_source": hub.label}

@tool("query_history")
def query_history(sensor_type: str, hours: int) -> dict:
    """查询指定传感器过去几小时的历史趋势摘要。当用户询问"最近几小时的数据趋势"时使用。"""
    if not isinstance(hours, int) or hours < 1 or hours > 24: return {"error": "hours 参数必须为 1-24 之间"}
    st = sensor_type.lower()
    if st not in SENSOR_SPECS: return {"error": f"不支持的传感器类型: {sensor_type}"}
    hub = _get_hub()
    if hub.is_serial:
        buf = _history_buffer[st]
        cutoff = datetime.now() - timedelta(hours=hours)
        pts = [(ts, v) for (ts, v) in buf if ts >= cutoff]
        if not pts: return {"error": f"暂无 {st} 落地历史数据，请先调用 get_sensor_data 采样"}
        data = [v for _, v in pts]; span_sec = (pts[-1][0] - pts[0][0]).total_seconds()
        note = f"基于已录得 {len(pts)} 条真实读数，覆盖 {span_sec / 60:.1f} 分钟"
        data_points = len(pts)
    else:
        sim = hub._sims[st]; now = time.time()
        n = hours * _HISTORY_SAMPLES_PER_HOUR
        data = [sim.reading_at(now - i * _HISTORY_INTERVAL_SEC).value for i in range(n)]
        note = "模拟数据源相位回溯"; data_points = n

    cfg = get_config(); threshold = cfg.sensor_thresholds.get(st, {})
    is_alert = False
    if threshold: is_alert = max(data) > threshold.get("max", float('inf')) or min(data) < threshold.get("min", float('-inf'))
    return {"sensor_type": st, "hours": hours, "mean": round(sum(data)/len(data), 2), "max": round(max(data), 2), "min": round(min(data), 2), "is_alert": is_alert, "data_points": data_points, "data_source": hub.label, "note": note}

@tool("check_threshold")
def check_threshold(sensor_type: str, value: float) -> dict:
    """判断指定传感器的读数是否超出了安全阈值。拿到读数后必须使用此工具判断。"""
    cfg = get_config(); sensor_key = sensor_type.lower(); threshold = cfg.sensor_thresholds.get(sensor_key)
    if not threshold: return {"error": f"未配置传感器 {sensor_key} 的阈值"}
    min_v, max_v = threshold["min"], threshold["max"]; is_breach = value < min_v or value > max_v
    return {"sensor_type": sensor_key, "value": value, "is_breached": is_breach, "threshold_min": min_v, "threshold_max": max_v, "message": f"读数 {value} 超出安全范围 [{min_v}, {max_v}]" if is_breach else "读数在安全范围内"}

@tool("search_manual")
def search_manual(query: str) -> str:
    """检索传感器知识库手册，获取传感器规格、工作电压等文档信息。遇到硬件规格问答时使用。"""
    try:
        vectorstore = load_vectorstore()
        docs = vectorstore.similarity_search(query, k=get_config().retrieve_top_k)
        if not docs: return "未检索到相关手册内容"
        return "\n\n".join([f"来源: {d.metadata.get('source', '未知')}\n内容: {d.page_content}" for d in docs])
    except Exception as e: return f"检索手册失败: {str(e)}"

@tool("generate_report")
def generate_report(findings: List[FindingItem]) -> dict:
    """根据巡检发现项生成结构化的巡检报告。完成多个传感器检查后汇总输出。"""
    abnormal_items = [f for f in findings if f.status == "异常"]
    suggestions = []
    if abnormal_items: suggestions.append("建议立即检查异常传感器及相关硬件连线")
    if all(f.status == "正常" for f in findings): suggestions.append("所有设备运行正常，建议保持当前监控频率")
    return {"report_title": "传感器巡检报告", "inspection_time": datetime.now().isoformat(), "total_items": len(findings), "normal_count": len(findings) - len(abnormal_items), "abnormal_count": len(abnormal_items), "abnormal_details": [f.model_dump() for f in abnormal_items], "suggestions": suggestions}