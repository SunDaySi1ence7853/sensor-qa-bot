# src/tools/sensor_tools.py
"""
传感器工具集。
提供给 Agent 调用的 5 个核心工具，用于数据采集、历史查询、阈值校验、手册检索和报告生成。

第八阶段·任务1 改造：工具函数不再现场造数，统一走 src/datasource.py 数据源层——
- config.yaml 的 data_source(simulated|serial)一行切换 模拟/真实硬件
- 串口模式：全部传感器共享一个串口，按协议帧 sensor_id 路由 + 各自量程校验；
  查无该传感器帧时如实报错，禁止编造数值
- 历史查询：模拟模式由数据源模型相位回溯(与实时读数同源可对账)；
  串口模式统计真实录得数据(内存环形缓冲，任务3升级 SQLite 24h 留存)，
  数据不足时在 note 中如实说明
"""

import logging
import math   # noqa: F401  保留:兼容第六阶段测试可能的模块级 patch
import random  # noqa: F401  保留:兼容第六阶段测试可能的模块级 patch
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from pydantic import BaseModel, Field
from langchain_core.tools import tool

from src.config import get_config
from src.datasource import (
    FakeSerial,
    SensorReading,
    SerialSource,
    SimulatedSource,
    get_datasource,
    resolve_source_type,
)
from src.vectorstore import load_vectorstore

logger = logging.getLogger(__name__)


# ==================== 数据模型 ====================
class FindingItem(BaseModel):
    sensor_type: str = Field(description="传感器类型，如 temperature/humidity")
    value: float = Field(description="读数")
    status: str = Field(description="状态：正常/异常")

class ReportInput(BaseModel):
    findings: List[FindingItem] = Field(description="巡检发现项列表")


# ==================== 数据源接入(第八阶段) ====================

SENSOR_SPECS = {
    # 曲线参数与第六阶段模拟逻辑完全一致(基线/幅度/噪声)，
    # 第六阶段的取值区间类断言不受影响
    "temperature": {
        "sensor_id": "temp", "unit": "°C",
        "base": 24.0, "amplitude": 4.0, "noise": 0.5,
        "period_sec": 3600.0, "value_range": (-40.0, 85.0),
    },
    "humidity": {
        "sensor_id": "humi", "unit": "%",
        "base": 50.0, "amplitude": 5.0, "noise": 0.5,
        "period_sec": 3600.0, "value_range": (0.0, 100.0),
    },
    "vibration": {
        "sensor_id": "vib", "unit": "g",
        "base": 1.25, "amplitude": 0.75, "noise": 0.5,
        "period_sec": 3600.0, "value_range": (0.0, 16.0),
    },
}

_VALID_TYPES = list(SENSOR_SPECS)

# 内存环形缓冲：串口模式真实读数落地(任务3将升级为 SQLite，留存 ≥24h)
_HISTORY_MAX_POINTS = 2880   # 24h × 每 30s 一条的上限，超出自动淘汰最旧
_history_buffer = {st: deque(maxlen=_HISTORY_MAX_POINTS) for st in SENSOR_SPECS}

# 模拟模式历史采样：每小时 4 个点(15 分钟一个)。
# 采样间隔 ≠ 正弦周期(1h)，避免整周期采样导致历史变成一条直线
_HISTORY_SAMPLES_PER_HOUR = 4
_HISTORY_INTERVAL_SEC = 3600.0 / _HISTORY_SAMPLES_PER_HOUR


class _SensorHub:
    """数据源网关：按当前配置为全部工具供数(进程内单例，见 _get_hub)。

    - simulated:每类传感器一个 SimulatedSource
    - serial:共享一个串口(协议帧 sensor_id 自描述)，本层负责 id 路由
      + 各传感器量程校验；真实串口不可用时自动降级 FakeSerial 并如实标注
    """

    _ROUTE_TRIES = 4   # 串口路由最多读几轮寻找目标传感器的帧

    def __init__(self):
        self.source_type = resolve_source_type()
        self._serial: Optional[SerialSource] = None
        self._sims: Dict[str, SimulatedSource] = {}

        if self.source_type == "serial":
            # 共享串口不做统一量程过滤(各传感器量程不同)，由 read() 分类型校验
            shared = get_datasource("serial", value_range=(None, None), max_retries=1)
            self._serial = shared
            if isinstance(shared.serial, FakeSerial):
                self.label = "serial(降级模拟)"
                logger.warning("真实串口不可用，串口模式降级为 FakeSerial(输出会如实标注)")
            else:
                self.label = "serial"
                logger.info("串口数据源已接通: %s", type(shared.serial).__name__)
        else:
            self.label = "simulated"

        for st, spec in SENSOR_SPECS.items():
            self._sims[st] = SimulatedSource(
                sensor_id=spec["sensor_id"], unit=spec["unit"],
                base=spec["base"], amplitude=spec["amplitude"],
                noise=spec["noise"], period_sec=spec["period_sec"],
                value_range=spec["value_range"],
            )

    @property
    def is_serial(self) -> bool:
        return self._serial is not None

    def read(self, sensor_type: str) -> SensorReading:
        """读一类传感器：模拟源直读；串口按帧内 sensor_id 路由。"""
        spec = SENSOR_SPECS[sensor_type]

        if self._serial is None:
            return self._sims[sensor_type].read()

        last_note = "串口无数据"
        for _ in range(self._ROUTE_TRIES):
            r = self._serial.read()   # 自带坏帧剔除 + 重试
            if not r.valid:
                last_note = r.note or "无数据"
                continue
            if r.sensor_id != spec["sensor_id"]:
                last_note = f"串口帧来自 {r.sensor_id}，非目标 {spec['sensor_id']}"
                continue
            lo, hi = spec["value_range"]
            if not (lo <= r.value <= hi):
                last_note = f"{spec['sensor_id']} 读数 {r.value} 超出量程 [{lo}, {hi}]"
                continue
            return r

        return SensorReading(
            sensor_id=spec["sensor_id"], value=float("nan"), unit=spec["unit"],
            timestamp=datetime.now(), valid=False,
            note=f"串口数据流中未见 {spec['sensor_id']} 帧({last_note})；当前硬件可能未接该传感器",
        )


_hub: Optional[_SensorHub] = None


def _get_hub() -> _SensorHub:
    global _hub
    if _hub is None:
        _hub = _SensorHub()
    return _hub


def reset_sensor_hub() -> None:
    """重置数据源网关。切换 data_source 配置后重启进程生效；
    Web 端热切换(任务2)可直接调用本函数。"""
    global _hub
    _hub = None


def _record(sensor_type: str, reading: SensorReading) -> None:
    """有效读数落地到环形缓冲，供串口模式历史查询。"""
    if reading.valid:
        _history_buffer[sensor_type].append((reading.timestamp, reading.value))


# ==================== 工具实现 ====================

@tool("get_sensor_data")
def get_sensor_data(sensor_type: str) -> dict:
    """
    获取指定传感器当前实时读数。
    当用户询问"当前温度/湿度是多少"时使用此工具。
    数据来自数据源层(模拟或真实串口，见返回的 data_source 字段)。
    若返回 error(如硬件未接该传感器或读取失败)，必须如实告知用户，禁止编造数值。

    Args:
        sensor_type (str): 传感器类型，支持 temperature(温度,单位°C), humidity(湿度,单位%), vibration(振动,单位g)

    Returns:
        dict: 包含传感器类型、当前值、单位、时间戳与数据来源的字典
    """
    st = sensor_type.lower()
    if st not in SENSOR_SPECS:
        return {"error": f"不支持的传感器类型: {sensor_type}。支持: {_VALID_TYPES}"}

    hub = _get_hub()
    reading = hub.read(st)
    _record(st, reading)

    if not reading.valid:
        return {"error": f"传感器读取失败: {reading.note}", "sensor_type": st}

    return {
        "sensor_type": st,
        "value": reading.value,
        "unit": SENSOR_SPECS[st]["unit"],
        "timestamp": reading.timestamp.isoformat(),
        "data_source": hub.label,
    }

@tool("query_history")
def query_history(sensor_type: str, hours: int) -> dict:
    """
    查询指定传感器过去几小时的历史趋势摘要。
    当用户询问"最近几小时的数据趋势/历史"时使用此工具。
    串口模式下统计真实录得数据，数据不足时会在 note 中如实说明；
    若返回 error(暂无落地数据)，如实告知用户，禁止编造历史。

    Args:
        sensor_type (str): 传感器类型 (temperature/humidity/vibration)

        hours (int): 查询的小时数，范围 1-24

    Returns:
        dict: 包含均值、最大值、最小值、是否超限的摘要字典
    """
    if not isinstance(hours, int) or hours < 1 or hours > 24:
        return {"error": "hours 参数必须为 1-24 之间的整数"}

    st = sensor_type.lower()
    if st not in SENSOR_SPECS:
        return {"error": f"不支持的传感器类型: {sensor_type}"}

    hub = _get_hub()

    if hub.is_serial:
        # 串口模式：只统计真实录得的数据，不编造
        buf = _history_buffer[st]
        cutoff = datetime.now() - timedelta(hours=hours)
        pts = [(ts, v) for (ts, v) in buf if ts >= cutoff]
        if not pts:
            return {
                "error": (
                    f"暂无 {st} 的落地历史数据：串口模式的历史随实时读取逐步积累"
                    f"(可先调用 get_sensor_data 采样；24h 持久化将在任务3接入)"
                ),
                "sensor_type": st, "hours": hours,
            }
        data = [v for _, v in pts]
        span_sec = (pts[-1][0] - pts[0][0]).total_seconds()
        note = (
            f"基于已录得 {len(pts)} 条真实读数，覆盖 {span_sec / 60:.1f} 分钟"
            + ("(不足请求时长，统计仅覆盖已录得部分)" if span_sec < hours * 3600 * 0.99 else "")
        )
        data_points = len(pts)
    else:
        # 模拟模式：数据源模型相位回溯，与实时读数同源、可对账
        sim = hub._sims[st]
        now = time.time()
        n = hours * _HISTORY_SAMPLES_PER_HOUR
        data = [sim.reading_at(now - i * _HISTORY_INTERVAL_SEC).value for i in range(n)]
        note = "模拟数据源相位回溯，与实时读数同源"
        data_points = n

    cfg = get_config()
    threshold = cfg.sensor_thresholds.get(st, {})
    is_alert = False
    if threshold:
        is_alert = max(data) > threshold.get("max", float('inf')) or min(data) < threshold.get("min", float('-inf'))

    return {
        "sensor_type": st,
        "hours": hours,
        "mean": round(sum(data) / len(data), 2),
        "max": round(max(data), 2),
        "min": round(min(data), 2),
        "is_alert": is_alert,
        "data_points": data_points,
        "data_source": hub.label,
        "note": note,
    }

@tool("check_threshold")
def check_threshold(sensor_type: str, value: float) -> dict:
    """
    判断指定传感器的读数是否超出了安全阈值。
    当拿到传感器读数后，必须使用此工具判断是否安全。

    Args:
        sensor_type (str): 传感器类型

        value (float): 待校验的读数值

    Returns:
        dict: 包含是否超限、阈值范围和当前值的字典
    """
    cfg = get_config()
    sensor_key = sensor_type.lower()
    threshold = cfg.sensor_thresholds.get(sensor_key)

    if not threshold:
        return {"error": f"未配置传感器 {sensor_key} 的阈值信息"}

    min_v, max_v = threshold["min"], threshold["max"]
    is_breach = value < min_v or value > max_v

    return {
        "sensor_type": sensor_key,
        "value": value,
        "is_breached": is_breach,
        "threshold_min": min_v,
        "threshold_max": max_v,
        "message": f"读数 {value} 超出安全范围 [{min_v}, {max_v}]" if is_breach else "读数在安全范围内"
    }

@tool("search_manual")
def search_manual(query: str) -> str:
    """
    检索传感器知识库手册，获取传感器规格、工作电压、封装尺寸、故障处理等文档信息。
    当遇到硬件规格、原理、故障排查方案等知识问答时使用此工具。

    Args:
        query (str): 检索关键词或问题

    Returns:
        str: 检索到的相关文档内容片段
    """
    try:
        vectorstore = load_vectorstore()
        docs = vectorstore.similarity_search(query, k=get_config().retrieve_top_k)
        if not docs:
            return "未检索到相关手册内容"
        return "\n\n".join([f"来源: {d.metadata.get('source', '未知')}\n内容: {d.page_content}" for d in docs])
    except Exception as e:
        return f"检索手册失败: {str(e)}。请确保已运行 sensor-qa-build 构建向量库。"

@tool("generate_report")
def generate_report(findings: List[FindingItem]) -> dict:
    """
    根据巡检发现项生成结构化的巡检报告。
    当完成多个传感器检查后，使用此工具汇总输出报告。

    Args:
        findings (List[FindingItem]): 巡检发现项列表，包含传感器类型、数值和状态

    Returns:
        dict: 结构化巡检报告
    """
    abnormal_items = [f for f in findings if f.status == "异常"]
    suggestions = []
    if abnormal_items:
        suggestions.append("建议立即检查异常传感器及相关硬件连线")
    if all(f.status == "正常" for f in findings):
        suggestions.append("所有设备运行正常，建议保持当前监控频率")

    return {
        "report_title": "传感器巡检报告",
        "inspection_time": datetime.now().isoformat(),
        "total_items": len(findings),
        "normal_count": len(findings) - len(abnormal_items),
        "abnormal_count": len(abnormal_items),
        "abnormal_details": [f.model_dump() for f in abnormal_items],
        "suggestions": suggestions
    }