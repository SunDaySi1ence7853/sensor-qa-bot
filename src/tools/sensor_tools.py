"""
传感器工具集(第八阶段任务1 + 24h 留存)。
- 工具函数不再现场造数，统一走 src/datasource.py 数据源层。
- 历史 buffer 只由后台采样线程写入(每 SAMPLE_INTERVAL_SEC 一条)，工具函数只读不写，
  保证 "N 条 × 采样间隔 = 覆盖时长" 可以对账。
"""

import logging
import math   # noqa: F401
import random  # noqa: F401
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

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

# ---- 采样与留存的对账关系 ----
#   SAMPLE_INTERVAL_SEC × _HISTORY_MAX_POINTS = HISTORY_RETENTION_SEC
#          30 s        ×        2880         =  86400 s = 24 h
# 改采样间隔必须同步重算 maxlen，tests 里有一条用例锁死这个等式，改错就红。
SAMPLE_INTERVAL_SEC = 30.0
HISTORY_RETENTION_SEC = 86400
_HISTORY_MAX_POINTS = 2880
_history_buffer = {st: deque(maxlen=_HISTORY_MAX_POINTS) for st in SENSOR_SPECS}
_buffer_lock = threading.Lock()

# 模拟模式的历史回溯步长(与采样线程无关：模拟源可按相位回算任意时刻)
_HISTORY_SAMPLES_PER_HOUR = 4
_HISTORY_INTERVAL_SEC = 3600.0 / _HISTORY_SAMPLES_PER_HOUR

# 连续无效读数告警：每累计 10 次(10 × 30s = 5 分钟没有有效数据)warning 一次
_FAIL_WARN_EVERY = 10
_fail_streak: Dict[str, int] = {st: 0 for st in SENSOR_SPECS}

class _SensorHub:
    _ROUTE_TRIES = 4
    def __init__(self):
        self.source_type = resolve_source_type()
        self._serial: Optional[SerialSource] = None
        self._sims: Dict[str, SimulatedSource] = {}
        # 采样线程与 Agent/面板会并发读同一个串口，readline 必须串行
        self._lock = threading.Lock()

        if self.source_type == "serial":
            shared = get_datasource("serial", value_range=(None, None), max_retries=1)
            self._serial = shared
            if isinstance(shared.serial, FakeSerial):
                self.label = "simulated"
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
        with self._lock:
            return self._read_unlocked(sensor_type)

    def _read_unlocked(self, sensor_type: str) -> SensorReading:
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

    def close(self) -> None:
        if self._serial is not None:
            try: self._serial.serial.close()
            except Exception: pass

_hub: Optional[_SensorHub] = None
_hub_lock = threading.Lock()

def _get_hub() -> _SensorHub:
    # 双重检查：采样线程和 Streamlit 首次同时进来时，只允许建一个 hub(否则第二个打开 COM 口会失败被迫降级)
    global _hub
    if _hub is None:
        with _hub_lock:
            if _hub is None: _hub = _SensorHub()
    return _hub

def reset_sensor_hub() -> None:
    global _hub
    with _hub_lock:
        old, _hub = _hub, None
    if old is not None: old.close()

# ---------------- buffer 读写 ----------------
def _record(sensor_type: str, reading: SensorReading) -> bool:
    if not reading.valid: return False
    with _buffer_lock:
        _history_buffer[sensor_type].append((reading.timestamp, reading.value))
    return True

def _snapshot(sensor_type: str) -> List[Tuple[datetime, float]]:
    with _buffer_lock:
        return list(_history_buffer[sensor_type])

def _mark_streak(sensor_type: str, ok: bool, note: str) -> None:
    prev = _fail_streak.get(sensor_type, 0)
    if ok:
        if prev >= _FAIL_WARN_EVERY:
            logger.info("%s 采样已恢复(此前连续 %d 次无效)", sensor_type, prev)
        _fail_streak[sensor_type] = 0
        return
    _fail_streak[sensor_type] = prev + 1
    if _fail_streak[sensor_type] % _FAIL_WARN_EVERY == 0:
        logger.warning("%s 连续 %d 次无有效读数，链路可能已断: %s",
                       sensor_type, _fail_streak[sensor_type], note)

# ---------------- 采样逻辑(纯函数，与线程分离) ----------------
def sample_once(sensor_types: Optional[List[str]] = None) -> Dict[str, bool]:
    """读一次 → 判 valid → 落 buffer。不睡眠、不抛异常，返回各传感器本轮是否成功落地。"""
    types = list(sensor_types) if sensor_types else list(_VALID_TYPES)
    try:
        hub = _get_hub()
    except Exception as exc:
        logger.warning("采样失败，数据源初始化异常: %s", exc)
        for st in types: _mark_streak(st, False, f"数据源初始化异常:{exc}")
        return {st: False for st in types}

    result: Dict[str, bool] = {}
    for st in types:
        try:
            reading = hub.read(st)
            ok, note = _record(st, reading), reading.note
        except Exception as exc:
            logger.warning("采样 %s 读取异常: %s", st, exc)
            ok, note = False, f"读取异常:{exc}"
        _mark_streak(st, ok, note)
        result[st] = ok
    return result

# ---------------- 采样线程(只负责定时调用 sample_once) ----------------
class _Sampler:
    THREAD_NAME = "sensor-sampler"

    def __init__(self, interval_sec: float):
        self.interval_sec = interval_sec
        self._stop_evt = threading.Event()
        self.ticked = threading.Event()  # 每完成一轮置位，供测试/自检同步，避免 sleep 等待
        # daemon=True：Ctrl+C 停服务时线程随主进程退出，不会占着端口不放
        self.thread = threading.Thread(target=self._run, name=self.THREAD_NAME, daemon=True)

    def start(self) -> None: self.thread.start()
    def is_alive(self) -> bool: return self.thread.is_alive()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_evt.set()
        self.thread.join(timeout)

    def _run(self) -> None:
        logger.info("采样线程启动，间隔 %.0fs", self.interval_sec)
        while not self._stop_evt.is_set():
            try:
                sample_once()
            except Exception:  # sample_once 本身不抛，这里兜底防止线程意外死亡
                logger.exception("采样线程本轮异常，继续下一轮")
            self.ticked.set()
            self._stop_evt.wait(self.interval_sec)  # 可被 stop() 立即唤醒，不用 sleep
        logger.info("采样线程已停止")

_sampler: Optional[_Sampler] = None
_sampler_lock = threading.Lock()

def start_sampler(interval_sec: float = SAMPLE_INTERVAL_SEC) -> _Sampler:
    """幂等启动：已在运行就直接返回原线程。Streamlit 每次 rerun 调用也只有一个采样线程。"""
    global _sampler
    with _sampler_lock:
        if _sampler is not None and _sampler.is_alive():
            return _sampler
        _sampler = _Sampler(interval_sec)
        _sampler.start()
        return _sampler

def stop_sampler(timeout: float = 5.0) -> None:
    global _sampler
    with _sampler_lock:
        s, _sampler = _sampler, None
    if s is not None: s.stop(timeout)

def get_sampler() -> Optional[_Sampler]:
    return _sampler

# ---------------- Agent 工具 ----------------
@tool("get_sensor_data")
def get_sensor_data(sensor_type: str) -> dict:
    """获取指定传感器当前实时读数。当用户询问"当前温度/湿度是多少"时使用此工具。"""
    st = sensor_type.lower()
    if st not in SENSOR_SPECS: return {"error": f"不支持的传感器类型: {sensor_type}"}
    hub = _get_hub()
    reading = hub.read(st)  # 只读不写 buffer：buffer 由采样线程按固定间隔写，保证可对账
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
        cutoff = datetime.now() - timedelta(hours=hours)
        pts = [(ts, v) for (ts, v) in _snapshot(st) if ts >= cutoff]
        if not pts:
            return {"error": f"暂无 {st} 落地历史数据：采样线程每 {SAMPLE_INTERVAL_SEC:.0f}s 落地一条，请确认已调用 start_sampler()"}
        data = [v for _, v in pts]
        n = len(pts)
        span_sec = (pts[-1][0] - pts[0][0]).total_seconds()
        expected_sec = (n - 1) * SAMPLE_INTERVAL_SEC
        missing = max(0, round(span_sec / SAMPLE_INTERVAL_SEC) + 1 - n)
        note = (f"基于采样线程落地的 {n} 条读数(每 {SAMPLE_INTERVAL_SEC:.0f}s 一条)，"
                f"首末跨度 {span_sec / 60:.1f} 分钟；(N-1)×间隔 = {n - 1}×{SAMPLE_INTERVAL_SEC:.0f}s = {expected_sec / 60:.1f} 分钟")
        if missing:
            note += f"；差额对应期间跳过 {missing} 个无效采样点"
        data_points = n
    else:
        sim = hub._sims[st]; now = time.time()
        n = hours * _HISTORY_SAMPLES_PER_HOUR
        data = [sim.reading_at(now - i * _HISTORY_INTERVAL_SEC).value for i in range(n)]
        note = f"模拟数据源相位回溯：{n} 条 × {_HISTORY_INTERVAL_SEC / 60:.0f} 分钟 = 覆盖 {hours} 小时"
        data_points = n

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