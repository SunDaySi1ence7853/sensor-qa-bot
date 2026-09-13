# src/tools/sensor_tools.py
"""
传感器工具集。
提供给 Agent 调用的 5 个核心工具，用于数据采集、历史查询、阈值校验、手册检索和报告生成。
"""

import math
import random
from datetime import datetime, timedelta
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from src.config import get_config
from src.vectorstore import load_vectorstore


# ==================== 数据模型 ====================
class FindingItem(BaseModel):
    sensor_type: str = Field(description="传感器类型，如 temperature/humidity")
    value: float = Field(description="读数")
    status: str = Field(description="状态：正常/异常")

class ReportInput(BaseModel):
    findings: List[FindingItem] = Field(description="巡检发现项列表")

# ==================== 工具实现 ====================

@tool("get_sensor_data")
def get_sensor_data(sensor_type: str) -> dict:
    """
    获取指定传感器当前实时读数。
    当用户询问"当前温度/湿度是多少"时使用此工具。
    
    Args:
        sensor_type (str): 传感器类型，支持 temperature(温度,单位°C), humidity(湿度,单位%), vibration(振动,单位g)
        
    Returns:
        dict: 包含传感器类型、当前值、单位和时间戳的字典
    """
    valid_types = ["temperature", "humidity", "vibration"]
    if sensor_type.lower() not in valid_types:
        return {"error": f"不支持的传感器类型: {sensor_type}。支持: {valid_types}"}
    
    # 模拟数据：量程合理 + 正弦波趋势 + 随机噪声
    # 利用当前秒数作为相位，模拟连续变化
    t = datetime.now().second / 60.0
    noise = random.uniform(-0.5, 0.5)
    
    if sensor_type.lower() == "temperature":
        # 温度范围 20-28°C
        value = 24.0 + 4.0 * math.sin(2 * math.pi * t) + noise
    elif sensor_type.lower() == "humidity":
        # 湿度范围 45-55%
        value = 50.0 + 5.0 * math.sin(2 * math.pi * t) + noise
    else: # vibration
        # 振动范围 0.5-2.0g
        value = 1.25 + 0.75 * math.sin(2 * math.pi * t) + noise
    
    return {
        "sensor_type": sensor_type.lower(),
        "value": round(value, 2),
        "unit": {"temperature": "°C", "humidity": "%", "vibration": "g"}[sensor_type.lower()],
        "timestamp": datetime.now().isoformat()
    }

@tool("query_history")
def query_history(sensor_type: str, hours: int) -> dict:
    """
    查询指定传感器过去几小时的历史趋势摘要。
    当用户询问"最近几小时的数据趋势/历史"时使用此工具。
    
    Args:
        sensor_type (str): 传感器类型 (temperature/humidity/vibration)
        hours (int): 查询的小时数，范围 1-24
        
    Returns:
        dict: 包含均值、最大值、最小值、是否超限的摘要字典
    """
    if not isinstance(hours, int) or hours < 1 or hours > 24:
        return {"error": "hours 参数必须为 1-24 之间的整数"}
    
    # 模拟历史数据统计
    data = [get_sensor_data.invoke({"sensor_type": sensor_type})["value"] for _ in range(hours)]
    
    cfg = get_config()
    threshold = cfg.sensor_thresholds.get(sensor_type.lower(), {})
    is_alert = False
    if threshold:
        is_alert = max(data) > threshold.get("max", float('inf')) or min(data) < threshold.get("min", float('-inf'))
        
    return {
        "sensor_type": sensor_type.lower(),
        "hours": hours,
        "mean": round(sum(data) / len(data), 2),
        "max": round(max(data), 2),
        "min": round(min(data), 2),
        "is_alert": is_alert
    }

@tool("check_threshold")
def check_threshold(sensor_type: str, value: float) -> dict:
    """
    判断指定传感器的读数是否超出了安全阈值。
    当拿到传感器读数后，必须使用此工具判断是否安全。
    
    Args:
        sensor_type (str): 传感器类型 (temperature/humidity/vibration)
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
        "abnormal_details": [f.dict() for f in abnormal_items],
        "suggestions": suggestions
    }