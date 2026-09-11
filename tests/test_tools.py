# tests/test_tools.py
"""测试传感器工具集"""
import pytest
from src.tools.sensor_tools import get_sensor_data, query_history, check_threshold, generate_report, FindingItem

def test_get_sensor_data_normal():
    """正常获取温度"""
    result = get_sensor_data.invoke({"sensor_type": "temperature"})
    assert "value" in result
    assert result["unit"] == "°C"
    assert 15.0 < result["value"] < 35.0  # 宽松断言范围

def test_get_sensor_data_invalid_type():
    """异常：不支持的传感器类型"""
    result = get_sensor_data.invoke({"sensor_type": "pressure"})
    assert "error" in result
    assert "不支持" in result["error"]

def test_query_history_normal():
    """正常查询历史"""
    result = query_history.invoke({"sensor_type": "humidity", "hours": 5})
    assert result["hours"] == 5
    assert "mean" in result

def test_query_history_invalid_hours():
    """异常：时间超范围"""
    result = query_history.invoke({"sensor_type": "humidity", "hours": 99})
    assert "error" in result

def test_check_threshold_normal():
    """正常：温度在范围内"""
    result = check_threshold.invoke({"sensor_type": "temperature", "value": 25.0})
    assert result["is_breached"] is False

def test_check_threshold_breach():
    """异常：温度超上限"""
    result = check_threshold.invoke({"sensor_type": "temperature", "value": 40.0})
    assert result["is_breached"] is True
    assert "超出安全范围" in result["message"]

def test_check_threshold_unconfigured():
    """异常：未配置阈值的传感器"""
    result = check_threshold.invoke({"sensor_type": "pressure", "value": 10.0})
    assert "error" in result
    assert "未配置" in result["error"]

def test_generate_report_all_normal():
    """正常：全正常报告"""
    findings = [
        FindingItem(sensor_type="temperature", value=25.0, status="正常"),
        FindingItem(sensor_type="humidity", value=50.0, status="正常")
    ]
    result = generate_report.invoke({"findings": [f.dict() for f in findings]})
    assert result["normal_count"] == 2
    assert result["abnormal_count"] == 0

def test_generate_report_with_abnormal():
    """正常：包含异常的报告"""
    findings = [
        FindingItem(sensor_type="vibration", value=6.0, status="异常")
    ]
    result = generate_report.invoke({"findings": [f.dict() for f in findings]})
    assert result["abnormal_count"] == 1
    assert len(result["suggestions"]) > 0

def test_generate_report_empty():
    """正常：空巡检项"""
    result = generate_report.invoke({"findings": []})
    assert result["total_items"] == 0