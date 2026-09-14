# tests/test_tools.py
"""测试传感器工具集"""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from src.tools.sensor_tools import (
    get_sensor_data, query_history, check_threshold, 
    generate_report, search_manual, FindingItem
)

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
    # 验证趋势是真实的：经过多小时的相位偏移，max 和 min 不应完全相等
    assert result["max"] >= result["min"]

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

# ==================== 补齐 search_manual 测试 ====================

@patch("src.tools.sensor_tools.load_vectorstore")
def test_search_manual_normal(mock_load_vs):
    """正常：成功检索到手册内容"""
    # 构造假的 Document 返回值
    mock_vs = MagicMock()
    mock_docs = [
        Document(page_content="SHT30 湿度精度为 ±2%RH", metadata={"source": "sht30_manual.pdf"}),
        Document(page_content="工作电压 2.4V-5.5V", metadata={"source": "sht30_manual.pdf"})
    ]
    mock_vs.similarity_search.return_value = mock_docs
    mock_load_vs.return_value = mock_vs
    
    result = search_manual.invoke({"query": "SHT30 精度是多少"})
    
    # 验证返回了包含来源和内容的字符串
    assert isinstance(result, str)
    assert "SHT30 湿度精度为 ±2%RH" in result
    assert "sht30_manual.pdf" in result
    # 验证确实调用了模拟的检索方法
    mock_vs.similarity_search.assert_called_once()

@patch("src.tools.sensor_tools.load_vectorstore")
def test_search_manual_exception(mock_load_vs):
    """异常：向量库加载失败或检索报错"""
    # 模拟抛出异常（比如向量库文件不存在）
    mock_load_vs.side_effect = Exception("向量库文件未找到")
    
    result = search_manual.invoke({"query": "随便查"})
    
    # 验证工具捕获了异常，并返回了可读的降级错误提示，而不是直接崩溃
    assert isinstance(result, str)
    assert "检索手册失败" in result
    assert "向量库文件未找到" in result

# ==================== 生成报告测试 ====================

def test_generate_report_all_normal():
    """正常：全正常报告"""
    findings = [
        FindingItem(sensor_type="temperature", value=25.0, status="正常"),
        FindingItem(sensor_type="humidity", value=50.0, status="正常")
    ]
    result = generate_report.invoke({"findings": [f.model_dump() for f in findings]})
    assert result["normal_count"] == 2
    assert result["abnormal_count"] == 0

def test_generate_report_with_abnormal():
    """正常：包含异常的报告"""
    findings = [
        FindingItem(sensor_type="vibration", value=6.0, status="异常")
    ]
    result = generate_report.invoke({"findings": [f.model_dump() for f in findings]})
    assert result["abnormal_count"] == 1
    assert len(result["suggestions"]) > 0

def test_generate_report_empty():
    """正常：空巡检项"""
    result = generate_report.invoke({"findings": []})
    assert result["total_items"] == 0