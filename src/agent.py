# src/agent.py
"""
ReAct Agent 模块。
使用 LangGraph 的预置 Agent，将工具集与 LLM 绑定，实现自主决策。
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent

from src.llm import get_llm
from src.tools.sensor_tools import (
    get_sensor_data,
    query_history,
    check_threshold,
    search_manual,
    generate_report
)

# 系统提示词：定义 Agent 角色与决策纪律
SYSTEM_PROMPT = """你是一个专业的工业传感器监控与诊断助手。

你的核心能力包括：
1. 查询当前传感器实时读数、历史趋势及进行安全阈值校验。
2. 检索传感器硬件手册，解答规格、工作原理及故障排查问题。
3. 汇总生成结构化巡检报告。

【决策纪律 - 必须严格遵守】
- 当用户询问当前读数或温度/湿度是否正常时，必须先调用 `get_sensor_data` 获取数据，紧接着调用 `check_threshold` 校验。
- 当用户询问历史趋势或异常处理时，调用 `query_history`；如果发现异常（is_alert=True），必须接着调用 `search_manual` 查阅处理方案。
- 当用户询问硬件规格（如工作电压、封装尺寸、适用场景）时，这是纯知识问题，必须调用 `search_manual`，**严禁滥用数据查询工具**。
- 当要求生成报告时，需先收集各传感器的状态，最后调用 `generate_report`。

请一步步思考，决定调用哪些工具以及调用的顺序，最终给出清晰的综合回答。
"""

def get_sensor_agent():
    """获取组装好的 ReAct Agent 实例。"""
    llm = get_llm(streaming=False)
    tools = [
        get_sensor_data,
        query_history,
        check_threshold,
        search_manual,
        generate_report
    ]
    agent = create_react_agent(
        llm,
        tools,
        prompt=SYSTEM_PROMPT
    )
    return agent