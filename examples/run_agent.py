# examples/run_agent.py
"""
ReAct Agent 交互演示。打印完整的思考→行动→观察链路。
"""
import os
import sys

# 将项目根目录加入 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agent import get_sensor_agent

def run_test_case(agent, question: str):
    """单轮测试，打印中间步骤"""
    print(f"\n{'='*60}")
    print(f"👤 用户提问: {question}")
    print(f"{'='*60}")
    
    inputs = {"messages": [("user", question)]}
    
    # 流式输出中间过程
    for chunk in agent.stream(inputs, {"recursion_limit": 15}):
        # LangGraph 的 stream 会按节点输出
        for node_name, state in chunk.items():
            if node_name == "agent":
                # 获取 agent 节点的最后一条消息
                last_msg = state["messages"][-1]
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    tools = [tc["name"] for tc in last_msg.tool_calls]
                    print(f"🧠 Thought & Action: 我需要调用工具 {tools}")
                elif last_msg.content:
                    print(f"✅ Final Answer: {last_msg.content}\n")
            elif node_name == "tools":
                # 获取 tools 节点的执行结果
                for msg in state["messages"]:
                    if msg.type == "tool":
                        print(f"👀 Observation: 工具返回 {msg.content[:200]}...") # 截断过长的输出

def main():
    agent = get_sensor_agent()
    
    # 任务4的5道复合大考题
    test_cases = [
        "当前车间温度正常吗？",
        "最近 2 小时湿度有什么趋势？异常的话查查手册怎么处理",
        "帮我出一份当前所有传感器的巡检报告",
        "DHT22 和 BMP280 哪个适合测量管道内气体压力？",
        "温度传感器读数异常，帮我诊断一下原因"
    ]
    
    for q in test_cases:
        run_test_case(agent, q)

if __name__ == "__main__":
    main()