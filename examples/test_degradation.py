# examples/test_degradation.py
"""
异常降级测试：人为破坏工具依赖的配置，验证 Agent 能否优雅兜底。
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 注意导入顺序：要在 get_sensor_agent 之前，先把阈值配置改坏
from src.config import get_config
cfg = get_config()

# 人为制造异常：把温度的阈值配置删除
if "temperature" in cfg.sensor_thresholds:
    # 利用 object.__setattr__ 绕开 dataclass(frozen=True) 的限制
    thresholds = cfg.sensor_thresholds.copy()
    del thresholds["temperature"]
    object.__setattr__(cfg, "sensor_thresholds", thresholds)

from src.agent import get_sensor_agent

def main():
    agent = get_sensor_agent()
    
    question = "当前车间温度正常吗？"
    print(f"\n{'='*60}")
    print(f"[异常降级测试] 人为破坏了 check_threshold 的配置")
    print(f"👤 用户提问: {question}")
    print(f"{'='*60}")
    
    inputs = {"messages": [("user", question)]}
    
    for chunk in agent.stream(inputs, {"recursion_limit": 15}):
        for node_name, state in chunk.items():
            if node_name == "agent":
                last_msg = state["messages"][-1]
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    tools = [tc["name"] for tc in last_msg.tool_calls]
                    print(f"🧠 Thought & Action: 我需要调用工具 {tools}")
                elif last_msg.content:
                    print(f"✅ Final Answer: {last_msg.content}\n")
            elif node_name == "tools":
                for msg in state["messages"]:
                    if msg.type == "tool":
                        print(f"👀 Observation: 工具返回 {msg.content[:200]}...")

if __name__ == "__main__":
    main()