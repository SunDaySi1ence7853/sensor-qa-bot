# examples/mini_agent.py
"""
手写最小 Agent 循环，理解工具调用底层机制。
"""
import os
import sys
import json
from dotenv import load_dotenv
from openai import OpenAI

# 将项目根目录加入 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import get_config

load_dotenv()

# 1. 定义两个 demo 工具
def calculate(expression: str) -> str:
    """简单计算器"""
    try:
        # 安全的简单表达式求值（仅限数字和加减乘除）
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "错误：包含非法字符"
        result = eval(expression)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算失败：{e}"

def check_temperature(value: float) -> str:
    """温度阈值判断"""
    if value > 35.0:
        return f"温度 {value}°C 过高，超限报警！"
    elif value < 10.0:
        return f"温度 {value}°C 过低，超限报警！"
    else:
        return f"温度 {value}°C 正常。"

# 2. 将工具映射成 OpenAI function schema
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算简单的数学表达式，如 '1+2*3'",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_temperature",
            "description": "判断温度是否超出安全阈值",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "温度值"}
                },
                "required": ["value"]
            }
        }
    }
]

available_functions = {
    "calculate": calculate,
    "check_temperature": check_temperature
}

def run_mini_agent():
    cfg = get_config()
    client = OpenAI(api_key=cfg.deepseek_api_key, base_url=cfg.deepseek_base_url)
    
    # 包含两轮对话：第一轮调工具，第二轮直接回答
    messages = [
        {"role": "system", "content": "你是一个传感器助手。如果需要计算或判断温度，请使用工具。"},
        {"role": "user", "content": "帮我算一下 12 * 7 等于多少，然后判断 38 度的温度是否正常？"}
    ]
    
    print("🤖 用户:", messages[-1]["content"])
    
    # 3. Agent 循环
    for _ in range(5):  # 限制最大轮数防止死循环
        response = client.chat.completions.create(
            model=cfg.deepseek_model,
            messages=messages,
            tools=tools_schema
        )
        msg = response.choices[0].message
        
        # 4. 解析是否有工具调用
        if msg.tool_calls:
            messages.append(msg)
            print(f"🛠️ LLM 决定调用工具: {[tc.function.name for tc in msg.tool_calls]}")
            for tc in msg.tool_calls:
                func_name = tc.function.name
                func_args = json.loads(tc.function.arguments)
                print(f"   -> 工具参数: {func_args}")
                
                # 执行对应函数
                result = available_functions[func_name](**func_args)
                print(f"   <- 执行结果: {result}")
                
                # 5. 结果回填
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result
                })
        else:
            # 没有工具调用，拿到最终回答
            print("✅ 最终回答:", msg.content)
            break

if __name__ == "__main__":
    run_mini_agent()