"""
DeepSeek API 调用封装。

本项目使用 DeepSeek 的 OpenAI 兼容接口：
base_url = https://api.deepseek.com
model = deepseek-v4-flash
"""

import os
from typing import List, Dict

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

if not DEEPSEEK_API_KEY:
    raise RuntimeError(
        "没有读取到 DEEPSEEK_API_KEY。请检查项目根目录下的 .env 文件。"
    )


client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)


def ask_deepseek(messages: List[Dict[str, str]]) -> str:
    """
    调用 DeepSeek 聊天接口。

    参数:
        messages: OpenAI 格式消息列表，例如:
            [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."}
            ]

    返回:
        模型回答文本
    """

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages,
        temperature=0.2,
        stream=False,
    )

    return response.choices[0].message.content.strip()