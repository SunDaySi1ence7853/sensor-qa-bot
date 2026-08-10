"""
对话管理逻辑。
"""

from src.api_client import ask_deepseek
from src.prompt import build_system_prompt


class SensorChat:
    """
    传感器问答助手。
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """
        重置会话，并重新加载知识库。
        """

        self.messages = [
            {
                "role": "system",
                "content": build_system_prompt(),
            }
        ]

    def ask(self, user_question: str) -> str:
        """
        接收用户问题，调用 DeepSeek API，返回回答。
        """

        self.messages.append(
            {
                "role": "user",
                "content": user_question,
            }
        )

        # 控制历史长度，避免上下文越来越长。
        # 保留 system + 最近 12 条消息。
        if len(self.messages) > 13:
            self.messages = [self.messages[0]] + self.messages[-12:]

        answer = ask_deepseek(self.messages)

        self.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        return answer