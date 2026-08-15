"""
对话管理逻辑。

本文件重点：
1. 支持普通问答。
2. 支持流式问答。
3. 修复历史消息截断后可能出现连续两条同角色消息的问题。
"""

from typing import Dict, Generator, List

from src.api_client import ChatResult, StreamEvent, ask_deepseek, ask_deepseek_stream
from src.prompt import build_system_prompt


class SensorChat:
    """
    传感器问答助手。
    """

    def __init__(self, max_history_messages: int = 12):
        self.max_history_messages = max_history_messages
        self.messages: List[Dict[str, str]] = []
        self.reset()

    def reset(self) -> None:
        """
        重置会话，并重新加载知识库。
        """

        self.messages = [
            {
                "role": "system",
                "content": build_system_prompt(force_reload=True),
            }
        ]

    def _trim_history(self) -> None:
        """
        截断历史消息。

        规则：
        1. 永远保留 system 消息。
        2. 只保留最近 max_history_messages 条 user/assistant 消息。
        3. 截断后不能出现连续两条相同 role。
        4. 如果最前面是 assistant，删掉它，保证历史从 user 开始。
        """

        if not self.messages:
            return

        system_message = self.messages[0]
        history = self.messages[1:]

        if len(history) <= self.max_history_messages:
            return

        history = history[-self.max_history_messages :]

        # 如果截断后第一条是 assistant，删除它。
        # 因为对话历史应该从 user 开始更合理。
        while history and history[0]["role"] == "assistant":
            history.pop(0)

        # 清理连续同角色消息。
        cleaned_history: List[Dict[str, str]] = []

        for msg in history:
            if not cleaned_history:
                cleaned_history.append(msg)
                continue

            if cleaned_history[-1]["role"] == msg["role"]:
                # 如果出现连续同角色，只保留后面这条。
                cleaned_history[-1] = msg
            else:
                cleaned_history.append(msg)

        self.messages = [system_message] + cleaned_history

    def _validate_no_consecutive_same_role(self) -> None:
        """
        开发期保护：确保不会出现连续两条相同 role 的消息。
        system 后面接 user 是允许的。
        """

        for i in range(1, len(self.messages)):
            prev_role = self.messages[i - 1]["role"]
            curr_role = self.messages[i]["role"]

            if prev_role == curr_role:
                raise RuntimeError(
                    f"消息历史异常：第 {i - 1} 条和第 {i} 条都是 {curr_role}"
                )

    def ask(self, user_question: str) -> ChatResult:
        """
        普通非流式问答。
        """

        self.messages.append(
            {
                "role": "user",
                "content": user_question,
            }
        )

        self._trim_history()
        self._validate_no_consecutive_same_role()

        try:
            result = ask_deepseek(self.messages)
        except Exception:
            # 如果 API 调用失败，移除刚刚追加的 user 消息，
            # 避免下一次提问时出现连续 user。
            if self.messages and self.messages[-1]["role"] == "user":
                self.messages.pop()
            raise

        self.messages.append(
            {
                "role": "assistant",
                "content": result.content,
            }
        )

        self._trim_history()
        self._validate_no_consecutive_same_role()

        return result

    def ask_stream(
        self,
        user_question: str,
    ) -> Generator[StreamEvent, None, None]:
        """
        流式问答。

        main.py 中会逐字打印 event.delta。
        流结束时，event.done=True，并带有完整 ChatResult。
        """

        self.messages.append(
            {
                "role": "user",
                "content": user_question,
            }
        )

        self._trim_history()
        self._validate_no_consecutive_same_role()

        final_result: ChatResult | None = None

        try:
            for event in ask_deepseek_stream(self.messages):
                if event.done:
                    final_result = event.result
                yield event
        except Exception:
            if self.messages and self.messages[-1]["role"] == "user":
                self.messages.pop()
            raise

        if final_result is None:
            if self.messages and self.messages[-1]["role"] == "user":
                self.messages.pop()
            raise RuntimeError("流式调用结束异常：没有收到最终结果。")

        self.messages.append(
            {
                "role": "assistant",
                "content": final_result.content,
            }
        )

        self._trim_history()
        self._validate_no_consecutive_same_role()