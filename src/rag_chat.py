"""
RAG 对话核心。

设计要点：
1. RAG：先检索知识库，把资料塞进 prompt。
2. 记忆：维护 history，按 MEMORY_TURNS 截断，成对存储，天然无连续同角色。
3. 流式与非流式共用同一套核心（DRY）：非流式 = 收集流式结果。
4. token 统计与费用估算。
"""

from dataclasses import dataclass, field
from typing import Generator, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.config import get_config
from src.llm import get_llm
from src.prompt import build_rag_prompt
from src.vectorstore import load_vectorstore


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_cny: float = 0.0


@dataclass
class ChatResult:
    content: str
    usage: TokenUsage
    sources: List[str] = field(default_factory=list)


@dataclass
class StreamEvent:
    delta: str = ""
    done: bool = False
    result: Optional[ChatResult] = None


def _estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    cfg = get_config()
    return (
        prompt_tokens / 1_000_000 * cfg.input_price_per_1m
        + completion_tokens / 1_000_000 * cfg.output_price_per_1m
    )


class SensorRAGChat:
    def __init__(self):
        cfg = get_config()
        self.top_k = cfg.retrieve_top_k
        self.memory_turns = cfg.memory_turns

        self.vectorstore = load_vectorstore()
        self.prompt = build_rag_prompt()
        self.history: List[BaseMessage] = []

    def reset(self) -> None:
        self.history = []

    def _retrieve(self, question: str):
        docs = self.vectorstore.similarity_search(question, k=self.top_k)
        context = "\n\n".join(
            f"[资料{i + 1}] {d.page_content}" for i, d in enumerate(docs)
        )
        sources = [
            d.metadata.get("source", "unknown").split("\\")[-1].split("/")[-1]
            for d in docs
        ]
        return context, sources

    def _trim_history(self) -> None:
        """
        history 里都是成对的 Human/AI 消息，
        只保留最近 memory_turns 轮（一轮 = 1条Human + 1条AI）。
        """
        max_msgs = self.memory_turns * 2
        if len(self.history) > max_msgs:
            self.history = self.history[-max_msgs:]

    def _core_stream(
        self, question: str
    ) -> Generator[StreamEvent, None, None]:
        """
        统一核心：始终以流式方式跑，逐块 yield。
        非流式接口只是把它收集起来。

        重要约定（关于失败回滚）：
            history 的更新（append 用户与助手消息）必须放在流式循环
            **全部结束、生成 done 事件之前**统一执行。
            这样，无论是在 _retrieve() 还是 llm.stream() 中途抛异常，
            history 都还没被改动，天然“失败不写入、无需回滚”。
            ——今后维护者请勿在下面的 for 循环内部 append history，
            否则中途失败会污染对话记忆。
        """
        context, sources = self._retrieve(question)

        messages = self.prompt.format_messages(
            context=context,
            history=self.history,
            question=question,
        )

        llm = get_llm(streaming=True)

        parts: List[str] = []
        prompt_tokens = 0
        completion_tokens = 0

        for chunk in llm.stream(messages):
            text = chunk.content or ""
            if text:
                parts.append(text)
                yield StreamEvent(delta=text, done=False)

            usage = getattr(chunk, "usage_metadata", None)
            if usage:
                prompt_tokens = usage.get("input_tokens", prompt_tokens)
                completion_tokens = usage.get("output_tokens", completion_tokens)

        full_text = "".join(parts).strip()
        total_tokens = prompt_tokens + completion_tokens

        result = ChatResult(
            content=full_text,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                estimated_cost_cny=_estimate_cost(prompt_tokens, completion_tokens),
            ),
            sources=sources,
        )

        # 只有走到这里（流式全部成功）才更新记忆，成对写入。
        # 中途任何异常都不会执行到这两行，因此记忆不会被污染。
        self.history.append(HumanMessage(content=question))
        self.history.append(AIMessage(content=full_text))
        self._trim_history()

        yield StreamEvent(done=True, result=result)

    def ask_stream(self, question: str) -> Generator[StreamEvent, None, None]:
        yield from self._core_stream(question)

    def ask(self, question: str) -> ChatResult:
        """
        非流式：复用流式核心，收集后返回。
        """
        result: Optional[ChatResult] = None
        for event in self._core_stream(question):
            if event.done:
                result = event.result
        assert result is not None
        return result