"""
RAG 问答核心逻辑。
"""

from dataclasses import dataclass
from typing import Generator

from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

from src.config import get_config
from src.llm import get_llm
from src.logging_config import get_logger
from src.prompt import build_rag_prompt
from src.vectorstore import load_vectorstore

logger = get_logger(__name__)


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
    sources: list[str]


@dataclass
class StreamEvent:
    delta: str = ""
    done: bool = False
    result: ChatResult | None = None


class SensorRAGChat:
   def __init__(self):
    logger.debug("正在初始化 SensorRAGChat...")
    self.cfg = get_config()
    self.vectorstore: FAISS = load_vectorstore()
    self.llm: ChatOpenAI = get_llm(streaming=False)
    self.llm_stream: ChatOpenAI = get_llm(streaming=True)
    self.history: list[BaseMessage] = []

    self.prompt = build_rag_prompt()

    logger.info("SensorRAGChat 初始化完成 | memory_turns=%d", self.cfg.memory_turns)

    def ask(self, question: str) -> ChatResult:
        logger.info("收到问题 | question=%r", question)

        docs = self.vectorstore.similarity_search(
            question, k=self.cfg.retrieve_top_k
        )
        logger.debug("检索到 %d 个文档片段", len(docs))
        for i, doc in enumerate(docs, 1):
            logger.debug(
                "Chunk %d/%d | source=%s | content=%s",
                i,
                len(docs),
                doc.metadata.get("source", "unknown"),
                doc.page_content[:100],
            )

        context = "\n\n".join(d.page_content for d in docs)
        sources = sorted(set(d.metadata.get("source", "") for d in docs))

        logger.debug("开始调用 LLM | history_len=%d", len(self.history))

        # 直接调 llm 拿 AIMessage（带 usage_metadata），不用 StrOutputParser
        messages = self.prompt.format_messages(
            context=context,
            question=question,
            history=self.history,
        )
        response = self.llm.invoke(messages)

        content = response.content
        usage_meta = getattr(response, "usage_metadata", None)

        if usage_meta:
            prompt_tokens = usage_meta.get("input_tokens", 0)
            completion_tokens = usage_meta.get("output_tokens", 0)
        else:
            logger.warning("LLM 响应缺少 usage_metadata，token 统计归零")
            prompt_tokens = 0
            completion_tokens = 0

        usage = self._calculate_usage(prompt_tokens, completion_tokens)

        logger.info(
            "LLM 响应完成 | tokens=%d | cost=¥%.6f",
            usage.total_tokens,
            usage.estimated_cost_cny,
        )
        logger.debug("LLM 回答内容：%s", content[:200])

        self.history.append(HumanMessage(content=question))
        self.history.append(AIMessage(content=content))
        self._trim_history()

        return ChatResult(content=content, usage=usage, sources=sources)

    def ask_stream(self, question: str) -> Generator[StreamEvent, None, None]:
        logger.info("收到流式问题 | question=%r", question)

        docs = self.vectorstore.similarity_search(
            question, k=self.cfg.retrieve_top_k
        )
        logger.debug("检索到 %d 个文档片段", len(docs))

        context = "\n\n".join(d.page_content for d in docs)
        sources = sorted(set(d.metadata.get("source", "") for d in docs))

        messages = self.prompt.format_messages(
            context=context,
            question=question,
            history=self.history,
        )

        logger.debug("开始流式调用 LLM | history_len=%d", len(self.history))

        chunks = []
        prompt_tokens = 0
        completion_tokens = 0

        for chunk in self.llm_stream.stream(messages):
            text = chunk.content
            if text:
                chunks.append(text)
                yield StreamEvent(delta=text)

            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                prompt_tokens = chunk.usage_metadata.get("input_tokens", 0)
                completion_tokens = chunk.usage_metadata.get("output_tokens", 0)

        full_response = "".join(chunks)
        usage = self._calculate_usage(prompt_tokens, completion_tokens)

        logger.info(
            "流式响应完成 | tokens=%d | cost=¥%.6f",
            usage.total_tokens,
            usage.estimated_cost_cny,
        )
        logger.debug("流式回答内容：%s", full_response[:200])

        self.history.append(HumanMessage(content=question))
        self.history.append(AIMessage(content=full_response))
        self._trim_history()

        yield StreamEvent(
            done=True,
            result=ChatResult(content=full_response, usage=usage, sources=sources),
        )

    def reset(self):
        self.history.clear()
        logger.info("对话历史已清空")

    def _trim_history(self):
        max_messages = self.cfg.memory_turns * 2
        if len(self.history) > max_messages:
            removed = len(self.history) - max_messages
            self.history = self.history[-max_messages:]
            logger.debug(
                "历史记录已裁剪 | 移除 %d 条，保留 %d 条", removed, max_messages
            )

    def _calculate_usage(
        self, prompt_tokens: int, completion_tokens: int
    ) -> TokenUsage:
        total = prompt_tokens + completion_tokens
        cost = (
            prompt_tokens / 1_000_000 * self.cfg.input_price_per_1m
            + completion_tokens / 1_000_000 * self.cfg.output_price_per_1m
        )
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
            estimated_cost_cny=cost,
        )