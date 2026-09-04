"""
RAG 对话核心。

功能：
- 基于检索增强的问答
- 多轮对话（带历史裁剪）
- 流式与非流式两种模式
- Token 用量与成本统计（带降级方案）
"""

from dataclasses import dataclass, field
from typing import Iterator, List, Optional

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
    sources: List[str] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass
class StreamEvent:
    delta: str = ""
    done: bool = False
    result: Optional[ChatResult] = None


class SensorRAGChat:
    """带检索、历史、token 统计的对话器。"""

    def __init__(self):
        self.cfg = get_config()
        self.vectorstore = load_vectorstore()
        self.llm = get_llm(streaming=False)
        self.llm_stream = get_llm(streaming=True)
        self.prompt = build_rag_prompt()
        self.history: List[BaseMessage] = []

    # ---------------- 公共 API ---------------- #

    def reset(self) -> None:
        """清空历史。"""
        self.history = []

    def ask(self, question: str) -> ChatResult:
        """非流式提问。"""
        docs = self.vectorstore.similarity_search(
            question, k=self.cfg.retrieve_top_k
        )
        context = self._format_context(docs)
        sources = self._extract_sources(docs)

        messages = self.prompt.format_messages(
            context=context,
            history=self.history,
            question=question,
        )

        response = self.llm.invoke(messages)
        content = response.content
        usage = self._extract_usage(response)

        self._append_history(question, content)
        return ChatResult(content=content, sources=sources, usage=usage)

    def ask_stream(self, question: str) -> Iterator[StreamEvent]:
        """流式提问。产出多个 delta 事件，最后一个 done=True 带完整结果。"""
        docs = self.vectorstore.similarity_search(
            question, k=self.cfg.retrieve_top_k
        )
        context = self._format_context(docs)
        sources = self._extract_sources(docs)

        messages = self.prompt.format_messages(
            context=context,
            history=self.history,
            question=question,
        )

        collected: List[str] = []
        last_chunk = None
        for chunk in self.llm_stream.stream(messages):
            last_chunk = chunk
            text = chunk.content or ""
            if text:
                collected.append(text)
                yield StreamEvent(delta=text, done=False)

        full_content = "".join(collected)
        usage = self._extract_usage(last_chunk)
        self._append_history(question, full_content)

        yield StreamEvent(
            delta="",
            done=True,
            result=ChatResult(
                content=full_content,
                sources=sources,
                usage=usage,
            ),
        )

    # ---------------- 内部工具 ---------------- #

    def _format_context(self, docs) -> str:
        if not docs:
            return "（未检索到相关内容）"
        return "\n\n".join(
            f"[{i + 1}] {d.page_content}" for i, d in enumerate(docs)
        )

    def _extract_sources(self, docs) -> List[str]:
        """
        从检索到的文档提取来源文件名（去重、按首次出现排序）。
        使用 os.path.basename 兼容 Windows/Linux 路径。
        """
        import os

        seen = []
        for d in docs:
            src = d.metadata.get("source", "")
            if not src:
                continue
            name = os.path.basename(src.replace("\\", "/"))
            if name not in seen:
                seen.append(name)
        return seen

    def _extract_usage(self, response) -> TokenUsage:
        """
        从 LLM 响应提取 token 用量。

        防御性设计：如果 usage_metadata 缺失或字段不完整，
        降级返回全零的 TokenUsage，而不是抛异常中断主流程。

        如果不这么改：
        用户换用不返回 usage_metadata 的模型（例如本地部署的 Ollama、
        某些第三方兼容 API），每次 ask() 都会在统计步骤崩溃，
        即使 LLM 已经生成了答案也无法返回给用户。
        """
        cfg = get_config()

        if response is None:
            return TokenUsage()

        meta = getattr(response, "usage_metadata", None)
        if not meta:
            return TokenUsage()

        # meta 可能是 dict，也可能是对象；统一按 dict 读取
        try:
            prompt_tokens = int(meta.get("input_tokens", 0) or 0)
            completion_tokens = int(meta.get("output_tokens", 0) or 0)
            total_tokens = int(
                meta.get("total_tokens", prompt_tokens + completion_tokens) or 0
            )
        except (AttributeError, TypeError, ValueError):
            # meta 结构异常时也降级为全零，不影响业务
            return TokenUsage()

        cost = (
            prompt_tokens / 1_000_000 * cfg.input_price_per_1m
            + completion_tokens / 1_000_000 * cfg.output_price_per_1m
        )

        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_cny=cost,
        )

    def _append_history(self, question: str, answer: str) -> None:
        self.history.append(HumanMessage(content=question))
        self.history.append(AIMessage(content=answer))
        self._trim_history()

    def _trim_history(self) -> None:
        """
        只保留最近 MEMORY_TURNS 轮对话（每轮 = human + ai 两条消息）。
        """
        max_messages = self.cfg.memory_turns * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]