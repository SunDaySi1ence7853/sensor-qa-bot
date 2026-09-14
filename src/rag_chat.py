"""
RAG 对话核心。

功能：
- 基于检索增强的问答
- 多轮对话（带历史裁剪，支持外部传入历史）
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
    history: List[BaseMessage] = field(default_factory=list)


@dataclass
class StreamEvent:
    delta: str = ""
    done: bool = False
    result: Optional[ChatResult] = None


class SensorRAGChat:
    """带检索、历史、token 统计的对话器（无状态版，适配 Web 端）。"""

    def __init__(self):
        self.cfg = get_config()
        self.vectorstore = load_vectorstore()
        self.llm = get_llm(streaming=False)
        self.llm_stream = get_llm(streaming=True)
        self.prompt = build_rag_prompt()

    # ---------------- 公共 API ---------------- #

    def ask(self, question: str, history: Optional[List[BaseMessage]] = None) -> ChatResult:
        """非流式提问。"""
        docs = self.vectorstore.similarity_search(
            question, k=self.cfg.retrieve_top_k
        )
        context = self._format_context(docs)
        sources = self._extract_sources(docs)
        
        current_history = history or []

        messages = self.prompt.format_messages(
            context=context,
            history=current_history,
            question=question,
        )

        response = self.llm.invoke(messages)
        content = response.content
        usage = self._extract_usage(response)

        updated_history = self._update_history(current_history, question, content)
        return ChatResult(content=content, sources=sources, usage=usage, history=updated_history)

    def ask_stream(self, question: str, history: Optional[List[BaseMessage]] = None) -> Iterator[StreamEvent]:
        """流式提问。产出多个 delta 事件，最后一个 done=True 带完整结果。"""
        docs = self.vectorstore.similarity_search(
            question, k=self.cfg.retrieve_top_k
        )
        context = self._format_context(docs)
        sources = self._extract_sources(docs)
        
        current_history = history or []

        messages = self.prompt.format_messages(
            context=context,
            history=current_history,
            question=question,
        )

        collected: List[str] = []
        aggregated = None
        
        for chunk in self.llm_stream.stream(messages):
            aggregated = chunk if aggregated is None else aggregated + chunk
            text = chunk.content or ""
            if text:
                collected.append(text)
                yield StreamEvent(delta=text, done=False)

        full_content = "".join(collected)
        usage = self._extract_usage(aggregated)
        
        updated_history = self._update_history(current_history, question, full_content)

        yield StreamEvent(
            delta="",
            done=True,
            result=ChatResult(
                content=full_content,
                sources=sources,
                usage=usage,
                history=updated_history
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
        cfg = get_config()
        if response is None:
            return TokenUsage()

        # 路径 1: usage_metadata
        meta = getattr(response, "usage_metadata", None)
        if meta:
            try:
                prompt_tokens = int(meta.get("input_tokens", 0) or 0)
                completion_tokens = int(meta.get("output_tokens", 0) or 0)
                total_tokens = int(meta.get("total_tokens", prompt_tokens + completion_tokens) or 0)
                if prompt_tokens or completion_tokens:
                    cost = (prompt_tokens / 1_000_000 * cfg.input_price_per_1m + 
                            completion_tokens / 1_000_000 * cfg.output_price_per_1m)
                    return TokenUsage(prompt_tokens, completion_tokens, total_tokens, cost)
            except Exception:
                pass

        # 路径 2: response_metadata
        resp_meta = getattr(response, "response_metadata", None)
        if resp_meta and isinstance(resp_meta, dict):
            token_usage = resp_meta.get("token_usage")
            if token_usage:
                try:
                    prompt_tokens = int(token_usage.get("prompt_tokens", 0) or 0)
                    completion_tokens = int(token_usage.get("completion_tokens", 0) or 0)
                    total_tokens = int(token_usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
                    cost = (prompt_tokens / 1_000_000 * cfg.input_price_per_1m + 
                            completion_tokens / 1_000_000 * cfg.output_price_per_1m)
                    return TokenUsage(prompt_tokens, completion_tokens, total_tokens, cost)
                except Exception:
                    pass

        return TokenUsage()

    def _update_history(self, history: List[BaseMessage], question: str, answer: str) -> List[BaseMessage]:
        """更新历史并裁剪，返回新的历史列表。"""
        new_history = list(history)
        new_history.append(HumanMessage(content=question))
        new_history.append(AIMessage(content=answer))
        
        max_messages = self.cfg.memory_turns * 2
        if len(new_history) > max_messages:
            new_history = new_history[-max_messages:]
        return new_history