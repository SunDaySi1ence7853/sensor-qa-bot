"""
RAG 对话核心。

功能：
- 基于检索增强的问答
- 多轮对话（带历史裁剪，支持外部传入历史）
- 查询重写（解决无主语追问的检索召回失败问题）
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
        current_history = history or []
        
        # 1. 查询重写：解决无主语追问导致检索失败的问题
        search_query = self._rewrite_query(question, current_history)
        
        # 2. 使用重写后的完整问题去检索
        docs = self.vectorstore.similarity_search(
            search_query, k=self.cfg.retrieve_top_k
        )
        context = self._format_context(docs)
        sources = self._extract_sources(docs)

        # 3. 构建 Prompt 时，依然使用原始 question，保持对话自然性
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
        current_history = history or []
        
        # 1. 查询重写
        search_query = self._rewrite_query(question, current_history)
        
        # 2. 检索
        docs = self.vectorstore.similarity_search(
            search_query, k=self.cfg.retrieve_top_k
        )
        context = self._format_context(docs)
        sources = self._extract_sources(docs)

        # 3. 构建 Prompt
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

    def _rewrite_query(self, question: str, history: List[BaseMessage]) -> str:
        """结合历史对话，将无主语追问改写为独立问题，提升向量检索召回率。"""
        if not history:
            return question
            
        # 提取最近一次用户提问
        last_user_q = ""
        for msg in reversed(history):
            if isinstance(msg, HumanMessage):
                last_user_q = msg.content
                break
                
        if not last_user_q:
            return question
            
        # 调用主 LLM 进行问题改写 (Prompt 尽量简短，降低延迟)
        rewrite_prompt = (
            f"你是一个查询改写助手。根据历史对话，将用户的【追问】改写为一个独立、完整的问题，"
            f"使其脱离对话上下文也能被向量检索系统理解。只输出改写后的问题，不要包含任何其他解释。\n"
            f"历史提问：{last_user_q}\n"
            f"用户追问：{question}\n"
            f"改写后的问题："
        )
        
        try:
            response = self.llm.invoke(rewrite_prompt)
            rewritten = response.content.strip().strip('"').strip()
            # 防止 LLM 乱说话，简单校验长度
            if rewritten and len(rewritten) < 100:
                return rewritten
            return question
        except Exception:
            # 改写失败时降级：直接用原问题去搜
            return question

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