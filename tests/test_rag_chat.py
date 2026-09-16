"""
测试 src/rag_chat.py

覆盖：
1. ask 方法返回内容和来源
2. ask 默认 history 参数为 None 时不报错
3. ask 传入 history 参数时正常处理
4. ask 返回结果包含 history 字段
5. ask 返回的 history 长度增长正确
6. ask 在 history 过长时安全处理 (验证截断或不崩溃)
7. ask_stream 输出 delta 字符串
8. ask_stream 最终输出包含 result 对象
9. ask_stream 最终输出的 result 包含 history 字段
10. ask_stream 传入 history 参数正常处理
11. ask 在召回为空时安全兜底
12. ask_stream 在召回为空时安全兜底
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from src.rag_chat import SensorRAGChat, ChatResult, StreamEvent, TokenUsage


@pytest.fixture
def mock_rag_chat():
    """
    模块级 Mock，通过 patch 依赖项来实例化 SensorRAGChat，
    避免真实加载向量库和 LLM 模型。
    """
    with patch('src.rag_chat.get_config') as mock_get_config, \
         patch('src.rag_chat.load_vectorstore') as mock_load_vectorstore, \
         patch('src.rag_chat.get_llm') as mock_get_llm, \
         patch('src.rag_chat.build_rag_prompt') as mock_build_prompt:
        
        # 1. Mock Config
        mock_cfg = MagicMock()
        mock_cfg.retrieve_top_k = 3
        mock_cfg.memory_turns = 5
        mock_cfg.input_price_per_1m = 0.01
        mock_cfg.output_price_per_1m = 0.02
        mock_get_config.return_value = mock_cfg
        
        # 2. Mock Vectorstore
        mock_vectorstore = MagicMock()
        mock_doc = MagicMock()
        mock_doc.page_content = "测试文档内容"
        mock_doc.metadata = {"source": "test_source.md"}
        mock_vectorstore.similarity_search.return_value = [mock_doc]
        mock_load_vectorstore.return_value = mock_vectorstore
        
        # 3. Mock LLM (非流式)
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "这是最终回答"
        mock_response.usage_metadata = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
        mock_response.response_metadata = None
        mock_llm.invoke.return_value = mock_response
        
        # 4. Mock LLM (流式) - 关键：必须支持 + 运算符，因为源码里有 aggregated + chunk
        mock_chunk = MagicMock()
        mock_chunk.content = "流式回答"
        mock_chunk.usage_metadata = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
        mock_chunk.__add__ = lambda self, other: self  # 支持 aggregated + chunk
        
        mock_llm_stream = MagicMock()
        mock_llm_stream.stream.return_value = iter([mock_chunk])
        
        # 根据 streaming 参数返回不同的 mock
        mock_get_llm.side_effect = lambda streaming=False: mock_llm_stream if streaming else mock_llm
        
        # 5. Mock Prompt
        mock_prompt = MagicMock()
        mock_prompt.format_messages.return_value = []
        mock_build_prompt.return_value = mock_prompt
        
        # 实例化被测对象
        chat = SensorRAGChat()
        
        # yield chat 对象和 mock 对象，方便测试中修改行为
        yield chat, mock_vectorstore, mock_llm, mock_llm_stream


class TestAskMethod:
    def test_ask_returns_content_and_sources(self, mock_rag_chat):
        """1. ask 方法返回内容和来源"""
        chat, _, _, _ = mock_rag_chat
        result = chat.ask(question="测试问题")
        assert isinstance(result, ChatResult)
        assert result.content == "这是最终回答"
        assert "test_source.md" in result.sources

    def test_ask_history_defaults_to_none(self, mock_rag_chat):
        """2. ask 默认 history 参数为 None 时不报错"""
        chat, _, _, _ = mock_rag_chat
        result = chat.ask(question="测试问题")
        assert isinstance(result, ChatResult)

    def test_ask_with_provided_history(self, mock_rag_chat):
        """3. ask 传入 history 参数时正常处理"""
        chat, _, _, _ = mock_rag_chat
        history = [HumanMessage(content="前一轮"), AIMessage(content="前一轮答")]
        result = chat.ask(question="测试问题", history=history)
        assert isinstance(result, ChatResult)

    def test_ask_returns_history_field(self, mock_rag_chat):
        """4. ask 返回结果包含 history 字段"""
        chat, _, _, _ = mock_rag_chat
        result = chat.ask(question="测试问题")
        assert hasattr(result, 'history')
        assert isinstance(result.history, list)

    def test_ask_history_grows(self, mock_rag_chat):
        """5. ask 返回的 history 长度增长正确"""
        chat, _, _, _ = mock_rag_chat
        history = [HumanMessage(content="前一轮"), AIMessage(content="前一轮答")]
        result = chat.ask(question="测试问题", history=history)
        # 原有 2 条 + 本次提问和回答 2 条 = 4 条
        assert len(result.history) == 4

    def test_ask_handles_overflow_history_safely(self, mock_rag_chat):
        """6. ask 在 history 过长时安全处理"""
        chat, _, _, _ = mock_rag_chat
        long_history = []
        for i in range(20):  # 40 条消息，远超 memory_turns=5 (max 10 条)
            long_history.append(HumanMessage(content=f"问题{i}"))
            long_history.append(AIMessage(content=f"回答{i}"))
            
        result = chat.ask(question="测试问题", history=long_history)
        assert isinstance(result, ChatResult)
        # 验证确实被截断到 max_messages (5 * 2 = 10)
        assert len(result.history) == 10


class TestAskStreamMethod:
    def test_ask_stream_yields_delta(self, mock_rag_chat):
        """7. ask_stream 输出 delta 字符串"""
        chat, _, _, _ = mock_rag_chat
        events = list(chat.ask_stream(question="测试问题"))
        deltas = [e.delta for e in events if hasattr(e, 'delta') and not e.done]
        assert len(deltas) > 0
        assert deltas[0] == "流式回答"

    def test_ask_stream_yields_final_result(self, mock_rag_chat):
        """8. ask_stream 最终输出包含 result 对象"""
        chat, _, _, _ = mock_rag_chat
        events = list(chat.ask_stream(question="测试问题"))
        final_event = events[-1]
        assert final_event.done is True
        assert isinstance(final_event.result, ChatResult)

    def test_ask_stream_final_event_has_history(self, mock_rag_chat):
        """9. ask_stream 最终输出的 result 包含 history 字段"""
        chat, _, _, _ = mock_rag_chat
        events = list(chat.ask_stream(question="测试问题"))
        final_event = events[-1]
        assert hasattr(final_event.result, 'history')
        assert isinstance(final_event.result.history, list)

    def test_ask_stream_with_provided_history(self, mock_rag_chat):
        """10. ask_stream 传入 history 参数正常处理"""
        chat, _, _, _ = mock_rag_chat
        history = [HumanMessage(content="前一轮"), AIMessage(content="前一轮答")]
        events = list(chat.ask_stream(question="测试问题", history=history))
        final_event = events[-1]
        assert final_event.done is True
        assert len(final_event.result.history) == len(history) + 2


class TestEdgeCases:
    def test_ask_handles_empty_retrieval(self, mock_rag_chat):
        """11. ask 在召回为空时安全兜底"""
        chat, mock_vectorstore, _, _ = mock_rag_chat
        mock_vectorstore.similarity_search.return_value = []
        result = chat.ask(question="没召回的问题")
        assert isinstance(result, ChatResult)
        assert len(result.sources) == 0
        assert isinstance(result.usage, TokenUsage)

    def test_ask_stream_handles_empty_retrieval(self, mock_rag_chat):
        """12. ask_stream 在召回为空时安全兜底"""
        chat, mock_vectorstore, _, _ = mock_rag_chat
        mock_vectorstore.similarity_search.return_value = []
        events = list(chat.ask_stream(question="没召回的问题"))
        final_event = events[-1]
        assert final_event.done is True
        assert isinstance(final_event.result, ChatResult)
        assert len(final_event.result.sources) == 0