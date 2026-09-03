"""
测试 src/rag_chat.py

覆盖：
1. ask 基本调用返回 ChatResult
2. history 正确成对追加
3. _trim_history 按 MEMORY_TURNS 保留
4. reset 清空历史
5. ask_stream 事件序列正确（delta 后 done）
6. token 统计正确
7. sources 正确提取（跨平台路径）
8. 检索为空时仍能工作
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src import rag_chat as rag_module
from src.rag_chat import ChatResult, SensorRAGChat, StreamEvent
from tests.conftest import FakeDoc, FakeLLM, FakeVectorStore


@pytest.fixture
def chat(valid_env, monkeypatch, fake_llm, fake_vectorstore):
    """
    构造一个所有外部依赖都被 mock 的 SensorRAGChat。
    """
    monkeypatch.setattr(
        rag_module, "load_vectorstore", lambda: fake_vectorstore
    )
    monkeypatch.setattr(
        rag_module, "get_llm", lambda streaming=False: fake_llm
    )
    return SensorRAGChat()


class TestAsk:
    def test_ask_returns_chat_result(self, chat):
        result = chat.ask("DHT22是什么？")
        assert isinstance(result, ChatResult)
        assert result.content == "Hello world!"

    def test_ask_returns_token_usage(self, chat):
        result = chat.ask("测试问题")
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.usage.total_tokens == 15
        # 成本 = 10/1M * 1 + 5/1M * 2 = 2e-5
        assert result.usage.estimated_cost_cny == pytest.approx(2e-5, rel=1e-6)

    def test_ask_returns_sources_from_metadata(self, chat):
        result = chat.ask("测试")
        assert "dht22.md" in result.sources
        assert "bmp280.md" in result.sources


class TestHistory:
    def test_history_appends_in_pairs(self, chat):
        chat.ask("Q1")
        assert len(chat.history) == 2
        assert isinstance(chat.history[0], HumanMessage)
        assert isinstance(chat.history[1], AIMessage)
        assert chat.history[0].content == "Q1"

    def test_history_grows_across_calls(self, chat):
        chat.ask("Q1")
        chat.ask("Q2")
        chat.ask("Q3")
        assert len(chat.history) == 6

    def test_reset_clears_history(self, chat):
        chat.ask("Q1")
        chat.ask("Q2")
        chat.reset()
        assert chat.history == []

    def test_trim_history_respects_memory_turns(
        self, valid_env, monkeypatch, fake_llm, fake_vectorstore
    ):
        valid_env.setenv("MEMORY_TURNS", "2")
        monkeypatch.setattr(rag_module, "load_vectorstore", lambda: fake_vectorstore)
        monkeypatch.setattr(rag_module, "get_llm", lambda streaming=False: fake_llm)

        chat = SensorRAGChat()
        chat.ask("Q1")
        chat.ask("Q2")
        chat.ask("Q3")
        # 只保留最近2轮 = 4条消息
        assert len(chat.history) == 4
        assert chat.history[0].content == "Q2"
        assert chat.history[-2].content == "Q3"


class TestStream:
    def test_ask_stream_yields_delta_then_done(self, chat):
        events = list(chat.ask_stream("测试"))
        # 3 个内容 chunk + 1 个 done
        delta_events = [e for e in events if not e.done]
        done_events = [e for e in events if e.done]
        assert len(delta_events) == 3
        assert len(done_events) == 1
        assert done_events[0].result is not None

    def test_stream_deltas_concatenate_to_full_content(self, chat):
        events = list(chat.ask_stream("测试"))
        deltas = "".join(e.delta for e in events if not e.done)
        assert deltas == "Hello world!"

    def test_stream_updates_history_only_at_end(self, chat):
        gen = chat.ask_stream("Q1")
        # 消费第一个 delta 事件，此时 history 还不应被更新
        first = next(gen)
        assert not first.done
        assert chat.history == []
        # 消费完
        list(gen)
        assert len(chat.history) == 2


class TestRetrieval:
    def test_empty_retrieval_still_works(
        self, valid_env, monkeypatch, fake_llm
    ):
        empty_store = FakeVectorStore(docs=[])
        monkeypatch.setattr(rag_module, "load_vectorstore", lambda: empty_store)
        monkeypatch.setattr(rag_module, "get_llm", lambda streaming=False: fake_llm)

        chat = SensorRAGChat()
        result = chat.ask("这个问题知识库里没有")
        assert result.content == "Hello world!"
        assert result.sources == []

    def test_retrieval_respects_top_k(
        self, valid_env, monkeypatch, fake_llm
    ):
        valid_env.setenv("RETRIEVE_TOP_K", "1")
        store = FakeVectorStore()
        monkeypatch.setattr(rag_module, "load_vectorstore", lambda: store)
        monkeypatch.setattr(rag_module, "get_llm", lambda streaming=False: fake_llm)

        chat = SensorRAGChat()
        chat.ask("测试")
        assert store.search_calls[0][1] == 1