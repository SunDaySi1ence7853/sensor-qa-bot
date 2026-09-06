"""
rag_chat.py 单元测试。

设计原则：
- 不真实调用 LLM 与向量库，全部用 fake 对象注入
- 通过 monkeypatch 替换 SensorRAGChat.__init__ 依赖，避免副作用
- 重点覆盖：
  1) sources 去重与文件名提取（Windows/Linux 路径都要通）
  2) usage 提取的降级路径（None、缺 usage_metadata、字段异常）
  3) 历史裁剪：memory_turns 生效
  4) 流式 ask_stream 的事件序列（多个 delta + 最后一个 done=True）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import pytest

from src.rag_chat import ChatResult, SensorRAGChat, StreamEvent, TokenUsage


# ============================================================
# 轻量 fake 对象
# ============================================================
@dataclass
class FakeDoc:
    """模拟 LangChain 的 Document。"""
    page_content: str
    metadata: dict


class FakeVectorStore:
    def __init__(self, docs: List[FakeDoc]):
        self._docs = docs
        self.last_k: Optional[int] = None

    def similarity_search(self, query: str, k: int = 3):
        self.last_k = k
        return self._docs[:k]


class FakeMessage:
    """模拟 LLM 返回的消息对象。"""
    def __init__(self, content: str, usage_metadata: Any = None):
        self.content = content
        self.usage_metadata = usage_metadata


class FakeLLM:
    def __init__(self, response: FakeMessage):
        self.response = response
        self.called_with = None

    def invoke(self, messages):
        self.called_with = messages
        return self.response


class FakeStreamLLM:
    """模拟流式 LLM：yield 出一串 chunk，最后一个 chunk 带 usage_metadata。"""
    def __init__(self, chunks: List[FakeMessage]):
        self.chunks = chunks

    def stream(self, messages):
        for c in self.chunks:
            yield c


class FakePrompt:
    def format_messages(self, context, history, question):
        # 只是把参数打包返回，测试断言时能看到
        return [{"context": context, "history": history, "question": question}]


class FakeConfig:
    retrieve_top_k = 2
    memory_turns = 3
    input_price_per_1m = 1.0
    output_price_per_1m = 2.0


# ============================================================
# 通用工厂：构造一个绕过真实依赖的 SensorRAGChat
# ============================================================
def _build_chat(monkeypatch, docs=None, response=None, stream_chunks=None):
    """
    绕过 __init__ 里的真实依赖（load_vectorstore / get_llm / build_rag_prompt），
    手动装配一个 SensorRAGChat 实例。
    """
    docs = docs or []
    response = response or FakeMessage("默认回答")
    stream_chunks = stream_chunks or [FakeMessage("默认")]

    # 用 __new__ 绕过 __init__，手动填字段
    chat = SensorRAGChat.__new__(SensorRAGChat)
    chat.cfg = FakeConfig()
    chat.vectorstore = FakeVectorStore(docs)
    chat.llm = FakeLLM(response)
    chat.llm_stream = FakeStreamLLM(stream_chunks)
    chat.prompt = FakePrompt()
    chat.history = []

    # rag_chat._extract_usage 里会调 get_config()，替换掉
    monkeypatch.setattr("src.rag_chat.get_config", lambda: FakeConfig())
    return chat


# ============================================================
# ask() 主流程
# ============================================================
def test_ask_returns_content_and_sources(monkeypatch):
    docs = [
        FakeDoc("DHT22 是温湿度传感器", {"source": "docs/dht22.md"}),
        FakeDoc("工作电压 3.3-5V", {"source": "docs/dht22.md"}),  # 重复来源
        FakeDoc("常用于气象站", {"source": "docs/applications.md"}),
    ]
    response = FakeMessage(
        content="DHT22 是温湿度传感器，工作电压 3.3-5V",
        usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
    )
    chat = _build_chat(monkeypatch, docs=docs, response=response)

    result = chat.ask("DHT22是什么？")

    assert isinstance(result, ChatResult)
    assert "DHT22" in result.content
    # top_k=2，所以只取前 2 篇；来源去重后只剩 dht22.md
    assert result.sources == ["dht22.md"]
    assert result.usage.prompt_tokens == 100
    assert result.usage.completion_tokens == 50
    # cost = 100/1M * 1.0 + 50/1M * 2.0 = 0.0002
    assert result.usage.estimated_cost_cny == pytest.approx(0.0002)


def test_ask_writes_to_history(monkeypatch):
    chat = _build_chat(
        monkeypatch,
        response=FakeMessage("答案 A"),
    )
    chat.ask("问题 A")
    assert len(chat.history) == 2
    assert chat.history[0].content == "问题 A"
    assert chat.history[1].content == "答案 A"


def test_reset_clears_history(monkeypatch):
    chat = _build_chat(monkeypatch, response=FakeMessage("x"))
    chat.ask("q1")
    chat.ask("q2")
    assert len(chat.history) == 4
    chat.reset()
    assert chat.history == []


# ============================================================
# _extract_sources：Windows / Linux 路径 + 去重
# ============================================================
def test_extract_sources_handles_windows_paths(monkeypatch):
    docs = [
        FakeDoc("x", {"source": r"C:\data\docs\dht22.md"}),
        FakeDoc("y", {"source": r"C:\data\docs\dht22.md"}),
        FakeDoc("z", {"source": r"C:\data\docs\bmp280.md"}),
    ]
    chat = _build_chat(monkeypatch, docs=docs)
    # 直接调内部方法验证
    sources = chat._extract_sources(docs)
    assert sources == ["dht22.md", "bmp280.md"]


def test_extract_sources_skips_missing_source(monkeypatch):
    docs = [
        FakeDoc("x", {}),  # 没有 source 字段
        FakeDoc("y", {"source": ""}),  # 空字符串
        FakeDoc("z", {"source": "docs/ok.md"}),
    ]
    chat = _build_chat(monkeypatch, docs=docs)
    sources = chat._extract_sources(docs)
    assert sources == ["ok.md"]


# ============================================================
# _extract_usage：防御性降级
# ============================================================
def test_extract_usage_returns_zero_when_response_none(monkeypatch):
    chat = _build_chat(monkeypatch)
    usage = chat._extract_usage(None)
    assert usage.prompt_tokens == 0
    assert usage.total_tokens == 0
    assert usage.estimated_cost_cny == 0.0


def test_extract_usage_returns_zero_when_metadata_missing(monkeypatch):
    chat = _build_chat(monkeypatch)
    response = FakeMessage("x", usage_metadata=None)
    usage = chat._extract_usage(response)
    assert usage.prompt_tokens == 0


def test_extract_usage_returns_zero_when_metadata_malformed(monkeypatch):
    chat = _build_chat(monkeypatch)
    # usage_metadata 不是 dict，触发 AttributeError 分支
    response = FakeMessage("x", usage_metadata="not-a-dict")
    usage = chat._extract_usage(response)
    assert usage.prompt_tokens == 0
    assert usage.total_tokens == 0


def test_extract_usage_computes_cost_correctly(monkeypatch):
    chat = _build_chat(monkeypatch)
    response = FakeMessage(
        "x",
        usage_metadata={"input_tokens": 1_000_000, "output_tokens": 500_000},
    )
    usage = chat._extract_usage(response)
    # 1_000_000/1M * 1.0 + 500_000/1M * 2.0 = 1.0 + 1.0 = 2.0
    assert usage.estimated_cost_cny == pytest.approx(2.0)
    # total_tokens 未提供时用 prompt+completion 兜底
    assert usage.total_tokens == 1_500_000


# ============================================================
# 历史裁剪
# ============================================================
def test_history_trimmed_to_memory_turns(monkeypatch):
    chat = _build_chat(monkeypatch, response=FakeMessage("a"))
    # memory_turns=3，最多保留 6 条消息
    for i in range(10):
        chat.ask(f"q{i}")
    assert len(chat.history) == 6
    # 最后一轮应该是 q9 / a
    assert chat.history[-2].content == "q9"


# ============================================================
# ask_stream 事件序列
# ============================================================
def test_ask_stream_yields_deltas_and_final_result(monkeypatch):
    chunks = [
        FakeMessage("Hello "),
        FakeMessage("World"),
        FakeMessage(
            "!",
            usage_metadata={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
        ),
    ]
    docs = [FakeDoc("ctx", {"source": "docs/a.md"})]
    chat = _build_chat(monkeypatch, docs=docs, stream_chunks=chunks)

    events = list(chat.ask_stream("你好"))

    # 3 个 delta + 1 个 done
    assert len(events) == 4
    assert [e.delta for e in events[:3]] == ["Hello ", "World", "!"]
    assert all(e.done is False for e in events[:3])

    final = events[-1]
    assert final.done is True
    assert final.delta == ""
    assert isinstance(final.result, ChatResult)
    assert final.result.content == "Hello World!"
    assert final.result.sources == ["a.md"]
    assert final.result.usage.prompt_tokens == 10


def test_ask_stream_writes_to_history(monkeypatch):
    chunks = [FakeMessage("完整"), FakeMessage("回答")]
    chat = _build_chat(monkeypatch, stream_chunks=chunks)

    list(chat.ask_stream("流式问题"))

    assert len(chat.history) == 2
    assert chat.history[0].content == "流式问题"
    assert chat.history[1].content == "完整回答"


def test_ask_stream_skips_empty_chunks(monkeypatch):
    chunks = [
        FakeMessage("A"),
        FakeMessage(""),   # 空 chunk，不应该产生 delta 事件
        FakeMessage("B"),
    ]
    chat = _build_chat(monkeypatch, stream_chunks=chunks)

    events = list(chat.ask_stream("q"))
    delta_events = [e for e in events if not e.done]
    assert [e.delta for e in delta_events] == ["A", "B"]
