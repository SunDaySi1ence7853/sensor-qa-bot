"""
共享 fixtures。

设计要点：
1. 每个测试前自动清空所有 lru_cache，避免测试间污染。
2. clean_env fixture：清除所有相关环境变量，测试从干净环境开始。
3. valid_env fixture：设置一套合法的默认配置，便于快速构造测试上下文。
"""

import pytest

from src import config, embeddings, llm

_ENV_KEYS = [
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_MODEL",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_INPUT_PRICE_PER_1M",
    "DEEPSEEK_OUTPUT_PRICE_PER_1M",
    "EMBEDDING_PROVIDER",
    "LOCAL_EMBEDDING_MODEL",
    "DEEPSEEK_EMBEDDING_MODEL",
    "RETRIEVE_TOP_K",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "VECTORSTORE_DIR",
    "MEMORY_TURNS",
]


@pytest.fixture(autouse=True)
def clear_all_caches():
    """
    每个测试前后都清空 lru_cache，保证测试隔离。
    """
    config.get_config.cache_clear()
    embeddings.get_embeddings.cache_clear()
    llm.get_llm.cache_clear()
    yield
    config.get_config.cache_clear()
    embeddings.get_embeddings.cache_clear()
    llm.get_llm.cache_clear()


@pytest.fixture
def clean_env(monkeypatch):
    """
    清除所有可能影响测试的环境变量。
    """
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


@pytest.fixture
def valid_env(clean_env):
    """
    设置一套合法的默认配置。
    """
    clean_env.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    clean_env.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    clean_env.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    clean_env.setenv("DEEPSEEK_INPUT_PRICE_PER_1M", "1.00")
    clean_env.setenv("DEEPSEEK_OUTPUT_PRICE_PER_1M", "2.00")
    clean_env.setenv("EMBEDDING_PROVIDER", "local")
    clean_env.setenv("LOCAL_EMBEDDING_MODEL", "shibing624/text2vec-base-chinese")
    clean_env.setenv("RETRIEVE_TOP_K", "3")
    clean_env.setenv("CHUNK_SIZE", "500")
    clean_env.setenv("CHUNK_OVERLAP", "80")
    clean_env.setenv("VECTORSTORE_DIR", "vectorstore")
    clean_env.setenv("MEMORY_TURNS", "6")
    return clean_env


# ==================== 用于 mock LLM / vectorstore 的伪对象 ====================

class FakeChunk:
    """伪造 LangChain 流式返回的 chunk。"""
    def __init__(self, content: str = "", usage_metadata=None):
        self.content = content
        self.usage_metadata = usage_metadata


class FakeLLM:
    """伪造 ChatOpenAI，用于 rag_chat 测试。"""
    def __init__(self, chunks=None, total_input_tokens=10, total_output_tokens=5):
        if chunks is None:
            chunks = ["Hello ", "world", "!"]
        self._chunks = chunks
        self._input_tokens = total_input_tokens
        self._output_tokens = total_output_tokens
        self.call_count = 0

    def stream(self, messages):
        self.call_count += 1
        for text in self._chunks:
            yield FakeChunk(content=text)
        # 最后一个 chunk 带 usage
        yield FakeChunk(
            content="",
            usage_metadata={
                "input_tokens": self._input_tokens,
                "output_tokens": self._output_tokens,
            },
        )


class FakeDoc:
    def __init__(self, content: str, source: str = "fake.txt"):
        self.page_content = content
        self.metadata = {"source": source}


class FakeVectorStore:
    def __init__(self, docs=None):
        self.docs = docs if docs is not None else [
            FakeDoc("DHT22 是一种数字温湿度传感器。", "dht22.md"),
            FakeDoc("BMP280 是气压传感器。", "bmp280.md"),
        ]
        self.search_calls = []

    def similarity_search(self, query: str, k: int = 3):
        self.search_calls.append((query, k))
        return self.docs[:k]


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def fake_vectorstore():
    return FakeVectorStore()