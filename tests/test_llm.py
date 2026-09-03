# tests/test_llm.py
from unittest.mock import MagicMock

from src import llm as llm_module


class TestLLM:
    def test_get_llm_creates_chat_openai(self, valid_env, monkeypatch):
        fake_cls = MagicMock()
        monkeypatch.setattr("langchain_openai.ChatOpenAI", fake_cls)
        llm_module.get_llm()
        fake_cls.assert_called_once()

    def test_get_llm_streaming_flag(self, valid_env, monkeypatch):
        fake_cls = MagicMock()
        monkeypatch.setattr("langchain_openai.ChatOpenAI", fake_cls)
        llm_module.get_llm(streaming=True)
        _, kwargs = fake_cls.call_args
        assert kwargs.get("streaming") is True