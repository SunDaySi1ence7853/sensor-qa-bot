"""
tests/test_cli.py

CLI 入口层测试（任务1，4 个必测场景）。

覆盖分支：
  - 缺少 DEEPSEEK_API_KEY  → exit 2
  - EMBEDDING_PROVIDER 非法 → exit 2
  - --help                 → exit 0
  - -q 单次提问（mock RAG）  → exit 0，输出 mock 内容
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from src.cli import app_chat

runner = CliRunner()


# ============================================================
# 场景 1：缺少 DEEPSEEK_API_KEY → exit 2
# ============================================================
def test_missing_api_key_exits_with_code_2(clean_env):
    """无 DEEPSEEK_API_KEY 时，_preflight_check 应退出并提示缺少 key。"""
    clean_env.setenv("EMBEDDING_PROVIDER", "local")
    # 明确不设置 DEEPSEEK_API_KEY

    result = runner.invoke(app_chat, ["-q", "随便问一句"])

    assert result.exit_code == 2
    assert "DEEPSEEK_API_KEY" in result.output


# ============================================================
# 场景 2：EMBEDDING_PROVIDER 非法 → exit 2
# ============================================================
def test_invalid_embedding_provider_exits_with_code_2(clean_env):
    """非法的 EMBEDDING_PROVIDER 应退出并在输出中包含该非法值。"""
    clean_env.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    clean_env.setenv("EMBEDDING_PROVIDER", "invalid_provider")

    result = runner.invoke(app_chat, ["-q", "随便问一句"])

    assert result.exit_code == 2
    # 非法 provider 在 get_config() 阶段抛 ValueError，被 _preflight_check 捕获
    # 输出里会包含 "配置加载失败" 或非法值本身
    assert "配置加载失败" in result.output or "invalid_provider" in result.output


# ============================================================
# 场景 3：--help → exit 0，帮助文本正常
# ============================================================
def test_help_exits_with_code_0():
    """--help 应正常返回 exit 0，并输出帮助文本。"""
    result = runner.invoke(app_chat, ["--help"])

    assert result.exit_code == 0
    # 帮助文本至少包含命令描述关键词
    assert "sensor-qa" in result.output or "问答" in result.output


# ============================================================
# 场景 4：-q 单次提问（mock SensorRAGChat）→ exit 0，输出 mock 内容
# ============================================================
def test_single_question_with_mock_rag(clean_env):
    """
    -q 单次提问：mock 掉 SensorRAGChat，验证 exit 0 且输出包含 mock 答案。

    patch 路径是 src.cli.SensorRAGChat（延迟导入在函数体内），
    不是 src.rag_chat.SensorRAGChat。
    """
    clean_env.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    clean_env.setenv("EMBEDDING_PROVIDER", "local")

    # 构造 mock 返回值，结构与真实 ChatResult / TokenUsage 一致
    from src.rag_chat import ChatResult, TokenUsage

    mock_result = ChatResult(
        content="DHT22 的测量温度范围是 -40°C 到 +80°C。",
        sources=["DHT22.md"],
        usage=TokenUsage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            estimated_cost_cny=0.000123,
        ),
    )

    # mock ask_stream：产出一个 done=True 的 StreamEvent
    from src.rag_chat import StreamEvent

    def mock_ask_stream(question):
        yield StreamEvent(delta="", done=True, result=mock_result)

    mock_chat_instance = MagicMock()
    mock_chat_instance.ask_stream.side_effect = mock_ask_stream

    with patch("src.rag_chat.SensorRAGChat", return_value=mock_chat_instance):
        result = runner.invoke(app_chat, ["-q", "DHT22的温度范围是多少"])

    assert result.exit_code == 0, f"非预期退出码，输出：\n{result.output}"
    assert "DHT22" in result.output or "-40" in result.output or "80" in result.output