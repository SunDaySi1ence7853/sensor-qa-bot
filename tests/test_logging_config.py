import io
import logging
import sys

import src.logging_config as lc


def test_console_handler_follows_current_stderr(monkeypatch, tmp_path):
    """初始化时的 stderr 被关闭后，日志应写到新的 stderr，而不是报 closed file。

    复现 test_cli 场景：CliRunner 替换 stderr → setup_logging → 关闭临时流。
    """
    monkeypatch.setattr(lc, "_INITIALIZED", False)
    temp = io.StringIO()
    monkeypatch.setattr(sys, "stderr", temp)
    lc.setup_logging(log_dir=tmp_path)
    temp.close()

    later = io.StringIO()
    monkeypatch.setattr(sys, "stderr", later)
    logging.getLogger("regress").warning("hello-after-close")

    out = later.getvalue()
    # 修复前 handleError 也会把 "Message: 'hello-after-close'" 打到 later，
    # 所以必须同时断言格式化后的正文和没有 Logging error，否则会误判为绿
    assert "Logging error" not in out, out
    assert "[WARNING] hello-after-close" in out, out