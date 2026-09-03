"""
统一日志配置。

设计要点：
1. 只在应用入口（main.py / build_index.py）调用一次 setup_logging()，
   其他模块通过 logging.getLogger(__name__) 拿 logger，配置由 root 继承。
2. 终端 handler 和文件 handler 独立控制级别：
   终端跟随 --debug 参数，文件永远记录 DEBUG 全量。
3. 使用 TimedRotatingFileHandler 按天轮转，避免日志无限增长。
4. logs 目录不存在时自动创建。
5. 幂等：多次调用 setup_logging() 不会重复添加 handler。
"""

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"

_CONSOLE_FORMAT = "[%(levelname)s] %(message)s"
_FILE_FORMAT = (
    "%(asctime)s | %(levelname)-7s | %(name)s:%(lineno)d | %(message)s"
)

_INITIALIZED = False


def setup_logging(debug: bool = False, log_dir: Path | None = None) -> None:
    """
    初始化日志系统。

    Args:
        debug: True 时终端输出 DEBUG，否则输出 INFO
        log_dir: 自定义日志目录，默认 logs/
    """
    global _INITIALIZED
    if _INITIALIZED:
        # 幂等：允许重复调用但只生效一次
        # 支持运行时切换级别
        _update_console_level(debug)
        return

    target_dir = log_dir or LOG_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    # 根 logger 设最低级别，让 handler 各自过滤
    root_logger.setLevel(logging.DEBUG)

    # 清理可能存在的默认 handler
    root_logger.handlers.clear()

    # ---------- 终端 handler ----------
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    console_handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT))
    console_handler.set_name("console")
    root_logger.addHandler(console_handler)

    # ---------- 全量文件 handler ----------
    app_file = TimedRotatingFileHandler(
        target_dir / "app.log",
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8",
    )
    app_file.setLevel(logging.DEBUG)
    app_file.setFormatter(logging.Formatter(_FILE_FORMAT))
    app_file.suffix = "%Y-%m-%d"
    app_file.set_name("app_file")
    root_logger.addHandler(app_file)

    # ---------- 错误文件 handler ----------
    error_file = TimedRotatingFileHandler(
        target_dir / "error.log",
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8",
    )
    error_file.setLevel(logging.ERROR)
    error_file.setFormatter(logging.Formatter(_FILE_FORMAT))
    error_file.suffix = "%Y-%m-%d"
    error_file.set_name("error_file")
    root_logger.addHandler(error_file)

    # 降低第三方库的噪音
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    _INITIALIZED = True

    logging.getLogger(__name__).debug(
        "日志系统已初始化 | debug=%s | log_dir=%s", debug, target_dir
    )


def _update_console_level(debug: bool) -> None:
    """
    运行时切换终端日志级别（不影响文件 handler）。
    """
    level = logging.DEBUG if debug else logging.INFO
    for h in logging.getLogger().handlers:
        if h.get_name() == "console":
            h.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """
    统一 logger 获取入口。虽然直接用 logging.getLogger 也行，
    但通过这个入口便于未来加自定义 adapter。
    """
    return logging.getLogger(name)