"""
pytest 公共 fixture。

要点：
- autouse 清空 lru_cache：防止 get_config / get_embeddings 在测试间串味
- clean_env 会同时禁用 load_dotenv，防止 .env 污染测试
- autouse 隔离 sensor_tools 全局状态：采样线程 / hub / buffer / 失败计数
"""

from __future__ import annotations

import logging
import os
import sys
import threading

import pytest

os.environ["SENSOR_DATA_SOURCE"] = "simulated"  # 测试固定模拟源，不随 config.yaml 的串口演示配置漂移

@pytest.fixture(autouse=True)
def _clear_caches():
    """每个测试前后清空所有 lru_cache 的模块级缓存。

    embeddings 只在**已经导入过**时清理：不主动 import，避免给只跑 tests/agents
    的干净机器（未装 torch / langchain_huggingface）强加 RAG 依赖 —— 那会让
    整轮 pytest 在收集阶段就崩掉，而不是只跳过 RAG 相关用例。
    """
    from src import config as config_mod

    def mods():
        found = [config_mod]
        embeddings_mod = sys.modules.get("src.embeddings")
        if embeddings_mod is not None:
            found.append(embeddings_mod)
        return found

    def clear_all():
        for mod in mods():
            for name in dir(mod):
                obj = getattr(mod, name)
                cache_clear = getattr(obj, "cache_clear", None)
                if callable(cache_clear):
                    cache_clear()

    clear_all()
    yield
    clear_all()

_SENSOR_TOOLS = "src.tools.sensor_tools"


def _reset_sensor_tools_state() -> list[str]:
    """把 sensor_tools 的全局状态恢复到初始值，返回仍存活的采样线程描述。

    只在模块已被导入时处理：不主动 import，避免给不相关的测试引入 langchain/vectorstore 依赖。
    顺序有讲究：先停线程，否则清完 buffer 它会接着写。
    """
    mod = sys.modules.get(_SENSOR_TOOLS)
    if mod is None:
        return []

    # 1. 停采样线程（stop 用 Event 唤醒，正常情况立即返回）
    mod.stop_sampler(timeout=2.0)
    # 绕过 start_sampler 直接 new _Sampler 的线程，stop_sampler 管不到，按线程名兜底检查
    leaked = [f"{t.name}(ident={t.ident})" for t in threading.enumerate()
              if t.name == mod._Sampler.THREAD_NAME and t.is_alive()]

    # 2. 重置数据源单例：下一条测试按当前环境变量重新解析 serial/simulated
    mod.reset_sensor_hub()

    # 3. 清 buffer：必须持锁，与 _record/_snapshot 保持同一把锁
    with mod._buffer_lock:
        for dq in mod._history_buffer.values():
            dq.clear()  # 原地清空，保留 deque 对象和 maxlen

    # 4. 清连续失败计数
    for key in mod._fail_streak:
        mod._fail_streak[key] = 0

    return leaked


@pytest.fixture(autouse=True)
def _isolate_sensor_tools():
    """每条测试前后重置采样线程 / hub / buffer / 失败计数，消除测试间竞态。"""
    _reset_sensor_tools_state()
    yield
    leaked = _reset_sensor_tools_state()
    if leaked:
        # 停不掉的线程会继续污染后续测试，在源头这条测试上直接报出来，而不是让后面的测试随机红
        pytest.fail(f"测试结束时采样线程仍存活：{leaked}", pytrace=False)

@pytest.fixture
def clean_env(monkeypatch):
    for key in list(os.environ.keys()):
        if key.startswith((
            "DEEPSEEK_", "EMBEDDING_", "LOCAL_", "VECTORSTORE_",
            "RETRIEVE_", "CHUNK_", "MEMORY_",
        )):
            monkeypatch.delenv(key, raising=False)

    # 从源头禁用：patch dotenv 库本身，
    # 这样 reload(src.config) 里的 `from dotenv import load_dotenv`
    # 拿到的也是空操作版本。
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **kw: False)

    return monkeypatch


_APP_LOG_HANDLERS = {"console", "app_file", "error_file"}


@pytest.fixture(autouse=True)
def _isolate_app_logging(monkeypatch, tmp_path):
    """日志系统隔离：每个测试都从未初始化状态开始，且不写真实 logs/。

    只清理 setup_logging 挂的具名 handler，不碰 pytest 的 LogCaptureHandler。
    """
    import src.logging_config as lc

    monkeypatch.setattr(lc, "_INITIALIZED", False)
    monkeypatch.setattr(lc, "LOG_DIR", tmp_path / "logs")
    root = logging.getLogger()
    saved_level = root.level
    yield
    for h in [h for h in root.handlers if h.get_name() in _APP_LOG_HANDLERS]:
        root.removeHandler(h)
        h.close()  # Windows 下不关闭的话 tmp_path 里的 .log 文件会被占用
    root.setLevel(saved_level)


@pytest.fixture
def valid_env(clean_env):
    """预设一套合法的环境变量，供不想从零设置的测试用。"""
    clean_env.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    clean_env.setenv("EMBEDDING_PROVIDER", "local")
    return clean_env