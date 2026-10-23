"""验证 conftest 对 sensor_tools 全局状态的隔离。

test_a 故意启动采样线程且不停止，模拟"测试中途断言失败、stop 没执行到"；
test_b 断言自己拿到的是干净环境。没有隔离 fixture 时 test_b 必红。
"""
import logging
import threading

import src.logging_config as lc
from src.tools import sensor_tools as st

_APP_HANDLERS = {"console", "app_file", "error_file"}


def _alive_sampler_threads():
    return [t for t in threading.enumerate()
            if t.name == st._Sampler.THREAD_NAME and t.is_alive()]


def test_a_pollute_on_purpose():
    s = st.start_sampler(interval_sec=0.05)
    assert s.ticked.wait(2.0), "采样线程 2s 内没完成一轮"
    assert any(len(dq) for dq in st._history_buffer.values()), "前置：buffer 应已有数据"
    # 故意不调用 stop_sampler()


def test_b_starts_clean():
    assert st.get_sampler() is None
    assert _alive_sampler_threads() == []
    assert all(len(dq) == 0 for dq in st._history_buffer.values())
    assert st._hub is None
    assert all(v == 0 for v in st._fail_streak.values())


def test_logging_leak_1_dirty(tmp_path):
    """故意初始化日志系统，模拟 test_cli 的行为。"""
    lc.setup_logging(log_dir=tmp_path)
    assert lc._INITIALIZED is True


def test_logging_leak_2_clean():
    """上一个测试挂上的 handler 不应泄漏到这里。"""
    leaked = [h.get_name() for h in logging.getLogger().handlers if h.get_name() in _APP_HANDLERS]
    assert leaked == [], f"日志 handler 泄漏到下一个测试：{leaked}"
    assert lc._INITIALIZED is False


def test_logging_never_writes_real_logs_dir():
    """测试期间 LOG_DIR 不应指向项目真实的 logs/。"""
    assert lc.LOG_DIR != lc.PROJECT_ROOT / "logs"
