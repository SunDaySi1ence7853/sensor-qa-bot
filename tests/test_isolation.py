"""验证 conftest 对 sensor_tools 全局状态的隔离。

test_a 故意启动采样线程且不停止，模拟"测试中途断言失败、stop 没执行到"；
test_b 断言自己拿到的是干净环境。没有隔离 fixture 时 test_b 必红。
"""
import threading

from src.tools import sensor_tools as st


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