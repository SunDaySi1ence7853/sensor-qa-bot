"""动态路由测试。
验收标准：
  - degraded → re_collect 节点出现在 routing_path
  - triggered_count == 0 → ai_analysis_lite 节点出现在 routing_path，LLM 未被调用
  - collector offline/None → 直接 supervisor，rule/ai/report 未运行
"""
import json
from datetime import timedelta

from src.agents.base import AgentFailure
import pytest

from src.agents.fakes import FAKE_MANUAL, FIXED_NOW, FakeLLM, FakeRetriever, make_readings
from src.agents.supervisor import PipelineConfig, _initial_state, build_pipeline
from src.agents.fakes import (
    FAKE_MANUAL, FIXED_NOW, FakeLLM, FakeRetriever, make_gapped_readings, make_readings,
)
from src.agents.collector import CollectorAgent

GOOD_LLM_RESP = json.dumps(
    {"root_cause": "散热不良", "recommendation": "检查风扇", "cited": [2]},
    ensure_ascii=False,
)


def _make_pipeline(rows=None, llm_resp=GOOD_LLM_RESP, docs=None, reader=None):
    if rows is not None:
        _reader = lambda s: rows
    elif reader is not None:
        _reader = reader
    else:
        _reader = lambda s: make_readings([23.0, 23.5, 24.0, 24.8, 25.5])
    cfg = PipelineConfig(
        llm=FakeLLM(llm_resp),
        retriever=FakeRetriever(docs if docs is not None else FAKE_MANUAL),
        reader=_reader,
        clock=lambda: FIXED_NOW,
    )
    return build_pipeline(cfg)


def _run(p, sensor="temperature"):
    return p.invoke(_initial_state(sensor, 1.0))


# ── 路径A：补采路径 ───────────────────────────────────────────────────────

class _StatefulReader:
    """第一次返回带时间洞的数据（degraded），第二次返回正常的上升趋势数据。"""
    def __init__(self):
        self.calls = 0

    def __call__(self, sensor_type: str):
        self.calls += 1
        if self.calls == 1:
            return make_gapped_readings([23.6] * 10)
        return make_readings([23.0, 23.5, 24.0, 24.8, 25.5])


def test_degraded_triggers_re_collect_node():
    # 前置：构造的数据必须真的判成 degraded，否则这条测试测的是空气
    probe = CollectorAgent(
        reader=lambda s: make_gapped_readings([23.6] * 10),
        clock=lambda: FIXED_NOW,
        interval_sec=30.0,
    ).run({"sensor_type": "temperature", "window_hours": 1.0})
    assert probe["data_quality"] == "degraded"

    result = _run(_make_pipeline(reader=_StatefulReader()))
    assert "re_collect" in result["routing_path"]


def test_re_collect_reader_called_twice():
    reader = _StatefulReader()
    _run(_make_pipeline(reader=reader))
    assert reader.calls == 2


def test_degraded_path_status_complete():
    result = _run(_make_pipeline(reader=_StatefulReader()))
    assert result["status"] == "complete"


# ── 路径B：全绿→AI 简化 ─────────────────────────────────────────────────

def test_all_green_routes_to_ai_lite():
    result = _run(_make_pipeline(rows=make_readings([23.6] * 10)))
    assert "ai_analysis_lite" in result["routing_path"]
    assert "ai_analysis" not in result["routing_path"]


def test_all_green_no_llm_call():
    llm = FakeLLM(GOOD_LLM_RESP)
    cfg = PipelineConfig(
        llm=llm,
        retriever=FakeRetriever(FAKE_MANUAL),
        reader=lambda s: make_readings([23.6] * 10),
        clock=lambda: FIXED_NOW,
    )
    pipeline = build_pipeline(cfg)
    pipeline.invoke(_initial_state("temperature", 1.0))
    assert llm.calls == []


def test_all_green_report_complete():
    result = _run(_make_pipeline(rows=make_readings([23.6] * 10)))
    assert result["status"] == "complete"
    assert result["report"] is not None


def test_all_green_ai_lite_recommendation_present():
    result = _run(_make_pipeline(rows=make_readings([23.6] * 10)))
    assert result["ai_analysis"]["recommendation"]


# ── 正常路径用完整 AI（回归验证）─────────────────────────────────────────

def test_triggered_rules_use_full_ai():
    result = _run(_make_pipeline())   # 上升趋势，R03 触发
    assert "ai_analysis" in result["routing_path"]
    assert "ai_analysis_lite" not in result["routing_path"]


# ── collector 失败→直接 supervisor ─────────────────────────────────────

def test_collector_fail_goes_directly_to_supervisor():
    result = _run(_make_pipeline(), sensor="pressure")
    assert "re_collect" not in result["routing_path"]
    assert "rule" not in result["routing_path"]
    assert result["status"] == "failed"


def test_collector_fail_no_downstream_errors():
    """动态路由下 collector 失败直达 supervisor，不产生 rule/ai/report 的 skip 错误。"""
    result = _run(_make_pipeline(), sensor="pressure")
    agent_names = [e["agent"] for e in result["errors"]]
    assert "rule" not in agent_names
    assert "ai_analysis" not in agent_names
    assert "report" not in agent_names
# ── collector offline → 直接 supervisor，判 failed ─────────────────────

@pytest.mark.parametrize(
    "case, rows, clock_shift",
    [
        ("空 buffer", [], timedelta(0)),
        # 数据仍在 1h 窗口内，但最新一帧离 now 超过 stale 阈值（3×30s）
        ("数据过旧", make_readings([23.6] * 10), timedelta(minutes=10)),
    ],
)
def test_offline_routes_to_supervisor_failed(case, rows, clock_shift):
    now = FIXED_NOW + clock_shift

    # 前置：构造的数据必须真的判成 offline，否则测的是空气
    probe = CollectorAgent(
        reader=lambda s: rows, clock=lambda: now, interval_sec=30.0,
    ).run({"sensor_type": "temperature", "window_hours": 1.0})
    assert probe["data_quality"] == "offline", f"{case}: 前置数据没判成 offline"

    cfg = PipelineConfig(
        llm=FakeLLM(GOOD_LLM_RESP),
        retriever=FakeRetriever(FAKE_MANUAL),
        reader=lambda s: rows,
        clock=lambda: now,
    )
    result = build_pipeline(cfg).invoke(_initial_state("temperature", 1.0))

    # 路由：直达 supervisor，下游一个都没跑
    for node in ("re_collect", "rule", "ai_analysis", "ai_analysis_lite", "report"):
        assert node not in result["routing_path"], f"{case}: {node} 不该运行"
    assert "supervisor" in result["routing_path"]

    # 状态：failed，且 errors 里说清楚原因（修复前是 partial + 空列表）
    assert result["status"] == "failed"
    assert any("offline" in e["reason"] for e in result["errors"])


# ── 补采失败 → 保留首次采集结果，管线继续 ──────────────────────────────

class _ReCollectFailReader:
    """第一次返回 degraded 数据，第二次（补采）抛 AgentFailure。"""
    def __init__(self):
        self.calls = 0

    def __call__(self, sensor_type: str):
        self.calls += 1
        if self.calls == 1:
            return make_gapped_readings([23.6] * 10)
        raise AgentFailure("collector", "模拟补采时数据源不可用")


def test_re_collect_failure_keeps_first_collect():
    first = CollectorAgent(
        reader=lambda s: make_gapped_readings([23.6] * 10),
        clock=lambda: FIXED_NOW,
        interval_sec=30.0,
    ).run({"sensor_type": "temperature", "window_hours": 1.0})
    assert first["data_quality"] == "degraded"

    reader = _ReCollectFailReader()
    result = _run(_make_pipeline(reader=reader))

    assert reader.calls == 2                          # 确实补采了一次
    assert "re_collect" in result["routing_path"]
    assert result["collector"] == first               # 首次产物没被覆盖
    assert any(
        e["agent"] == "re_collect" and "补采失败" in e["reason"]
        for e in result["errors"]
    )
    assert "rule" in result["routing_path"]           # 补采失败不中断，继续往下走

# ── routing_path 完整性 ──────────────────────────────────────────────────

def test_routing_path_populated_on_happy_path():
    result = _run(_make_pipeline())
    assert "collector" in result["routing_path"]
    assert "supervisor" in result["routing_path"]
    assert "report" in result["routing_path"]