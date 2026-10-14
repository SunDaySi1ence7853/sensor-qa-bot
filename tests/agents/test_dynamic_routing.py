"""动态路由测试。
验收标准：
  - degraded → re_collect 节点出现在 routing_path
  - triggered_count == 0 → ai_analysis_lite 节点出现在 routing_path，LLM 未被调用
  - collector offline/None → 直接 supervisor，rule/ai/report 未运行
"""
import json

import pytest

from src.agents.fakes import FAKE_MANUAL, FIXED_NOW, FakeLLM, FakeRetriever, make_readings
from src.agents.supervisor import PipelineConfig, _initial_state, build_pipeline

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
    """第一次返回 degraded 数据（少帧），第二次返回正常数据。"""
    def __init__(self):
        self.calls = 0

    def __call__(self, sensor_type: str):
        self.calls += 1
        if self.calls == 1:
            return make_readings([23.6, 23.7])          # 只有 2 帧，触发 degraded
        return make_readings([23.0, 23.5, 24.0, 24.8, 25.5])


def test_degraded_triggers_re_collect_node():
    reader = _StatefulReader()
    result = _run(_make_pipeline(reader=reader))
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


# ── routing_path 完整性 ──────────────────────────────────────────────────

def test_routing_path_populated_on_happy_path():
    result = _run(_make_pipeline())
    assert "collector" in result["routing_path"]
    assert "supervisor" in result["routing_path"]
    assert "report" in result["routing_path"]