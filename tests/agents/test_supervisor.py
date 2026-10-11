import json

import pytest

from src.agents.fakes import FAKE_MANUAL, FIXED_NOW, FakeLLM, FakeRetriever, make_readings
from src.agents.supervisor import PipelineConfig, _initial_state, build_pipeline

# ── 共用 fixture ────────────────────────────────────────────────────────

GOOD_LLM_RESP = json.dumps(
    {"root_cause": "散热不良", "recommendation": "检查风扇", "cited": [2]},
    ensure_ascii=False,
)


def _pipeline(rows=None, llm_resp=GOOD_LLM_RESP, docs=None):
    if rows is None:
        rows = make_readings([23.0, 23.5, 24.0, 24.8, 25.5])
    if docs is None:
        docs = FAKE_MANUAL
    cfg = PipelineConfig(
        llm=FakeLLM(llm_resp),
        retriever=FakeRetriever(docs),
        reader=lambda s: rows,
        clock=lambda: FIXED_NOW,
    )
    return build_pipeline(cfg)


def _run(pipeline, sensor="temperature", hours=1.0):
    return pipeline.invoke(_initial_state(sensor, hours))


# ── 正常链路 ────────────────────────────────────────────────────────────

def test_full_pipeline_status_complete():
    result = _run(_pipeline())
    assert result["status"] == "complete"
    assert result["errors"] == []


def test_full_pipeline_report_populated():
    result = _run(_pipeline())
    report = result["report"]
    assert report is not None
    assert report["summary"] and report["confidence"] > 0
    assert len(report["evidence_chain"]) >= 3
    assert report["timestamp"] == FIXED_NOW.isoformat(timespec="seconds")


def test_full_pipeline_all_agent_outputs_present():
    result = _run(_pipeline())
    assert result["collector"] is not None
    assert result["rule"] is not None
    assert result["ai_analysis"] is not None
    assert result["report"] is not None


def test_data_propagates_through_nodes():
    """每层产物来自上一层，不是硬编结果。"""
    rows = make_readings([24.5] * 5)
    result = _run(_pipeline(rows=rows))
    assert result["collector"]["readings"][0]["value"] == 24.5
    assert result["rule"]["items"]  # 规则引擎跑了


# ── Collector 失败 ──────────────────────────────────────────────────────

def test_collector_failure_sets_status_failed():
    result = _run(_pipeline(), sensor="pressure")  # 不支持的传感器
    assert result["status"] == "failed"
    assert any(e["agent"] == "collector" for e in result["errors"])


def test_collector_failure_downstream_all_skip():
    result = _run(_pipeline(), sensor="pressure")
    assert result["rule"] is None
    assert result["ai_analysis"] is None
    assert result["report"] is None


def test_collector_failure_cascades_to_three_skip_errors():
    result = _run(_pipeline(), sensor="pressure")
    # collector 本身 1 条 + rule/ai/report 各跳过 1 条 = 4
    assert len(result["errors"]) == 4


# ── AI Agent 失败（知识库为空）─────────────────────────────────────────

def test_ai_failure_status_partial():
    result = _run(_pipeline(docs=[]))    # 空检索库 → AIAnalysisAgent 抛 AgentFailure
    assert result["status"] == "partial"
    assert result["collector"] is not None
    assert result["rule"] is not None
    assert result["ai_analysis"] is None
    assert result["report"] is None


def test_ai_failure_no_llm_call():
    """空库时不应调 LLM。"""
    llm = FakeLLM(GOOD_LLM_RESP)
    cfg = PipelineConfig(
        llm=llm,
        retriever=FakeRetriever([]),
        reader=lambda s: make_readings([23.6] * 5),
        clock=lambda: FIXED_NOW,
    )
    _run(build_pipeline(cfg))
    assert llm.calls == []


# ── 错误累计 ────────────────────────────────────────────────────────────

def test_errors_accumulate_not_overwrite():
    """reducer 保证每个节点的错误都在列表里，不互相覆盖。"""
    result = _run(_pipeline(), sensor="pressure")
    agents = [e["agent"] for e in result["errors"]]
    assert len(agents) == len(set(agents)) + 0  # 每 agent 最多一条
    assert "collector" in agents


def test_error_record_shape():
    result = _run(_pipeline(), sensor="pressure")
    for err in result["errors"]:
        assert "agent" in err and "reason" in err
        assert isinstance(err["reason"], str) and err["reason"]


# ── 初始状态辅助函数 ────────────────────────────────────────────────────

def test_initial_state_defaults():
    s = _initial_state("humidity", 2.0)
    assert s["sensor_type"] == "humidity"
    assert s["window_hours"] == 2.0
    assert s["errors"] == []
    assert s["status"] == "running"
    assert s["collector"] is None