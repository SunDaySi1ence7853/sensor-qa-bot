import json

import pytest

from src.agents import AgentFailure, AIAnalysisAgent
from src.agents.ai_agent import SYSTEM_PROMPT
from src.agents.fakes import FAKE_MANUAL, FakeLLM, FakeRetriever, make_readings

RULE_WARN = {
    "items": [{"rule_id": "R03_RISING_TREND", "level": "warn", "evidence": "末 5 点单调上升，累计 +2.5"}],
    "triggered_count": 1,
}
GOOD = {"root_cause": "散热不良", "recommendation": "检查风扇", "cited": [2]}


def _inp():
    return {"readings": make_readings([23.0, 23.5, 24.0, 24.8, 25.5]),
            "rule_result": RULE_WARN, "sensor_type": "temperature"}


def _llm(obj):
    return FakeLLM(json.dumps(obj, ensure_ascii=False))


def test_happy_path_returns_cited_original_text():
    out = AIAnalysisAgent(_llm(GOOD), FakeRetriever(FAKE_MANUAL)).run(_inp())
    assert out["references"] == [FAKE_MANUAL[1]]
    assert out["root_cause"] == "散热不良"


def test_prompt_carries_rule_evidence_and_numbered_refs():
    llm = _llm(GOOD)
    AIAnalysisAgent(llm, FakeRetriever(FAKE_MANUAL)).run(_inp())
    system, user = llm.calls[0]
    assert system == SYSTEM_PROMPT
    assert "R03_RISING_TREND" in user and "[1]" in user and "[2]" in user


def test_query_targets_abnormal_rules():
    retriever = FakeRetriever(FAKE_MANUAL)
    AIAnalysisAgent(_llm(GOOD), retriever).run(_inp())
    assert "持续上升" in retriever.queries[0]


def test_no_knowledge_refuses_without_calling_llm():
    llm = _llm(GOOD)
    with pytest.raises(AgentFailure):
        AIAnalysisAgent(llm, FakeRetriever([])).run(_inp())
    assert llm.calls == []


@pytest.mark.parametrize("bad_output", [
    "我觉得是风扇坏了",                                                       # 非 JSON
    json.dumps({"root_cause": "x", "recommendation": "y", "cited": [5]}),   # 引用越界
    json.dumps({"root_cause": "x", "recommendation": "y", "cited": []}),    # 无引用
])
def test_ungrounded_output_raises(bad_output):
    with pytest.raises(AgentFailure):
        AIAnalysisAgent(FakeLLM(bad_output), FakeRetriever(FAKE_MANUAL)).run(_inp())


def test_fenced_json_accepted():
    fenced = "```json\n" + json.dumps(GOOD, ensure_ascii=False) + "\n```"
    out = AIAnalysisAgent(FakeLLM(fenced), FakeRetriever(FAKE_MANUAL)).run(_inp())
    assert out["recommendation"] == "检查风扇"