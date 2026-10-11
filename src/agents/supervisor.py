"""固定管线：collector → rule → ai_analysis → report → supervisor → END

施工约定（来自任务1变更记录）：
- AgentFailure 在节点内 except，不往图外传播
- 上游产物缺失时节点主动跳过，向 errors 追加一项，下游同理
- supervisor 节点是 10-18 动态路由的挂载点，现在只判 status
- 真实 DeepSeek / search_manual 在这里注入；骨架期传 FakeLLM / FakeRetriever
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

from langgraph.graph import END, StateGraph

from .base import AgentFailure
from .collector import CollectorAgent
from .ai_agent import AIAnalysisAgent
from .report_agent import ReportAgent
from .rule_agent import RuleAgent
from .state import DiagnosisState

LLM = Callable[[str, str], str]
Retriever = Callable[[str], list]
Reader = Callable[[str], list]
Clock = Callable[[], datetime]


@dataclass
class PipelineConfig:
    llm: LLM
    retriever: Retriever
    reader: Optional[Reader] = None          # None → 用真实 sensor_tools._history_buffer
    clock: Optional[Clock] = None            # None → datetime.now
    interval_sec: float = 30.0


def build_pipeline(cfg: PipelineConfig):
    """编译并返回固定管线的 CompiledGraph。"""

    _collector = CollectorAgent(
        reader=cfg.reader,
        clock=cfg.clock,
        interval_sec=cfg.interval_sec,
    )
    _rule = RuleAgent()
    _ai = AIAnalysisAgent(llm=cfg.llm, retriever=cfg.retriever)
    _report = ReportAgent(clock=cfg.clock)

    # ── 节点函数 ────────────────────────────────────────────────────────

    def node_collector(state: DiagnosisState) -> dict:
        try:
            out = _collector.run({
                "sensor_type": state["sensor_type"],
                "window_hours": state["window_hours"],
            })
            return {"collector": out}
        except AgentFailure as e:
            return {"errors": [{"agent": e.agent, "reason": e.reason}]}

    def node_rule(state: DiagnosisState) -> dict:
        if state.get("collector") is None:
            return {"errors": [{"agent": "rule", "reason": "上游 collector 未产出，跳过"}]}
        try:
            out = _rule.run({
                "readings": state["collector"]["readings"],
                "sensor_type": state["sensor_type"],
            })
            return {"rule": out}
        except AgentFailure as e:
            return {"errors": [{"agent": e.agent, "reason": e.reason}]}

    def node_ai(state: DiagnosisState) -> dict:
        if state.get("rule") is None:
            return {"errors": [{"agent": "ai_analysis", "reason": "上游 rule 未产出，跳过"}]}
        try:
            out = _ai.run({
                "readings": state["collector"]["readings"],
                "rule_result": state["rule"],
                "sensor_type": state["sensor_type"],
            })
            return {"ai_analysis": out}
        except AgentFailure as e:
            return {"errors": [{"agent": e.agent, "reason": e.reason}]}

    def node_report(state: DiagnosisState) -> dict:
        if state.get("ai_analysis") is None:
            return {"errors": [{"agent": "report", "reason": "上游 ai_analysis 未产出，跳过"}]}
        try:
            out = _report.run({
                "collector": state["collector"],
                "rule": state["rule"],
                "ai_analysis": state["ai_analysis"],
            })
            return {"report": out}
        except AgentFailure as e:
            return {"errors": [{"agent": e.agent, "reason": e.reason}]}

    def node_supervisor(state: DiagnosisState) -> dict:
        """汇总管线状态；10-18 在这里改成条件边实现动态路由。"""
        errors = state.get("errors") or []
        if state.get("report") is not None:
            # 报告产出：即使有中间错误（如数据质量降级），也算完成
            return {"status": "complete"}
        collector_failed = any(e["agent"] == "collector" for e in errors)
        if collector_failed:
            return {"status": "failed"}
        return {"status": "partial"}

    # ── 图装配 ──────────────────────────────────────────────────────────

    g = StateGraph(DiagnosisState)
    g.add_node("collector", node_collector)
    g.add_node("rule", node_rule)
    g.add_node("ai_analysis", node_ai)
    g.add_node("report", node_report)
    g.add_node("supervisor", node_supervisor)

    g.set_entry_point("collector")
    g.add_edge("collector", "rule")
    g.add_edge("rule", "ai_analysis")
    g.add_edge("ai_analysis", "report")
    g.add_edge("report", "supervisor")
    g.add_edge("supervisor", END)

    return g.compile()


def _initial_state(sensor_type: str, window_hours: float) -> DiagnosisState:
    """构造干净的初始状态，便于 invoke 调用方不用手写全部字段。"""
    return {
        "sensor_type": sensor_type,
        "window_hours": window_hours,
        "collector": None,
        "rule": None,
        "ai_analysis": None,
        "report": None,
        "errors": [],
        "status": "running",
    }


if __name__ == "__main__":
    from .fakes import FAKE_MANUAL, FIXED_NOW, FakeLLM, FakeRetriever, make_readings

    rows = make_readings([23.0, 23.5, 24.0, 24.8, 25.5])
    cfg = PipelineConfig(
        llm=FakeLLM(json.dumps({
            "root_cause": "温度单调上升，符合散热不良特征",
            "recommendation": "检查散热风扇与通风口，必要时降载",
            "cited": [2],
        }, ensure_ascii=False)),
        retriever=FakeRetriever(FAKE_MANUAL),
        reader=lambda s: rows,
        clock=lambda: FIXED_NOW,
    )

    pipeline = build_pipeline(cfg)
    result = pipeline.invoke(_initial_state("temperature", 1.0))

    print("===== 状态 =====")
    print("status:", result["status"])
    print("errors:", result["errors"])
    print("===== 报告摘要 =====")
    if result["report"]:
        print(result["report"]["summary"])
        print("置信度:", result["report"]["confidence"])
        print("依据链:")
        for line in result["report"]["evidence_chain"]:
            print(" ", line)