"""动态管线：
  正常路径：collector → rule → ai_analysis → report → supervisor → END
  补采路径：collector →[degraded]→ re_collect → rule → ai_analysis → report → supervisor → END
  全绿路径：collector → rule →[all_green]→ ai_analysis_lite → report → supervisor → END

路由规则（挂载点，10-18 落地）：
  route_after_collector:
    None / offline → supervisor（直接判 failed，后续节点不跑）
    degraded       → re_collect（补采一次，不管结果都继续）
    ok             → rule
  route_after_rule:
    None           → report（直接跳 AI，report 会记缺失）
    triggered == 0 → ai_analysis_lite（全绿快速确认，跳 RAG）
    triggered > 0  → ai_analysis（完整 RAG + LLM 根因分析）

任务2 变更记录：Supervisor 按 except AgentFailure 捕获（见设计稿变更记录）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
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
    reader: Optional[Reader] = None
    clock: Optional[Clock] = None
    interval_sec: float = 30.0


def build_pipeline(cfg: PipelineConfig):
    """编译并返回动态路由管线的 CompiledGraph。"""

    _collector = CollectorAgent(
        reader=cfg.reader,
        clock=cfg.clock,
        interval_sec=cfg.interval_sec,
    )
    _rule = RuleAgent()
    _ai = AIAnalysisAgent(llm=cfg.llm, retriever=cfg.retriever)
    _report = ReportAgent(clock=cfg.clock)

    # ── 节点函数 ─────────────────────────────────────────────────────────

    def node_collector(state: DiagnosisState) -> dict:
        try:
            out = _collector.run({
                "sensor_type": state["sensor_type"],
                "window_hours": state["window_hours"],
            })
            return {"collector": out, "routing_path": ["collector"]}
        except AgentFailure as e:
            return {
                "errors": [{"agent": e.agent, "reason": e.reason}],
                "routing_path": ["collector"],
            }

    def node_re_collect(state: DiagnosisState) -> dict:
        """补采节点：degraded 时尝试重新拉一次数据，结果不管好坏都继续。"""
        try:
            out = _collector.run({
                "sensor_type": state["sensor_type"],
                "window_hours": state["window_hours"],
            })
            return {"collector": out, "routing_path": ["re_collect"]}
        except AgentFailure as e:
            # 补采也失败：保留原始产物，记录并继续
            return {
                "errors": [{"agent": "re_collect", "reason": f"补采失败：{e.reason}"}],
                "routing_path": ["re_collect"],
            }

    def node_rule(state: DiagnosisState) -> dict:
        if state.get("collector") is None:
            return {
                "errors": [{"agent": "rule", "reason": "上游 collector 未产出，跳过"}],
                "routing_path": ["rule(skip)"],
            }
        try:
            out = _rule.run({
                "readings": state["collector"]["readings"],
                "sensor_type": state["sensor_type"],
            })
            return {"rule": out, "routing_path": ["rule"]}
        except AgentFailure as e:
            return {
                "errors": [{"agent": e.agent, "reason": e.reason}],
                "routing_path": ["rule(fail)"],
            }

    def node_ai(state: DiagnosisState) -> dict:
        if state.get("rule") is None:
            return {
                "errors": [{"agent": "ai_analysis", "reason": "上游 rule 未产出，跳过"}],
                "routing_path": ["ai_analysis(skip)"],
            }
        try:
            out = _ai.run({
                "readings": state["collector"]["readings"],
                "rule_result": state["rule"],
                "sensor_type": state["sensor_type"],
            })
            return {"ai_analysis": out, "routing_path": ["ai_analysis"]}
        except AgentFailure as e:
            return {
                "errors": [{"agent": e.agent, "reason": e.reason}],
                "routing_path": ["ai_analysis(fail)"],
            }

    def node_ai_lite(state: DiagnosisState) -> dict:
        """全绿快速确认节点：跳过 RAG，不消耗 token，直接产出无异常结论。"""
        if state.get("rule") is None:
            return {
                "errors": [{"agent": "ai_analysis_lite", "reason": "上游 rule 未产出，跳过"}],
                "routing_path": ["ai_analysis_lite(skip)"],
            }
        out = {
            "root_cause": "规则全部正常，当前读数在安全范围内，无需深入分析",
            "recommendation": "维持当前运行状态，按计划执行周期巡检",
            "references": [],
        }
        return {"ai_analysis": out, "routing_path": ["ai_analysis_lite"]}

    def node_report(state: DiagnosisState) -> dict:
        if state.get("ai_analysis") is None:
            return {
                "errors": [{"agent": "report", "reason": "上游 ai_analysis 未产出，跳过"}],
                "routing_path": ["report(skip)"],
            }
        try:
            out = _report.run({
                "collector": state["collector"],
                "rule": state["rule"],
                "ai_analysis": state["ai_analysis"],
            })
            return {"report": out, "routing_path": ["report"]}
        except AgentFailure as e:
            return {
                "errors": [{"agent": e.agent, "reason": e.reason}],
                "routing_path": ["report(fail)"],
            }

    def node_supervisor(state: DiagnosisState) -> dict:
        """汇总管线状态。动态路由在 collector/rule 之后的条件边；这里只判 status。

        failed 的两种来源：
          - collector 抛 AgentFailure（errors 里有 collector 记录）
          - collector 正常返回但 data_quality=offline（collector 按设计不记 error，
            由这里补记，否则会落到 partial 且 errors 为空）
        """
        errors = state.get("errors") or []
        if state.get("report") is not None:
            return {"status": "complete", "routing_path": ["supervisor"]}

        collector_failed = any(e["agent"] == "collector" for e in errors)
        if collector_failed:
            return {"status": "failed", "routing_path": ["supervisor"]}

        collector = state.get("collector")
        if collector is not None and collector.get("data_quality") == "offline":
            return {
                "status": "failed",
                "errors": [{
                    "agent": "collector",
                    "reason": "data_quality=offline：窗口内无有效读数或最新数据过旧，未进入诊断",
                }],
                "routing_path": ["supervisor"],
            }

        return {"status": "partial", "routing_path": ["supervisor"]}

    # ── 路由函数 ─────────────────────────────────────────────────────────

    def route_after_collector(state: DiagnosisState) -> str:
        if state.get("collector") is None:
            return "supervisor"                              # collector 完全失败，直接结束
        quality = state["collector"].get("data_quality", "ok")
        if quality == "offline":
            return "supervisor"                              # 传感器离线，直接结束
        if quality == "degraded":
            return "re_collect"                              # 数据降级，先补采
        return "rule"                                        # 正常路径

    def route_after_rule(state: DiagnosisState) -> str:
        if state.get("rule") is None:
            return "report"                                  # rule 失败，跳到 report 记录
        triggered = state["rule"].get("triggered_count", 0)
        if triggered == 0:
            return "ai_analysis_lite"                        # 规则全绿，AI 简化
        return "ai_analysis"                                 # 有告警，完整 RAG 分析

    # ── 图装配 ───────────────────────────────────────────────────────────

    g = StateGraph(DiagnosisState)
    g.add_node("collector", node_collector)
    g.add_node("re_collect", node_re_collect)
    g.add_node("rule", node_rule)
    g.add_node("ai_analysis", node_ai)
    g.add_node("ai_analysis_lite", node_ai_lite)
    g.add_node("report", node_report)
    g.add_node("supervisor", node_supervisor)

    g.set_entry_point("collector")
    g.add_conditional_edges(
        "collector",
        route_after_collector,
        {"re_collect": "re_collect", "rule": "rule", "supervisor": "supervisor"},
    )
    g.add_edge("re_collect", "rule")
    g.add_conditional_edges(
        "rule",
        route_after_rule,
        {"ai_analysis": "ai_analysis", "ai_analysis_lite": "ai_analysis_lite", "report": "report"},
    )
    g.add_edge("ai_analysis", "report")
    g.add_edge("ai_analysis_lite", "report")
    g.add_edge("report", "supervisor")
    g.add_edge("supervisor", END)

    return g.compile()


def _initial_state(sensor_type: str, window_hours: float) -> DiagnosisState:
    return {
        "sensor_type": sensor_type,
        "window_hours": window_hours,
        "collector": None,
        "rule": None,
        "ai_analysis": None,
        "report": None,
        "errors": [],
        "routing_path": [],
        "status": "running",
    }