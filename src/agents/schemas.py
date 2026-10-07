"""四个 Agent 的 IO schema，按第三阶段任务1设计稿施工，语义不改。"""
from __future__ import annotations

from typing import Literal, TypedDict

DataQuality = Literal["ok", "degraded", "offline"]
Level = Literal["normal", "warn", "fault"]


class Reading(TypedDict):
    timestamp: str      # ISO8601，秒级
    value: float


# ---------- 数据采集 Agent ----------
class CollectorInput(TypedDict):
    sensor_type: str            # "temperature" / "humidity"
    window_hours: float         # 诊断窗口长度


class CollectorOutput(TypedDict):
    readings: list[Reading]     # 窗口内有效读数，按时间升序
    data_quality: DataQuality   # ok / degraded（有缺口或坏帧）/ offline（无数据或过旧）
    missing_count: int          # 按采样间隔推算缺失的点数
    bad_frame_count: int        # 时间戳或数值不合法的帧数


# ---------- 规则诊断 Agent ----------
class RuleItem(TypedDict):
    rule_id: str
    level: Level
    evidence: str               # 判定依据，带具体数值


class RuleInput(TypedDict):
    readings: list[Reading]
    sensor_type: str


class RuleOutput(TypedDict):
    items: list[RuleItem]       # 每条规则都出一项，正常也列出，便于审计
    triggered_count: int        # level != normal 的条数


# ---------- AI 分析 Agent ----------
class AIAnalysisInput(TypedDict):
    readings: list[Reading]
    rule_result: RuleOutput
    sensor_type: str


class AIAnalysisOutput(TypedDict):
    root_cause: str
    recommendation: str
    references: list[str]       # 被引用的知识库原文


# ---------- 报告生成 Agent ----------
class ReportInput(TypedDict):
    collector: CollectorOutput
    rule: RuleOutput
    ai_analysis: AIAnalysisOutput


class ReportOutput(TypedDict):
    summary: str
    confidence: float           # 0.0 - 1.0
    evidence_chain: list[str]
    timestamp: str