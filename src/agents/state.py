"""DiagnosisState：整条管线的唯一数据容器。

字段读写矩阵见设计稿；routing_path 由各节点追加，用于调试和日志。
"""
from __future__ import annotations

import operator
from typing import Annotated, Literal, Optional

from typing_extensions import TypedDict

from .schemas import AIAnalysisOutput, CollectorOutput, ReportOutput, RuleOutput

Status = Literal["running", "partial", "complete", "failed"]


class DiagnosisState(TypedDict):
    # 入参
    sensor_type: str
    window_hours: float

    # Agent 产物
    collector: Optional[CollectorOutput]
    rule: Optional[RuleOutput]
    ai_analysis: Optional[AIAnalysisOutput]
    report: Optional[ReportOutput]

    # 管线元数据
    errors: Annotated[list[dict], operator.add]
    routing_path: Annotated[list[str], operator.add]
    status: Status