"""DiagnosisState：整条管线的唯一数据容器，按设计稿"谁写谁读"矩阵落地。

各字段读写权限（任务2施工时请核对设计稿，不许在矩阵外写字段）：
  sensor_type / window_hours  ← 入口，外部写，内部只读
  collector                   ← CollectorAgent 写；RuleAgent / AIAnalysisAgent / ReportAgent 读
  rule                        ← RuleAgent 写；AIAnalysisAgent / ReportAgent 读
  ai_analysis                 ← AIAnalysisAgent 写；ReportAgent 读
  report                      ← ReportAgent 写；Supervisor / Web 层读
  errors                      ← 所有节点均可追加（reducer 保证不覆盖），Supervisor 汇总读
  status                      ← Supervisor 写，外部只读
"""
from __future__ import annotations

import operator
from typing import Annotated, Literal, Optional

from typing_extensions import TypedDict

from .schemas import AIAnalysisOutput, CollectorOutput, ReportOutput, RuleOutput

Status = Literal["running", "partial", "complete", "failed"]


class DiagnosisState(TypedDict):
    # 入参 ────────────────────────────────────────────────
    sensor_type: str
    window_hours: float

    # Agent 产物 ──────────────────────────────────────────
    collector: Optional[CollectorOutput]
    rule: Optional[RuleOutput]
    ai_analysis: Optional[AIAnalysisOutput]
    report: Optional[ReportOutput]

    # 管线元数据 ──────────────────────────────────────────
    errors: Annotated[list[dict], operator.add]  # 追加 reducer；节点只返回新增项
    status: Status