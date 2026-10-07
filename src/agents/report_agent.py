from __future__ import annotations

from datetime import datetime
from typing import Callable, Optional

from .base import AgentFailure
from .rule_agent import LEVEL_RANK
from .schemas import ReportInput, ReportOutput

ROLE = """报告生成 Agent（确定性汇总，不调 LLM）
职责：汇总采集、规则、AI 三份产物，产出带置信度与依据链的诊断报告。
不做：改写诊断等级（以规则为准）、补写 AI 没给的结论。
失败信号：任一前序产物缺失 → 抛 AgentFailure（部分报告由任务2 Supervisor 决定是否生成）。
"""

LEVEL_CN = {"normal": "正常", "warn": "警告", "fault": "故障"}
QUALITY_WEIGHT = {"ok": 1.0, "degraded": 0.7, "offline": 0.3}


class ReportAgent:
    name = "report"

    def __init__(self, clock: Optional[Callable[[], datetime]] = None):
        self._clock = clock or datetime.now

    def run(self, inp: ReportInput) -> ReportOutput:
        for key in ("collector", "rule", "ai_analysis"):
            if not inp.get(key):
                raise AgentFailure(self.name, f"缺少前序产物 {key}，拒绝出完整报告")
        c, r, a = inp["collector"], inp["rule"], inp["ai_analysis"]
        items = r.get("items") or []
        if not items:
            raise AgentFailure(self.name, "规则诊断结果为空")

        overall = max((i["level"] for i in items), key=LEVEL_RANK.get)
        quality = c.get("data_quality", "offline")
        refs = a.get("references") or []

        # 置信度 = 数据质量权重 × 知识库依据充分度（0 条 0.5，满 3 条 1.0），公式写死可复算
        confidence = round(QUALITY_WEIGHT.get(quality, 0.3) * (0.5 + 0.5 * min(len(refs), 3) / 3), 2)

        chain = [
            f"[采集] {len(c.get('readings', []))} 条读数，质量={quality}，"
            f"缺口 {c.get('missing_count', 0)}，坏帧 {c.get('bad_frame_count', 0)}"
        ]
        chain += [f"[规则 {i['rule_id']}] {LEVEL_CN[i['level']]}：{i['evidence']}" for i in items]
        chain.append(f"[AI] 根因：{a['root_cause']}")
        chain += [f"[知识库 {k}] {ref[:80]}" for k, ref in enumerate(refs, 1)]

        summary = (
            f"诊断结论：{LEVEL_CN[overall]}（{r.get('triggered_count', 0)} 条规则触发）。"
            f"根因：{a['root_cause']}。建议：{a['recommendation']}"
        )
        if quality != "ok":
            summary += f"（注意：数据质量 {quality}，置信度已下调）"

        return {
            "summary": summary,
            "confidence": confidence,
            "evidence_chain": chain,
            "timestamp": self._clock().isoformat(timespec="seconds"),
        }


if __name__ == "__main__":
    import json

    from .fakes import FAKE_MANUAL, FIXED_NOW, make_readings

    report = ReportAgent(clock=lambda: FIXED_NOW).run({
        "collector": {
            "readings": make_readings([23.0, 23.5, 24.0, 24.8, 25.5]),
            "data_quality": "ok", "missing_count": 0, "bad_frame_count": 0,
        },
        "rule": {
            "items": [
                {"rule_id": "R01_LATEST_THRESHOLD", "level": "normal", "evidence": "最新值 25.5"},
                {"rule_id": "R03_RISING_TREND", "level": "warn", "evidence": "末 5 点单调上升，累计 +2.5"},
            ],
            "triggered_count": 1,
        },
        "ai_analysis": {
            "root_cause": "符合散热不良特征",
            "recommendation": "检查散热风扇与通风口",
            "references": [FAKE_MANUAL[1]],
        },
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))