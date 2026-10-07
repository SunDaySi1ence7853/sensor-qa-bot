from __future__ import annotations

import json
import re
from typing import Callable

from .base import AgentFailure
from .schemas import AIAnalysisInput, AIAnalysisOutput

SYSTEM_PROMPT = """你是工业传感器故障诊断系统中的「AI 分析 Agent」。
你的唯一职责：基于【读数摘要】【规则诊断结果】【知识库片段】三类输入，推测根因并给出处置建议。

硬性纪律：
1. 只能依据给定的知识库片段下结论，每条结论都必须能对应到片段编号。
2. 不得重新判定正常/警告/故障，等级以规则诊断结果为准；你只解释"为什么"和"怎么办"。
3. 片段不足以支撑根因时，root_cause 写"知识库依据不足：<缺什么>"，cited 留空，不得编造。
4. 只输出一个 JSON 对象，不要输出其他任何文字：
{"root_cause": "...", "recommendation": "...", "cited": [片段编号, ...]}
"""

SENSOR_CN = {"temperature": "温度", "humidity": "湿度"}
# 规则编号 → 检索关键词，让检索围绕"哪里异常"而不是泛泛地搜
RULE_KEYWORDS = {
    "R01_LATEST_THRESHOLD": "超限 阈值",
    "R02_WINDOW_EXTREME": "超限 历史异常",
    "R03_RISING_TREND": "持续上升 原因",
    "R04_JUMP": "读数跳变 接线 传感器故障",
}

LLM = Callable[[str, str], str]          # (system, user) -> text
Retriever = Callable[[str], list]        # query -> 知识库片段列表


class AIAnalysisAgent:
    name = "ai_analysis"

    def __init__(self, llm: LLM, retriever: Retriever):
        # 真实 DeepSeek 与 search_manual 在任务2 Supervisor 装配时注入；骨架期用 fake
        self._llm = llm
        self._retriever = retriever

    def run(self, inp: AIAnalysisInput) -> AIAnalysisOutput:
        sensor = inp.get("sensor_type", "")
        rule_result = inp.get("rule_result") or {}
        items = rule_result.get("items") or []
        if not items:
            raise AgentFailure(self.name, "缺少规则诊断结果，无法做有依据的分析")

        query = self._build_query(sensor, items)
        refs = [r for r in (self._retriever(query) or []) if isinstance(r, str) and r.strip()]
        if not refs:
            # 没有依据就不调 LLM：既守来源纪律，也不浪费 token
            raise AgentFailure(self.name, f"知识库无相关片段（query={query!r}），拒绝无依据分析")

        user_prompt = self._build_user_prompt(sensor, inp.get("readings") or [], items, refs)
        parsed = self._parse(self._llm(SYSTEM_PROMPT, user_prompt), len(refs))
        return {
            "root_cause": parsed["root_cause"],
            "recommendation": parsed["recommendation"],
            "references": [refs[i - 1] for i in parsed["cited"]],
        }

    @staticmethod
    def _build_query(sensor: str, items: list) -> str:
        abnormal = [i for i in items if i.get("level") != "normal"]
        name = SENSOR_CN.get(sensor, sensor)
        if not abnormal:
            return f"{name} 正常范围 日常维护"
        keywords = " ".join(RULE_KEYWORDS.get(i["rule_id"], i["rule_id"]) for i in abnormal)
        return f"{name} 异常 {keywords}"

    @staticmethod
    def _summarize(readings: list) -> str:
        values = [r["value"] for r in readings if isinstance(r.get("value"), (int, float))]
        if not values:
            return "无有效读数"
        mean = round(sum(values) / len(values), 2)
        return f"{len(values)} 点，min={min(values)} max={max(values)} mean={mean} 最新={values[-1]}"

    def _build_user_prompt(self, sensor: str, readings: list, items: list, refs: list) -> str:
        rules_text = "\n".join(f"- {i['rule_id']} [{i['level']}] {i['evidence']}" for i in items)
        refs_text = "\n".join(f"[{k}] {ref}" for k, ref in enumerate(refs, 1))
        return (
            f"【传感器】{SENSOR_CN.get(sensor, sensor)}\n"
            f"【读数摘要】{self._summarize(readings)}\n"
            f"【规则诊断结果】\n{rules_text}\n"
            f"【知识库片段】\n{refs_text}"
        )

    def _parse(self, text: str, ref_count: int) -> dict:
        # 兼容模型包了 ```json 围栏的情况，只取第一个 {...}
        match = re.search(r"\{.*\}", text or "", re.S)
        if not match:
            raise AgentFailure(self.name, f"LLM 输出不是 JSON：{(text or '')[:80]!r}")
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as e:
            raise AgentFailure(self.name, f"LLM 输出 JSON 解析失败：{e}") from e

        root, rec, cited = data.get("root_cause"), data.get("recommendation"), data.get("cited")
        if not isinstance(root, str) or not root.strip() or not isinstance(rec, str) or not rec.strip():
            raise AgentFailure(self.name, "LLM 输出缺少 root_cause 或 recommendation")
        if not isinstance(cited, list) or not cited:
            raise AgentFailure(self.name, f"LLM 未引用任何知识库片段：{root}")
        if any(isinstance(i, bool) or not isinstance(i, int) or not 1 <= i <= ref_count for i in cited):
            raise AgentFailure(self.name, f"引用编号越界：{cited}，可用 1~{ref_count}")
        return {"root_cause": root.strip(), "recommendation": rec.strip(), "cited": list(dict.fromkeys(cited))}


if __name__ == "__main__":
    from .fakes import FAKE_MANUAL, FakeLLM, FakeRetriever, make_readings
    from .rule_agent import RuleAgent

    readings = make_readings([23.0, 23.5, 24.0, 24.8, 25.5])
    rule_result = RuleAgent().run({"readings": readings, "sensor_type": "temperature"})
    llm = FakeLLM(json.dumps({
        "root_cause": "温度单调上升 2.5°C，符合片段[2]所述散热不良特征",
        "recommendation": "检查散热风扇运转与通风口，必要时降载",
        "cited": [2],
    }, ensure_ascii=False))
    retriever = FakeRetriever(FAKE_MANUAL)

    out = AIAnalysisAgent(llm, retriever).run(
        {"readings": readings, "rule_result": rule_result, "sensor_type": "temperature"}
    )
    print("===== 检索 query =====")
    print(retriever.queries[0])
    print("===== 发给 LLM 的 user prompt =====")
    print(llm.calls[0][1])
    print("===== 输出 =====")
    print(json.dumps(out, ensure_ascii=False, indent=2))