from __future__ import annotations

from typing import Callable, Optional

from .base import AgentFailure
from .schemas import RuleInput, RuleItem, RuleOutput

ROLE = """规则诊断 Agent（纯规则引擎，零 LLM、零 token）
职责：对读数逐条跑规则，输出正常/警告/故障及带数值的依据。
不做：根因推测、处置建议、调用任何模型。
失败信号：sensor_type 无阈值配置 / 无有效读数 → 抛 AgentFailure（不把"没数据"判成"正常"）。
"""

# 阈值为骨架期占位值，任务3按手册和现场工况校准后移入 config.yaml
THRESHOLDS = {
    "temperature": {"warn": (10.0, 35.0), "fault": (0.0, 45.0), "jump": 5.0, "trend_rise": 2.0},
    "humidity": {"warn": (20.0, 80.0), "fault": (10.0, 90.0), "jump": 15.0, "trend_rise": 10.0},
}
TREND_POINTS = 5
LEVEL_RANK = {"normal": 0, "warn": 1, "fault": 2}

Rule = Callable[[list, dict], RuleItem]


def _band_level(v: float, th: dict) -> str:
    flo, fhi = th["fault"]
    wlo, whi = th["warn"]
    if v < flo or v > fhi:
        return "fault"
    if v < wlo or v > whi:
        return "warn"
    return "normal"


def rule_latest_threshold(values: list, th: dict) -> RuleItem:
    """阈值型：最新值落在哪个区间"""
    v = values[-1]
    return {
        "rule_id": "R01_LATEST_THRESHOLD",
        "level": _band_level(v, th),
        "evidence": f"最新值 {v}，告警带 {th['warn']}，故障带 {th['fault']}",
    }


def rule_window_extreme(values: list, th: dict) -> RuleItem:
    """阈值型：窗口内是否有点越界（抓住已恢复的历史异常）"""
    levels = [_band_level(v, th) for v in values]
    worst = max(levels, key=LEVEL_RANK.get)
    out_count = sum(lv != "normal" for lv in levels)
    return {
        "rule_id": "R02_WINDOW_EXTREME",
        "level": worst,
        "evidence": f"窗口 {len(values)} 点，min={min(values)} max={max(values)}，越界 {out_count} 点",
    }


def rule_rising_trend(values: list, th: dict) -> RuleItem:
    """趋势型：末 N 点单调上升且累计升幅达标"""
    tail = values[-TREND_POINTS:]
    if len(tail) < TREND_POINTS:
        return {
            "rule_id": "R03_RISING_TREND",
            "level": "normal",
            "evidence": f"点数不足 {len(tail)}/{TREND_POINTS}，不判趋势",
        }
    rising = all(b > a for a, b in zip(tail, tail[1:]))
    rise = round(tail[-1] - tail[0], 2)
    level = "warn" if rising and rise >= th["trend_rise"] else "normal"
    return {
        "rule_id": "R03_RISING_TREND",
        "level": level,
        "evidence": f"末 {TREND_POINTS} 点 {tail}，{'单调上升' if rising else '非单调'}，"
                    f"累计 {rise:+}，阈值 {th['trend_rise']}",
    }


def rule_jump(values: list, th: dict) -> RuleItem:
    """突变型：相邻两点跳变过大，多指向接线或传感器问题"""
    if len(values) < 2:
        return {"rule_id": "R04_JUMP", "level": "normal", "evidence": "点数不足 2，不判跳变"}
    max_delta = round(max(abs(b - a) for a, b in zip(values, values[1:])), 2)
    return {
        "rule_id": "R04_JUMP",
        "level": "warn" if max_delta > th["jump"] else "normal",
        "evidence": f"相邻最大跳变 {max_delta}，阈值 {th['jump']}",
    }


DEFAULT_RULES: list[Rule] = [rule_latest_threshold, rule_window_extreme, rule_rising_trend, rule_jump]


class RuleAgent:
    name = "rule"

    def __init__(self, rules: Optional[list[Rule]] = None, thresholds: Optional[dict] = None):
        self._rules = rules or DEFAULT_RULES
        self._thresholds = thresholds or THRESHOLDS

    def run(self, inp: RuleInput) -> RuleOutput:
        sensor = inp.get("sensor_type")
        if sensor not in self._thresholds:
            raise AgentFailure(self.name, f"传感器 {sensor!r} 无阈值配置")
        values = [
            r["value"] for r in (inp.get("readings") or [])
            if isinstance(r.get("value"), (int, float)) and not isinstance(r.get("value"), bool)
        ]
        if not values:
            raise AgentFailure(self.name, "无有效读数，规则引擎拒绝出结论")

        th = self._thresholds[sensor]
        items = [rule(values, th) for rule in self._rules]
        return {"items": items, "triggered_count": sum(i["level"] != "normal" for i in items)}


if __name__ == "__main__":
    import json

    from .fakes import make_readings

    cases = {
        "平稳（老板复核过的那组数）": [23.6] * 8 + [23.7] * 2,
        "持续升温": [23.0, 23.5, 24.0, 24.8, 25.5],
        "突变+超限": [23.6, 23.6, 23.6, 50.0],
    }
    for label, values in cases.items():
        out = RuleAgent().run({"readings": make_readings(values), "sensor_type": "temperature"})
        print(f"===== {label} =====")
        print(json.dumps(out, ensure_ascii=False, indent=2))