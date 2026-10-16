from __future__ import annotations

from typing import Callable, Optional

from .base import AgentFailure
from .rule_config import load_rule_thresholds
from .schemas import RuleInput, RuleItem, RuleOutput

ROLE = """规则诊断 Agent（纯规则引擎，零 LLM、零 token）
职责：对读数逐条跑规则，输出正常/警告/故障及带数值的依据。
不做：根因推测、处置建议、调用任何模型。
失败信号：sensor_type 无阈值配置 / 无有效读数 → 抛 AgentFailure（不把"没数据"判成"正常"）。
阈值来源：config.yaml 的 rule_engine 段（任务3 起不再硬编码）。
"""

TREND_POINTS = 5
LEVEL_RANK = {"normal": 0, "warn": 1, "fault": 2}

# 全局参数的兜底值，只给"测试里手写的旧阈值 dict"用。
# 生产路径走 config.yaml，加载器会要求这些字段必须写全，所以这里不构成第二份真相源。
RULE_PARAM_DEFAULTS = {
    "trend_points": TREND_POINTS,
    "flatline_frames": 40,
    "flatline_epsilon": 0.0,
    "oscillation_min_jumps": 3,
    "sustained_frames": 10,
}

Rule = Callable[[list, dict], RuleItem]


# ---------- 工具函数 ----------
def _p(th: dict, key: str):
    return th.get(key, RULE_PARAM_DEFAULTS[key])


def _skip(rule_id: str, key: str) -> RuleItem:
    return {"rule_id": rule_id, "level": "normal", "evidence": f"未配置 {key}，跳过本规则"}


def _band_level(v: float, th: dict) -> str:
    flo, fhi = th["fault"]
    wlo, whi = th["warn"]
    if v < flo or v > fhi:
        return "fault"
    if v < wlo or v > whi:
        return "warn"
    return "normal"


def _strictly(tail: list, rising: bool) -> bool:
    pairs = zip(tail, tail[1:])
    return all(b > a for a, b in pairs) if rising else all(b < a for a, b in pairs)


# ==================== 阈值型 ====================
def rule_latest_threshold(values: list, th: dict) -> RuleItem:
    """R01 阈值型：最新值落在哪个区间"""
    v = values[-1]
    return {
        "rule_id": "R01_LATEST_THRESHOLD",
        "level": _band_level(v, th),
        "evidence": f"最新值 {v}，告警带 {th['warn']}，故障带 {th['fault']}",
    }


def rule_window_extreme(values: list, th: dict) -> RuleItem:
    """R02 阈值型：窗口内是否有点越界（抓住已恢复的历史异常）"""
    levels = [_band_level(v, th) for v in values]
    worst = max(levels, key=LEVEL_RANK.get)
    out_count = sum(lv != "normal" for lv in levels)
    return {
        "rule_id": "R02_WINDOW_EXTREME",
        "level": worst,
        "evidence": f"窗口 {len(values)} 点，min={min(values)} max={max(values)}，越界 {out_count} 点",
    }


def rule_physical_range(values: list, th: dict) -> RuleItem:
    """R05 阈值型（物理量程）：超出传感器量程的读数在物理上不可能 → 判传感器故障，而非环境异常"""
    rid = "R05_PHYSICAL_RANGE"
    if "physical" not in th:
        return _skip(rid, "physical")
    lo, hi = th["physical"]
    bad = [v for v in values if v < lo or v > hi]
    if not bad:
        return {"rule_id": rid, "level": "normal",
                "evidence": f"全部 {len(values)} 点在物理量程 [{lo}, {hi}] 内"}
    return {
        "rule_id": rid,
        "level": "fault",
        "evidence": f"{len(bad)} 点超出物理量程 [{lo}, {hi}]（首个越界值 {bad[0]}），"
                    f"读数物理上不可能，判定传感器故障而非环境异常",
    }


# ==================== 趋势型 ====================
def rule_rising_trend(values: list, th: dict) -> RuleItem:
    """R03 趋势型：末 N 点单调上升且累计升幅达标"""
    n = int(_p(th, "trend_points"))
    tail = values[-n:]
    if len(tail) < n:
        return {
            "rule_id": "R03_RISING_TREND",
            "level": "normal",
            "evidence": f"点数不足 {len(tail)}/{n}，不判趋势",
        }
    rising = all(b > a for a, b in zip(tail, tail[1:]))
    rise = round(tail[-1] - tail[0], 2)
    level = "warn" if rising and rise >= th["trend_rise"] else "normal"
    return {
        "rule_id": "R03_RISING_TREND",
        "level": level,
        "evidence": f"末 {n} 点 {tail}，{'单调上升' if rising else '非单调'}，"
                    f"累计 {rise:+}，阈值 {th['trend_rise']}",
    }


def rule_falling_trend(values: list, th: dict) -> RuleItem:
    """R06 趋势型：末 N 点单调下降且累计降幅达标（空调过冷、传感器脱落贴到冷面等）"""
    rid = "R06_FALLING_TREND"
    if "trend_fall" not in th:
        return _skip(rid, "trend_fall")
    n = int(_p(th, "trend_points"))
    tail = values[-n:]
    if len(tail) < n:
        return {"rule_id": rid, "level": "normal", "evidence": f"点数不足 {len(tail)}/{n}，不判趋势"}
    falling = _strictly(tail, rising=False)
    change = round(tail[-1] - tail[0], 2)
    level = "warn" if falling and -change >= th["trend_fall"] else "normal"
    return {
        "rule_id": rid,
        "level": level,
        "evidence": f"末 {n} 点 {tail}，{'单调下降' if falling else '非单调'}，"
                    f"累计 {change:+}，阈值 -{th['trend_fall']}",
    }


# ==================== 跳变型 ====================
def rule_jump(values: list, th: dict) -> RuleItem:
    """R04 突变型：相邻两点跳变过大，多指向接线或传感器问题"""
    if len(values) < 2:
        return {"rule_id": "R04_JUMP", "level": "normal", "evidence": "点数不足 2，不判跳变"}
    max_delta = round(max(abs(b - a) for a, b in zip(values, values[1:])), 2)
    return {
        "rule_id": "R04_JUMP",
        "level": "warn" if max_delta > th["jump"] else "normal",
        "evidence": f"相邻最大跳变 {max_delta}，阈值 {th['jump']}",
    }


def rule_oscillation(values: list, th: dict) -> RuleItem:
    """R08 跳变型：反复大幅跳变（接触不良、电磁干扰）。单个尖峰是一上一下 2 次，不算"""
    rid = "R08_OSCILLATION"
    k = int(_p(th, "oscillation_min_jumps"))
    big = [round(b - a, 2) for a, b in zip(values, values[1:]) if abs(b - a) > th["jump"]]
    return {
        "rule_id": rid,
        "level": "warn" if len(big) >= k else "normal",
        "evidence": f"超阈值跳变 {len(big)} 次（阈值 {th['jump']}，要求 ≥{k} 次），明细 {big[:5]}",
    }


# ==================== 传感器健康 ====================
def rule_flatline(values: list, th: dict) -> RuleItem:
    """R07 健康型：长时间读数纹丝不动 → 传感器卡死或采集链路在重放旧值"""
    rid = "R07_FLATLINE"
    n = int(_p(th, "flatline_frames"))
    eps = float(_p(th, "flatline_epsilon"))
    tail = values[-n:]
    if len(tail) < n:
        return {"rule_id": rid, "level": "normal", "evidence": f"点数不足 {len(tail)}/{n}，不判冻结"}
    spread = round(max(tail) - min(tail), 4)
    return {
        "rule_id": rid,
        "level": "warn" if spread <= eps else "normal",
        "evidence": f"末 {n} 点极差 {spread}，冻结阈值 ≤{eps}"
                    f"（真实传感器有噪声，长时间完全不变指向卡死或链路重放）",
    }


# ==================== 组合型（单传感器内多条件，见设计变更记录）====================
def rule_approaching_limit(values: list, th: dict) -> RuleItem:
    """R09 组合型：尚在告警带内 ∧ 距告警线 ≤ 余量 ∧ 末 N 点单调朝告警线走 → 即将越限"""
    rid = "R09_APPROACHING_LIMIT"
    if "approach_margin" not in th:
        return _skip(rid, "approach_margin")
    n = int(_p(th, "trend_points"))
    tail = values[-n:]
    if len(tail) < n:
        return {"rule_id": rid, "level": "normal", "evidence": f"点数不足 {len(tail)}/{n}，不判越限预警"}
    v = values[-1]
    if _band_level(v, th) != "normal":
        return {"rule_id": rid, "level": "normal",
                "evidence": f"最新值 {v} 已越出告警带，由 R01 负责，本规则只管尚未越限的情况"}
    wlo, whi = th["warn"]
    m = th["approach_margin"]
    gap_hi, gap_lo = round(whi - v, 2), round(v - wlo, 2)
    if _strictly(tail, rising=True) and gap_hi <= m:
        return {"rule_id": rid, "level": "warn",
                "evidence": f"最新值 {v} 距告警上限 {whi} 仅 {gap_hi}（余量 {m}），"
                            f"且末 {n} 点单调上升 {tail}，预计即将越限"}
    if _strictly(tail, rising=False) and gap_lo <= m:
        return {"rule_id": rid, "level": "warn",
                "evidence": f"最新值 {v} 距告警下限 {wlo} 仅 {gap_lo}（余量 {m}），"
                            f"且末 {n} 点单调下降 {tail}，预计即将越限"}
    return {"rule_id": rid, "level": "normal",
            "evidence": f"最新值 {v}，距上限 {gap_hi} / 距下限 {gap_lo}，余量 {m}，不构成越限预警"}


def rule_sustained_excursion(values: list, th: dict) -> RuleItem:
    """R10 组合型：越出告警带 ∧ 末尾连续持续 ≥ N 帧 → 不是偶发尖峰，升级为故障"""
    rid = "R10_SUSTAINED_EXCURSION"
    need = int(_p(th, "sustained_frames"))
    run = 0
    for v in reversed(values):
        if _band_level(v, th) == "normal":
            break
        run += 1
    if run >= need:
        return {"rule_id": rid, "level": "fault",
                "evidence": f"末尾连续 {run} 点越出告警带 {th['warn']}（要求 ≥{need} 点），"
                            f"属于持续越限而非偶发尖峰，升级为故障"}
    return {"rule_id": rid, "level": "normal",
            "evidence": f"末尾连续越界 {run} 点，未达持续阈值 {need}"}


def rule_step_out_of_band(values: list, th: dict) -> RuleItem:
    """R11 组合型：大跳变 ∧ 由带内一步跳到带外 → 没有渐变过程，疑似传感器或接线故障"""
    rid = "R11_STEP_OUT_OF_BAND"
    for i, (a, b) in enumerate(zip(values, values[1:])):
        if abs(b - a) > th["jump"] and _band_level(a, th) == "normal" and _band_level(b, th) != "normal":
            return {"rule_id": rid, "level": "warn",
                    "evidence": f"第 {i + 1}→{i + 2} 点由带内 {a} 一步跳到 {b}"
                                f"（跳变 {round(b - a, 2):+}，阈值 {th['jump']}），没有渐变过程直接越界。"
                                f"温湿度是缓变量，这更像传感器或接线故障，需与 R01 的越限结论交叉核对"}
    return {"rule_id": rid, "level": "normal", "evidence": "未发现一步跳出告警带的阶跃"}


DEFAULT_RULES: list[Rule] = [
    rule_latest_threshold,      # R01 阈值
    rule_window_extreme,        # R02 阈值
    rule_rising_trend,          # R03 趋势
    rule_jump,                  # R04 跳变
    rule_physical_range,        # R05 阈值（物理量程）
    rule_falling_trend,         # R06 趋势
    rule_flatline,              # R07 健康
    rule_oscillation,           # R08 跳变
    rule_approaching_limit,     # R09 组合
    rule_sustained_excursion,   # R10 组合
    rule_step_out_of_band,      # R11 组合
]


class RuleAgent:
    name = "rule"

    def __init__(self, rules: Optional[list[Rule]] = None, thresholds: Optional[dict] = None):
        self._rules = rules or DEFAULT_RULES
        self._thresholds = thresholds or load_rule_thresholds()

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