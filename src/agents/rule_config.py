"""规则引擎配置加载：唯一真相源是项目根目录 config.yaml 的 rule_engine 段。

加载即强校验：缺字段、类型错、区间嵌套错，都在构造 RuleAgent 时当场报错，
不等到诊断跑到一半才发现。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"

BAND_KEYS = ("physical", "warn", "fault")
SCALAR_KEYS = ("jump", "trend_rise", "trend_fall", "approach_margin")
# 全局参数：key -> (类型, 最小值)
PARAM_SPECS = {
    "trend_points": (int, 2),
    "flatline_frames": (int, 2),
    "flatline_epsilon": (float, 0.0),
    "oscillation_min_jumps": (int, 2),
    "sustained_frames": (int, 1),
}


class RuleConfigError(ValueError):
    """规则配置不合法。属于部署错误，不是运行时数据问题，所以不用 AgentFailure。"""


def _band(where: str, raw) -> tuple[float, float]:
    if not (isinstance(raw, (list, tuple)) and len(raw) == 2):
        raise RuleConfigError(f"{where} 必须是 [下限, 上限]，当前：{raw!r}")
    try:
        lo, hi = float(raw[0]), float(raw[1])
    except (TypeError, ValueError):
        raise RuleConfigError(f"{where} 必须是数字，当前：{raw!r}")
    if lo >= hi:
        raise RuleConfigError(f"{where} 下限必须小于上限，当前：{raw!r}")
    return (lo, hi)  # 必须是 float tuple：旧 evidence 按 "(10.0, 35.0)" 格式打印


def _positive(where: str, raw) -> float:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        raise RuleConfigError(f"{where} 必须是数字，当前：{raw!r}")
    if v <= 0:
        raise RuleConfigError(f"{where} 必须大于 0，当前：{v}")
    return v


def _param(where: str, key: str, raw):
    typ, minimum = PARAM_SPECS[key]
    try:
        v = typ(raw)
    except (TypeError, ValueError):
        raise RuleConfigError(f"{where}.{key} 必须是 {typ.__name__}，当前：{raw!r}")
    if v < minimum:
        raise RuleConfigError(f"{where}.{key} 不能小于 {minimum}，当前：{v}")
    return v


def parse_rule_config(data: Optional[dict]) -> dict:
    """把 yaml 解析结果转成 RuleAgent 要的阈值 dict：{sensor_type: th}。"""
    section = (data or {}).get("rule_engine")
    if not isinstance(section, dict):
        raise RuleConfigError("config.yaml 缺少 rule_engine 段")

    params = section.get("params")
    if not isinstance(params, dict):
        raise RuleConfigError("rule_engine 缺少 params 段")
    for key in PARAM_SPECS:
        if key not in params:
            raise RuleConfigError(f"rule_engine.params 缺少 {key}")

    thresholds = section.get("thresholds")
    if not isinstance(thresholds, dict) or not thresholds:
        raise RuleConfigError("rule_engine 缺少 thresholds 段或为空")

    out: dict = {}
    for sensor, raw in thresholds.items():
        where = f"rule_engine.thresholds.{sensor}"
        if not isinstance(raw, dict):
            raise RuleConfigError(f"{where} 必须是映射")

        th: dict = {}
        for key in BAND_KEYS:
            if key not in raw:
                raise RuleConfigError(f"{where} 缺少 {key}")
            th[key] = _band(f"{where}.{key}", raw[key])
        for key in SCALAR_KEYS:
            if key not in raw:
                raise RuleConfigError(f"{where} 缺少 {key}")
            th[key] = _positive(f"{where}.{key}", raw[key])

        (plo, phi), (flo, fhi), (wlo, whi) = th["physical"], th["fault"], th["warn"]
        if not (plo <= flo < wlo < whi < fhi <= phi):
            raise RuleConfigError(
                f"{where} 区间嵌套错误：要求 物理下限 ≤ 故障下限 < 告警下限 < 告警上限 < 故障上限 ≤ 物理上限，"
                f"当前 physical={th['physical']} fault={th['fault']} warn={th['warn']}"
            )

        # 全局参数先铺底，传感器级同名字段覆盖
        for key in PARAM_SPECS:
            th[key] = _param("rule_engine.params", key, params[key])
            if key in raw:
                th[key] = _param(where, key, raw[key])

        out[sensor] = th
    return out


def load_rule_thresholds(path: Union[str, Path, None] = None) -> dict:
    p = Path(path) if path else CONFIG_PATH
    if not p.exists():
        raise RuleConfigError(f"找不到规则配置文件：{p}")
    try:
        with p.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise RuleConfigError(f"{p} 不是合法 YAML：{e}") from e
    return parse_rule_config(data)