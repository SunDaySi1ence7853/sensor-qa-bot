"""规则配置加载与校验。

前几条读真实 config.yaml，拦住手滑改错 yaml；
后两条守卫专治"复制粘贴忘删旧段"和"注释写了测试没写"。
"""
import copy
import re

import pytest
import yaml

from src.agents.rule_config import (
    CONFIG_PATH, RuleConfigError, load_rule_thresholds, parse_rule_config,
)
from src.config import get_config

_VALID = {
    "rule_engine": {
        "params": {
            "trend_points": 5, "flatline_frames": 40, "flatline_epsilon": 0.0,
            "oscillation_min_jumps": 3, "sustained_frames": 10,
        },
        "thresholds": {
            "temperature": {
                "physical": [-40, 80], "warn": [10, 35], "fault": [0, 45],
                "jump": 5, "trend_rise": 2, "trend_fall": 2, "approach_margin": 2,
            },
        },
    }
}


def _valid():
    return copy.deepcopy(_VALID)


# ---------- 重复键检测用的 Loader ----------
class _NoDupLoader(yaml.SafeLoader):
    """PyYAML 遇到重复键会静默用后者覆盖前者，所以 data_source 段贴两遍不会报错。这里改成直接报出来。"""


def _no_dup_mapping(loader, node, deep=False):
    seen: dict = {}
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        line = key_node.start_mark.line + 1
        if key in seen:
            raise ValueError(f"config.yaml 第 {line} 行键 {key!r} 重复（首次出现在第 {seen[key]} 行）")
        seen[key] = line
    return loader.construct_mapping(node, deep=deep)


_NoDupLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_dup_mapping)


# ---------- 真实 config.yaml ----------
def test_real_config_loads_temperature_and_humidity():
    th = load_rule_thresholds()
    assert {"temperature", "humidity"} <= set(th)


def test_real_config_bands_are_float_tuples():
    """旧 evidence 按 '(10.0, 35.0)' 打印，读出来是 list 或 int 都会让旧断言失败。"""
    for sensor, th in load_rule_thresholds().items():
        for key in ("physical", "warn", "fault"):
            assert isinstance(th[key], tuple), f"{sensor}.{key}"
            assert all(isinstance(x, float) for x in th[key]), f"{sensor}.{key}"


def test_warn_band_matches_legacy_sensor_thresholds():
    """config.yaml 注释承诺的一致性测试就是这一条。

    config.py 的 sensor_thresholds（旧面板在用）和规则告警带是同一组数，两边必须一致。
    如果 .env 里设了 SENSOR_THRESHOLDS 导致失败，说明两处真的已经分叉，这正是本测试要拦的。
    """
    legacy = get_config().sensor_thresholds
    for sensor, th in load_rule_thresholds().items():
        assert sensor in legacy, f"{sensor} 在 rule_engine 有配置，但 config.py 的 sensor_thresholds 里没有"
        expected = (float(legacy[sensor]["min"]), float(legacy[sensor]["max"]))
        assert th["warn"] == expected, f"{sensor}：yaml warn={th['warn']}，config.py={expected}"


def test_real_config_has_no_duplicate_keys():
    """病根守卫：复制粘贴后忘删旧段。重复键会被 PyYAML 静默吞掉，只能靠这条测试抓。"""
    yaml.load(CONFIG_PATH.read_text(encoding="utf-8"), Loader=_NoDupLoader)


def test_yaml_comment_references_point_to_real_tests():
    """反空头支票：config.yaml 注释里引用的 test_rule_config.py::xxx，必须真的存在于本文件。"""
    text = CONFIG_PATH.read_text(encoding="utf-8")
    refs = re.findall(r"test_rule_config\.py::(\w+)", text)
    assert refs, "config.yaml 注释应注明一致性由哪条测试锁定"
    missing = [r for r in refs if not callable(globals().get(r))]
    assert not missing, f"config.yaml 注释引用了不存在的测试：{missing}"


# ---------- 校验逻辑 ----------
def test_missing_section_raises():
    with pytest.raises(RuleConfigError, match="rule_engine"):
        parse_rule_config({"data_source": "serial"})


def test_bad_nesting_raises():
    data = _valid()
    data["rule_engine"]["thresholds"]["temperature"]["warn"] = [10, 50]  # 告警上限 > 故障上限
    with pytest.raises(RuleConfigError, match="嵌套"):
        parse_rule_config(data)


def test_missing_key_raises_with_name():
    data = _valid()
    del data["rule_engine"]["thresholds"]["temperature"]["jump"]
    with pytest.raises(RuleConfigError, match="jump"):
        parse_rule_config(data)


def test_missing_file_raises(tmp_path):
    with pytest.raises(RuleConfigError, match="找不到"):
        load_rule_thresholds(tmp_path / "nope.yaml")


def test_params_merged_and_sensor_override():
    data = _valid()
    data["rule_engine"]["thresholds"]["temperature"]["sustained_frames"] = 20
    th = parse_rule_config(data)["temperature"]
    assert th["trend_points"] == 5          # 继承全局参数
    assert th["sustained_frames"] == 20     # 传感器级覆盖
    assert th["warn"] == (10.0, 35.0)       # int 转成了 float tuplepython -m pytest tests/agents/test_rule_config.py -q