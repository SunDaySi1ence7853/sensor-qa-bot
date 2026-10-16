"""任务3 新增规则 R05~R11 的单测，以及兼容性守卫。"""
import pytest

from src.agents.fakes import make_readings
from src.agents.rule_agent import (
    DEFAULT_RULES, RuleAgent,
    rule_approaching_limit, rule_falling_trend, rule_flatline, rule_oscillation,
    rule_physical_range, rule_rising_trend, rule_step_out_of_band, rule_sustained_excursion,
)

TH = {
    "physical": (-40.0, 80.0), "warn": (10.0, 35.0), "fault": (0.0, 45.0),
    "jump": 5.0, "trend_rise": 2.0, "trend_fall": 2.0, "approach_margin": 2.0,
    "trend_points": 5, "flatline_frames": 40, "flatline_epsilon": 0.0,
    "oscillation_min_jumps": 3, "sustained_frames": 10,
}
# 任务3 之前的阈值格式：没有新字段，用来测兼容
LEGACY_TH = {"warn": (10.0, 35.0), "fault": (0.0, 45.0), "jump": 5.0, "trend_rise": 2.0}


# ---------- R05 物理量程 ----------
def test_r05_out_of_physical_range_is_fault():
    item = rule_physical_range([23.0, 85.0], TH)
    assert item["level"] == "fault"
    assert "85.0" in item["evidence"] and "80.0" in item["evidence"]


def test_r05_beyond_fault_band_but_physical_is_normal():
    """50℃ 超故障带（R01 管），但物理上可能 → R05 不报，两者分工明确"""
    assert rule_physical_range([23.0, 50.0], TH)["level"] == "normal"


# ---------- R06 下降趋势 ----------
def test_r06_falling_trend_warn():
    assert rule_falling_trend([30.0, 29.4, 28.8, 28.1, 27.5], TH)["level"] == "warn"


def test_r06_small_drop_normal():
    assert rule_falling_trend([30.0, 29.9, 29.8, 29.7, 29.6], TH)["level"] == "normal"


# ---------- R07 冻结 ----------
def test_r07_flatline_warn():
    assert rule_flatline([24.1] * 40, TH)["level"] == "warn"


def test_r07_short_constant_is_not_flatline():
    """守卫：全绿 fixture 是 [23.6]*N 的常数序列，帧数不够时绝不能判成冻结，否则路由会走错"""
    item = rule_flatline([23.6] * 39, TH)
    assert item["level"] == "normal"
    assert "39/40" in item["evidence"]


def test_r07_sensor_noise_not_flatline():
    assert rule_flatline([24.1, 24.2] * 20, TH)["level"] == "normal"


# ---------- R08 反复跳变 ----------
def test_r08_repeated_jumps_warn():
    assert rule_oscillation([23.0, 30.0, 23.0, 30.0, 23.0], TH)["level"] == "warn"


def test_r08_single_spike_normal():
    assert rule_oscillation([23.0, 30.0, 23.0], TH)["level"] == "normal"


# ---------- R09 即将越限 ----------
def test_r09_approaching_upper_warn():
    item = rule_approaching_limit([32.7, 32.9, 33.1, 33.3, 33.5], TH)
    assert item["level"] == "warn"
    assert "35.0" in item["evidence"]


def test_r09_approaching_lower_warn():
    assert rule_approaching_limit([12.0, 11.8, 11.6, 11.4, 11.2], TH)["level"] == "warn"


def test_r09_near_limit_but_flat_normal():
    assert rule_approaching_limit([33.5] * 5, TH)["level"] == "normal"


def test_r09_already_out_of_band_defers_to_r01():
    item = rule_approaching_limit([33.0, 34.0, 35.0, 36.0, 37.0], TH)
    assert item["level"] == "normal"
    assert "R01" in item["evidence"]


# ---------- R10 持续越限 ----------
def test_r10_sustained_out_of_band_is_fault():
    """36℃ 只在告警带外、没到故障线，但持续 10 帧 → 升级为 fault"""
    assert rule_sustained_excursion([36.0] * 10, TH)["level"] == "fault"


def test_r10_short_excursion_normal():
    assert rule_sustained_excursion([23.6] * 5 + [36.0] * 9, TH)["level"] == "normal"


def test_r10_recovered_resets():
    assert rule_sustained_excursion([36.0] * 12 + [23.6], TH)["level"] == "normal"


# ---------- R11 阶跃越界 ----------
def test_r11_step_into_fault_band_warn():
    item = rule_step_out_of_band([23.6, 23.6, 23.6, 50.0], TH)
    assert item["level"] == "warn"
    assert "23.6" in item["evidence"] and "50.0" in item["evidence"]


def test_r11_gradual_rise_out_of_band_normal():
    assert rule_step_out_of_band([33.0, 34.0, 35.0, 36.0, 37.0], TH)["level"] == "normal"


def test_r11_spike_within_band_normal():
    assert rule_step_out_of_band([23.6, 30.0, 23.6], TH)["level"] == "normal"


# ---------- 兼容性守卫 ----------
def test_r03_evidence_format_unchanged_with_default_points():
    item = rule_rising_trend([23.0, 23.5, 24.0, 24.8, 25.5], LEGACY_TH)
    assert item["evidence"] == "末 5 点 [23.0, 23.5, 24.0, 24.8, 25.5]，单调上升，累计 +2.5，阈值 2.0"


def test_legacy_thresholds_without_new_keys_do_not_crash():
    out = RuleAgent(thresholds={"temperature": LEGACY_TH}).run(
        {"readings": make_readings([23.6] * 5), "sensor_type": "temperature"}
    )
    assert len(out["items"]) == len(DEFAULT_RULES)
    assert out["triggered_count"] == 0
    skipped = {i["rule_id"] for i in out["items"] if "未配置" in i["evidence"]}
    assert skipped == {"R05_PHYSICAL_RANGE", "R06_FALLING_TREND", "R09_APPROACHING_LIMIT"}


def test_default_rules_at_least_10_with_unique_ids():
    out = RuleAgent().run({"readings": make_readings([23.6] * 5), "sensor_type": "temperature"})
    ids = [i["rule_id"] for i in out["items"]]
    assert len(ids) >= 10
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("values", [
    [23.6] * 5,
    [23.6] * 8 + [23.7] * 2,
    [23.6] * 10,
], ids=["green5", "boss_reviewed", "gapped_values"])
def test_all_green_fixtures_stay_untriggered(values):
    """路由守卫：这些是动态路由测试里走 lite 路径的数据，任何新规则让它们触发，lite 路由就会断"""
    out = RuleAgent().run({"readings": make_readings(values), "sensor_type": "temperature"})
    fired = [i["rule_id"] for i in out["items"] if i["level"] != "normal"]
    assert fired == []