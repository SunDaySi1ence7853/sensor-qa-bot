"""任务3 验收：5 个异常场景加 1 个对照组，规则集精确相等，依据可复算。"""
import pytest

from src.agents.rule_config import load_rule_thresholds
from src.agents.scenarios import SCENARIOS, fired, run_scenario


def test_scenario_data_shape_preconditions():
    """前置：先确认数据形态真的是场景描述的那样，避免测试测了个寂寞（吸取 degraded 那次的教训）"""
    th = load_rule_thresholds()
    t, h = th["temperature"], th["humidity"]
    s = {x.key: x for x in SCENARIOS}

    tail_out = 0
    for v in reversed(s["S1"].values):
        if t["warn"][0] <= v <= t["warn"][1]:
            break
        tail_out += 1
    assert tail_out >= t["sustained_frames"]

    v2 = s["S2"].values
    assert t["warn"][1] - v2[-1] <= t["approach_margin"]
    assert v2[-1] - v2[-t["trend_points"]] < t["trend_rise"]   # 升幅不够 R03，预警只能来自 R09

    assert len(s["S3"].values) >= t["flatline_frames"]

    v4 = s["S4"].values
    assert max(abs(b - a) for a, b in zip(v4, v4[1:])) > t["jump"]
    assert max(v4) <= t["warn"][1]                              # 尖峰在带内，否则测到的是 R11 而不是 R04

    assert max(s["S5"].values) > h["physical"][1]


@pytest.mark.parametrize("sc", SCENARIOS, ids=lambda s: s.key)
def test_scenario_exact_rule_set(sc):
    out = run_scenario(sc)
    got = fired(out)
    assert got == sc.expected, (
        f"{sc.key} {sc.title}：多报 {set(got) - set(sc.expected)}，"
        f"漏报 {set(sc.expected) - set(got)}，实际 {got}"
    )
    assert out["triggered_count"] == len(sc.expected)


@pytest.mark.parametrize("sc", [s for s in SCENARIOS if s.evidence_must_contain], ids=lambda s: s.key)
def test_scenario_evidence_cites_numbers(sc):
    """可解释性：没看过代码的人，读 evidence 就能复算结论"""
    items = {i["rule_id"]: i for i in run_scenario(sc)["items"]}
    for rid, needles in sc.evidence_must_contain.items():
        for needle in needles:
            assert needle in items[rid]["evidence"], f"{rid} 依据缺少 {needle!r}：{items[rid]['evidence']}"