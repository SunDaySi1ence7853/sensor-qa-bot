"""任务3 验收场景：测试与演示共用一份数据，预期结果写死在这里。

运行：python -m src.agents.scenarios
全部场景通过时退出码为 0，否则为 1（check.ps1 靠这个判断）。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

from .fakes import make_readings
from .rule_agent import RuleAgent


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    sensor_type: str
    values: tuple
    expected: dict                    # rule_id -> level，精确相等（多报、漏报、级别错都算失败）
    story: str                        # 现场含义：为什么是这些规则
    evidence_must_contain: dict = field(default_factory=dict)  # rule_id -> 必须出现的数字


_S1_RAMP = [30.0, 32.0, 34.0, 36.0, 38.0, 40.0, 42.0, 44.0, 46.0]
_S1_HOLD = [46.0, 46.2, 46.1, 46.3, 46.2, 46.4, 46.3, 46.5, 46.4, 46.6, 46.5, 46.7]

SCENARIOS: list[Scenario] = [
    Scenario(
        "S1", "机柜持续过热", "temperature", tuple(_S1_RAMP + _S1_HOLD),
        {"R01_LATEST_THRESHOLD": "fault", "R02_WINDOW_EXTREME": "fault", "R10_SUSTAINED_EXCURSION": "fault"},
        "风扇停转后温度逐步爬升，停在 46℃ 附近。当前值超故障线（R01）、窗口内越界（R02），"
        "且连续 18 帧在带外（R10），说明是持续过热而不是偶发尖峰。升温是渐变的，所以 R04/R11 不报",
        {"R01_LATEST_THRESHOLD": ("46.7", "45.0"), "R10_SUSTAINED_EXCURSION": ("连续 18 点", "≥10")},
    ),
    Scenario(
        "S2", "即将越限", "temperature", (31.0, 31.5, 32.0, 32.5, 32.7, 32.9, 33.1, 33.3, 33.5),
        {"R09_APPROACHING_LIMIT": "warn"},
        "33.5℃ 还在告警带内，但单调爬升，距告警线只剩 1.5℃。R01 不报（未越限），"
        "R03 不报（末 5 点升幅 0.8 < 2.0），只有组合规则 R09 能提前预警",
        {"R09_APPROACHING_LIMIT": ("33.5", "35.0", "1.5")},
    ),
    Scenario(
        "S3", "传感器冻结", "temperature", tuple([24.1] * 45),
        {"R07_FLATLINE": "warn"},
        "45 帧（约 22 分钟）读数纹丝不动，而真实 DHT22 有 ±0.1 级噪声，指向传感器卡死或链路重放旧值。"
        "读数本身在正常带内，所以阈值类规则全部不报",
        {"R07_FLATLINE": ("末 40 点", "极差 0.0")},
    ),
    Scenario(
        "S4", "接线松动（单帧尖峰）", "temperature", tuple([23.6] * 6 + [30.0] + [23.6] * 6),
        {"R04_JUMP": "warn"},
        "单帧跳到 30℃ 后立即回落。尖峰在告警带内，所以 R11 不报；只有一上一下 2 次跳变，所以 R08 不报。"
        "只报 R04 跳变，指向接线或接触问题",
        {"R04_JUMP": ("6.4", "5.0")},
    ),
    Scenario(
        "S5", "湿度传感器损坏", "humidity", (50.2, 50.4, 50.1, 50.3, 105.0),
        {"R01_LATEST_THRESHOLD": "fault", "R02_WINDOW_EXTREME": "fault", "R04_JUMP": "warn",
         "R05_PHYSICAL_RANGE": "fault", "R11_STEP_OUT_OF_BAND": "warn"},
        "湿度从 50% 一步跳到 105%。R05 判定物理上不可能，是传感器故障；R11 阶跃越界、R04 跳变从侧面印证。"
        "R01/R02 仍按故障带报 fault，这是已知限制：它们还不区分环境越限和传感器越限，见设计变更记录",
        {"R05_PHYSICAL_RANGE": ("105.0", "[0.0, 100.0]"), "R11_STEP_OUT_OF_BAND": ("50.3", "105.0")},
    ),
    Scenario(
        "S0", "对照组：正常波动", "temperature",
        (23.5, 23.7, 23.6, 23.8, 23.6, 23.7, 23.5, 23.6, 23.7, 23.6),
        {},
        "正常小幅波动，11 条规则全部不报。证明规则库不是什么都报",
    ),
]


def run_scenario(sc: Scenario) -> dict:
    return RuleAgent().run({"readings": make_readings(list(sc.values)), "sensor_type": sc.sensor_type})


def fired(out: dict) -> dict:
    return {i["rule_id"]: i["level"] for i in out["items"] if i["level"] != "normal"}


def _fmt(d: dict) -> str:
    return "空集" if not d else ", ".join(f"{k}={v}" for k, v in sorted(d.items()))


if __name__ == "__main__":
    passed = 0
    for sc in SCENARIOS:
        out = run_scenario(sc)
        got = fired(out)
        ok = got == sc.expected
        passed += ok
        print(f"===== {sc.key} {sc.title} {'✓' if ok else '✗'} =====")
        print(f"  现场：{sc.story}")
        print(f"  预期：{_fmt(sc.expected)}")
        print(f"  实际：{_fmt(got)}")
        for item in out["items"]:
            if item["level"] != "normal":
                print(f"    [{item['level']:<5}] {item['rule_id']}：{item['evidence']}")
        print()
    print(f"场景通过 {passed}/{len(SCENARIOS)}")
    sys.exit(0 if passed == len(SCENARIOS) else 1)