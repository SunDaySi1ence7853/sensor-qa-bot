"""
路径A：数据质量降级 → 补采 → 规则有告警 → 完整 AI 分析
路径B：数据质量正常 → 规则全绿 → AI 简化（跳过 RAG）
路径C：数据离线（offline）→ 直达 supervisor → failed，rule/AI/report 均不运行
每段打印都由运行事实推出（reader 调用次数、规则触发数、LLM/RAG 调用次数），
不手写结论：哪一段没发生，打印就如实显示，测试也会红。
三条路径共用 _print_result，禁止单独手打结果行。
"""
from __future__ import annotations

import json

from .fakes import (
    FAKE_MANUAL, FIXED_NOW, FakeLLM, FakeRetriever, make_gapped_readings, make_readings,
)
from .supervisor import PipelineConfig, _initial_state, build_pipeline

_LLM_RESP = json.dumps(
    {"root_cause": "温度单调上升，符合散热不良特征", "recommendation": "检查散热风扇与通风口", "cited": [2]},
    ensure_ascii=False,
)

SEP = "=" * 60


def _find_rule_output(result: dict):
    """在 state 里找规则引擎输出（同时带 items 和 triggered_count 的 dict）。

    不写死 state 键名，避免 demo 和 state 契约耦合；找不到时返回 None，打印会如实显示。
    """
    for value in result.values():
        if isinstance(value, dict) and "items" in value and "triggered_count" in value:
            return value
    return None


def _ai_mode(result: dict) -> str:
    path = result["routing_path"]
    if "ai_analysis_lite" in path:
        return "lite（跳过 RAG，LLM 零调用）"
    if "ai_analysis" in path:
        return "完整 RAG + LLM"
    return "未运行"


def _print_result(label: str, result: dict, stats: dict | None = None) -> None:
    stats = stats or {}
    print(f"\n{SEP}")
    print(f"【{label}】")
    print(f"  路由路径   : {' → '.join(result['routing_path'])}")

    # 数据质量由 collector 产出推出，不手写
    collector = result.get("collector")
    if collector is not None:
        print(f"  数据质量   : {collector['data_quality']}（有效读数 {len(collector['readings'])} 帧）")

    if "first_batch" in stats:
        print(f"  首采       : {stats['first_batch']}")
    if "reader_calls" in stats:
        n = stats["reader_calls"]
        print(f"  采集       : reader 调用 {n} 次（{'含补采' if n >= 2 else '无补采'}）")

    rules = _find_rule_output(result)
    if rules is None:
        ran = any(n.startswith("rule") for n in result["routing_path"])
        print(f"  规则       : {'（state 中未找到规则输出）' if ran else '未运行'}")
    else:
        fired = [f"{i['rule_id']}={i['level']}" for i in rules["items"] if i["level"] != "normal"]
        verdict = "规则告警" if fired else "规则全绿"
        print(f"  规则       : {verdict}，触发 {rules['triggered_count']} 条 {fired or ''}".rstrip())

    print(f"  AI 路径    : {_ai_mode(result)}")
    if "llm" in stats and "retriever" in stats:
        print(f"  调用统计   : LLM {len(stats['llm'].calls)} 次，RAG 检索 {len(stats['retriever'].queries)} 次")

    print(f"  最终状态   : {result['status']}")
    print(f"  错误列表   : {result['errors'] or '（无）'}")
    report = result.get("report")
    if report:
        print(f"  报告摘要   : {report['summary']}")
        print(f"  置信度     : {report['confidence']}")
    print(SEP)


def _run_path_a() -> tuple[dict, dict]:
    """路径A 本体：返回 (state 结果, 运行事实)。测试直接断言运行事实，不依赖打印文本。"""
    stats = {
        "reader_calls": 0,
        "first_batch": "10 帧，前 5 帧整体前移 300s（帧间有洞 → 应判 degraded）",
        "llm": FakeLLM(_LLM_RESP),
        "retriever": FakeRetriever(FAKE_MANUAL),
    }

    def stateful_reader(sensor_type: str):
        stats["reader_calls"] += 1
        if stats["reader_calls"] == 1:
            return make_gapped_readings([23.6] * 10)          # 帧间有洞 → degraded
        return make_readings([23.0, 23.5, 24.0, 24.8, 25.5])  # 补采：单调升 2.5 → R03 告警

    cfg = PipelineConfig(
        llm=stats["llm"],
        retriever=stats["retriever"],
        reader=stateful_reader,
        clock=lambda: FIXED_NOW,
    )
    result = build_pipeline(cfg).invoke(_initial_state("temperature", 1.0))
    return result, stats


def _run_path_b() -> tuple[dict, dict]:
    stats = {"llm": FakeLLM(_LLM_RESP), "retriever": FakeRetriever(FAKE_MANUAL)}
    cfg = PipelineConfig(
        llm=stats["llm"],
        retriever=stats["retriever"],
        reader=lambda s: make_readings([23.6] * 10),   # 老板复核过的那组数
        clock=lambda: FIXED_NOW,
    )
    result = build_pipeline(cfg).invoke(_initial_state("temperature", 1.0))
    return result, stats


def run_path_a_degraded_then_warn():
    """路径A：第一次采集 degraded → re_collect → 规则触发告警 → 完整 AI。"""
    result, stats = _run_path_a()
    _print_result("路径A：degraded → 补采 → 告警 → 完整AI", result, stats)
    return result


def run_path_b_ok_all_green():
    """路径B：数据质量正常，规则全绿 → AI 简化（跳 RAG，不调 LLM）。"""
    result, stats = _run_path_b()
    _print_result("路径B：正常 → 全绿 → AI简化（跳RAG）", result, stats)
    assert stats["llm"].calls == [], "全绿路径不应调 LLM"
    print("  ✓ 确认：全绿路径 LLM 调用次数 = 0")
    return result


def _run_path_c() -> tuple[dict, dict]:
    """路径C 本体：buffer 为空 → offline → 直达 supervisor。"""
    stats = {
        "reader_calls": 0,
        "first_batch": "空 buffer（传感器无数据 → 应判 offline）",
        "llm": FakeLLM(_LLM_RESP),
        "retriever": FakeRetriever(FAKE_MANUAL),
    }

    def empty_reader(sensor_type: str):
        stats["reader_calls"] += 1
        return []

    cfg = PipelineConfig(
        llm=stats["llm"],
        retriever=stats["retriever"],
        reader=empty_reader,
        clock=lambda: FIXED_NOW,
    )
    result = build_pipeline(cfg).invoke(_initial_state("temperature", 1.0))
    return result, stats


def run_path_c_offline():
    """路径C：传感器离线 → 直达 supervisor → failed，下游零调用。"""
    result, stats = _run_path_c()
    _print_result("路径C：offline → 直达 supervisor → failed", result, stats)
    assert result["status"] == "failed", "离线路径应判 failed"
    assert stats["llm"].calls == [] and stats["retriever"].queries == [], "离线路径不应调 LLM/RAG"
    print("  ✓ 确认：离线路径 LLM 0 次、RAG 0 次，状态 failed")
    return result


if __name__ == "__main__":
    print("动态路由演示")
    run_path_a_degraded_then_warn()
    run_path_b_ok_all_green()
    run_path_c_offline()