"""演示动态路由的两条路径，输出可直接粘进群。

路径A：数据质量降级 → 补采 → 规则有告警 → 完整 AI 分析
路径B：数据质量正常 → 规则全绿 → AI 简化（跳过 RAG）
"""
import json

from .fakes import FAKE_MANUAL, FIXED_NOW, FakeLLM, FakeRetriever, make_readings
from .supervisor import PipelineConfig, _initial_state, build_pipeline

_LLM_RESP = json.dumps(
    {"root_cause": "温度单调上升，符合散热不良特征", "recommendation": "检查散热风扇与通风口", "cited": [2]},
    ensure_ascii=False,
)

SEP = "=" * 60


def _print_result(label: str, result: dict) -> None:
    print(f"\n{SEP}")
    print(f"【{label}】")
    print(f"  路由路径   : {' → '.join(result['routing_path'])}")
    print(f"  最终状态   : {result['status']}")
    print(f"  错误列表   : {result['errors'] or '（无）'}")
    if result.get("report"):
        r = result["report"]
        print(f"  报告摘要   : {r['summary']}")
        print(f"  置信度     : {r['confidence']}")
        print(f"  AI 路径    : {'lite（无 RAG）' if not r.get('evidence_chain', [''])[-1].startswith('[知识库') else '完整 RAG'}")
    print(SEP)


def run_path_a_degraded_then_warn():
    """路径A：第一次采集 degraded → re_collect → 规则触发告警 → 完整 AI。"""
    call_count = 0

    def stateful_reader(sensor_type: str):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # 首次采集：缺帧，质量降级
            return make_readings([23.6, 23.6, 23.7], gap_before=True)
        # 补采：读数恢复，但温度已呈上升趋势
        return make_readings([23.0, 23.5, 24.0, 24.8, 25.5])

    cfg = PipelineConfig(
        llm=FakeLLM(_LLM_RESP),
        retriever=FakeRetriever(FAKE_MANUAL),
        reader=stateful_reader,
        clock=lambda: FIXED_NOW,
    )
    pipeline = build_pipeline(cfg)
    result = pipeline.invoke(_initial_state("temperature", 1.0))
    _print_result("路径A：降级→补采→告警→完整AI", result)
    return result


def run_path_b_ok_all_green():
    """路径B：数据质量正常，规则全绿 → AI 简化（跳 RAG，不调 LLM）。"""
    llm = FakeLLM(_LLM_RESP)

    cfg = PipelineConfig(
        llm=llm,
        retriever=FakeRetriever(FAKE_MANUAL),
        reader=lambda s: make_readings([23.6] * 10),   # 老板复核过的那组数
        clock=lambda: FIXED_NOW,
    )
    pipeline = build_pipeline(cfg)
    result = pipeline.invoke(_initial_state("temperature", 1.0))
    _print_result("路径B：正常→全绿→AI简化（跳RAG）", result)
    assert llm.calls == [], "全绿路径不应调 LLM"
    print("  ✓ 确认：全绿路径 LLM 调用次数 = 0")
    return result


if __name__ == "__main__":
    print("动态路由演示")
    run_path_a_degraded_then_warn()
    run_path_b_ok_all_green()