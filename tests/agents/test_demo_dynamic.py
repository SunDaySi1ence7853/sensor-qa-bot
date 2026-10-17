"""demo 路径A 守卫：四段路由（degraded → 补采 → 告警 → 完整AI）必须真实发生且完整打印。

背景：2d5ecfe 修 reader 时误删了路径A 的 _print_result，被验收点名两次。
"""
from src.agents.demo_dynamic import _find_rule_output, _run_path_a, run_path_a_degraded_then_warn


def test_path_a_four_stages_actually_happen():
    """断言运行事实，不断言打印文本：防止"打印写了，事情没发生"。"""
    result, stats = _run_path_a()

    assert stats["reader_calls"] == 2, f"应首采 + 补采共 2 次，实际 {stats['reader_calls']}"

    rules = _find_rule_output(result)
    assert rules is not None, "state 中找不到规则输出"
    fired = {i["rule_id"] for i in rules["items"] if i["level"] != "normal"}
    assert "R03_RISING_TREND" in fired, f"补采数据应触发 R03，实际触发 {fired}"

    assert "ai_analysis_lite" not in result["routing_path"], f"应走完整 AI：{result['routing_path']}"
    assert len(stats["llm"].calls) >= 1, "完整 AI 应调用 LLM"
    assert len(stats["retriever"].queries) >= 1, "完整 AI 应走 RAG 检索"


def test_path_a_prints_all_four_stages(capsys):
    run_path_a_degraded_then_warn()
    out = capsys.readouterr().out
    for needle in ("路径A", "帧间有洞", "含补采", "规则告警", "完整 RAG + LLM", "路由路径"):
        assert needle in out, f"路径A 打印缺少 {needle!r}\n---\n{out}"