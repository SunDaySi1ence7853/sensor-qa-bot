"""传感器诊断管线 · 路由可视化（Streamlit 初版）。

用途：把 `src.agents.demo_dynamic` 的三条路由（A 补采 / B 全绿 / C 离线）画到页面上，
评委不读终端也能看到「数据质量 → 走了哪条路 → 谁被调用 → 最终状态」。

特点：
- 不调 LLM、不读向量库、不需要 API Key（全用 src.agents.fakes 的替身），现场断网也能跑；
- 页面上的每个数字都从管线返回的 state / stats 推出，不手写结论；
- 演示数据是构造的，页面上明确标注，不冒充真实用户数据。

运行：streamlit run streamlit_diagnosis.py
"""
from __future__ import annotations

import streamlit as st

from src.agents.demo_dynamic import (
    _ai_mode,
    _find_rule_output,
    _run_path_a,
    _run_path_b,
    _run_path_c,
)

st.set_page_config(page_title="诊断管线 · 路由可视化", page_icon="🧭", layout="wide")

PATHS = {
    "路径A：degraded → 补采 → 告警 → 完整AI": _run_path_a,
    "路径B：正常 → 全绿 → AI 简化（跳 RAG）": _run_path_b,
    "路径C：offline → 直达 supervisor → failed": _run_path_c,
}

STAGE_LABELS = {
    "collector": "采集",
    "re_collect": "补采",
    "rule": "规则",
    "ai_analysis": "AI 分析",
    "ai_analysis_lite": "AI 简化",
    "report": "报告",
    "supervisor": "汇总",
}


def _stage_chip(node: str) -> str:
    """把一个 routing_path 节点画成一枚徽章；skip / fail 变体单独标色。"""
    label = STAGE_LABELS.get(node, node)
    if "(skip)" in node:
        return f"⬜ ~~{label}（跳过）~~"
    if "(fail)" in node:
        return f"🟥 **{label}（失败）**"
    return f"🟦 **{label}**"


def render(result: dict, stats: dict) -> None:
    errors = result.get("errors") or []
    rules = _find_rule_output(result)
    collector = result.get("collector")
    status = result["status"]

    # ── 路由轨迹 ────────────────────────────────────────────────
    st.markdown("### 🧭 路由轨迹")
    st.markdown("  →  ".join(_stage_chip(n) for n in result["routing_path"]))

    # ── 关键事实 ────────────────────────────────────────────────
    st.markdown("### 📌 关键事实")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("最终状态", status)
    c2.metric(
        "数据质量",
        collector["data_quality"] if collector else "未产出",
        help="collector 如实上报；offline 不会进入规则与 AI",
    )
    c3.metric(
        "有效读数",
        f"{len(collector['readings'])} 帧" if collector else "0 帧",
    )
    c4.metric("规则触发", f"{rules['triggered_count']} 条" if rules else "未运行")
    c5.metric(
        "LLM 调用",
        len(stats["llm"].calls) if "llm" in stats else "—",
        help="全绿路径与离线路径都应保持 0 次",
    )

    st.caption(
        f"AI 路径：{_ai_mode(result)}　·　"
        f"RAG 检索：{len(stats['retriever'].queries) if 'retriever' in stats else 0} 次　·　"
        f"reader 调用：{stats.get('reader_calls', '—')} 次"
    )

    # ── 规则明细 ────────────────────────────────────────────────
    st.markdown("### 📏 规则明细")
    if rules is None:
        ran = any(n.startswith("rule") for n in result["routing_path"])
        st.info("规则未运行（数据不可信时不该下结论）" if not ran else "规则节点执行了但没有产出")
    else:
        fired = [i for i in rules["items"] if i["level"] != "normal"]
        if fired:
            st.warning("触发的规则：" + "、".join(f"`{i['rule_id']}`={i['level']}" for i in fired))
        else:
            st.success("规则全绿：0 条触发，因此跳过 RAG 与 LLM")

    # ── 错误与报告 ──────────────────────────────────────────────
    col_err, col_rep = st.columns(2)
    with col_err:
        st.markdown("### ⚠️ 错误列表")
        if errors:
            for e in errors:
                st.error(f"**{e['agent']}**：{e['reason']}")
        else:
            st.caption("（无）")
    with col_rep:
        st.markdown("### 📄 报告")
        report = result.get("report")
        if report:
            st.write(report["summary"])
            st.caption(f"置信度：{report['confidence']}")
        else:
            st.caption("未生成报告（管线在上游就停了，不会硬编一份出来）")

    with st.expander("🔍 原始 state（对账用）"):
        st.json({k: v for k, v in result.items() if k != "routing_path"})


# ── 页面 ────────────────────────────────────────────────────────
st.title("🧭 传感器诊断管线 · 路由可视化")
st.caption(
    "同一套管线、三份数据、三条路线。演示数据为构造数据；"
    "页面数字全部来自管线返回值，不手写结论。"
)

with st.sidebar:
    st.header("选择场景")
    choice = st.radio("路径", list(PATHS), label_visibility="collapsed")
    st.divider()
    st.caption(
        "口令版：`python -m src.agents.demo_dynamic`\n\n"
        "不依赖网络、不调 LLM、不需要 API Key。"
    )
    st.caption("离线路径的意义：数据不可信时，管线宁可不说话。")

runner = PATHS[choice]
try:
    _result, _stats = runner()
except AssertionError as exc:  # demo 自带断言：全绿不应调 LLM、离线应判 failed
    st.error(f"演示自查失败：{exc}")
    st.stop()

render(_result, _stats)
