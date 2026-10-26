"""传感器诊断管线 · 路由可视化 + 一键诊断（Streamlit）。

两个模式：
- 演示：跑 `src.agents.demo_dynamic` 的三条构造路径（A 补采 / B 全绿 / C 离线）；
- 实测：读当前采样 buffer 跑一次真实诊断（collector → rule → [AI] → report → supervisor）。

页面上的每个数字都从管线返回值推出，不手写结论；诊断结果可导出 Markdown / JSON。

运行：streamlit run streamlit_diagnosis.py
"""
from __future__ import annotations

from datetime import datetime

import streamlit as st

from src.agents.demo_dynamic import (
    _ai_mode,
    _find_rule_output,
    _run_path_a,
    _run_path_b,
    _run_path_c,
)
from src.agents.live_diagnose import diagnose_live, to_json, to_markdown

st.set_page_config(page_title="诊断管线 · 路由可视化", page_icon="🧭", layout="wide")

MODE_DEMO = "演示：三条构造路径"
MODE_LIVE = "实测：当前采样缓冲区"

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


def show_routing_path(result: dict) -> None:
    st.markdown("### 🧭 路由轨迹")
    st.markdown("  →  ".join(_stage_chip(n) for n in result["routing_path"]))


def render(result: dict, stats: dict) -> None:
    errors = result.get("errors") or []
    rules = _find_rule_output(result)
    collector = result.get("collector")
    status = result["status"]

    # ── 路由轨迹 ────────────────────────────────────────────────
    show_routing_path(result)

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
        len(stats["llm"].calls) if stats.get("llm") is not None else "—",
        help="全绿路径与离线路径都应保持 0 次",
    )

    st.caption(
        f"AI 路径：{_ai_mode(result)}　·　"
        f"RAG 检索：{len(stats['retriever'].queries) if stats.get('retriever') is not None else 0} 次　·　"
        f"reader 调用：{stats.get('reader_calls', '—')} 次"
    )
    if stats.get("ai_note"):
        st.caption(f"ℹ️ {stats['ai_note']}")

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


def export_buttons(result: dict, stats: dict, tag: str,
                   sensor: str = "", hours: float | None = None) -> None:
    """把当前结果导成 Markdown / JSON。按钮只触发下载，不写服务端文件。"""
    md = to_markdown(result, stats, sensor_type=sensor, window_hours=hours)
    payload = to_json(
        result, stats, source=tag, sensor_type=sensor, window_hours=hours,
        generated_at=datetime.now().isoformat(timespec="seconds"),
    )
    c1, c2, _ = st.columns([1, 1, 2])
    c1.download_button("⬇️ 下载报告（Markdown）", md,
                       file_name="sensor_diagnosis.md", mime="text/markdown")
    c2.download_button("⬇️ 下载原始 state（JSON）", payload,
                       file_name="sensor_diagnosis.json", mime="application/json")


# ── 页面 ────────────────────────────────────────────────────────
st.title("🧭 传感器诊断管线 · 路由可视化")
st.caption(
    "同一套管线：数据质量 → 路由 → 规则 / AI → 报告。"
    "页面数字全部来自管线返回值，不手写结论。"
)

with st.sidebar:
    st.header("模式")
    mode = st.radio("模式", [MODE_DEMO, MODE_LIVE], label_visibility="collapsed")
    st.divider()

    if mode == MODE_DEMO:
        st.caption("构造数据，用来演示三条路由。")
        choice = st.radio("路径", list(PATHS), label_visibility="collapsed")
        st.divider()
        st.caption("口令版：`python -m src.agents.demo_dynamic`")
    else:
        from src.tools.sensor_tools import (
            SAMPLE_INTERVAL_SEC, _history_buffer, get_sampler, start_sampler,
        )

        st.caption("数据来自后台采样线程写入的 buffer，时间用真实 now。")
        _sampler = get_sampler()
        if _sampler is not None and _sampler.is_alive():
            st.caption("后台采样：运行中")
        else:
            st.caption("后台采样：未运行（buffer 为空时诊断会判 offline）")
            if st.button("启动后台采样"):
                start_sampler()
                st.rerun()
        sensor = st.selectbox("传感器", ["temperature", "humidity"])
        hours = st.selectbox("窗口（小时）", [1.0, 2.0, 6.0], index=0)
        use_ai = st.checkbox(
            "规则触发时调用 AI",
            value=False,
            help="需要 DEEPSEEK_API_KEY 与已构建的向量库；不开则只做规则诊断。",
        )
        st.caption(
            f"buffer：温度 {len(_history_buffer['temperature'])} 条，"
            f"湿度 {len(_history_buffer['humidity'])} 条（每 {SAMPLE_INTERVAL_SEC:.0f}s 一条）"
        )

if mode == MODE_DEMO:
    runner = PATHS[choice]
    try:
        _result, _stats = runner()
    except AssertionError as exc:  # demo 自带断言：全绿不应调 LLM、离线应判 failed
        st.error(f"演示自查失败：{exc}")
        st.stop()

    render(_result, _stats)
    st.divider()
    export_buttons(_result, _stats, tag=choice)

else:
    st.info(
        "实测模式：点下面的按钮，用当前 buffer 跑一次真实诊断。"
        "规则全绿会直接出 lite 结论（不调模型）；传感器离线会直接判 failed。"
    )
    if st.button("▶️ 运行诊断", type="primary"):
        with st.spinner("正在诊断…"):
            st.session_state["live"] = diagnose_live(sensor, hours, use_ai=use_ai)
        st.session_state["live_meta"] = {"sensor": sensor, "hours": hours, "use_ai": use_ai}

    if "live" in st.session_state:
        _result, _stats = st.session_state["live"]
        _meta = st.session_state.get("live_meta", {})
        render(_result, _stats)
        st.divider()
        export_buttons(_result, _stats, tag="live",
                       sensor=_meta.get("sensor", ""), hours=_meta.get("hours"))
    else:
        st.caption("尚未运行。可先在演示模式里看三条构造路径。")
