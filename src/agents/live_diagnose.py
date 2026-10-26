"""实测模式：拿当前采样 buffer 跑一次真实诊断。

和 `demo_dynamic` 的区别：
- 数据来自 `src.tools.sensor_tools._history_buffer`（后台采样线程写入），不是构造数据；
- 时间用真实的 now，因此会真的触发 offline / degraded 判定；
- AI 默认关闭：规则全绿本来就走 lite（不调模型）；规则触发时才需要显式 use_ai，
  没开或装配失败时如实返回 partial + 错误原因，不静默编一个结论。

依赖边界：本模块只 import agents 层。LLM 与向量库适配器都是函数内延迟导入，
没装 ML 栈时 import 本模块仍然可用，只是 use_ai=True 会返回明确的失败原因。
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Callable, Optional

from .base import AgentFailure
from .supervisor import PipelineConfig, _initial_state, build_pipeline

LLM = Callable[[str, str], str]      # (system, user) -> text
Retriever = Callable[[str], list]    # query -> 知识库片段
Reader = Callable[[str], list]       # sensor_type -> buffer 读数
Clock = Callable[[], datetime]


# ── 计数器：让页面/报告能显示"到底调了几次" ──────────────────────────
class _CountedLLM:
    def __init__(self, fn: Optional[LLM] = None):
        self._fn = fn
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if self._fn is None:
            raise AgentFailure("ai_analysis", "未启用 AI：本次只做规则诊断")
        return self._fn(system, user)


class _CountedRetriever:
    def __init__(self, fn: Optional[Retriever] = None):
        self._fn = fn
        self.queries: list[str] = []

    def __call__(self, query: str) -> list:
        self.queries.append(query)
        return list(self._fn(query)) if self._fn else []


class _CountedReader:
    def __init__(self, fn: Reader):
        self._fn = fn
        self.calls = 0

    def __call__(self, sensor_type: str) -> list:
        self.calls += 1
        return list(self._fn(sensor_type))


def _buffer_reader(sensor_type: str) -> list:
    """默认读数来源：后台采样线程写的 buffer（拷贝一份，避免遍历时被写入）。"""
    from src.tools.sensor_tools import _history_buffer

    return list(_history_buffer.get(sensor_type, []))


def build_ai_adapters() -> tuple[Optional[LLM], Optional[Retriever], str]:
    """尝试装配真实 LLM + 向量库检索。返回 (llm, retriever, 失败原因)。"""
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from src.config import get_config, require_api_key
        from src.llm import get_llm

        require_api_key()
        chat = get_llm(streaming=False)

        def llm(system: str, user: str) -> str:
            reply = chat.invoke([
                SystemMessage(content=system),
                HumanMessage(content=user),
            ])
            return reply.content if isinstance(reply.content, str) else str(reply.content)

    except Exception as exc:  # 缺 Key / 缺依赖 / 初始化失败，都不该让页面崩
        return None, None, f"AI 不可用（{type(exc).__name__}）：{exc}"

    try:
        from src.vectorstore import load_vectorstore

        store = load_vectorstore()
        k = get_config().retrieve_top_k

        def retriever(query: str) -> list:
            return [d.page_content for d in store.similarity_search(query, k=k)]

    except Exception as exc:
        return None, None, f"知识库不可用（{type(exc).__name__}）：{exc}"

    return llm, retriever, ""


def diagnose_live(
    sensor_type: str,
    window_hours: float,
    *,
    reader: Optional[Reader] = None,
    clock: Optional[Clock] = None,
    use_ai: bool = False,
    llm: Optional[LLM] = None,
    retriever: Optional[Retriever] = None,
) -> tuple[dict, dict]:
    """跑一次实测诊断，返回 (state, stats)。

    stats["ai_enabled"] / stats["ai_note"] 会让 UI 能如实说明"这次到底有没有调模型"。
    """
    note = ""
    if use_ai and (llm is None or retriever is None):
        llm, retriever, note = build_ai_adapters()

    ai_enabled = llm is not None and retriever is not None
    counted_llm = _CountedLLM(llm)
    counted_retriever = _CountedRetriever(retriever)
    counted_reader = _CountedReader(reader or _buffer_reader)

    cfg = PipelineConfig(
        llm=counted_llm,
        retriever=counted_retriever,
        reader=counted_reader,
        clock=clock,
    )
    result = build_pipeline(cfg).invoke(_initial_state(sensor_type, window_hours))

    stats = {
        "ai_enabled": ai_enabled,
        "ai_note": note if use_ai else "未启用 AI：本次只做规则诊断",
        "llm": counted_llm,
        "retriever": counted_retriever,
        "reader_calls": counted_reader.calls,
    }
    return result, stats


# ── 导出 ─────────────────────────────────────────────────────────────
def to_markdown(result: dict, stats: Optional[dict] = None,
                sensor_type: str = "", window_hours: float | None = None) -> str:
    """把一次诊断结果导成 Markdown 报告（给评委看/存档用）。"""
    stats = stats or {}
    collector = result.get("collector") or {}
    report = result.get("report") or {}
    ai = result.get("ai_analysis") or {}
    rule = None
    for value in result.values():
        if isinstance(value, dict) and "items" in value and "triggered_count" in value:
            rule = value
            break

    lines = [
        "# 传感器诊断报告",
        "",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 传感器：{sensor_type or '（未标注）'}　窗口：{window_hours if window_hours is not None else '（未标注）'} 小时",
        f"- 最终状态：**{result.get('status')}**",
        f"- 路由路径：{' → '.join(result.get('routing_path') or [])}",
    ]
    if collector:
        lines += [
            f"- 数据质量：{collector.get('data_quality')}（有效读数 {len(collector.get('readings') or [])} 帧，"
            f"缺口 {collector.get('missing_count', 0)}，坏帧 {collector.get('bad_frame_count', 0)}）",
        ]
    if "llm" in stats:
        lines.append(
            f"- 模型调用：{len(getattr(stats['llm'], 'calls', []))} 次，"
            f"知识库检索：{len(getattr(stats.get('retriever'), 'queries', []))} 次"
        )

    lines += ["", "## 规则诊断", ""]
    if rule is None:
        lines.append("规则未运行。")
    else:
        lines.append(f"触发 {rule.get('triggered_count', 0)} 条：")
        lines.append("")
        for item in rule.get("items") or []:
            mark = "⛔" if item.get("level") == "fault" else "⚠️" if item.get("level") == "warn" else "·"
            lines.append(f"- {mark} `{item['rule_id']}` [{item['level']}] {item.get('evidence', '')}")

    lines += ["", "## 结论", ""]
    if report:
        lines.append(report.get("summary", ""))
        lines.append("")
        lines.append(f"置信度：{report.get('confidence')}")
    else:
        lines.append("未生成报告（管线在上游停止）。")

    if ai.get("root_cause"):
        lines += ["", "## 根因与建议", "", f"- 根因：{ai['root_cause']}", f"- 建议：{ai['recommendation']}"]

    errors = result.get("errors") or []
    if errors:
        lines += ["", "## 错误与跳过原因", ""]
        lines += [f"- `{e['agent']}`：{e['reason']}" for e in errors]

    if report.get("evidence_chain"):
        lines += ["", "## 证据链", ""]
        lines += [f"{i}. {step}" for i, step in enumerate(report["evidence_chain"], 1)]

    return "\n".join(lines).rstrip() + "\n"


def to_json(result: dict, stats: Optional[dict] = None, **meta) -> str:
    """导出原始 state（不含不可序列化对象），便于复查什么真的发生了。"""
    payload = {
        "meta": meta,
        "state": result,
        "calls": {
            "llm": len(getattr((stats or {}).get("llm"), "calls", [])),
            "retrieval": len(getattr((stats or {}).get("retriever"), "queries", [])),
            "reader": (stats or {}).get("reader_calls"),
            "ai_enabled": (stats or {}).get("ai_enabled"),
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n"
