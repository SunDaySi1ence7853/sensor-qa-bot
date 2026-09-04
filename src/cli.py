"""
CLI 入口模块。

提供两个命令：
- sensor-qa       : 启动交互式问答
- sensor-qa-build : 构建/重建向量知识库

防御性设计：
1. 启动前校验关键配置（API Key、embedding provider），
   避免让用户在真正调用 API 时才看到晦涩的 401/连接错误。
2. 所有未预期异常在最外层被捕获，只向终端输出简洁的中文提示，
   完整 traceback 只写入日志文件（配合第三关日志系统）。
3. --debug 参数用于切换日志级别，无需修改代码即可诊断问题。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.config import get_config

console = Console()
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# 两个独立的 Typer app，分别绑定到两个命令入口
# ------------------------------------------------------------
app_chat = typer.Typer(
    name="sensor-qa",
    help="传感器知识库问答机器人 - 交互模式",
    add_completion=False,
)
app_build = typer.Typer(
    name="sensor-qa-build",
    help="传感器知识库问答机器人 - 构建向量库",
    add_completion=False,
)


# ============================================================
# 公共工具函数
# ============================================================
def _setup_logging(debug: bool) -> None:
    """
    初始化日志级别。第三关会把这里替换为完整的结构化日志方案。
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _preflight_check() -> None:
    """
    启动前配置自检。任何硬性缺失直接退出，避免半路崩溃。

    如果不做此检查：
    用户在缺 DEEPSEEK_API_KEY 时启动 sensor-qa，会一路走到 LLM 调用才
    抛出 openai.AuthenticationError，用户根本看不懂到底是网络问题、
    Key 无效、还是根本没配。
    """
    try:
        cfg = get_config()
    except Exception as e:
        console.print(f"[red]❌ 配置加载失败：[/red]{e}")
        console.print(
            "[dim]请检查项目根目录下的 .env 文件是否存在且格式正确。[/dim]"
        )
        sys.exit(2)

    if not cfg.api_key:
        console.print(
            "[red]❌ 缺少 DEEPSEEK_API_KEY[/red]\n"
            "[dim]请在 .env 中添加：DEEPSEEK_API_KEY=sk-xxxxxxxx[/dim]"
        )
        sys.exit(2)

    if cfg.embedding_provider not in ("local", "deepseek"):
        console.print(
            f"[red]❌ 非法的 EMBEDDING_PROVIDER：{cfg.embedding_provider}[/red]\n"
            "[dim]仅支持 'local' 或 'deepseek'。[/dim]"
        )
        sys.exit(2)


# ============================================================
# 命令 1：sensor-qa —— 交互式问答
# ============================================================
@app_chat.callback(invoke_without_command=True)
def chat_main(
    debug: bool = typer.Option(
        False, "--debug", help="开启 DEBUG 级别日志"
    ),
    question: Optional[str] = typer.Option(
        None, "--question", "-q",
        help="单次提问模式：给定问题后输出答案立即退出",
    ),
    no_stream: bool = typer.Option(
        False, "--no-stream", help="关闭流式输出（一次性打印完整答案）"
    ),
) -> None:
    """启动交互式问答，或使用 -q 进行单次提问。"""
    _setup_logging(debug)
    _preflight_check()

    # 延迟导入：避免 --help 时也把重量级依赖全加载一遍
    from src.rag_chat import SensorRAGChat

    try:
        chat = SensorRAGChat()
    except FileNotFoundError as e:
        console.print(f"[red]❌ 向量库未就绪：[/red]{e}")
        console.print(
            "[yellow]💡 请先运行：[/yellow][bold]sensor-qa-build[/bold]"
        )
        sys.exit(3)
    except Exception as e:
        logger.exception("初始化 RAG 失败")
        console.print(f"[red]❌ 初始化失败：{e}[/red]")
        sys.exit(1)

    # --- 单次提问模式 ---
    if question:
        _ask_once(chat, question, stream=not no_stream)
        return

    # --- 交互模式 ---
    _run_interactive(chat, stream=not no_stream)


def _ask_once(chat, question: str, stream: bool) -> None:
    """执行单次提问并打印结果。"""
    try:
        if stream:
            _print_stream(chat, question)
        else:
            result = chat.ask(question)
            _print_result(result)
    except KeyboardInterrupt:
        console.print("\n[yellow]已中断[/yellow]")
        sys.exit(130)
    except Exception as e:
        logger.exception("提问失败")
        console.print(f"[red]❌ 提问失败：{e}[/red]")
        sys.exit(1)


def _run_interactive(chat, stream: bool) -> None:
    """启动交互式 REPL。"""
    console.print(
        Panel.fit(
            "[bold green]🤖 Sensor QA Bot[/bold green]\n"
            "[dim]输入问题回车提问；输入 [cyan]/reset[/cyan] 清空对话历史；"
            "输入 [cyan]/quit[/cyan] 或 Ctrl+D 退出。[/dim]",
            border_style="green",
        )
    )

    while True:
        try:
            question = console.input("\n[bold cyan]❓ [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]再见！[/yellow]")
            return

        if not question:
            continue

        if question in ("/quit", "/exit", ":q"):
            console.print("[yellow]再见！[/yellow]")
            return

        if question == "/reset":
            chat.reset()
            console.print("[dim]（已清空对话历史）[/dim]")
            continue

        try:
            if stream:
                _print_stream(chat, question)
            else:
                result = chat.ask(question)
                _print_result(result)
        except Exception as e:
            logger.exception("提问失败")
            console.print(f"[red]❌ 出错：{e}[/red]")


def _print_stream(chat, question: str) -> None:
    """流式打印答案，最后附加 sources / token 信息。"""
    console.print("\n[bold cyan]📝 回答：[/bold cyan]")

    final_result = None
    for event in chat.ask_stream(question):
        if event.done:
            final_result = event.result
            break
        console.print(event.delta, end="", soft_wrap=True)
    console.print()  # 换行

    if final_result:
        _print_footer(final_result)


def _print_result(result) -> None:
    """非流式：一次性打印。"""
    console.print("\n[bold cyan]📝 回答：[/bold cyan]")
    console.print(Markdown(result.content))
    _print_footer(result)


def _print_footer(result) -> None:
    """打印来源与 token 用量。"""
    if result.sources:
        console.print(
            f"\n[dim]📎 来源：{', '.join(result.sources)}[/dim]"
        )
    u = result.usage
    console.print(
        f"[dim]🔢 tokens: prompt={u.prompt_tokens}, "
        f"completion={u.completion_tokens}, "
        f"cost≈¥{u.estimated_cost_cny:.6f}[/dim]"
    )


# ============================================================
# 命令 2：sensor-qa-build —— 构建向量库
# ============================================================
@app_build.callback(invoke_without_command=True)
def build_main(
    source_dir: Path = typer.Option(
        Path("knowledge_base"),
        "--source", "-s",
        help="知识库源文档目录（支持 .md / .txt）",
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="强制重建（覆盖已有向量库）",
    ),
    debug: bool = typer.Option(
        False, "--debug", help="开启 DEBUG 级别日志"
    ),
) -> None:
    """构建或更新本地向量知识库。"""
    _setup_logging(debug)
    _preflight_check()

    if not source_dir.exists():
        console.print(
            f"[red]❌ 源文档目录不存在：{source_dir}[/red]\n"
            f"[dim]请创建目录并放入 .md / .txt 格式的知识库文件。[/dim]"
        )
        sys.exit(2)

    # 延迟导入
    try:
        from src.build_vectorstore import build_vectorstore
    except ImportError:
        console.print(
            "[red]❌ 未找到 src/build_vectorstore.py[/red]\n"
            "[dim]请确认项目已包含向量库构建脚本。[/dim]"
        )
        sys.exit(1)

    try:
        console.print(
            f"[bold]📦 正在从 [cyan]{source_dir}[/cyan] 构建向量库...[/bold]"
        )
        build_vectorstore(source_dir=source_dir, force=force)
        console.print("[bold green]✅ 向量库构建完成[/bold green]")
    except FileExistsError:
        console.print(
            "[yellow]⚠ 向量库已存在。[/yellow] "
            "如需重建请加 [cyan]--force[/cyan]。"
        )
        sys.exit(1)
    except Exception as e:
        logger.exception("构建向量库失败")
        console.print(f"[red]❌ 构建失败：{e}[/red]")
        sys.exit(1)


# ============================================================
# 便于本地 `python -m src.cli` 调试
# ============================================================
if __name__ == "__main__":
    app_chat()