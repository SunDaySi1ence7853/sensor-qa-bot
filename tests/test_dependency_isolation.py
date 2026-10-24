"""依赖隔离守卫：纯采样 / 诊断路径不能被 RAG 全家桶焊死。

背景（老板在干净沙箱里实测发现）：
`src/tools/__init__.py` 曾在顶层 `from src.vectorstore import load_vectorstore`，
而 `import src.tools.sensor_tools` 必然先执行包的 `__init__`，于是
sensor_tools → vectorstore → embeddings → langchain_huggingface / faiss 全被拉起来。
新机器不装完整 ML 栈时，pytest 连收集都过不了。

修法是把 vectorstore 的 import 延迟到 search_manual 内部。本文件用子进程守住这条线：
pytest 主进程早就 import 过这些包了，同进程内断言 sys.modules 没有意义。
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 这些包一旦被拉起，就说明"最小安装路径"又断了
_HEAVY = [
    "torch",
    "transformers",
    "sentence_transformers",
    "langchain_huggingface",
    "faiss",
    "langchain_community",
    "langchain_openai",
]

_PROBE = """
import sys
import src.tools.sensor_tools
import src.agents.demo_dynamic
heavy = [name for name in {heavy!r} if name in sys.modules]
print("|".join(heavy))
"""


def _run_probe() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _PROBE.format(heavy=_HEAVY)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )


def test_sampling_path_does_not_import_ml_stack():
    """sampling / agents 只依赖 langchain-core、pydantic、pyserial，不该碰 ML 栈。"""
    proc = _run_probe()
    assert proc.returncode == 0, f"子进程导入失败：\n{proc.stderr}"
    leaked = [name for name in proc.stdout.strip().split("|") if name]
    assert leaked == [], f"纯采样路径拉起了重依赖：{leaked}"
