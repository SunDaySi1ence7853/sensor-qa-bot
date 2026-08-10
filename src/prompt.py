"""
系统提示词和知识库加载逻辑。
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"


def load_knowledge() -> str:
    """
    读取 knowledge 目录下所有 .txt / .md 文档。
    """

    if not KNOWLEDGE_DIR.exists():
        KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        return ""

    docs = []

    files = sorted(
        [
            p
            for p in KNOWLEDGE_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in [".txt", ".md"]
        ]
    )

    for file_path in files:
        try:
            content = file_path.read_text(encoding="utf-8").strip()
            if content:
                docs.append(
                    f"\n\n===== 文档：{file_path.name} =====\n{content}"
                )
        except UnicodeDecodeError:
            print(f"警告：{file_path.name} 不是 UTF-8 编码，已跳过。")
        except Exception as e:
            print(f"警告：读取 {file_path.name} 失败：{e}")

    return "\n".join(docs)


def build_system_prompt() -> str:
    """
    构建系统提示词。
    """

    knowledge = load_knowledge()

    if not knowledge:
        knowledge = "当前知识库为空。"

    return f"""
你是“传感器问答助手 v0.1”，你的任务是回答传感器技术问题。

你必须严格遵守以下规则：

1. 只能基于【知识库内容】回答问题。
2. 如果知识库中没有明确信息，必须回答：
   “抱歉，我的知识库中没有相关信息，无法确定。”
3. 不允许凭常识、经验或想象补充知识库没有的参数。
4. 如果用户追问“刚才那个”“它的精度”“这种传感器”等，要结合对话历史理解上下文。
5. 回答要简洁、准确，适合初学者阅读。
6. 如果问题涉及选型，优先从量程、精度、输出信号、环境适应性、应用场景几个方面回答。
7. 默认使用中文回答。

【知识库内容】
{knowledge}
"""