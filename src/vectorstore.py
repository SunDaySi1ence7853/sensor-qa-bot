"""
向量库加载与构建。

安全说明：
FAISS 使用 pickle 序列化，加载时必须显式开启
allow_dangerous_deserialization。本模块仅加载由本项目
build 命令生成、存储在受控本地路径下的向量库文件。
"""

from pathlib import Path

from langchain_community.vectorstores import FAISS

from src.config import get_config
from src.embeddings import get_embeddings


def load_vectorstore() -> FAISS:
    """
    加载本地 FAISS 向量库。

    安全说明（重要）：
    ------------------------------------------------------------------
    allow_dangerous_deserialization=True 会对 FAISS 存储的 pickle 文件
    执行反序列化。pickle 文件可以在加载时执行任意 Python 代码，
    加载来源不可信的向量库文件等同于远程代码执行（RCE）。

    当前项目的安全前提：
      1. 向量库只由本项目的 build 命令在本地生成；
      2. 向量库目录（VECTORSTORE_DIR）只有开发者/部署者本人可写；
      3. 不接受任何来自用户上传、网络下载的向量库文件。

    如果未来需要支持从外部加载向量库，必须先：
      - 校验文件签名/哈希；或
      - 改用安全的序列化格式（例如把索引和 docstore 分开存 JSON+bin）。

    如果不加这段注释：
    团队成员在扩展功能时（比如允许用户"导入知识库"），可能不知道
    这行参数背后的风险，直接把外部文件塞进 load_local，
    等价于给攻击者开了一个 RCE 入口。
    ------------------------------------------------------------------
    """
    cfg = get_config()
    vectorstore_dir = Path(cfg.vectorstore_dir)

    if not vectorstore_dir.exists():
        raise FileNotFoundError(
            f"向量库目录不存在: {vectorstore_dir}\n"
            f"请先运行 `python -m src.build_vectorstore` 构建知识库。"
        )

    index_file = vectorstore_dir / "index.faiss"
    if not index_file.exists():
        raise FileNotFoundError(
            f"向量库文件不完整，缺少 index.faiss: {vectorstore_dir}\n"
            f"请重新运行 `python -m src.build_vectorstore` 构建知识库。"
        )

    embeddings = get_embeddings()

    # 安全说明见上方 docstring：仅加载本项目在本地生成的向量库文件。
    return FAISS.load_local(
        str(vectorstore_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )