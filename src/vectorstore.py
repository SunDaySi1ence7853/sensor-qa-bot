"""
向量库加载与构建。

安全说明：
FAISS 使用 pickle 序列化，加载时必须显式开启
allow_dangerous_deserialization。本模块仅加载由本项目
build 命令生成、存储在受控本地路径下的向量库文件。
"""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pypdf import PdfReader  # 新增：引入 pypdf

from src.config import get_config
from src.embeddings import get_embeddings
from src.logging_config import get_logger

logger = get_logger(__name__)

# 项目根目录：src/vectorstore.py -> src/ -> 项目根
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 知识库目录（放待索引的原始 txt/md/pdf 文件）
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge_base"

# 允许作为知识源的文件后缀（新增 .pdf）
SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


def _extract_pdf_text(path: Path) -> str:
    """使用 pypdf 提取 PDF 文本。"""
    reader = PdfReader(str(path))
    text_parts = []
    for page in reader.pages:
        # extract_text() 在遇到扫描版/图片版 PDF 时会返回空字符串或 None
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    return "\n".join(text_parts).strip()


def _load_documents() -> list[Document]:
    """
    扫描 KNOWLEDGE_DIR，读取所有 .txt/.md/.pdf 文件为 Document 列表。

    行为约定（被测试用例锁定）：
      - 目录不存在：抛 RuntimeError("知识库目录不存在: ...")
      - 目录为空（无支持的文件）：抛 RuntimeError("知识库为空: ...")
      - 单个文件读失败（如非 UTF-8、损坏的 PDF）：warning 日志，跳过
      - 扫描版 PDF 提取为空：warning 日志，跳过（防御性设计）
      - 所有文件都读失败：抛 RuntimeError("所有文件都无法读取")
      - 不支持的文件后缀：静默忽略
    """
    if not KNOWLEDGE_DIR.exists():
        raise RuntimeError(
            f"知识库目录不存在: {KNOWLEDGE_DIR}\n"
            f"请在项目根目录下创建 knowledge/ 文件夹，并放入 .txt / .md / .pdf 文件。"
        )

    candidates = [
        p for p in KNOWLEDGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    ]

    if not candidates:
        raise RuntimeError(
            f"知识库为空: {KNOWLEDGE_DIR}\n"
            f"请在其中放入至少一个 .txt、.md 或 .pdf 文件。"
        )

    docs: list[Document] = []
    failed: list[str] = []

    for path in candidates:
        try:
            if path.suffix.lower() == ".pdf":
                text = _extract_pdf_text(path)
                if not text:
                    # 防御性设计：扫描版 PDF 没有文字层，提取为空，跳过并警告
                    logger.warning("跳过无法提取文本的 PDF（可能是扫描版或纯图片）：%s", path.name)
                    failed.append(path.name)
                    continue
            else:
                text = path.read_text(encoding="utf-8")
                
        except (UnicodeDecodeError, OSError, Exception) as e:
            logger.warning("跳过无法读取的文件 %s：%s", path.name, e)
            failed.append(path.name)
            continue

        if not text.strip():
            logger.warning("跳过空文件 %s", path.name)
            continue

        docs.append(
            Document(
                page_content=text,
                metadata={"source": str(path)},
            )
        )

    if not docs:
        raise RuntimeError(
            f"所有文件都无法读取（共 {len(failed)} 个）：{failed}\n"
            f"请检查文件编码或 PDF 是否包含可提取的文字层。"
        )

    return docs


def load_vectorstore() -> FAISS:
    """
    加载本地 FAISS 向量库。

    安全说明（重要）：
    allow_dangerous_deserialization=True 会执行 pickle 反序列化。
    仅加载本项目 build 命令生成的向量库文件。
    """
    cfg = get_config()
    vectorstore_dir = PROJECT_ROOT / cfg.vectorstore_dir

    if not vectorstore_dir.exists():
        raise RuntimeError(
            f"向量库不存在: {vectorstore_dir}\n"
            f"请先运行 `python -m src.build_vectorstore` 构建知识库。"
        )

    index_file = vectorstore_dir / "index.faiss"
    if not index_file.exists():
        raise RuntimeError(
            f"向量库不存在或不完整，缺少 index.faiss: {vectorstore_dir}\n"
            f"请重新运行 `python -m src.build_vectorstore` 构建知识库。"
        )

    embeddings = get_embeddings()

    try:
        return FAISS.load_local(
            str(vectorstore_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception as e:
        raise RuntimeError(
            f"向量库加载失败: {vectorstore_dir}\n"
            f"文件可能已损坏，请重新运行 `python -m src.build_vectorstore`。\n"
            f"原始错误：{e}"
        ) from e