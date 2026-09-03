"""
向量库构建与加载（FAISS）。
"""

from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_config
from src.embeddings import get_embeddings
from src.logging_config import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"


def _load_documents():
    cfg = get_config()

    if not KNOWLEDGE_DIR.exists():
        logger.error("知识库目录不存在：%s", KNOWLEDGE_DIR)
        raise RuntimeError(
            f"知识库目录不存在：{KNOWLEDGE_DIR}\n"
            "请先运行：python scripts\\create_sample_knowledge.py"
        )

    files = sorted(
        p
        for p in KNOWLEDGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in (".txt", ".md")
    )

    if not files:
        logger.error("知识库为空：%s", KNOWLEDGE_DIR)
        raise RuntimeError(
            f"知识库为空：{KNOWLEDGE_DIR}\n"
            "请先运行：python scripts\\create_sample_knowledge.py"
        )

    logger.info("发现 %d 个候选文件", len(files))

    raw_docs = []
    skipped = []

    for f in files:
        try:
            loader = TextLoader(str(f), encoding="utf-8")
            raw_docs.extend(loader.load())
            logger.debug("已加载文件：%s", f.name)
        except UnicodeDecodeError:
            skipped.append(f.name)
            logger.warning("跳过文件（编码错误，非 UTF-8）：%s", f.name)
        except Exception as e:
            skipped.append(f.name)
            logger.warning("跳过文件（加载失败）：%s | 原因：%s", f.name, e)

    if not raw_docs:
        logger.error("知识库中所有文件都无法读取（共 %d 个）", len(files))
        raise RuntimeError(
            f"知识库中所有文件都无法读取（共 {len(files)} 个文件）。\n"
            f"请检查文件编码是否为 UTF-8。"
        )

    if skipped:
        logger.