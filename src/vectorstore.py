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
        logger.info("已加载 %d 个文件，跳过 %d 个", len(files) - len(skipped), len(skipped))
    else:
        logger.info("已加载全部 %d 个文件", len(files))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )

    docs = splitter.split_documents(raw_docs)
    logger.info("文档切分完成：%d chunks", len(docs))
    logger.debug(
        "切分参数 | chunk_size=%d | chunk_overlap=%d",
        cfg.chunk_size,
        cfg.chunk_overlap,
    )

    return docs


def build_vectorstore() -> FAISS:
    """
    从 knowledge/ 目录构建向量库。
    供 build_index.py 使用。
    """
    logger.info("开始构建向量库...")
    docs = _load_documents()
    embeddings = get_embeddings()

    logger.info("正在生成 embeddings（首次运行或模型未缓存时可能较慢）...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    logger.info("向量库构建完成")

    return vectorstore


def load_vectorstore() -> FAISS:
    """
    加载已保存的向量库。
    供 main.py 使用。
    """
    cfg = get_config()
    vectorstore_path = PROJECT_ROOT / cfg.vectorstore_dir

    index_file = vectorstore_path / "index.faiss"
    if not index_file.exists():
        logger.error("向量库不存在：%s", vectorstore_path)
        raise RuntimeError(
            f"向量库不存在：{vectorstore_path}\n"
            "请先运行：python build_index.py"
        )

    logger.info("正在加载向量库：%s", vectorstore_path)

    embeddings = get_embeddings()

    try:
        vectorstore = FAISS.load_local(
            str(vectorstore_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("向量库加载完成")
        return vectorstore
    except Exception as e:
        logger.exception("向量库加载失败：%s", vectorstore_path)
        raise RuntimeError(
            f"向量库加载失败：{vectorstore_path}\n"
            f"原因：{e}\n"
            "建议重新构建：python build_index.py"
        )