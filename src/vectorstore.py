"""
向量库构建与加载（FAISS）。
"""

from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_config
from src.embeddings import get_embeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"


def _load_documents():
    """
    读取 knowledge 下所有 .txt / .md，并切分成 chunk。
    """
    cfg = get_config()

    if not KNOWLEDGE_DIR.exists():
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
        raise RuntimeError(
            f"知识库为空：{KNOWLEDGE_DIR}\n"
            "请先运行：python scripts\\create_sample_knowledge.py"
        )

    raw_docs = []
    for f in files:
        loader = TextLoader(str(f), encoding="utf-8")
        raw_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", "。", "；", " ", ""],
    )
    return splitter.split_documents(raw_docs)


def build_vectorstore() -> FAISS:
    """
    构建向量库并保存到本地。
    """
    cfg = get_config()
    docs = _load_documents()

    print(f"共加载 {len(docs)} 个文本块，开始生成向量...")

    embeddings = get_embeddings()
    store = FAISS.from_documents(docs, embeddings)

    save_dir = PROJECT_ROOT / cfg.vectorstore_dir
    store.save_local(str(save_dir))

    print(f"向量库已保存到：{save_dir}")
    return store


def load_vectorstore() -> FAISS:
    """
    加载已保存的向量库。
    """
    cfg = get_config()
    save_dir = PROJECT_ROOT / cfg.vectorstore_dir

    if not (save_dir / "index.faiss").exists():
        raise RuntimeError(
            "还没有构建向量库。请先运行：python build_index.py"
        )

    embeddings = get_embeddings()
    return FAISS.load_local(
        str(save_dir),
        embeddings,
        # allow_dangerous_deserialization=True 会用 pickle 反序列化。
        # 风险仅在于加载“别人给的、来源不明的”向量库文件。
        # 这里加载的是本程序 build_index.py 亲自构建、保存在本地的文件，
        # 来源可信、内容自控，因此开启是安全的。
        allow_dangerous_deserialization=True,
    )