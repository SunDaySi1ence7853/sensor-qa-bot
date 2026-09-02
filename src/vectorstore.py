"""
向量库构建与加载（FAISS）。

新增：
1. 跳过非文本文件和编码错误文件，而不是直接崩溃
2. 检查切分后的文档数量，防止空知识库进入构建流程
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
    
    新增容错：跳过编码错误文件，警告用户。
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
    skipped = []
    
    for f in files:
        try:
            loader = TextLoader(str(f), encoding="utf-8")
            raw_docs.extend(loader.load())
        except UnicodeDecodeError:
            skipped.append(f.name)
            print(f"⚠️  跳过文件（编码错误）：{f.name}")
        except Exception as e:
            skipped.append(f.name)
            print(f"⚠️  跳过文件（加载失败）：{f.name}，原因：{e}")

    if not raw_docs:
        raise RuntimeError(
            f"知识库中所有文件都无法读取（共 {len(files)} 个文件）。\n"
            f"请检查文件编码是否为 UTF-8，或文件内容是否为空。"
        )

    if skipped:
        print(f"ℹ️  成功加载 {len(files) - len(skipped)}/{len(files)} 个文件")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", "。", "；", " ", ""],
    )
    
    chunks = splitter.split_documents(raw_docs)
    
    if not chunks:
        raise RuntimeError(
            "文档切分后为空，可能所有文件都是空白文件。\n"
            "请检查 knowledge 目录下的文件内容。"
        )
    
    return chunks


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
    save_dir.mkdir(parents=True, exist_ok=True)
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
            f"向量库不存在：{save_dir}\n"
            "请先运行：python build_index.py"
        )

    embeddings = get_embeddings()
    
    try:
        return FAISS.load_local(
            str(save_dir),
            embeddings,
            # allow_dangerous_deserialization=True 会用 pickle 反序列化。
            # 风险仅在于加载"别人给的、来源不明的"向量库文件。
            # 这里加载的是本程序 build_index.py 亲自构建、保存在本地的文件，
            # 来源可信、内容自控，因此开启是安全的。
            allow_dangerous_deserialization=True,
        )
    except Exception as e:
        raise RuntimeError(
            f"向量库加载失败：{e}\n"
            f"可能原因：\n"
            f"1. FAISS 索引文件损坏，请删除 {save_dir} 后重新运行 python build_index.py\n"
            f"2. 切换了 embedding 模型但未重建索引\n"
            f"3. FAISS 版本不兼容"
        )