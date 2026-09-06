# src/build_vectorstore.py
"""
向量知识库构建脚本。

CLI 调用入口：sensor-qa-build
也可直接运行：python -m src.build_vectorstore
"""

from __future__ import annotations

import shutil
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_config
from src.embeddings import get_embeddings


def build_vectorstore(
    source_dir: Path = Path("knowledge_base"),
    force: bool = False,
) -> None:
    """
    从 source_dir 读取 .md / .txt 文档，构建 FAISS 向量库并保存到本地。

    Args:
        source_dir: 知识库源文档目录
        force:      True 时强制重建（先删除已有向量库）

    Raises:
        FileNotFoundError: source_dir 不存在
        FileExistsError:   向量库已存在且 force=False
    """
    cfg = get_config()
    source_dir = Path(source_dir)
    vectorstore_dir = Path(cfg.vectorstore_dir)

    if not source_dir.exists():
        raise FileNotFoundError(f"源文档目录不存在：{source_dir}")

    if vectorstore_dir.exists():
        if not force:
            raise FileExistsError(
                f"向量库已存在：{vectorstore_dir}，使用 --force 强制重建"
            )
        shutil.rmtree(vectorstore_dir)

    # 加载文档
    # 注意：Python 原生 glob 不支持 {md,txt} 花括号语法，需分别加载后合并
    docs = []
    for pattern in ["**/*.md", "**/*.txt"]:
        loader = DirectoryLoader(
            str(source_dir),
            glob=pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
        )
        docs.extend(loader.load())

    if not docs:
        raise ValueError(f"目录 {source_dir} 中未找到任何 .md / .txt 文件")

    # 切块
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    # 构建向量库
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 保存
    vectorstore_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(vectorstore_dir))


if __name__ == "__main__":
    build_vectorstore()