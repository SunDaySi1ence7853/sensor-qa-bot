# src/build_vectorstore.py
"""
向量知识库构建脚本。

CLI 调用入口：sensor-qa-build
也可直接运行：python -m src.build_vectorstore [--force]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_config
from src.embeddings import get_embeddings
# 直接复用我们在 vectorstore.py 写好的安全加载逻辑和目录常量
from src.vectorstore import _load_documents, KNOWLEDGE_DIR


def build_vectorstore(force: bool = False) -> None:
    """
    从 KNOWLEDGE_DIR 读取 .md / .txt / .pdf 文档，构建 FAISS 向量库并保存到本地。

    Args:
        force: True 时强制重建（先删除已有向量库）

    Raises:
        RuntimeError: 源目录不存在或无有效文档 (由 _load_documents 抛出)
        FileExistsError: 向量库已存在且 force=False
    """
    cfg = get_config()
    vectorstore_dir = Path(cfg.vectorstore_dir)

    if vectorstore_dir.exists():
        if not force:
            raise FileExistsError(
                f"向量库已存在：{vectorstore_dir}，使用 --force 强制重建"
            )
        print(f"检测到 --force 参数，正在删除旧向量库：{vectorstore_dir}")
        shutil.rmtree(vectorstore_dir)

    # 1. 加载文档（调用 vectorstore.py 中的安全防御性加载逻辑）
    print(f"开始从 {KNOWLEDGE_DIR} 加载文档...")
    docs = _load_documents()
    print(f"成功加载 {len(docs)} 份文档。")

    # 2. 切块
    print("正在进行文本切块...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    print(f"共生成 {len(chunks)} 个文本块。")

    # 3. 构建向量库
    print("开始生成向量并构建 FAISS 索引（这可能需要一点时间）...")
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 4. 保存
    vectorstore_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(vectorstore_dir))
    print(f"✅ 向量库构建成功并保存至：{vectorstore_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建传感器问答知识库向量库")
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="强制重建向量库（覆盖现有）"
    )
    args = parser.parse_args()

    try:
        build_vectorstore(force=args.force)
    except Exception as e:
        print(f"\n❌ 构建失败：{e}", file=sys.stderr)
        sys.exit(1)