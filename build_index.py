"""
一次性构建向量索引。

用法：
python build_index.py
"""

from src.vectorstore import build_vectorstore


def main() -> None:
    print("=" * 50)
    print("开始构建传感器知识库向量索引")
    print("=" * 50)
    build_vectorstore()
    print("完成。现在可以运行：python main.py")


if __name__ == "__main__":
    main()