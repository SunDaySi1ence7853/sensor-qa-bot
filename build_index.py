#!/usr/bin/env python
"""
构建向量库索引。

运行：python build_index.py
"""

import sys
from pathlib import Path

from src.config import get_config
from src.logging_config import setup_logging
from src.vectorstore import PROJECT_ROOT, build_vectorstore


def main():
    import argparse

    parser = argparse.ArgumentParser(description="构建传感器知识库向量索引")
    parser.add_argument("--debug", action="store_true", help="开启 DEBUG 日志")
    args = parser.parse_args()

    setup_logging(debug=args.debug)

    print("=" * 60)
    print("构建传感器知识库向量索引")
    print("=" * 60)

    try:
        cfg = get_config()
        vectorstore = build_vectorstore()

        save_path = PROJECT_ROOT / cfg.vectorstore_dir
        save_path.mkdir(parents=True, exist_ok=True)

        vectorstore.save_local(str(save_path))

        print(f"\n  ！！向量库已保存到：{save_path}")
        print("\n现在可以运行：python main.py")

    except Exception as e:
        print(f"\n ！！构建失败：{e}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()