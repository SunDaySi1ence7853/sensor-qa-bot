#!/usr/bin/env python
"""
传感器知识库问答机器人（命令行交互版）。

运行：python main.py
退出：输入 exit / quit / q
"""

import sys

from src.config import require_api_key
from src.logging_config import setup_logging
from src.rag_chat import SensorRAGChat


def main():
    import argparse

    parser = argparse.ArgumentParser(description="传感器知识库问答机器人")
    parser.add_argument("--debug", action="store_true", help="开启 DEBUG 日志")
    args = parser.parse_args()

    # 初始化日志（应用入口调用一次）
    setup_logging(debug=args.debug)

    print("=" * 60)
    print("传感器知识库问答机器人 v0.3")
    print("=" * 60)
    print("提示：输入 'exit' / 'quit' / 'q' 退出，'reset' 清空历史\n")

    try:
        require_api_key()
    except RuntimeError as e:
        print(f"\n❌ {e}\n", file=sys.stderr)
        sys.exit(1)

    try:
        chat = SensorRAGChat()
    except Exception as e:
        print(f"\n❌ 初始化失败：{e}\n", file=sys.stderr)
        sys.exit(1)

    while True:
        try:
            user_input = input("\n🤔 你的问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n再见！")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print("\n再见！")
            break

        if user_input.lower() == "reset":
            chat.reset()
            print("✅ 对话历史已清空")
            continue

        try:
            print("\n🤖 回答：", end="", flush=True)

            for event in chat.ask_stream(user_input):
                if not event.done:
                    print(event.delta, end="", flush=True)
                else:
                    result = event.result
                    print("\n")
                    print(
                        f"💰 本次消耗：{result.usage.total_tokens} tokens "
                        f"（¥{result.usage.estimated_cost_cny:.6f}）"
                    )
                    if result.sources:
                        print(f"📚 知识来源：{', '.join(result.sources)}")

        except KeyboardInterrupt:
            print("\n\n⚠️ 回答被中断")
            continue
        except Exception as e:
            print(f"\n\n❌ 错误：{e}\n", file=sys.stderr)
            continue


if __name__ == "__main__":
    main()