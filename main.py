"""
传感器问答助手 v0.2（RAG 版）
命令行入口。

普通输出：python main.py
流式输出：python main.py --stream
"""

import argparse

from src.rag_chat import ChatResult, SensorRAGChat


def print_banner(stream: bool) -> None:
    print("=" * 50)
    print("传感器问答助手 v0.2 (RAG)")
    print("DeepSeek Chat + 本地向量检索")
    print("=" * 50)
    print("1. 直接输入传感器相关问题。")
    print("2. clear 重置对话历史。")
    print("3. exit 退出。")
    print(f"4. 当前模式：{'流式输出' if stream else '普通输出'}")
    print("=" * 50)


def print_result_meta(result: ChatResult) -> None:
    print("\n" + "-" * 50)
    if result.sources:
        print("参考资料：" + "、".join(result.sources))
    print("本次消耗：")
    print(f"  输入 tokens：{result.usage.prompt_tokens}")
    print(f"  输出 tokens：{result.usage.completion_tokens}")
    print(f"  总 tokens：{result.usage.total_tokens}")
    print(f"  估算费用：约 {result.usage.estimated_cost_cny:.6f} 元")
    print("-" * 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="传感器问答助手 v0.2")
    parser.add_argument(
        "--stream", action="store_true", help="启用流式输出"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chat = SensorRAGChat()
    print_banner(stream=args.stream)

    while True:
        user_input = input("\n你：").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("助手：再见！")
            break
        if user_input.lower() == "clear":
            chat.reset()
            print("助手：对话历史已清空。")
            continue

        try:
            if args.stream:
                print("助手：", end="", flush=True)
                final = None
                for event in chat.ask_stream(user_input):
                    if not event.done:
                        print(event.delta, end="", flush=True)
                    else:
                        final = event.result
                print()
                if final:
                    print_result_meta(final)
            else:
                print("助手：正在检索并思考...")
                result = chat.ask(user_input)
                print(f"\n助手：{result.content}")
                print_result_meta(result)
        except Exception as e:
            print(f"\n助手：调用失败，错误信息：{e}")


if __name__ == "__main__":
    main()