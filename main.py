"""
传感器问答助手 v0.1
命令行入口。

支持：
1. 普通输出：
   python main.py

2. 流式输出：
   python main.py --stream
"""

import argparse

from src.api_client import TokenUsage
from src.chat import SensorChat


def print_banner(stream: bool) -> None:
    print("=" * 50)
    print("传感器问答助手 v0.1")
    print("DeepSeek API + 本地传感器知识库")
    print("=" * 50)
    print("使用说明：")
    print("1. 直接输入传感器相关问题。")
    print("2. 输入 clear 重置对话并重新加载知识库。")
    print("3. 输入 exit 退出程序。")
    print(f"4. 当前输出模式：{'流式输出' if stream else '普通输出'}")
    print("=" * 50)


def print_usage(usage: TokenUsage) -> None:
    """
    打印 token 使用量和费用估算。
    """

    print("\n" + "-" * 50)
    print("本次消耗统计：")
    print(f"输入 tokens：{usage.prompt_tokens}")
    print(f"输出 tokens：{usage.completion_tokens}")
    print(f"总 tokens：{usage.total_tokens}")
    print(f"估算费用：约 {usage.estimated_cost_cny:.6f} 元")
    print("-" * 50)


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    """

    parser = argparse.ArgumentParser(
        description="传感器问答助手 v0.1"
    )

    parser.add_argument(
        "--stream",
        action="store_true",
        help="启用流式输出模式，让回答逐字显示。",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    chat = SensorChat()
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
            print("助手：会话已重置，知识库已重新加载。")
            continue

        try:
            if args.stream:
                print("助手：", end="", flush=True)

                final_result = None

                for event in chat.ask_stream(user_input):
                    if not event.done:
                        print(event.delta, end="", flush=True)
                    else:
                        final_result = event.result

                print()

                if final_result is not None:
                    print_usage(final_result.usage)

            else:
                print("助手：正在思考...")
                result = chat.ask(user_input)
                print(f"\n助手：{result.content}")
                print_usage(result.usage)

        except Exception as e:
            print(f"\n助手：调用失败，错误信息：{e}")


if __name__ == "__main__":
    main()