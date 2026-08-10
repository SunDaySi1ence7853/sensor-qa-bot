"""
传感器问答助手 v0.1
命令行入口。
"""

from src.chat import SensorChat


def print_banner() -> None:
    print("=" * 50)
    print("传感器问答助手 v0.1")
    print("DeepSeek API + 本地传感器知识库")
    print("=" * 50)
    print("使用说明：")
    print("1. 直接输入传感器相关问题。")
    print("2. 输入 clear 重置对话并重新加载知识库。")
    print("3. 输入 exit 退出程序。")
    print("=" * 50)


def main() -> None:
    chat = SensorChat()
    print_banner()

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
            print("助手：正在思考...")
            answer = chat.ask(user_input)
            print(f"\n助手：{answer}")
        except Exception as e:
            print(f"\n助手：调用失败，错误信息：{e}")


if __name__ == "__main__":
    main()