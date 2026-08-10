# 传感器问答助手 v0.1

这是一个基于 Python + DeepSeek API 的命令行传感器技术问答助手。

## 功能

- 支持命令行多轮问答
- 支持本地传感器知识库
- 支持 DeepSeek API
- 不知道的问题会回答“知识库中没有相关信息”

## 项目结构

```text
sensor-qa-bot/
├── knowledge/          # 传感器知识文档
├── scripts/            # 辅助脚本
├── src/
│   ├── api_client.py   # DeepSeek API 调用封装
│   ├── chat.py         # 对话逻辑
│   └── prompt.py       # 提示词和知识库加载
├── main.py             # 程序入口
├── requirements.txt
└── README.md
```

## 安装

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## 配置 API Key

在项目根目录创建 `.env`：

```env
DEEPSEEK_API_KEY=你的DeepSeek_API_Key
DEEPSEEK_MODEL=deepseek-v4-flash
```

## 生成示例知识库

```powershell
python scripts\create_sample_knowledge.py
```

## 运行

```powershell
python main.py
```

## 命令

- `exit`：退出程序
- `clear`：清空历史并重新加载知识库