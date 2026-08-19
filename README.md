# 传感器问答助手 v0.2 (RAG)

基于 Python + LangChain + DeepSeek + FAISS 的传感器技术命令行问答助手。

## 功能
- RAG：本地向量检索，仅把相关资料喂给模型，不再全量塞知识库
- 对话记忆：多轮上下文
- 流式输出：`--stream`
- Token 消耗与费用估算
- Embedding 可切换：本地模型 / DeepSeek 官方接口

## 环境要求
- Python ≥ 3.10（代码使用了 3.10+ 的类型语法）

## 安装
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## 配置
填写根目录 `.env`，至少填 `DEEPSEEK_API_KEY`。

关键配置：
- `EMBEDDING_PROVIDER`：`local`（默认，推荐）或 `deepseek`
- `RETRIEVE_TOP_K`：检索返回的资料块数量
- `DEEPSEEK_INPUT_PRICE_PER_1M` / `DEEPSEEK_OUTPUT_PRICE_PER_1M`：费用估算单价
- `MEMORY_TURNS`：对话记忆保留轮数

## 使用步骤
```powershell
python scripts\create_sample_knowledge.py   # 首次生成知识库（可选）
python build_index.py                        # 构建向量索引
python main.py                               # 普通问答
python main.py --stream                      # 流式问答
```

## 命令
- `exit`：退出
- `clear`：清空对话历史

## 换 embedding 后必须重建索引
修改 `EMBEDDING_PROVIDER` 或 embedding 模型后，务必重新运行 `python build_index.py`。

## 关于 DeepSeek Embedding
DeepSeek 提供了 `deepseek-embedding-v1`（端点 `/v1/embeddings`），
但社区信息存在矛盾，建议先用 curl 验证接口可用后再把 `EMBEDDING_PROVIDER` 设为 `deepseek`。