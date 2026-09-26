# Sensor QA Bot 🤖

> 基于 RAG（检索增强生成）的**传感器知识库中文问答机器人**  
> DeepSeek LLM + LangChain + FAISS + HuggingFace Embeddings

---

## ✨ 功能特性

- 📚 **本地知识库检索**：基于 FAISS 向量库，纯本地运行、隐私可控
- 🧠 **RAG 问答**：结合检索到的上下文由 DeepSeek 生成中文答案
- 💬 **多轮对话**：自动裁剪历史，防止 token 爆炸
- 🌊 **流式输出**：边生成边打印，交互体验丝滑
- 🔢 **Token 统计**：实时显示 prompt / completion / 估算成本（人民币）
- 🛡 **防御性设计**：base_url 智能拼接、usage_metadata 缺失降级、配置自检

---

## 📦 安装

### 方式一：pip 直接安装（推荐）

```bash
pip install git+https://github.com/SunDaySi1ence7853/sensor-qa-bot.git
```

### 方式二：本地克隆开发安装

```bash
git clone https://github.com/SunDaySi1ence7853/sensor-qa-bot.git
cd sensor-qa-bot
pip install -e .

# 若要参与开发，安装可选依赖：
pip install -e ".[dev]"
```

安装完成后，命令行会新增两个可执行命令：

| 命令 | 作用 |
|------|------|
| `sensor-qa` | 启动交互式问答 |
| `sensor-qa-build` | 构建/重建向量知识库 |

---

## ⚙️ 配置

在项目根目录创建 `.env` 文件：

```ini
# ---------- DeepSeek LLM ----------
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_INPUT_PRICE_PER_1M=1.00
DEEPSEEK_OUTPUT_PRICE_PER_1M=2.00

# ---------- Embedding ----------
# local: 使用本地 HuggingFace 模型（默认，无需联网）
# deepseek: 使用 DeepSeek 提供的 embedding API
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=shibing624/text2vec-base-chinese
# DEEPSEEK_EMBEDDING_MODEL=deepseek-embedding

# ---------- 检索与对话 ----------
RETRIEVE_TOP_K=3
CHUNK_SIZE=500
CHUNK_OVERLAP=80
VECTORSTORE_DIR=vectorstore
MEMORY_TURNS=6
```

> 💡 `DEEPSEEK_BASE_URL` 无论填 `https://api.deepseek.com` 还是 `https://api.deepseek.com/v1`，都会被安全地规范化，不会出现 `/v1/v1` 的重复路径。

---

## 🚀 快速开始

### 1. 准备知识库

将 `.md` / `.txt` 格式的传感器文档放入 `knowledge_base/` 目录：

```
knowledge_base/
├── dht22.md
├── bmp280.md
└── mpu6050.md
```

### 2. 构建向量库

```bash
sensor-qa-build
```

若已存在向量库，需要强制重建：

```bash
sensor-qa-build --force
```

自定义源文档目录：

```bash
sensor-qa-build --source ./my_docs
```

### 3. 开始提问

```bash
# 交互模式
sensor-qa

# 单次提问模式
sensor-qa -q "DHT22 的工作电压是多少？"

# 关闭流式输出
sensor-qa --no-stream

# 打开 debug 日志
sensor-qa --debug
```

### 4. 交互示例

```
🤖 Sensor QA Bot
输入问题回车提问；输入 /reset 清空对话历史；输入 /quit 退出。

❓ DHT22 和 DHT11 有什么区别？

📝 回答：
DHT22 相比 DHT11 精度更高（温度 ±0.5°C vs ±2°C），
测量范围更广（-40~80°C vs 0~50°C），但采样周期更长（2s vs 1s）……

📎 来源：dht22.md, dht11.md
🔢 tokens: prompt=412, completion=98, cost≈¥0.000608
```

---
## 系统架构

mermaid
flowchart TB
    subgraph L1[数据源层 src/datasource.py]
        ESP32[ESP32 + 传感器<br/>A线: 真实串口] --> SS[SerialSource]
        FS[FakeSerial<br/>B线: 降级兜底] -.-> SS
        SIM[SimulatedSource<br/>正弦+噪声/相位回溯]
    end
    subgraph L2[采样层 src/tools/sensor_tools.py]
        HUB[_SensorHub 单例<br/>路由 + 串口锁]
        SAMPLER[采样线程 daemon<br/>每30s sample_once]
        BUF[(history buffer<br/>maxlen 2880 = 24h)]
        SS --> HUB
        SIM --> HUB
        SAMPLER --> HUB
        SAMPLER --> BUF
    end
    subgraph L3[Agent 层 LangGraph + DeepSeek]
        TOOLS[get_sensor_data / query_history<br/>check_threshold / search_manual / generate_report]
        TOOLS --> HUB
        TOOLS --> BUF
    end
    subgraph L4[Web 层 Streamlit]
        CHAT[对话区 ReAct 过程] --> TOOLS
        PANEL[监控面板] --> TOOLS
    end
对账关系：采样间隔 30s × buffer 容量 2880 = 86400s = 24h(由 `TestRetentionContract` 锁定)。


## 🌐 Web 界面使用说明

本项目提供基于 Streamlit 的 Web 交互界面，方便评委与用户直接在浏览器中体验 RAG 问答。

### 1. 安装依赖

确保已安装最新依赖（包含 Streamlit）：

```bash
pip install -r requirements.txt
```

---

## 🐍 Python API

```python
from src.rag_chat import SensorRAGChat

chat = SensorRAGChat()

# 非流式
result = chat.ask("BMP280 的量程是多少？")
print(result.content)
print(result.sources)
print(result.usage.estimated_cost_cny)

# 流式
for event in chat.ask_stream("请对比 DHT22 和 SHT30"):
    if event.done:
        print("\n[来源]", event.result.sources)
    else:
        print(event.delta, end="", flush=True)
```

---

## 🗂 项目结构

```
sensor-qa-bot/
├── pyproject.toml         # 项目配置 & CLI 入口
├── README.md
├── .env                   # 环境变量（用户自建，勿提交 git）
├── knowledge_base/        # 知识库源文档
├── vectorstore/           # 生成的 FAISS 向量库
├── logs/                  # 日志输出目录
├── src/
│   ├── __init__.py
│   ├── config.py          # 配置加载 & 校验
│   ├── embeddings.py      # Embedding 工厂（含 base_url 规范化）
│   ├── vectorstore.py     # 向量库加载
│   ├── build_vectorstore.py  # 向量库构建脚本
│   ├── prompt.py          # Prompt 模板
│   ├── llm.py             # LLM 工厂
│   ├── rag_chat.py        # RAG 对话核心
│   └── cli.py             # CLI 入口
└── tests/                 # pytest 单元测试
```

---

## 🧪 开发

```bash
# 运行测试
pytest

# 覆盖率报告
pytest --cov=src --cov-report=term-missing

# 代码格式化
black src/ tests/

# 静态检查
ruff check src/ tests/
```

---

## 📊 测试与评估结果 (任务四)

| 指标 | 结果 |
|------|------|
| 评测集规模 | 20 题 |
| 总命中率 | 85% (Top-K=5) |
| 平均检索耗时 | 0.026s |

---

## 🔬 模拟数据合理性说明

本项目的模拟传感器数据生成机制基于真实工业车间场景设计，各类传感器参数设定依据如下：

1. **温度传感器**
   - **量程设定**：15℃ ~ 35℃。车间常规环境温度区间，涵盖昼夜温差与季节交替。
   - **噪声模型**：±0.5℃ 的高斯噪声。模拟传感器自身测量精度与环境微小波动。
   - **周期设定**：24小时为周期的正弦波。符合车间白天生产发热、夜间降温的现实物理规律。

2. **湿度传感器**
   - **量程设定**：30% ~ 70% RH。工业电子车间的标准安全湿度区间。
   - **噪声模型**：±2% 的高斯噪声。模拟通风系统启停造成的局部湿度波动。
   - **周期设定**：与温度反相位的正弦波。温度升高通常伴随相对湿度下降（绝对湿度不变前提下）。

3. **振动传感器**
   - **量程设定**：0 ~ 5g 加速度。覆盖设备正常运转（~1g）及轻微异常（>2g）的监控需求。
   - **噪声模型**：±0.1g 的随机噪声。模拟机械设备运转时的固有震动底噪。
   - **周期设定**：设备启停阶段呈上升趋势，平稳运行时维持基础震动值。

---

## ⚠️ 已知限制

### 知识库覆盖度

当前知识库(`knowledge_base/*.md`)已补齐核心传感器（如 DHT22、BMP280、MPU6050）的工作电压、封装尺寸及认证信息。
若后续新增传感器型号，请确保在对应 md 文件中补充这些字段，并执行以下命令重建向量库：

```bash
sensor-qa-build --force
```

---

## 🛡 防御性设计要点

| 模块 | 措施 | 防御的场景 |
|------|------|-----------|
| `embeddings.py` | `_normalize_base_url()` 规范化 URL | 用户 `BASE_URL` 末尾带 `/v1` 时避免 `/v1/v1` 导致 404 |
| `rag_chat.py` | `_extract_usage()` 降级返回全零 | 模型不返回 `usage_metadata` 时不崩溃 |
| `vectorstore.py` | 反序列化风险注释 | 提醒后续维护者不要接受外部向量库文件（pickle RCE 风险） |
| `cli.py` | `_preflight_check()` | 缺 API Key / 非法配置时立即中文报错 |

---

## 📄 许可证

[MIT](LICENSE) © 2026 F1NeM0t1oN
