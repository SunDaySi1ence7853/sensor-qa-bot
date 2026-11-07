# 从零演练（方案 A）：干净机器上能不能跑起来

> 目的：证明"换台机器、没有 .env、没有向量库、没有 venv"时，仓库照样能装、能跑、能演示。
> 这份文件里每一步都能直接粘到 PowerShell 执行；末尾是本机的实测结果。

## 0. 前提：确认从"干净状态"开始

```powershell
$dst = "$env:TEMP\sqb-fresh"
if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
```

## 1. 全新 clone（不带任何私有文件）

```powershell
git clone https://github.com/SunDaySi1ence7853/sensor-qa-bot.git $dst
Set-Location $dst

# 确认三个私有物都不在：应输出 False False False
"env=$(Test-Path .env)  vectorstore=$(Test-Path vectorstore)  venv=$(Test-Path venv)"
```

> `.env`、`vectorstore/`、`venv/` 都在 `.gitignore` 里，所以 clone 下来天然没有。
> 如果你想更狠一点，可以先把本地的这三样改名，再对本地目录跑同样的流程。

## 2. 建 venv 并装依赖（按 README）

```powershell
py -3.11 -m venv venv                     # 没有 py 就用 python -m venv venv
.\venv\Scripts\python.exe -m pip install -U pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt   # pytest 属于开发依赖，必须单独装
```

> ⚠️ 只装 `requirements.txt` 跑不了 pytest —— 这是本仓库已知的、README 里写明的分工（运行依赖 / 开发依赖）。
>
> 只想跑演示、不想下 torch 那一套（约 1–2 GB）时，用最小依赖：
> `pip install -r requirements-minimal.txt pytest`，代价是全量 pytest 里 RAG 相关文件收集不了。

## 3. 三道闸门

```powershell
.\check.ps1
```

脚本会优先用 `.\venv\Scripts\python.exe`，所以**不需要先激活 venv**。

期望：

```
使用解释器：...\sqb-fresh\venv\Scripts\python.exe
===== 闸门 1/3 pytest 全量 =====  223 passed
===== 闸门 2/3 动态路由 demo =====  路径C ... failed / exit 0
===== 闸门 3/3 场景验收 =====     场景通过 6/6
✅ 三道闸门全部通过，可以 push
```

## 4. 演示页

```powershell
.\venv\Scripts\python.exe -m streamlit run streamlit_diagnosis.py --server.headless false
```

浏览器打开 http://localhost:8501 ，左侧「演示：三条构造路径」→ 切 A/B/C；再切「实测」，点「启动后台采样」→「运行诊断」。

> 演示页不需要 API Key、不需要向量库，断网也能跑。

## 5. 验收标准

| 检查项 | 通过标准 |
|--------|----------|
| clone 后私有物 | `.env` / `vectorstore` / `venv` 均为 False |
| 安装 | `pip install -r requirements.txt` 退出码 0 |
| 闸门 1 | 223 passed，0 failed |
| 闸门 2 | exit code 0，输出含三条路径（A/B/C） |
| 闸门 3 | 场景通过 6/6 |
| 演示页 | 三条路径可切换、指标与终端一致；实测模式能跑出 complete 或 failed |

## 常见坑

| 现象 | 原因 | 处理 |
|---|---|---|
| 退出码 9009 | 终端没激活 venv，`python` 找不到 | 用 `.\venv\Scripts\python.exe`，或直接 `.\check.ps1`（已内置） |
| 输出中文乱码 / `✓` 导致 UnicodeEncodeError | 接管道后 Windows 默认 GBK | 先 `[Console]::OutputEncoding=[Text.Encoding]::UTF8; $env:PYTHONIOENCODING='utf-8'` |
| pip 卡在 torch 下载 | 完整依赖包含 torch | 换镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple`，或改用 `requirements-minimal.txt` |
| 演示页报缺少向量库 | 误开了 `streamlit_app.py` | 演示入口是 `streamlit_diagnosis.py` |

---

# 本机实测结果（2026-11-07）

> 下面这些是真实执行输出，不是预期值。实测对象：commit `c9ec42e`，Windows 11 + Python 3.11.9。

| 步骤 | 实际输出 |
|---|---|
| clone | `exit 0`；`env=False vectorstore=False venv=False`（三个私有物都没有） |
| 安装 | `requirements.txt` exit 0；`requirements-dev.txt` exit 0；共 116 个包 |
| 闸门 1/3 | `223 passed, 1 warning in 7.27s` |
| 闸门 2/3 | exit code **0**，三条路径完整输出（A/B/C，44 行） |
| 闸门 3/3 | `场景通过 6/6` |
| 演示页 | `streamlit_diagnosis.py` → **HTTP 200**（7863 字节） |
| 解释器定位 | `.\check.ps1` 自动使用 `...\sqb-fresh2\venv\Scripts\python.exe`（未激活 venv 也能跑） |
| GitHub Actions | run #1，`windows-latest`，不装 ML 栈，结论 **success**（约 54 秒） |

**这一轮演练抓到的真问题**：`requirements.txt` 里没有 pytest（它属于开发依赖）。第一次按 README 装完直接跑 `.\check.ps1`，第一道闸门就报 `No module named pytest`。已修：

- README 把"运行依赖 / 开发依赖"分开写清，并明确警告只装前者跑不了验收；
- `check.ps1` 增加前置检查：缺 pytest 时输出"请先执行 `pip install -r requirements-dev.txt`"，而不是抛一串 ModuleNotFoundError。

**这次证明的是**：干净机器上 clone → 装 → 三闸门 → 演示页，全链路通。
**这次没有证明的**：真实传感器、真实用户、跨操作系统（只测了 Windows + 一个 Linux 容器思路的 CI）。
