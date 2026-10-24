# langchain-community 迁移评估

> 结论先说：**现在不迁。** 不是懒得迁，是查完之后没有可迁的落点；同时把风险面收敛到一处，并写清什么条件下必须迁。

## 一段话版本（可直接粘给老板）

> `langchain-community` 在 0.4.2 里已经自己打上 sunset 警告（指向 langchain-community#674），但我核了一遍迁移落点：`langchain-classic` 1.0.8 里确实有 `FAISS`，可它是从 community 再导出的转发，导进去反而提示"请改回 community"；PyPI 上的 `langchain-faiss` 0.1.1 是第三方包（作者 zbmain），wheel 只有 1 KB、`__init__.py` 是空文件、没有任何依赖声明，不是官方集成，不能用。本仓库对 community 的依赖只有 2 处 import、4 个方法（`from_documents` / `save_local` / `load_local` / `similarity_search`），所以我选择**不迁，改为收口**：把 FAISS 的引用集中在 `src/vectorstore.py` 一层，版本钉住并在 requirements 里显式声明；一旦 community 停止发版、或我们升级到不兼容的 langchain 主版本，就按"用 faiss-cpu 直接写 60 行薄封装"的路线落地——那时代码只需要改这一个文件，不动调用方。

## 支撑事实（可复现）

| 项 | 事实 | 怎么核实 |
|---|---|---|
| 依赖面 | 全仓库只有 2 处 `langchain_community`：`src/vectorstore.py:12`、`src/build_vectorstore.py:16` | `Get-ChildItem src -Recurse -Filter *.py \| Select-String langchain_community` |
| API 面 | 只用到 4 个方法：`FAISS.from_documents` / `save_local` / `load_local` / `similarity_search` | 同上 grep `FAISS\.` |
| 现状版本 | langchain 1.4.0、langchain-community 0.4.2、langchain-core 1.6.1、faiss-cpu 1.15.0 | `pip show` |
| 上游态度 | community 导入即告警："being sunset and is no longer actively maintained"，指向 langchain-community#674 | 跑任意测试即可看到 |
| 候选 A | `langchain-classic` 1.0.8 的 `FAISS` 是 community 的转发，导入时反而提示改回 community | `python -c "from langchain_classic.vectorstores import FAISS"` |
| 候选 B | PyPI `langchain-faiss` 0.1.1 为第三方包：wheel 1 KB、`__init__.py` 0 字节、METADATA 无依赖、作者 zbmain | `pip download langchain-faiss==0.1.1 --no-deps` 后解开 wheel |

## 为什么不选另外两条路

- **迁到 langchain-classic**：它只是过渡期的转发层，FAISS 实体仍在 community 里。迁过去等于多一个包、少零个依赖。
- **迁到 langchain-faiss**：不是官方包。空实现，装上之后 `import langchain_faiss` 能用但没有 `FAISS` 类，属于"包名对、内容空"。
- **换向量库（Chroma / Qdrant）**：当前是本地离线、单机、几千条文档的量级，换库只增加部署面和不确定性，收益为零。

## 触发迁移的条件（写死在这里，避免下次再讨论一遍）

满足任一条就动手：

1. `langchain-community` 连续两个 langchain 主版本不再发版，或安装时报错；
2. 升级 langchain 主版本后 `langchain_community.vectorstores.FAISS` 被移除；
3. 需要多向量库后端 / 需要增量更新（当前 FAISS 是静态的，这一点也写在 README 的已知限制里）。

## 迁移预案（到那天怎么做）

保持 `src/vectorstore.py` 对外只暴露 `load_vectorstore()` / `build_vectorstore()`，内部换成 `faiss-cpu` + 自己的 `save_local`/`load_local`（等价于 `index.faiss` + pickle 的 docstore，注意沿用现有的 `allow_dangerous_deserialization` 安全约束），调用方零改动。
