# Agentic RAG

一个 CLI-first 的本地知识库 RAG 系统。项目面向本地文档检索与问答：先把文档写入本地向量库，再通过命令行进行检索，或基于检索结果生成带来源的回答。

当前支持两类主要使用方式：

- `search`：只返回检索结果，适合作为主 Agent 或其他工具链中的检索组件。
- `ask`：执行检索增强生成，返回答案、来源和置信度，适合作为轻量本地知识库问答工具。

## 特性

- 本地优先：默认使用 Ollama 提供聊天模型和 embedding 模型。
- 持久化向量库：使用 Chroma 存储文档分块和向量。
- 文档导入：支持 `.md`、`.txt`、`.pdf` 文件，也支持递归导入目录。
- PDF 页码来源：PDF 按页加载，检索结果可以带上页码信息。
- 双输出格式：默认输出可读文本，也可通过 `--json` 输出稳定 JSON。
- 来源追踪：检索和问答结果包含 source、page、score 等信息。

## 环境要求

- Python 3.10+
- 本地可访问的 Ollama 服务
- 可用于聊天和向量化的 Ollama 模型

默认模型配置来自 `src/agentic_rag/config.py`：

- 聊天模型：`qwen3.5:0.8b`
- Embedding 模型：`qwen3-embedding:0.6b`

如需使用默认模型，可以先准备：

```bash
ollama pull qwen3.5:0.8b
ollama pull qwen3-embedding:0.6b
```

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

安装后会提供 `rag` 命令。

开发依赖：

```bash
pip install -e ".[dev]"
```

## 配置

项目会自动读取 `.env`。未配置时使用默认值：

```bash
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_CHAT_MODEL=qwen3.5:0.8b
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:0.6b
OLLAMA_TIMEOUT_SECONDS=30
OLLAMA_THINK=false
CHROMA_PERSIST_DIR=./storage/chroma
CHROMA_COLLECTION=agentic_rag
```

`CHROMA_PERSIST_DIR` 指向本地 Chroma 持久化目录。默认的 `storage/chroma/` 属于运行时数据，不适合提交到版本库。

## 使用

查看当前配置和向量库状态：

```bash
rag inspect
```

导入单个文件：

```bash
rag ingest docs/project.md
```

导入整个目录：

```bash
rag ingest docs
```

检索知识库：

```bash
rag search "项目支持哪些文档格式？" --top-k 5
```

检索并输出 JSON：

```bash
rag search "项目支持哪些文档格式？" --json
```

基于知识库问答：

```bash
rag ask "search 和 ask 有什么区别？" --top-k 5
```

问答并输出 JSON：

```bash
rag ask "search 和 ask 有什么区别？" --json
```

## 数据流程

1. `rag ingest` 读取文件或目录。
2. `DocumentLoader` 跳过不支持的文件类型和空内容。
3. `TextSplitter` 将文档切成带重叠的文本块。
4. `OllamaEmbeddingModel` 生成 embedding。
5. `ChromaVectorStore` 将分块、向量和元数据写入 Chroma。
6. `rag search` 根据 query 检索相似文本块。
7. `rag ask` 将检索结果作为上下文交给聊天模型生成回答。

## 输出说明

`search` 默认输出包含：

- 原始 query
- 命中结果数量
- 每条结果的 source、page、score
- 命中的文本内容

`ask` 默认输出包含：

- 原始 query
- confidence：`high`、`medium` 或 `low`
- 生成答案
- 来源列表

使用 `--json` 时，输出基于 Pydantic schema 的紧凑 JSON，适合脚本或上层 Agent 调用。

## 当前行为说明

- 仅支持 `.md`、`.txt`、`.pdf`。
- 目录导入会递归扫描文件。
- 不支持的文件会被跳过并发出 warning。
- 空文本文件、无可提取文本的 PDF 页面会被跳过。
- 当 `ask` 没有检索到足够上下文时，会返回“无法从当前知识库中确定。”
- 检索分数由 Chroma distance 归一化到 `[0.0, 1.0]`。

## 作为 Python 模块使用

```python
from agentic_rag.factory import create_pipeline

pipeline = create_pipeline()
response = pipeline.search("项目支持哪些文档格式？", top_k=5)

for result in response.results:
    print(result.source, result.score)
```
