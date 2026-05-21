# Agentic RAG

一个 CLI-first 的本地知识库 RAG 系统。项目面向本地文档检索与问答：先把文档写入本地向量库，再通过命令行进行检索，或基于检索结果生成带来源的回答。

当前支持两类主要使用方式：

- `search`：只返回检索结果，适合作为主 Agent 或其他工具链中的检索组件。
- `ask`：执行检索增强生成，返回答案、来源和置信度，适合作为轻量本地知识库问答工具。

## 特性

- 本地优先：默认使用 Ollama 提供聊天模型和 embedding 模型。
- 多 provider：支持 Ollama 和 OpenAI-compatible API，可分别配置 chat / embedding provider。
- 持久化向量库：使用 Chroma 存储文档分块和向量。
- Hybrid Search：默认同时使用 Chroma vector search 和 SQLite FTS5 keyword search。
- 文档导入：支持 `.md`、`.txt`、`.pdf` 文件，也支持递归导入目录。
- Agentic Ask：可选启用 query rewrite、multi-query retrieval、context judge 和 fallback policy。
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
MODEL_PROVIDER=ollama
CHAT_MODEL_PROVIDER=ollama
EMBEDDING_MODEL_PROVIDER=ollama
OPENAI_COMPATIBLE_BASE_URL=
OPENAI_COMPATIBLE_API_KEY=
OPENAI_COMPATIBLE_CHAT_MODEL=
OPENAI_COMPATIBLE_EMBEDDING_MODEL=
OPENAI_COMPATIBLE_TIMEOUT_SECONDS=30
CHROMA_PERSIST_DIR=./storage/chroma
CHROMA_COLLECTION=agentic_rag
SEARCH_STRATEGY=hybrid
KEYWORD_INDEX_PATH=./storage/keyword.sqlite
HYBRID_VECTOR_WEIGHT=0.65
HYBRID_CANDIDATE_MULTIPLIER=4
CHUNK_STRATEGY=semantic
CHUNK_SIZE_CHARS=800
CHUNK_OVERLAP_CHARS=120
CHUNK_MIN_CHARS=120
SEMANTIC_BREAKPOINT_THRESHOLD=0.35
SEMANTIC_PAGE_MERGE_MIN_SCORE=0.55
SEMANTIC_MAX_UNITS_PER_CHUNK=12
AGENTIC_CONTEXT_MIN_SCORE=0.45
AGENTIC_CONTEXT_MIN_CHARS=80
AGENTIC_CONTEXT_MAX_CHARS=4000
AGENTIC_MULTI_QUERY_COUNT=3
```

`CHROMA_PERSIST_DIR` 指向本地 Chroma 持久化目录。默认的 `storage/chroma/` 属于运行时数据，不适合提交到版本库。`KEYWORD_INDEX_PATH` 指向本地 SQLite keyword index，同样属于运行时数据。

`SEARCH_STRATEGY` 支持：

- `hybrid`：默认策略。并行执行 Chroma vector search 与 SQLite keyword search，再用简单加权分数合并排序。
- `vector`：只使用 Chroma vector search，便于回退和 debug。
- `keyword`：只使用 SQLite FTS5 keyword search；search / ask 检索阶段不需要 embedding 查询。

SQLite keyword index 会保存 source 生命周期记录、chunk 原文和内部 keyword text；不保存 source 级完整原文。keyword text 只用于检索，不会作为 `search` / `ask` 的结果内容返回。中文 keyword text 只生成 bigram / trigram，不生成单字 token，因此单字中文 query 不保证命中。

`HYBRID_VECTOR_WEIGHT` 控制 hybrid 排序中 vector score 的权重；keyword score 权重自动为 `1 - HYBRID_VECTOR_WEIGHT`。例如默认 `0.65` 表示 vector 占 65%，keyword 占 35%。

`CHUNK_STRATEGY` 支持：

- `semantic`：默认策略。先由 `DocumentBuilder` 按硬断点构建 sections、清洗 PDF 装饰行 / 页码 / 排版换行，并合并过小 blocks；`SemanticChunker` 再基于 blocks 判断语义边界。ingest 路径优先使用当前 embedding provider；没有传入 embedding model 的内部调用会 fallback 到 scikit-learn 字符 n-gram TF-IDF。
- `character`：按字数窗口切分，带固定重叠。

`CHUNK_SIZE_CHARS`、`CHUNK_OVERLAP_CHARS` 和 `CHUNK_MIN_CHARS` 按“字数”计算：中文、英文、数字计入，空白和标点不计入。`CHUNK_SIZE_CHARS` 在 `semantic` 下是最大字数预算，不表示每 N 字固定切一刀；`CHUNK_MIN_CHARS` 会影响 builder 的小 block 合并和 chunker 的小 chunk 合并；`CHUNK_OVERLAP_CHARS` 只对 `character` 策略生效。semantic 阈值是启发式默认值，可按语料继续微调。

如需通过 llama.cpp、vLLM、LM Studio 或其他 OpenAI 格式服务接入模型，可把 provider 切到 `openai_compatible`，并配置对应 base URL 与模型名。

## 使用

查看当前配置和向量库状态：

```bash
rag inspect
```

输出 JSON：

```bash
rag inspect --json
```

导入单个文件：

```bash
rag ingest docs/project.md
```

导入整个目录：

```bash
rag ingest docs
```

导入并输出 JSON：

```bash
rag ingest docs --json
```

`rag ingest` 不会覆盖已存在的 source。需要重新导入同一个 source 时，先删除旧 chunks：

```bash
rag de-ingest docs/project.md
rag ingest docs/project.md
```

删除并输出 JSON：

```bash
rag de-ingest docs/project.md --json
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

启用 Agentic Ask：

```bash
rag ask "search 和 ask 有什么区别？" --agentic
```

输出 Agentic Debug 信息：

```bash
rag ask "search 和 ask 有什么区别？" --agentic --debug --json
```

## 数据流程

1. `rag ingest` 读取文件或目录。
2. `DocumentLoader` 跳过不支持的文件类型和空内容。
3. 当前 chunk strategy 将文档切成 chunks。
   - `semantic`：`DocumentBuilder` 先做硬断点切 section、段落清洗和小 block 合并；`SemanticChunker` 再使用 embedding 相似度判断语义边界，PDF 可在相邻页之间合并连续 chunk。
   - `character`：按字数窗口切分，并使用固定重叠。
4. SQLite keyword index 保存 source 生命周期记录、chunk 原文和内部 keyword text。
5. 当前 embedding provider 为最终 chunks 生成 embedding。
6. `ChromaVectorStore` 将 chunks、向量和元数据写入 Chroma。
7. `rag search` 根据 `SEARCH_STRATEGY` 执行 vector / keyword / hybrid 检索。
8. `rag ask` 将检索结果作为上下文交给聊天模型生成回答。
9. `rag ask --agentic` 会先改写 query、生成多个检索 query、合并去重检索结果，再判断 context 是否足够。

## 输出说明

`search` 默认输出包含：

- 原始 query
- 命中结果数量
- 每条结果的 source、page、score；跨页 chunk 会显示类似 `page=3-4`
- 命中的文本内容

`ask` 默认输出包含：

- 原始 query
- confidence：`high`、`medium` 或 `low`
- 生成答案
- 来源列表

使用 `--json` 时，输出基于 Pydantic schema 的紧凑 JSON，适合脚本或上层 Agent 调用。

`rag ask --agentic --debug --json` 会在默认 `AnswerResponse` 字段外额外输出 `debug` 对象，包含 rewritten query、retrieval queries、context 判断、fallback reason 和选中的 source ids。

## 当前行为说明

- 仅支持 `.md`、`.txt`、`.pdf`。
- 目录导入会递归扫描文件。
- `rag ingest` 不覆盖已存在 source；更新前必须先 `rag de-ingest <source>`。
- 不支持的文件会被跳过并发出 warning。
- 空文本文件、无可提取文本的 PDF 页面会被跳过。
- PDF loader 仍按页输出 document；PDF 文本清洗在 builder 阶段完成；semantic chunker 可以合并相邻页的连续语义 chunk。
- 当 `ask` 没有检索到足够上下文时，会返回“无法从当前知识库中确定。”
- `rag search` 始终保持单 query 检索，不启用 rewrite 或 multi-query。
- Query rewrite 和 multi-query retrieval 只在 `rag ask --agentic` 下启用。
- 默认 hybrid 检索会要求 Chroma 和 SQLite keyword index 拥有一致的 source/chunk；旧的 Chroma-only 数据需要先 `rag de-ingest <source>` 再重新 ingest。
- vector 检索分数由 Chroma distance 归一化到 `[0.0, 1.0]`；hybrid 分数来自 vector score 与 keyword score 的简单加权，不是 LLM rerank。

## 作为 Python 模块使用

```python
from agentic_rag.factory import create_pipeline

pipeline = create_pipeline()
response = pipeline.search("项目支持哪些文档格式？", top_k=5)

for result in response.results:
    print(result.source, result.score)
```

Agentic Ask：

```python
from agentic_rag.factory import create_agentic_service

service = create_agentic_service()
response = service.ask("项目支持哪些文档格式？", top_k=5)
print(response.answer)
```
