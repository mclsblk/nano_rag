# Agentic RAG

一个 API-first 的本地知识库 RAG 系统。项目面向本地文档检索与问答：通过 HTTP API 管理文件、collection 和 ingest job，再执行检索或基于检索结果生成带来源的回答。

当前支持两类主要使用方式：

- API：主要集成入口，适合被应用、脚本、Web UI 或上层 Agent 调用。
- CLI：管理和 debug client，适合本地排查、冒烟测试和脚本化维护。

## 特性

- 本地优先：默认使用 Ollama 提供聊天模型和 embedding 模型。
- 多 provider：支持 Ollama 和 OpenAI-compatible API，可分别配置 chat / embedding provider。
- 持久化向量库：使用 Chroma 存储文档分块和向量。
- Hybrid Search：默认同时使用 Chroma vector search 和 SQLite FTS5 keyword search。
- 托管文件与 collection：文件先进入本地 file storage，再按需 ingest 到一个或多个 collection。
- 文档加载：默认支持 `.md`、`.txt`、`.pdf` 文件；`auto` / `visual` loader 策略下可处理 `.png`、`.jpg`、`.jpeg`、`.webp` 图片。
- Agentic Ask：可选启用 query rewrite、multi-query retrieval、context judge 和 fallback policy。
- PDF 页码来源：PDF 按页加载，检索结果可以带上页码信息。
- 双输出格式：默认输出可读文本，也可通过 `--json` 输出稳定 JSON。
- 来源追踪：检索和问答结果包含 source、page、score 等信息。
- API 服务：提供 upload、collection、ingest job、search、ask、de-ingest 和 delete 流程。
- 本地可用：提供 ready check、稳定错误 JSON、request id、可选 API key auth 和可选 CORS。

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
OPENAI_COMPATIBLE_VISUAL_MODEL=
OPENAI_COMPATIBLE_TIMEOUT_SECONDS=30
DOCUMENT_LOAD_STRATEGY=text
VISUAL_MODEL_PROVIDER=openai_compatible
VISUAL_MIN_TEXT_CHARS=40
CHROMA_PERSIST_DIR=./storage/chroma
CHROMA_COLLECTION=agentic_rag
SEARCH_STRATEGY=hybrid
KEYWORD_INDEX_PATH=./storage/keyword.sqlite
SYSTEM_DB_PATH=./storage/system.sqlite
FILE_STORAGE_DIR=./storage/files
UPLOAD_DIR=./storage/uploads
MAX_UPLOAD_MB=50
API_KEY=
CORS_ORIGINS=
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

`CHROMA_PERSIST_DIR` 指向本地 Chroma 持久化目录。默认的 `storage/chroma/` 属于运行时数据，不适合提交到版本库。`SYSTEM_DB_PATH` 保存 file / collection / registry / job 系统状态；`FILE_STORAGE_DIR` 保存托管文件副本；`UPLOAD_DIR` 保存 API 上传暂存文件；`KEYWORD_INDEX_PATH` 仍保留旧默认 keyword index 配置，新 collection 默认使用 `storage/keyword/<collection_id>.sqlite`。

`API_KEY` 为空时不启用认证；配置后，除 `/health`、`/ready`、`/docs` 和 `/openapi.json` 外，API 请求需要携带 `Authorization: Bearer <API_KEY>` 或 `X-API-Key: <API_KEY>`。`CORS_ORIGINS` 为空时不启用 CORS；配置多个浏览器来源时使用逗号分隔。

`SEARCH_STRATEGY` 支持：

- `hybrid`：默认策略。并行执行 Chroma vector search 与 SQLite keyword search，再用简单加权分数合并排序。
- `vector`：只使用 Chroma vector search，便于回退和 debug。
- `keyword`：只使用 SQLite FTS5 keyword search；search / ask 检索阶段不需要 embedding 查询。

SQLite keyword index 会保存 collection 内的 chunk 原文和基于正文生成的内部 keyword text；不保存文件生命周期状态。file / collection / registry 生命周期状态保存在 system SQLite 中。`source` 只作为展示和溯源字段，不再作为生命周期主键。keyword text 只用于检索，不会作为 `search` / `ask` 的结果内容返回。metadata 不参与 keyword text 生成，只通过结果里的 source、page 等字段用于溯源。中文 keyword text 只生成 bigram / trigram，不生成单字 token，因此单字中文 query 不保证命中。

keyword query 会先区分 API 名、英文/数字专名、版本号等高价值 token，以及“功能”“作用”“参数”等泛化意图 token。keyword score 表示 query 满足程度，会综合 required token 覆盖、optional token 覆盖、精确专名命中、BM25 信号和正文结构信号；它不是简单的 FTS 排名倒数。升级 keyword tokenization 或 score 逻辑后，已入库旧数据需要先对对应 `file_id + collection_id` 执行 de-ingest，再重新 ingest。

`HYBRID_VECTOR_WEIGHT` 控制 hybrid 排序中 vector score 的权重；keyword score 权重自动为 `1 - HYBRID_VECTOR_WEIGHT`。例如默认 `0.65` 表示 vector 占 65%，keyword 占 35%。

`CHUNK_STRATEGY` 支持：

- `semantic`：默认策略。先由 `DocumentBuilder` 按硬断点构建 sections、清洗 PDF 装饰行 / 页码 / 排版换行，并合并过小 blocks；`SemanticChunker` 再基于 blocks 判断语义边界。ingest 路径优先使用当前 embedding provider；没有传入 embedding model 的内部调用会 fallback 到 scikit-learn 字符 n-gram TF-IDF。
- `character`：按字数窗口切分，带固定重叠。

`CHUNK_SIZE_CHARS`、`CHUNK_OVERLAP_CHARS` 和 `CHUNK_MIN_CHARS` 按“字数”计算：中文、英文、数字计入，空白和标点不计入。`CHUNK_SIZE_CHARS` 在 `semantic` 下是最大字数预算，不表示每 N 字固定切一刀；`CHUNK_MIN_CHARS` 会影响 builder 的小 block 合并和 chunker 的小 chunk 合并；`CHUNK_OVERLAP_CHARS` 只对 `character` 策略生效。semantic 阈值是启发式默认值，可按语料继续微调。

如需通过 llama.cpp、vLLM、LM Studio 或其他 OpenAI 格式服务接入模型，可把 provider 切到 `openai_compatible`，并配置对应 base URL 与模型名。`DOCUMENT_LOAD_STRATEGY` 默认为 `text`；设置为 `auto` 或 `visual` 时，PDF 低文本页或图片文件会通过 `VISUAL_MODEL_PROVIDER` 对应的视觉模型抽取文本。

## API 使用

启动 API 服务：

```bash
rag-api --host 127.0.0.1 --port 8000
```

检查进程和本地依赖：

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
```

上传文件：

```bash
curl -X POST http://127.0.0.1:8000/v1/files/upload \
  -F "file=@docs/project.md"
```

创建 collection：

```bash
curl -X POST http://127.0.0.1:8000/v1/collections \
  -H "Content-Type: application/json" \
  -d '{"name":"docs","description":"local docs"}'
```

创建 ingest job：

```bash
curl -X POST http://127.0.0.1:8000/v1/ingest/jobs \
  -H "Content-Type: application/json" \
  -d '{"file_id":"<file_id>","collection_id":"<collection_id>"}'
```

轮询 job：

```bash
curl http://127.0.0.1:8000/v1/jobs/<job_id>
```

检索和问答：

```bash
curl -X POST http://127.0.0.1:8000/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query":"项目支持哪些文档格式？","collection_id":"<collection_id>","top_k":5}'

curl -X POST http://127.0.0.1:8000/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"search 和 ask 有什么区别？","collection_id":"<collection_id>","top_k":5}'
```

从 collection 中移除文件索引，再删除文件或 collection：

```bash
curl -X DELETE http://127.0.0.1:8000/v1/collections/<collection_id>/files/<file_id>
curl -X DELETE http://127.0.0.1:8000/v1/files/<file_id>
curl -X DELETE http://127.0.0.1:8000/v1/collections/<collection_id>
```

所有业务错误返回稳定 JSON：

```json
{"error":{"type":"RegistryError","message":"..."}}
```

每个响应都会带有 `X-Request-ID`。客户端也可以传入同名请求头，方便串联日志。

## CLI 使用

查看当前配置和向量库状态：

```bash
rag inspect
```

输出 JSON：

```bash
rag inspect --json
```

创建 collection：

```bash
rag collection create docs --json
```

导入托管文件并得到稳定 `file_id`：

```bash
rag file import docs/project.md --json
```

把托管文件 ingest 到指定 collection：

```bash
rag ingest <file_id> --collection <collection_id>
```

同一个 `file_id` 可以 ingest 到多个 collection。重复 ingest 同一个 active 的 `file_id + collection_id` 会直接失败，不会静默覆盖。

查看托管文件、collection 和 registry 状态：

```bash
rag file list --json
rag collection list --json
rag registry list --json
```

从某个 collection 中 de-ingest 文件：

```bash
rag de-ingest <file_id> --collection <collection_id> --json
```

删除 file 或 collection 前，必须先 de-ingest 所有关联的 active registry record：

```bash
rag file delete <file_id> --json
rag collection delete <collection_id> --json
```

检索知识库：

```bash
rag search "项目支持哪些文档格式？" --collection <collection_id> --top-k 5
```

检索并输出 JSON：

```bash
rag search "项目支持哪些文档格式？" --collection <collection_id> --json
```

输出完整检索调试 metadata：

```bash
rag search "项目支持哪些文档格式？" --collection <collection_id> --debug --json
```

基于知识库问答：

```bash
rag ask "search 和 ask 有什么区别？" --collection <collection_id> --top-k 5
```

问答并输出 JSON：

```bash
rag ask "search 和 ask 有什么区别？" --collection <collection_id> --json
```

启用 Agentic Ask：

```bash
rag ask "search 和 ask 有什么区别？" --collection <collection_id> --agentic
```

输出 Agentic Debug 信息：

```bash
rag ask "search 和 ask 有什么区别？" --collection <collection_id> --agentic --debug --json
```

## 数据流程

1. `rag file import` 计算文件内容 hash，把原始文件复制到托管 file storage，并登记稳定 `file_id`。
2. `rag collection create` 创建 collection 元数据，默认使用独立的 Chroma collection 和 keyword SQLite 文件。
3. `rag ingest <file_id> --collection <collection_id>` 由 registry 检查该 file 是否可进入目标 collection。
4. `DocumentLoader` 加载托管文件，跳过不支持的文件类型和空内容。
5. 当前 chunk strategy 将文档切成 chunks。
   - `semantic`：`DocumentBuilder` 先做硬断点切 section、段落清洗和小 block 合并；`SemanticChunker` 再使用 embedding 相似度判断语义边界，PDF 可在相邻页之间合并连续 chunk。
   - `character`：按字数窗口切分，并使用固定重叠。
6. `rag/indexer.py` 为 chunks 注入 `file_id`、`collection_id`，写入目标 collection 的 SQLite keyword index 和 Chroma。
7. registry 将对应 `file_id + collection_id` 标记为 `indexed`，并记录 chunk 数量。
8. `rag search --collection` 根据 `SEARCH_STRATEGY` 在指定 collection 内执行 vector / keyword / hybrid 检索。
9. `rag ask --collection` 将检索结果作为上下文交给聊天模型生成回答。
10. `rag ask --agentic --collection` 会先改写 query、生成多个检索 query、合并去重检索结果，再判断 context 是否足够。

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

非 JSON 模式下，长耗时命令会显示简短 spinner。使用 `--json` 时，spinner 会关闭，输出保持紧凑 JSON，适合脚本或上层 Agent 调用。默认 JSON 会过滤每条结果的内部 metadata，只保留 source、file、collection、page 相关溯源字段。

使用 `--debug --json` 时，会输出完整 result metadata，包括 retrieval mode、vector score、keyword score、raw BM25、matched tokens 和 coverage 等调试字段。`rag ask --agentic --debug --json` 还会在默认 `AnswerResponse` 字段外额外输出 `debug` 对象，包含 rewritten query、retrieval queries、context 判断、fallback reason 和选中的 source ids。

## 当前行为说明

- `text` loader 策略支持 `.md`、`.txt`、`.pdf`；`auto` / `visual` loader 策略额外支持 `.png`、`.jpg`、`.jpeg`、`.webp`。
- 目录导入会递归扫描文件。
- `rag ingest` 不覆盖已存在的 `file_id + collection_id`；更新前必须先 `rag de-ingest <file_id> --collection <collection_id>`。
- 不支持的文件会被跳过并发出 warning。
- 空文本文件、无可提取文本的 PDF 页面会被跳过。
- PDF loader 仍按页输出 document；PDF 文本清洗在 builder 阶段完成；semantic chunker 可以合并相邻页的连续语义 chunk。
- 当 `ask` 没有检索到足够上下文时，会返回“无法从当前知识库中确定。”
- `rag search` 始终保持单 query 检索，不启用 rewrite 或 multi-query。
- Query rewrite 和 multi-query retrieval 只在 `rag ask --agentic` 下启用。
- 默认 hybrid 检索会要求 Chroma 和 SQLite keyword index 拥有一致的 source/chunk；旧的 Chroma-only 数据需要先 `rag de-ingest <source>` 再重新 ingest。
- keyword text 和 document tokenization 不会自动迁移；升级 keyword 检索逻辑后，旧 source 也需要先 `rag de-ingest <source>` 再重新 ingest，才能完整受益。
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
