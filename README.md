# Agentic RAG

CLI-first local Agentic RAG system.

This repository is currently in phase 8: v0.1 CLI wiring.

Supported document formats are `.md`, `.txt`, and `.pdf`. Unsupported files are warned and skipped during directory loading. Empty files, including PDFs with no extractable text, are warned and skipped for now.

Chroma persists to `CHROMA_PERSIST_DIR` and uses `CHROMA_COLLECTION` as the collection name. Retrieval normalizes Chroma distances into `SearchResult.score` values in `[0.0, 1.0]`.

PDF files are loaded page by page so retrieved context can include page numbers in source labels.

Formatter JSON output is compact and based on Pydantic response schemas. Human-readable output includes source, page, and score labels.
