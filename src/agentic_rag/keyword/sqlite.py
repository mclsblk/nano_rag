from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any

from agentic_rag.core import Chunk, Document, KeywordStoreError, SearchResult
from agentic_rag.keyword.tokenizer import build_keyword_text, tokenize


class SQLiteKeywordStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._initialize()
        except Exception as exc:
            raise KeywordStoreError(f"Failed to initialize SQLite keyword store: {self.path}") from exc

    def add_documents(self, documents: list[Document]) -> None:
        if not documents:
            return

        try:
            with self._connect() as connection:
                for source, source_documents in _group_documents_by_source(documents).items():
                    metadata = _source_metadata(source_documents)
                    connection.execute(
                        """
                        INSERT INTO sources (source, metadata_json, created_at)
                        VALUES (?, ?, ?)
                        """,
                        (
                            source,
                            _json_dumps(metadata),
                            datetime.now(timezone.utc).isoformat(),
                        ),
                    )
        except sqlite3.IntegrityError as exc:
            raise KeywordStoreError("Source already exists in SQLite keyword store.") from exc
        except Exception as exc:
            raise KeywordStoreError("Failed to add documents to SQLite keyword store.") from exc

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        try:
            with self._connect() as connection:
                for chunk in chunks:
                    source = _chunk_source(chunk)
                    metadata = dict(chunk.metadata)
                    connection.execute(
                        """
                        INSERT INTO chunks (
                            id, source, content, metadata_json, chunk_index, page_start, page_end
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            chunk.id,
                            source,
                            chunk.content,
                            _json_dumps(metadata),
                            _optional_int(metadata.get("chunk_index")),
                            _optional_int(metadata.get("page_start")),
                            _optional_int(metadata.get("page_end")),
                        ),
                    )
                    connection.execute(
                        """
                        INSERT INTO chunk_fts (chunk_id, source, keyword_text)
                        VALUES (?, ?, ?)
                        """,
                        (chunk.id, source, build_keyword_text(chunk)),
                    )
        except sqlite3.IntegrityError as exc:
            raise KeywordStoreError("Chunk already exists in SQLite keyword store.") from exc
        except Exception as exc:
            raise KeywordStoreError("Failed to add chunks to SQLite keyword store.") from exc

    def source_exists(self, source: str) -> bool:
        try:
            with self._connect() as connection:
                row = connection.execute("SELECT 1 FROM sources WHERE source = ? LIMIT 1", (source,)).fetchone()
        except Exception as exc:
            raise KeywordStoreError(f"Failed to check source in SQLite keyword store: {source}") from exc

        return row is not None

    def delete_by_source(self, source: str) -> int:
        try:
            with self._connect() as connection:
                row = connection.execute("SELECT COUNT(*) AS count FROM chunks WHERE source = ?", (source,)).fetchone()
                deleted_chunks = int(row["count"]) if row is not None else 0
                connection.execute("DELETE FROM chunk_fts WHERE source = ?", (source,))
                connection.execute("DELETE FROM chunks WHERE source = ?", (source,))
                connection.execute("DELETE FROM sources WHERE source = ?", (source,))
        except Exception as exc:
            raise KeywordStoreError(f"Failed to delete source from SQLite keyword store: {source}") from exc

        return deleted_chunks

    def keyword_search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            raise KeywordStoreError("top_k must be greater than 0.")

        tokens = tokenize(query)
        if not tokens:
            return []

        match_query = _fts_or_query(tokens)
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT
                        chunks.id,
                        chunks.source,
                        chunks.content,
                        chunks.metadata_json,
                        bm25(chunk_fts) AS raw_bm25
                    FROM chunk_fts
                    JOIN chunks ON chunks.id = chunk_fts.chunk_id
                    WHERE chunk_fts MATCH ?
                    ORDER BY raw_bm25 ASC
                    LIMIT ?
                    """,
                    (match_query, top_k),
                ).fetchall()
        except Exception as exc:
            raise KeywordStoreError("Failed to query SQLite keyword store.") from exc

        results: list[SearchResult] = []
        for index, row in enumerate(rows, start=1):
            metadata = _json_loads(row["metadata_json"])
            raw_bm25 = float(row["raw_bm25"])
            keyword_score = 1.0 / index
            metadata.update(
                {
                    "keyword_score": keyword_score,
                    "raw_bm25": raw_bm25,
                    "retrieval_mode": "keyword",
                }
            )
            results.append(
                SearchResult(
                    id=str(row["id"]),
                    content=str(row["content"]),
                    score=keyword_score,
                    source=str(row["source"]),
                    metadata=metadata,
                )
            )

        return results

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT id, content, metadata_json FROM chunks WHERE id = ?",
                    (chunk_id,),
                ).fetchone()
        except Exception as exc:
            raise KeywordStoreError(f"Failed to get chunk from SQLite keyword store: {chunk_id}") from exc

        if row is None:
            return None
        return Chunk(
            id=str(row["id"]),
            content=str(row["content"]),
            metadata=_json_loads(row["metadata_json"]),
        )

    def count_sources(self) -> int:
        return self._count("sources")

    def count_chunks(self) -> int:
        return self._count("chunks")

    def list_sources(self) -> list[str]:
        try:
            with self._connect() as connection:
                rows = connection.execute("SELECT source FROM sources ORDER BY source").fetchall()
        except Exception as exc:
            raise KeywordStoreError("Failed to list sources from SQLite keyword store.") from exc

        return [str(row["source"]) for row in rows]

    def _count(self, table: str) -> int:
        try:
            with self._connect() as connection:
                row = connection.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
        except Exception as exc:
            raise KeywordStoreError(f"Failed to count rows in SQLite keyword store table: {table}") from exc

        return int(row["count"]) if row is not None else 0

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sources (
                    source TEXT PRIMARY KEY,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    chunk_index INTEGER,
                    page_start INTEGER,
                    page_end INTEGER
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                    chunk_id UNINDEXED,
                    source UNINDEXED,
                    keyword_text,
                    tokenize='unicode61'
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path))
        connection.row_factory = sqlite3.Row
        return connection


def _group_documents_by_source(documents: list[Document]) -> dict[str, list[Document]]:
    grouped: dict[str, list[Document]] = defaultdict(list)
    for document in documents:
        grouped[_document_source(document)].append(document)
    return dict(grouped)


def _source_metadata(documents: list[Document]) -> dict[str, Any]:
    metadata = dict(documents[0].metadata) if documents else {}
    metadata.pop("page_number", None)
    return metadata


def _document_source(document: Document) -> str:
    source = document.metadata.get("source")
    return source if isinstance(source, str) and source else document.id


def _chunk_source(chunk: Chunk) -> str:
    source = chunk.metadata.get("source")
    if isinstance(source, str) and source:
        return source
    if ":chunk:" in chunk.id:
        return chunk.id.split(":chunk:", 1)[0]
    return chunk.id


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _json_dumps(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _json_loads(value: str) -> dict[str, Any]:
    loaded = json.loads(value)
    return loaded if isinstance(loaded, dict) else {}


def _fts_or_query(tokens: list[str]) -> str:
    return " OR ".join(_quote_fts_token(token) for token in tokens)


def _quote_fts_token(token: str) -> str:
    return f'"{token.replace(chr(34), chr(34) + chr(34))}"'
