from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from agentic_rag.core import Chunk, Document, SearchResult
from agentic_rag.keyword.tokenizer import analyze_query, build_keyword_text


class PostgresChunkRepository:
    def __init__(self, store, collection_id: str | None = None) -> None:
        self.store = store
        self.collection_id = collection_id

    def add_documents(self, documents: list[Document]) -> None:
        if not documents:
            return
        now = _now()
        with self.store.connect() as connection:
            for source, source_documents in _group_documents_by_source(documents).items():
                metadata = _source_metadata(source_documents)
                connection.execute(
                    """
                    INSERT INTO chunk_sources (source, file_id, collection_id, metadata_json, created_at)
                    VALUES (%s, %s, %s, %s::jsonb, %s)
                    ON CONFLICT (source) DO UPDATE SET
                        file_id = excluded.file_id,
                        collection_id = excluded.collection_id,
                        metadata_json = excluded.metadata_json
                    """,
                    (
                        source,
                        _source_file_id(source_documents),
                        _source_collection_id(source_documents) or self.collection_id,
                        _json_dumps(metadata),
                        now,
                    ),
                )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        now = _now()
        with self.store.connect() as connection:
            for chunk in chunks:
                metadata = dict(chunk.metadata)
                source = _chunk_source(chunk)
                connection.execute(
                    """
                    INSERT INTO chunk_sources (source, file_id, collection_id, metadata_json, created_at)
                    VALUES (%s, %s, %s, %s::jsonb, %s)
                    ON CONFLICT (source) DO NOTHING
                    """,
                    (
                        source,
                        _chunk_file_id(chunk),
                        _chunk_collection_id(chunk) or self.collection_id,
                        _json_dumps(metadata),
                        now,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO chunks (
                        id, source, file_id, collection_id, content, metadata_json,
                        keyword_text, chunk_index, page_start, page_end,
                        index_status, created_at, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, 'indexing', %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        content = excluded.content,
                        metadata_json = excluded.metadata_json,
                        keyword_text = excluded.keyword_text,
                        chunk_index = excluded.chunk_index,
                        page_start = excluded.page_start,
                        page_end = excluded.page_end,
                        index_status = 'indexing',
                        updated_at = excluded.updated_at
                    """,
                    (
                        chunk.id,
                        source,
                        _chunk_file_id(chunk),
                        _chunk_collection_id(chunk) or self.collection_id,
                        chunk.content,
                        _json_dumps(metadata),
                        build_keyword_text(chunk),
                        _optional_int(metadata.get("chunk_index")),
                        _optional_int(metadata.get("page_start")),
                        _optional_int(metadata.get("page_end")),
                        now,
                        now,
                    ),
                )

    def add_embeddings(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        now = _now()
        with self.store.connect() as connection:
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                metadata = dict(chunk.metadata)
                source = _chunk_source(chunk)
                connection.execute(
                    """
                    INSERT INTO chunk_sources (source, file_id, collection_id, metadata_json, created_at)
                    VALUES (%s, %s, %s, %s::jsonb, %s)
                    ON CONFLICT (source) DO NOTHING
                    """,
                    (
                        source,
                        _chunk_file_id(chunk),
                        _chunk_collection_id(chunk) or self.collection_id,
                        _json_dumps(metadata),
                        now,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO chunks (
                        id, source, file_id, collection_id, content, metadata_json,
                        keyword_text, embedding, chunk_index, page_start, page_end,
                        index_status, created_at, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, 'indexing', %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        content = excluded.content,
                        metadata_json = excluded.metadata_json,
                        keyword_text = CASE
                            WHEN chunks.keyword_text = '' THEN excluded.keyword_text
                            ELSE chunks.keyword_text
                        END,
                        embedding = excluded.embedding,
                        chunk_index = excluded.chunk_index,
                        page_start = excluded.page_start,
                        page_end = excluded.page_end,
                        updated_at = excluded.updated_at
                    """,
                    (
                        chunk.id,
                        source,
                        _chunk_file_id(chunk),
                        _chunk_collection_id(chunk) or self.collection_id,
                        chunk.content,
                        _json_dumps(metadata),
                        build_keyword_text(chunk),
                        embedding,
                        _optional_int(metadata.get("chunk_index")),
                        _optional_int(metadata.get("page_start")),
                        _optional_int(metadata.get("page_end")),
                        now,
                        now,
                    ),
                )

    def mark_file_indexed(self, file_id: str, collection_id: str) -> int:
        with self.store.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE chunks
                SET index_status = 'indexed', updated_at = %s
                WHERE file_id = %s AND collection_id = %s
                """,
                (_now(), file_id, collection_id),
            )
            return cursor.rowcount

    def delete_by_file(self, file_id: str, collection_id: str | None = None) -> int:
        resolved_collection_id = collection_id or self.collection_id
        if resolved_collection_id:
            where_sql = "file_id = %s AND collection_id = %s"
            params: tuple[Any, ...] = (file_id, resolved_collection_id)
        else:
            where_sql = "file_id = %s"
            params = (file_id,)
        with self.store.connect() as connection:
            cursor = connection.execute(f"DELETE FROM chunks WHERE {where_sql}", params)
            connection.execute(
                """
                DELETE FROM chunk_sources
                WHERE file_id = %s
                  AND NOT EXISTS (
                      SELECT 1 FROM chunks WHERE chunks.source = chunk_sources.source
                  )
                """,
                (file_id,),
            )
            return cursor.rowcount

    def count_chunks(self) -> int:
        sql = "SELECT COUNT(*) AS count FROM chunks WHERE index_status = 'indexed'"
        params: tuple[Any, ...] = ()
        if self.collection_id:
            sql += " AND collection_id = %s"
            params = (self.collection_id,)
        with self.store.connect() as connection:
            row = connection.execute(sql, params).fetchone()
        return int(row["count"]) if row is not None else 0

    def count_sources(self) -> int:
        sql = """
            SELECT COUNT(DISTINCT source) AS count
            FROM chunks
            WHERE index_status = 'indexed'
        """
        params: tuple[Any, ...] = ()
        if self.collection_id:
            sql += " AND collection_id = %s"
            params = (self.collection_id,)
        with self.store.connect() as connection:
            row = connection.execute(sql, params).fetchone()
        return int(row["count"]) if row is not None else 0

    def list_sources(self) -> list[str]:
        sql = """
            SELECT DISTINCT source
            FROM chunks
            WHERE index_status = 'indexed'
        """
        params: tuple[Any, ...] = ()
        if self.collection_id:
            sql += " AND collection_id = %s"
            params = (self.collection_id,)
        sql += " ORDER BY source"
        with self.store.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [str(row["source"]) for row in rows]

    def get_by_ids(self, ids: list[str]) -> list[SearchResult]:
        if not ids:
            return []
        with self.store.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, content, source, metadata_json
                FROM chunks
                WHERE id = ANY(%s)
                  AND index_status = 'indexed'
                """,
                (ids,),
            ).fetchall()
        by_id = {_row_id(row): _row_result(row) for row in rows}
        return [by_id[result_id] for result_id in ids if result_id in by_id]

    def vector_search(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        params: list[Any] = []
        collection_filter = ""
        if self.collection_id:
            collection_filter = "AND collection_id = %s"
            params.append(self.collection_id)
        params.append(top_k)
        with self.store.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    id, content, source, metadata_json,
                    embedding <=> %s AS raw_distance
                FROM chunks
                WHERE index_status = 'indexed'
                  AND embedding IS NOT NULL
                  {collection_filter}
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                [query_embedding, query_embedding, *params],
            ).fetchall()
        return [_vector_result(row) for row in rows]

    def keyword_search(self, query: str, top_k: int) -> list[SearchResult]:
        query_text = _keyword_query_text(query)
        if not query_text:
            return []
        params: list[Any] = [query_text]
        collection_filter = ""
        if self.collection_id:
            collection_filter = "AND collection_id = %s"
            params.append(self.collection_id)
        params.append(top_k)
        with self.store.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    id, content, source, metadata_json,
                    ts_rank_cd(search_vector, plainto_tsquery('simple', %s)) AS keyword_score
                FROM chunks
                WHERE index_status = 'indexed'
                  AND search_vector @@ plainto_tsquery('simple', %s)
                  {collection_filter}
                ORDER BY keyword_score DESC
                LIMIT %s
                """,
                [query_text, *params],
            ).fetchall()
        return [_keyword_result(row, rank) for rank, row in enumerate(rows, start=1)]


def collection_id_from_keyword_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    name = Path(path).name
    if name.endswith(".sqlite"):
        return name.removesuffix(".sqlite")
    return Path(path).stem or None


def _keyword_query_text(query: str) -> str:
    keyword_query = analyze_query(query)
    tokens = [*keyword_query.required_tokens, *keyword_query.optional_tokens]
    return " ".join(tokens)


def _row_id(row: dict[str, Any]) -> str:
    return str(row["id"])


def _row_result(row: dict[str, Any]) -> SearchResult:
    metadata = _json_loads(row["metadata_json"])
    return SearchResult(
        id=str(row["id"]),
        content=str(row["content"]),
        source=str(row["source"]) if row.get("source") is not None else None,
        metadata=metadata,
    )


def _vector_result(row: dict[str, Any]) -> SearchResult:
    result = _row_result(row)
    raw_distance = row.get("raw_distance")
    metadata = dict(result.metadata)
    if raw_distance is not None:
        metadata["raw_distance"] = float(raw_distance)
    return result.model_copy(update={"score": _normalize_distance(raw_distance), "metadata": metadata})


def _keyword_result(row: dict[str, Any], rank: int) -> SearchResult:
    score = _normalize_keyword_score(row.get("keyword_score"))
    result = _row_result(row)
    metadata = dict(result.metadata)
    metadata.update(
        {
            "keyword_score": score,
            "keyword_rank": rank,
            "keyword_matched_tokens": [],
            "keyword_content_matched_tokens": [],
            "keyword_required_coverage": 0.0,
            "keyword_optional_coverage": 0.0,
            "keyword_exact_token_bonus": 0.0,
            "keyword_bm25_signal": score,
            "keyword_structure_bonus": 0.0,
        }
    )
    return result.model_copy(update={"score": score, "metadata": metadata})


def _normalize_distance(distance: Any) -> float | None:
    if distance is None:
        return None
    return max(0.0, min(1.0, 1.0 - float(distance)))


def _normalize_keyword_score(score: Any) -> float:
    if score is None:
        return 0.0
    return max(0.0, min(1.0, float(score)))


def _group_documents_by_source(documents: list[Document]) -> dict[str, list[Document]]:
    grouped: dict[str, list[Document]] = defaultdict(list)
    for document in documents:
        grouped[_document_source(document)].append(document)
    return dict(grouped)


def _source_metadata(documents: list[Document]) -> dict[str, Any]:
    metadata = dict(documents[0].metadata) if documents else {}
    metadata.pop("page_number", None)
    return metadata


def _source_file_id(documents: list[Document]) -> str | None:
    if not documents:
        return None
    value = documents[0].metadata.get("file_id")
    return value if isinstance(value, str) and value else None


def _source_collection_id(documents: list[Document]) -> str | None:
    if not documents:
        return None
    value = documents[0].metadata.get("collection_id")
    return value if isinstance(value, str) and value else None


def _document_source(document: Document) -> str:
    source = document.metadata.get("source")
    return source if isinstance(source, str) and source else document.id


def _chunk_file_id(chunk: Chunk) -> str | None:
    value = chunk.metadata.get("file_id")
    return value if isinstance(value, str) and value else None


def _chunk_collection_id(chunk: Chunk) -> str | None:
    value = chunk.metadata.get("collection_id")
    return value if isinstance(value, str) and value else None


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


def _json_loads(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    loaded = json.loads(str(value))
    return loaded if isinstance(loaded, dict) else {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
