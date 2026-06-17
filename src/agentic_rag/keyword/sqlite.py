from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any

from agentic_rag.core import Chunk, Document, KeywordStoreError, SearchResult
from agentic_rag.keyword.query_vocab import QUERY_TOKEN_ALIASES
from agentic_rag.keyword.tokenizer import KeywordQuery, analyze_query, build_keyword_text, tokenize


_CANDIDATE_MULTIPLIER = 8
_MIN_CANDIDATES = 50


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
                        INSERT INTO sources (source, file_id, metadata_json, created_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            source,
                            _source_file_id(source_documents),
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
                            id, source, file_id, content, metadata_json, chunk_index, page_start, page_end
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            chunk.id,
                            source,
                            _chunk_file_id(chunk),
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

    def delete_by_file_id(self, file_id: str) -> int:
        try:
            with self._connect() as connection:
                rows = connection.execute("SELECT id FROM chunks WHERE file_id = ?", (file_id,)).fetchall()
                chunk_ids = [str(row["id"]) for row in rows]
                deleted_chunks = len(chunk_ids)
                for chunk_id in chunk_ids:
                    connection.execute("DELETE FROM chunk_fts WHERE chunk_id = ?", (chunk_id,))
                connection.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
                connection.execute("DELETE FROM sources WHERE file_id = ?", (file_id,))
        except Exception as exc:
            raise KeywordStoreError(f"Failed to delete file from SQLite keyword store: {file_id}") from exc

        return deleted_chunks

    def keyword_search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            raise KeywordStoreError("top_k must be greater than 0.")

        keyword_query = analyze_query(query)
        query_tokens = [*keyword_query.required_tokens, *keyword_query.optional_tokens]
        if not query_tokens:
            return []

        try:
            with self._connect() as connection:
                rows = _keyword_candidate_rows(connection, keyword_query, limit=_candidate_limit(top_k))
        except Exception as exc:
            raise KeywordStoreError("Failed to query SQLite keyword store.") from exc

        scored_rows = _score_keyword_rows(rows, keyword_query)
        return [_search_result(row, score_data) for row, score_data in scored_rows[:top_k]]

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
                    file_id TEXT,
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
                    file_id TEXT,
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
            _ensure_column(connection, "sources", "file_id", "TEXT")
            _ensure_column(connection, "chunks", "file_id", "TEXT")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path))
        connection.row_factory = sqlite3.Row
        return connection


def _keyword_candidate_rows(
    connection: sqlite3.Connection,
    keyword_query: KeywordQuery,
    *,
    limit: int,
) -> list[sqlite3.Row]:
    if keyword_query.required_tokens:
        rows = _fetch_keyword_rows(connection, _fts_and_query(keyword_query.required_tokens), limit)
        if rows:
            return rows

    relaxed_tokens = [*keyword_query.required_tokens, *keyword_query.optional_tokens]
    return _fetch_keyword_rows(connection, _fts_or_query(relaxed_tokens), limit)


def _fetch_keyword_rows(connection: sqlite3.Connection, match_query: str, limit: int) -> list[sqlite3.Row]:
    return connection.execute(
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
        (match_query, limit),
    ).fetchall()


def _score_keyword_rows(
    rows: list[sqlite3.Row],
    keyword_query: KeywordQuery,
) -> list[tuple[sqlite3.Row, dict[str, Any]]]:
    raw_bm25_values = [float(row["raw_bm25"]) for row in rows]
    scored: list[tuple[sqlite3.Row, dict[str, Any]]] = []

    for index, row in enumerate(rows, start=1):
        content_tokens = _text_tokens(str(row["content"]))
        required_matches = _matched_tokens(keyword_query.required_tokens, content_tokens)
        optional_matches = _matched_tokens(keyword_query.optional_tokens, content_tokens)

        required_coverage = _coverage(required_matches, keyword_query.required_tokens)
        optional_coverage = _coverage(optional_matches, keyword_query.optional_tokens)
        exact_token_bonus = _exact_token_bonus(required_matches)
        structure_bonus = _structure_bonus(str(row["content"]), keyword_query)
        bm25_signal = _normalized_bm25(float(row["raw_bm25"]), raw_bm25_values)

        score = (
            0.55 * required_coverage
            + 0.20 * optional_coverage
            + 0.15 * exact_token_bonus
            + 0.05 * bm25_signal
            + 0.05 * structure_bonus
        )
        if keyword_query.required_tokens and required_coverage == 0.0:
            score = min(score, 0.20)

        scored.append(
            (
                row,
                {
                    "keyword_score": _clamp_score(score),
                    "keyword_rank": index,
                    "keyword_matched_tokens": _dedupe_tokens([*required_matches, *optional_matches]),
                    "keyword_content_matched_tokens": _dedupe_tokens([*required_matches, *optional_matches]),
                    "keyword_required_coverage": required_coverage,
                    "keyword_optional_coverage": optional_coverage,
                    "keyword_exact_token_bonus": exact_token_bonus,
                    "keyword_bm25_signal": bm25_signal,
                    "keyword_structure_bonus": structure_bonus,
                },
            )
        )

    scored.sort(key=lambda item: (item[1]["keyword_score"], -float(item[0]["raw_bm25"])), reverse=True)
    return scored


def _search_result(row: sqlite3.Row, score_data: dict[str, Any]) -> SearchResult:
    metadata = _json_loads(row["metadata_json"])
    raw_bm25 = float(row["raw_bm25"])
    keyword_score = float(score_data["keyword_score"])
    metadata.update(
        {
            "keyword_score": keyword_score,
            "raw_bm25": raw_bm25,
            "retrieval_mode": "keyword",
            **score_data,
        }
    )
    return SearchResult(
        id=str(row["id"]),
        content=str(row["content"]),
        score=keyword_score,
        source=str(row["source"]),
        metadata=metadata,
    )


def _candidate_limit(top_k: int) -> int:
    return max(top_k * _CANDIDATE_MULTIPLIER, _MIN_CANDIDATES)


def _text_tokens(text: str) -> set[str]:
    return set(tokenize(text))


def _matched_tokens(query_tokens: tuple[str, ...], keyword_tokens: set[str]) -> list[str]:
    return [token for token in query_tokens if _token_matches(token, keyword_tokens)]


def _token_matches(token: str, keyword_tokens: set[str]) -> bool:
    return token in keyword_tokens or any(alias in keyword_tokens for alias in QUERY_TOKEN_ALIASES.get(token, ()))


def _coverage(matches: list[str], query_tokens: tuple[str, ...]) -> float:
    if not query_tokens:
        return 0.0
    return len(matches) / len(query_tokens)


def _dedupe_tokens(tokens: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


def _exact_token_bonus(required_matches: list[str]) -> float:
    if not required_matches:
        return 0.0
    return 1.0 if any(token.isascii() and any(character.isalpha() for character in token) for token in required_matches) else 0.5


def _structure_bonus(content: str, keyword_query: KeywordQuery) -> float:
    folded = content.casefold()
    first_required = min(
        (position for token in keyword_query.required_tokens if (position := folded.find(token)) >= 0),
        default=-1,
    )
    if first_required < 0:
        return 0.0
    if any(token in keyword_query.optional_tokens for token in ("功能", "作用")):
        if "功能说明" in folded[:80] and first_required <= 160:
            return 1.0
    if "函数原型" in folded[:160] and first_required <= 160:
        return 0.6
    return 0.0


def _normalized_bm25(raw_bm25: float, raw_bm25_values: list[float]) -> float:
    if not raw_bm25_values:
        return 0.0
    best = min(raw_bm25_values)
    worst = max(raw_bm25_values)
    if best == worst:
        return 1.0
    return (worst - raw_bm25) / (worst - best)


def _clamp_score(score: float) -> float:
    return max(0.0, min(1.0, score))


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


def _document_source(document: Document) -> str:
    source = document.metadata.get("source")
    return source if isinstance(source, str) and source else document.id


def _chunk_file_id(chunk: Chunk) -> str | None:
    value = chunk.metadata.get("file_id")
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


def _json_loads(value: str) -> dict[str, Any]:
    loaded = json.loads(value)
    return loaded if isinstance(loaded, dict) else {}


def _ensure_column(connection: sqlite3.Connection, table: str, column: str, column_type: str) -> None:
    rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    if any(str(row["name"]) == column for row in rows):
        return
    connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")


def _fts_or_query(tokens: list[str]) -> str:
    return " OR ".join(_quote_fts_token(token) for token in tokens)


def _fts_and_query(tokens: tuple[str, ...]) -> str:
    return " AND ".join(_quote_fts_token(token) for token in tokens)


def _quote_fts_token(token: str) -> str:
    return f'"{token.replace(chr(34), chr(34) + chr(34))}"'
