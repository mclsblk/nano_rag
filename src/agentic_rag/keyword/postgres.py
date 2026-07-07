from __future__ import annotations

from pathlib import Path

from agentic_rag.config import Settings, load_settings
from agentic_rag.core import Chunk, Document, KeywordStoreError, SearchResult
from agentic_rag.file_sys.store import SystemStore
from agentic_rag.postgres.chunks import PostgresChunkRepository, collection_id_from_keyword_path


class PostgresKeywordStore:
    def __init__(
        self,
        path: str | Path | None = None,
        settings: Settings | None = None,
        collection_id: str | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.collection_id = collection_id or collection_id_from_keyword_path(path)
        self.repository = PostgresChunkRepository(
            SystemStore(
                self.settings.postgres.database_url,
                embedding_dimension=self.settings.postgres.embedding_dimension,
            ),
            collection_id=self.collection_id,
        )

    def add_documents(self, documents: list[Document]) -> None:
        try:
            self.repository.add_documents(documents)
        except Exception as exc:
            raise KeywordStoreError("Failed to add documents to Postgres keyword store.") from exc

    def add_chunks(self, chunks: list[Chunk]) -> None:
        try:
            self.repository.add_chunks(chunks)
        except Exception as exc:
            raise KeywordStoreError("Failed to add chunks to Postgres keyword store.") from exc

    def delete_by_file_id(self, file_id: str) -> int:
        try:
            return self.repository.delete_by_file(file_id, self.collection_id)
        except Exception as exc:
            raise KeywordStoreError(f"Failed to delete file from Postgres keyword store: {file_id}") from exc

    def keyword_search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            raise KeywordStoreError("top_k must be greater than 0.")
        try:
            return self.repository.keyword_search(query, top_k)
        except Exception as exc:
            raise KeywordStoreError("Failed to query Postgres keyword store.") from exc

    def count_sources(self) -> int:
        try:
            return self.repository.count_sources()
        except Exception as exc:
            raise KeywordStoreError("Failed to count sources in Postgres keyword store.") from exc

    def count_chunks(self) -> int:
        try:
            return self.repository.count_chunks()
        except Exception as exc:
            raise KeywordStoreError("Failed to count chunks in Postgres keyword store.") from exc

    def list_sources(self) -> list[str]:
        try:
            return self.repository.list_sources()
        except Exception as exc:
            raise KeywordStoreError("Failed to list sources from Postgres keyword store.") from exc
