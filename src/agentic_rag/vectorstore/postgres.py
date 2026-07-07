from __future__ import annotations

from agentic_rag.config import Settings, load_settings
from agentic_rag.core import Chunk, ModelError, SearchResult, VectorStoreError
from agentic_rag.file_sys.store import SystemStore
from agentic_rag.models import EmbeddingModel
from agentic_rag.postgres.chunks import PostgresChunkRepository
from agentic_rag.vectorstore.base import VectorStore


class PostgresVectorStore(VectorStore):
    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        settings: Settings | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        if embedding_model is None:
            raise VectorStoreError("PostgresVectorStore requires an explicit embedding model.")
        self.embedding_model = embedding_model
        self.collection_id = collection_name
        self.repository = PostgresChunkRepository(
            SystemStore(
                self.settings.postgres.database_url,
                embedding_dimension=self.settings.postgres.embedding_dimension,
            ),
            collection_id=collection_name,
        )

    def add_documents(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        try:
            embeddings = self.embedding_model.embed_documents([chunk.content for chunk in chunks])
            self.repository.add_embeddings(chunks, embeddings)
        except ModelError:
            raise
        except Exception as exc:
            raise VectorStoreError("Failed to add documents to Postgres.") from exc

    def delete_by_file(self, file_id: str, collection_id: str) -> int:
        try:
            return self.repository.delete_by_file(file_id, collection_id)
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to delete file from Postgres chunks: {file_id} + {collection_id}"
            ) from exc

    def mark_file_indexed(self, file_id: str, collection_id: str) -> int:
        return self.repository.mark_file_indexed(file_id, collection_id)

    def get_by_ids(self, ids: list[str]) -> list[SearchResult]:
        try:
            return self.repository.get_by_ids(ids)
        except Exception as exc:
            raise VectorStoreError("Failed to get chunks from Postgres by id.") from exc

    def count_chunks(self) -> int:
        try:
            return self.repository.count_chunks()
        except Exception as exc:
            raise VectorStoreError("Failed to count chunks in Postgres.") from exc

    def list_sources(self) -> list[str]:
        try:
            return self.repository.list_sources()
        except Exception as exc:
            raise VectorStoreError("Failed to list sources in Postgres.") from exc

    def similarity_search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            raise VectorStoreError("top_k must be greater than 0.")
        try:
            query_embedding = self.embedding_model.embed_query(query)
            return self.repository.vector_search(query_embedding, top_k)
        except ModelError:
            raise
        except Exception as exc:
            raise VectorStoreError("Failed to query Postgres vector search.") from exc
