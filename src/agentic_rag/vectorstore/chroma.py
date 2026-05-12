from pathlib import Path
from typing import Any

import chromadb

from agentic_rag.config import Settings, load_settings
from agentic_rag.core import Chunk, ModelError, SearchResult, VectorStoreError
from agentic_rag.models import EmbeddingModel, OllamaEmbeddingModel
from agentic_rag.vectorstore.base import VectorStore


class ChromaVectorStore(VectorStore):
    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.embedding_model = embedding_model or OllamaEmbeddingModel(self.settings)
        self.persist_dir = Path(self.settings.chroma_persist_dir)

        try:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(self.persist_dir))
            self.collection = self.client.get_or_create_collection(
                name=self.settings.chroma_collection,
            )
        except Exception as exc:
            raise VectorStoreError("Failed to initialize Chroma vector store.") from exc

    def add_documents(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        try:
            embeddings = self.embedding_model.embed_documents([chunk.content for chunk in chunks])
            self.collection.upsert(
                ids=[chunk.id for chunk in chunks],
                documents=[chunk.content for chunk in chunks],
                embeddings=embeddings,
                metadatas=[_sanitize_metadata(chunk.metadata) for chunk in chunks],
            )
        except ModelError:
            raise
        except Exception as exc:
            raise VectorStoreError("Failed to add documents to Chroma.") from exc

    def similarity_search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            raise VectorStoreError("top_k must be greater than 0.")

        try:
            query_embedding = self.embedding_model.embed_query(query)
            raw = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except ModelError:
            raise
        except Exception as exc:
            raise VectorStoreError("Failed to query Chroma.") from exc

        return self._parse_query_results(raw)

    def estimate_confidence(self, results: list[SearchResult]) -> str:
        if not results:
            return "low"

        best_score = max((result.score or 0.0) for result in results)
        if best_score >= 0.75:
            return "high"
        if best_score >= 0.45:
            return "medium"
        return "low"

    def _parse_query_results(self, raw: dict[str, Any]) -> list[SearchResult]:
        ids = _first_result_list(raw.get("ids"))
        documents = _first_result_list(raw.get("documents"))
        metadatas = _first_result_list(raw.get("metadatas"))
        distances = _first_result_list(raw.get("distances"))

        results: list[SearchResult] = []
        for index, result_id in enumerate(ids):
            metadata = dict(metadatas[index] or {}) if index < len(metadatas) else {}
            distance = distances[index] if index < len(distances) else None
            if distance is not None:
                metadata["raw_distance"] = distance

            results.append(
                SearchResult(
                    id=str(result_id),
                    content=str(documents[index]) if index < len(documents) else "",
                    score=_normalize_distance(distance),
                    source=metadata.get("source") if isinstance(metadata.get("source"), str) else None,
                    metadata=metadata,
                )
            )

        return results


def _first_result_list(value: Any) -> list[Any]:
    if not value:
        return []
    if isinstance(value, list) and value and isinstance(value[0], list):
        return value[0]
    if isinstance(value, list):
        return value
    return []


def _normalize_distance(distance: Any) -> float | None:
    if distance is None:
        return None
    try:
        score = 1.0 / (1.0 + float(distance))
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, score))


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    sanitized: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        else:
            sanitized[key] = str(value)
    return sanitized
