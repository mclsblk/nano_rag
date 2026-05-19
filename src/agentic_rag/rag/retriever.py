from typing import Any

from agentic_rag.core import IndexConsistencyError, SearchResponse, SearchResult
from agentic_rag.keyword import KeywordStore
from agentic_rag.vectorstore import VectorStore


class Retriever:
    def __init__(self, vectorstore: VectorStore) -> None:
        self.vectorstore = vectorstore

    def search(self, query: str, top_k: int = 5) -> SearchResponse:
        results = [_as_vector_result(result) for result in self.vectorstore.similarity_search(query, top_k=top_k)]
        return SearchResponse(query=query, results=results)

    def estimate_confidence(self, results: list[SearchResult]) -> str:
        return _estimate_confidence(results)


class KeywordRetriever:
    def __init__(self, keyword_store: KeywordStore) -> None:
        self.keyword_store = keyword_store

    def search(self, query: str, top_k: int = 5) -> SearchResponse:
        return SearchResponse(query=query, results=self.keyword_store.keyword_search(query, top_k=top_k))

    def estimate_confidence(self, results: list[SearchResult]) -> str:
        return _estimate_confidence(results)


class HybridRetriever:
    def __init__(
        self,
        vectorstore: VectorStore,
        keyword_store: KeywordStore,
        vector_weight: float = 0.65,
        candidate_multiplier: int = 4,
    ) -> None:
        self.vectorstore = vectorstore
        self.keyword_store = keyword_store
        self.vector_weight = vector_weight
        self.candidate_multiplier = candidate_multiplier

    def search(self, query: str, top_k: int = 5) -> SearchResponse:
        self._check_index_consistency()
        candidate_k = max(top_k, top_k * self.candidate_multiplier)
        vector_results = self.vectorstore.similarity_search(query, top_k=candidate_k)
        keyword_results = self.keyword_store.keyword_search(query, top_k=candidate_k)

        candidates: dict[str, dict[str, Any]] = {}
        for rank, result in enumerate(vector_results, start=1):
            entry = candidates.setdefault(result.id, {"result": result})
            entry["vector_rank"] = rank
            entry["vector_score"] = result.score

        for rank, result in enumerate(keyword_results, start=1):
            entry = candidates.setdefault(result.id, {"result": result})
            entry["keyword_rank"] = rank
            entry["keyword_score"] = result.score
            entry["raw_bm25"] = result.metadata.get("raw_bm25")
            if "vector_rank" not in entry:
                entry["result"] = result

        scored = [_hybrid_result(candidate, self.vector_weight) for candidate in candidates.values()]
        scored.sort(key=lambda result: result.score or 0.0, reverse=True)
        return SearchResponse(query=query, results=scored[:top_k])

    def estimate_confidence(self, results: list[SearchResult]) -> str:
        return _estimate_confidence(results)

    def _check_index_consistency(self) -> None:
        vector_sources = set(self.vectorstore.list_sources())
        keyword_sources = set(self.keyword_store.list_sources())
        vector_count = self.vectorstore.count_chunks()
        keyword_count = self.keyword_store.count_chunks()

        if vector_sources == keyword_sources and vector_count == keyword_count:
            return

        missing_keyword = sorted(vector_sources - keyword_sources)
        missing_vector = sorted(keyword_sources - vector_sources)
        details = [
            f"chroma_chunks={vector_count}",
            f"keyword_chunks={keyword_count}",
        ]
        if missing_keyword:
            details.append(f"missing_keyword_sources={missing_keyword}")
        if missing_vector:
            details.append(f"missing_vector_sources={missing_vector}")

        raise IndexConsistencyError(
            "Hybrid search requires Chroma and SQLite keyword indexes to contain the same sources and chunks. "
            f"{'; '.join(details)}. Run `rag de-ingest <source>` and ingest the source again."
        )


def _as_vector_result(result: SearchResult) -> SearchResult:
    metadata = dict(result.metadata)
    metadata.update({"retrieval_mode": "vector", "vector_score": result.score})
    return result.model_copy(update={"metadata": metadata})


def _hybrid_result(candidate: dict[str, Any], vector_weight: float) -> SearchResult:
    result: SearchResult = candidate["result"]
    metadata = dict(result.metadata)
    vector_score = candidate.get("vector_score", 0.0)
    keyword_score = candidate.get("keyword_score", 0.0)
    keyword_weight = 1.0 - vector_weight

    hybrid_score = vector_weight * vector_score + keyword_weight * keyword_score

    metadata.update({"retrieval_mode": "hybrid", "hybrid_score": hybrid_score})
    if vector_score is not None:
        metadata["vector_score"] = vector_score
    if keyword_score is not None:
        metadata["keyword_score"] = keyword_score
    if candidate.get("raw_bm25") is not None:
        metadata["raw_bm25"] = candidate["raw_bm25"]

    return result.model_copy(update={"metadata": metadata, "score": hybrid_score})


def _estimate_confidence(results: list[SearchResult]) -> str:
    if not results:
        return "low"

    best_score = max((result.score or 0.0) for result in results)
    if best_score >= 0.75:
        return "high"
    if best_score >= 0.45:
        return "medium"
    return "low"
