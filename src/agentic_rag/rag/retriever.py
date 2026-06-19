from collections.abc import Callable
from typing import Any

from agentic_rag.config import Settings
from agentic_rag.core import (
    PUBLIC_METADATA_KEYS,
    IndexConsistencyError,
    SearchDebugResponse,
    SearchResponse,
    SearchResult,
)
from agentic_rag.core.exceptions import ConfigurationError
from agentic_rag.keyword import KeywordStore
from agentic_rag.vectorstore import VectorStore


_KEYWORD_METADATA_KEYS = (
    "keyword_score",
    "raw_bm25",
    "keyword_rank",
    "keyword_matched_tokens",
    "keyword_content_matched_tokens",
    "keyword_required_coverage",
    "keyword_optional_coverage",
    "keyword_exact_token_bonus",
    "keyword_bm25_signal",
    "keyword_structure_bonus",
)
_DEBUG_EVIDENCE_KEYS = (
    "retrieval_mode",
    "hybrid_score",
    "vector_score",
    "keyword_score",
    "raw_distance",
    *_KEYWORD_METADATA_KEYS,
)
_SCORE_BREAKDOWN_KEYS = ("hybrid_score", "vector_score", "keyword_score", "raw_distance")


class Retriever:
    def __init__(self, vectorstore: VectorStore) -> None:
        self.vectorstore = vectorstore

    def search(self, query: str, top_k: int = 5) -> SearchResponse:
        results = [_as_vector_result(result) for result in self.vectorstore.similarity_search(query, top_k=top_k)]
        return SearchResponse(query=query, results=results)

    def search_debug(
        self,
        query: str,
        *,
        collection_id: str,
        top_k: int = 5,
        include_content: bool = False,
    ) -> SearchDebugResponse:
        results = [_as_vector_result(result) for result in self.vectorstore.similarity_search(query, top_k=top_k)]
        return SearchDebugResponse(
            query=query,
            collection_id=collection_id,
            top_k=top_k,
            strategy="vector",
            results=[_debug_result_data(result, include_content) for result in results],
            diagnostics={
                "vector_candidates": [_debug_result_data(result, include_content) for result in results],
                "keyword_candidates": [],
            },
        )

    def estimate_confidence(self, results: list[SearchResult]) -> str:
        return _estimate_confidence(results)


class KeywordRetriever:
    def __init__(self, keyword_store: KeywordStore) -> None:
        self.keyword_store = keyword_store

    def search(self, query: str, top_k: int = 5) -> SearchResponse:
        results = [
            _as_keyword_result(result, rank)
            for rank, result in enumerate(self.keyword_store.keyword_search(query, top_k=top_k), start=1)
        ]
        return SearchResponse(query=query, results=results)

    def search_debug(
        self,
        query: str,
        *,
        collection_id: str,
        top_k: int = 5,
        include_content: bool = False,
    ) -> SearchDebugResponse:
        results = [
            _as_keyword_result(result, rank)
            for rank, result in enumerate(self.keyword_store.keyword_search(query, top_k=top_k), start=1)
        ]
        return SearchDebugResponse(
            query=query,
            collection_id=collection_id,
            top_k=top_k,
            strategy="keyword",
            results=[_debug_result_data(result, include_content) for result in results],
            diagnostics={
                "vector_candidates": [],
                "keyword_candidates": [_debug_result_data(result, include_content) for result in results],
            },
        )

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
        scored, _vector_results, _keyword_results, _candidate_k = self._search_candidates(query, top_k)
        return SearchResponse(query=query, results=scored[:top_k])

    def search_debug(
        self,
        query: str,
        *,
        collection_id: str,
        top_k: int = 5,
        include_content: bool = False,
    ) -> SearchDebugResponse:
        scored, vector_results, keyword_results, candidate_k = self._search_candidates(query, top_k)
        vector_candidates = [_as_vector_result(result) for result in vector_results]
        keyword_candidates = [
            _as_keyword_result(result, rank)
            for rank, result in enumerate(keyword_results, start=1)
        ]
        return SearchDebugResponse(
            query=query,
            collection_id=collection_id,
            top_k=top_k,
            strategy="hybrid",
            results=[_debug_result_data(result, include_content) for result in scored[:top_k]],
            diagnostics={
                "candidate_k": candidate_k,
                "vector_weight": self.vector_weight,
                "keyword_weight": 1.0 - self.vector_weight,
                "vector_candidates": [_debug_result_data(result, include_content) for result in vector_candidates],
                "keyword_candidates": [_debug_result_data(result, include_content) for result in keyword_candidates],
            },
        )

    def estimate_confidence(self, results: list[SearchResult]) -> str:
        return _estimate_confidence(results)

    def _search_candidates(
        self,
        query: str,
        top_k: int,
    ) -> tuple[list[SearchResult], list[SearchResult], list[SearchResult], int]:
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
            entry["keyword_score"] = result.score
            entry["keyword_metadata"] = _keyword_metadata(result, rank)
            if "vector_rank" not in entry:
                entry["result"] = result

        scored = [_hybrid_result(candidate, self.vector_weight) for candidate in candidates.values()]
        scored.sort(key=lambda result: result.score or 0.0, reverse=True)
        return scored, vector_results, keyword_results, candidate_k

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


def create_retriever(
    settings: Settings,
    vectorstore_factory: Callable[[], VectorStore],
    keyword_store_factory: Callable[[], KeywordStore],
) -> Retriever | KeywordRetriever | HybridRetriever:
    search = settings.search
    strategy = search.strategy.strip().lower().replace("-", "_")

    if strategy == "vector":
        return Retriever(vectorstore_factory())
    if strategy == "keyword":
        return KeywordRetriever(keyword_store_factory())
    if strategy == "hybrid":
        return HybridRetriever(
            vectorstore=vectorstore_factory(),
            keyword_store=keyword_store_factory(),
            vector_weight=search.hybrid_vector_weight,
            candidate_multiplier=search.hybrid_candidate_multiplier,
        )

    raise ConfigurationError(
        f"Unsupported search strategy: {search.strategy}. Supported strategies: vector, keyword, hybrid."
    )


def _as_vector_result(result: SearchResult) -> SearchResult:
    metadata = dict(result.metadata)
    raw_distance = metadata.pop("raw_distance", None)
    retrieval = dict(result.retrieval)
    retrieval.update({"retrieval_mode": "vector", "vector_score": result.score})
    if raw_distance is not None:
        retrieval["raw_distance"] = raw_distance
    return result.model_copy(update={"metadata": metadata, "retrieval": retrieval})


def _as_keyword_result(result: SearchResult, rank: int) -> SearchResult:
    metadata, keyword_data = _pop_keyword_data(result.metadata)
    retrieval = dict(result.retrieval)
    retrieval.update({"retrieval_mode": "keyword", "keyword_rank": rank})
    retrieval.update(keyword_data)
    if result.score is not None:
        retrieval["keyword_score"] = result.score
    return result.model_copy(update={"metadata": metadata, "retrieval": retrieval})


def _hybrid_result(candidate: dict[str, Any], vector_weight: float) -> SearchResult:
    result: SearchResult = candidate["result"]
    metadata, _keyword_data = _pop_keyword_data(result.metadata)
    raw_distance = metadata.pop("raw_distance", None)
    vector_score = candidate.get("vector_score", 0.0)
    keyword_score = candidate.get("keyword_score", 0.0)
    keyword_weight = 1.0 - vector_weight

    hybrid_score = vector_weight * vector_score + keyword_weight * keyword_score

    retrieval = dict(result.retrieval)
    retrieval.update({"retrieval_mode": "hybrid", "hybrid_score": hybrid_score})
    if raw_distance is not None:
        retrieval["raw_distance"] = raw_distance
    if vector_score is not None:
        retrieval["vector_score"] = vector_score
    if keyword_score is not None:
        retrieval["keyword_score"] = keyword_score
    if vector_rank := candidate.get("vector_rank"):
        retrieval["vector_rank"] = vector_rank
    retrieval.update(candidate.get("keyword_metadata", {}))

    return result.model_copy(update={"metadata": metadata, "retrieval": retrieval, "score": hybrid_score})


def _debug_result_data(result: SearchResult, include_content: bool) -> dict[str, Any]:
    retrieval = {key: result.retrieval[key] for key in _DEBUG_EVIDENCE_KEYS if key in result.retrieval}
    score_breakdown = {key: result.retrieval[key] for key in _SCORE_BREAKDOWN_KEYS if key in result.retrieval}
    if result.score is not None:
        score_breakdown["score"] = result.score

    data: dict[str, Any] = {
        "id": result.id,
        "score": result.score,
        "source": result.source,
        "metadata": {key: result.metadata[key] for key in PUBLIC_METADATA_KEYS if key in result.metadata},
        "retrieval": retrieval,
        "score_breakdown": score_breakdown,
    }
    if include_content:
        data["content"] = result.content
    return data


def _keyword_metadata(result: SearchResult, rank: int) -> dict[str, Any]:
    _metadata, metadata = _pop_keyword_data(result.metadata)
    metadata["keyword_rank"] = rank
    if result.score is not None:
        metadata["keyword_score"] = result.score
    return metadata


def _pop_keyword_data(metadata: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    cleaned = dict(metadata)
    keyword_data: dict[str, Any] = {}
    for key in _KEYWORD_METADATA_KEYS:
        value = cleaned.pop(key, None)
        if value is not None:
            keyword_data[key] = value
    return cleaned, keyword_data


def _estimate_confidence(results: list[SearchResult]) -> str:
    if not results:
        return "low"

    best_score = max((result.score or 0.0) for result in results)
    if best_score >= 0.75:
        return "high"
    if best_score >= 0.45:
        return "medium"
    return "low"
