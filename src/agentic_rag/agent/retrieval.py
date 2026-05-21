from agentic_rag.core import SearchResult
from agentic_rag.rag import Retriever


class MultiQueryRetriever:
    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever

    def search(self, queries: list[str], top_k: int = 5, max_results: int | None = None) -> list[SearchResult]:
        merged: dict[str, SearchResult] = {}

        for query in _dedupe_queries(queries):
            for result in self.retriever.search(query, top_k=top_k).results:
                existing = merged.get(result.id)
                if existing is None or _score_value(result.score) > _score_value(existing.score):
                    merged[result.id] = result

        results = sorted(merged.values(), key=lambda result: result.score or 0.0, reverse=True)
        if max_results is not None:
            return results[:max_results]
        return results


def _dedupe_queries(queries: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()

    for query in queries:
        cleaned = query.strip()
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)

    return deduped


def _score_value(score: float | None) -> float:
    return score if score is not None else -1.0
