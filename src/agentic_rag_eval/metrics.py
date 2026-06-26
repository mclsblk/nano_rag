from typing import Any

from agentic_rag_eval.cases import EvalCase


def score_case(
    case: EvalCase,
    search_response: dict[str, Any] | None = None,
    debug_response: dict[str, Any] | None = None,
    latency_ms: float | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    top_results = []
    debug_evidence = []
    source_rank = file_rank = keyword_rank = None

    if error is None:
        results = [item for item in (search_response or {}).get("results", []) if isinstance(item, dict)]
        debug_results = [item for item in (debug_response or {}).get("results", []) if isinstance(item, dict)]
        debug_by_id = {item.get("id"): item for item in debug_results}

        for rank, result in enumerate(results, start=1):
            metadata = _dict(result.get("metadata"))
            source = str(result.get("source") or metadata.get("source") or "")
            file_id = metadata.get("file_id")
            retrieval = _dict(debug_by_id.get(result.get("id"), {}).get("retrieval"))
            tokens = [str(token).lower() for token in retrieval.get("keyword_matched_tokens", [])]
            content = str(result.get("content") or "").lower()

            if source_rank is None and case.expected_sources:
                source_rank = _rank_if(any(_matches(source, expected) for expected in case.expected_sources), rank)
            if file_rank is None and case.expected_file_ids:
                file_rank = _rank_if(file_id in set(case.expected_file_ids), rank)
            if keyword_rank is None and case.expected_keywords:
                keyword_rank = _rank_if(
                    all(keyword.lower() in content or keyword.lower() in tokens for keyword in case.expected_keywords),
                    rank,
                )

            top_results.append({"rank": rank, "id": result.get("id"), "source": source, "file_id": file_id, "score": result.get("score")})
            debug_evidence.append({"rank": rank, "id": result.get("id"), "source": source, "score": result.get("score"), "retrieval": retrieval})

    first_rank = min([rank for rank in (source_rank, file_rank, keyword_rank) if rank is not None], default=None)
    return {
        "id": case.id,
        "query": case.query,
        "collection_id": case.collection_id,
        "expected": {"sources": case.expected_sources, "file_ids": case.expected_file_ids, "keywords": case.expected_keywords},
        "source_hit": None if not case.expected_sources else source_rank is not None,
        "file_hit": None if not case.expected_file_ids else file_rank is not None,
        "keyword_hit": None if not case.expected_keywords else keyword_rank is not None,
        "first_hit_rank": first_rank,
        "recall_at_k": 1.0 if first_rank is not None else 0.0,
        "mrr": (1.0 / first_rank) if first_rank is not None else 0.0,
        "retrieval_evidence_present": any(item["retrieval"] for item in debug_evidence),
        "latency_ms": latency_ms,
        "top_results": top_results,
        "debug_evidence": debug_evidence,
        "error": error,
    }


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _matches(value: str, expected: str) -> bool:
    value_lower = value.lower()
    expected_lower = expected.lower()
    return bool(value_lower) and (expected_lower in value_lower or value_lower in expected_lower)


def _rank_if(condition: bool, rank: int) -> int | None:
    return rank if condition else None
