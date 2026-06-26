from typing import Any


def build_report(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [result for result in case_results if result.get("error") is None]
    return {
        "summary": {
            "case_count": len(case_results),
            "error_count": len(case_results) - len(valid),
            "recall_at_k": _mean_number(valid, "recall_at_k"),
            "mrr": _mean_number(valid, "mrr"),
            "source_hit_rate": _mean_bool(valid, "source_hit"),
            "file_hit_rate": _mean_bool(valid, "file_hit"),
            "keyword_hit_rate": _mean_bool(valid, "keyword_hit"),
            "retrieval_evidence_coverage": _mean_bool(valid, "retrieval_evidence_present"),
            "avg_latency_ms": _mean_number(valid, "latency_ms"),
        },
        "cases": case_results,
    }


def _mean_number(results: list[dict[str, Any]], key: str) -> float | None:
    values = [float(value) for result in results if isinstance((value := result.get(key)), (int, float))]
    if not values:
        return None
    return sum(values) / len(values)


def _mean_bool(results: list[dict[str, Any]], key: str) -> float | None:
    values = [value for result in results if isinstance((value := result.get(key)), bool)]
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)
