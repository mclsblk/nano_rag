from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalCase:
    id: str
    query: str
    collection_id: str
    expected_sources: list[str]
    expected_file_ids: list[str]
    expected_keywords: list[str]
    top_k: int | None = None


def load_cases(path: str | Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"Line {line_number} is empty.")
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Line {line_number} is not valid JSON.") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"Line {line_number} must be a JSON object.")
        cases.append(_case_from_dict(raw, line_number))
    return cases


def _case_from_dict(raw: dict[str, Any], line_number: int) -> EvalCase:
    case_id = _required_string(raw, "id", line_number)
    query = _required_string(raw, "query", line_number)
    collection_id = _required_string(raw, "collection_id", line_number)
    top_k = raw.get("top_k")
    if top_k is not None:
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"Line {line_number} field top_k must be a positive integer.")

    return EvalCase(
        id=case_id,
        query=query,
        collection_id=collection_id,
        expected_sources=_string_list(raw, "expected_sources", line_number),
        expected_file_ids=_string_list(raw, "expected_file_ids", line_number),
        expected_keywords=_string_list(raw, "expected_keywords", line_number),
        top_k=top_k,
    )


def _required_string(raw: dict[str, Any], key: str, line_number: int) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Line {line_number} field {key} must be a non-empty string.")
    return value


def _string_list(raw: dict[str, Any], key: str, line_number: int) -> list[str]:
    value = raw.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Line {line_number} field {key} must be a list of strings.")
    return value
