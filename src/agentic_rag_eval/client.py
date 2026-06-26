import time
from typing import Any

import requests

from agentic_rag_eval.cases import EvalCase


class ApiClient:
    def __init__(self, base_url: str, api_key: str = "", timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def search(self, case: EvalCase, top_k: int) -> tuple[dict[str, Any], float]:
        return self._post(
            "/v1/search",
            {
                "query": case.query,
                "collection_id": case.collection_id,
                "top_k": case.top_k or top_k,
            },
        )

    def search_debug(self, case: EvalCase, top_k: int) -> tuple[dict[str, Any], float]:
        return self._post(
            "/v1/search/debug",
            {
                "query": case.query,
                "collection_id": case.collection_id,
                "top_k": case.top_k or top_k,
            },
        )

    def _post(self, path: str, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
        start = time.perf_counter()
        response = requests.post(
            f"{self.base_url}{path}",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        if not response.ok:
            raise RuntimeError(f"{path} failed with status {response.status_code}: {response.text}")
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError(f"{path} returned a non-object JSON response.")
        return data, latency_ms

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
