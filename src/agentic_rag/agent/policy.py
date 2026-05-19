from agentic_rag.config import Settings
from agentic_rag.core import SearchResult
from agentic_rag.rag.prompts import FALLBACK_ANSWER


NO_RESULTS = "no_results"
BEST_SCORE_BELOW_MIN_SCORE = "best_score_below_min_score"
CONTENT_CHARS_BELOW_MIN_CHARS = "content_chars_below_min_chars"


class AgentPolicy:
    def __init__(self, min_score: float = 0.45, min_chars: int = 80) -> None:
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between 0.0 and 1.0.")
        if min_chars < 0:
            raise ValueError("min_chars must be greater than or equal to 0.")

        self.min_score = min_score
        self.min_chars = min_chars

    @classmethod
    def from_settings(cls, settings: Settings) -> "AgentPolicy":
        agentic = settings.agentic
        return cls(
            min_score=agentic.context_min_score,
            min_chars=agentic.context_min_chars,
        )

    def is_context_sufficient(self, results: list[SearchResult]) -> bool:
        return self.fallback_reason(results) is None

    def fallback_reason(self, results: list[SearchResult]) -> str | None:
        selected_results = self.select_results_for_context(results)
        if not selected_results:
            return NO_RESULTS

        best_score = max((result.score or 0.0) for result in selected_results)
        if best_score < self.min_score:
            return BEST_SCORE_BELOW_MIN_SCORE

        content_chars = sum(len(result.content.strip()) for result in selected_results)
        if content_chars < self.min_chars:
            return CONTENT_CHARS_BELOW_MIN_CHARS

        return None

    def fallback_answer(self) -> str:
        return FALLBACK_ANSWER

    def select_results_for_context(self, results: list[SearchResult]) -> list[SearchResult]:
        return sorted(
            [result for result in results if result.content.strip()],
            key=lambda result: result.score or 0.0,
            reverse=True,
        )
