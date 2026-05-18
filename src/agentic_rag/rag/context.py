from typing import Any

from agentic_rag.core import SearchResult


class ContextBuilder:
    def __init__(self, max_chars: int = 4000) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be greater than 0.")
        self.max_chars = max_chars

    def build(self, results: list[SearchResult]) -> str:
        sections: list[str] = []
        current_chars = 0

        for index, result in enumerate(_sorted_results_with_content(results), start=1):
            content = result.content.strip()
            section = "\n".join(
                [
                    f"[{index}] [source: {_source_label(result)} | page: {_page_label(result.metadata)} | score: {_score_label(result.score)}]",
                    content,
                ]
            )
            separator_chars = 2 if sections else 0
            remaining_chars = self.max_chars - current_chars - separator_chars
            if remaining_chars <= 0:
                break

            if len(section) > remaining_chars:
                label, _, section_content = section.partition("\n")
                content_budget = remaining_chars - len(label) - 1
                if content_budget <= 0:
                    break
                section = "\n".join([label, section_content[:content_budget].rstrip()])

            sections.append(section)
            current_chars += separator_chars + len(section)

            if current_chars >= self.max_chars:
                break

        return "\n\n".join(sections)


def _sorted_results_with_content(results: list[SearchResult]) -> list[SearchResult]:
    return sorted(
        [result for result in results if result.content.strip()],
        key=lambda result: result.score or 0.0,
        reverse=True,
    )


def _source_label(result: SearchResult) -> str:
    if result.source:
        return result.source
    source = result.metadata.get("source")
    return source if isinstance(source, str) else "unknown"


def _page_label(metadata: dict[str, Any]) -> str:
    page_start = metadata.get("page_start")
    page_end = metadata.get("page_end")
    if isinstance(page_start, int) and isinstance(page_end, int):
        if page_start == page_end:
            return str(page_start)
        return f"{page_start}-{page_end}"
    if "page_number" in metadata:
        return str(metadata["page_number"])
    if "page" in metadata:
        return str(metadata["page"])
    if isinstance(metadata.get("page_index"), int):
        return str(metadata["page_index"] + 1)
    if metadata.get("page_count") == 1:
        return "1"
    return "n/a"


def _score_label(score: float | None) -> str:
    if score is None:
        return "n/a"
    return f"{score:.3f}"
