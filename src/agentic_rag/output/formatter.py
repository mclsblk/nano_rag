from typing import Any

from pydantic import BaseModel

from agentic_rag.core import AnswerResponse, OutputFormatError, SearchResponse, SearchResult


class OutputFormatter:
    def __init__(self, content_preview_chars: int | None = None) -> None:
        if content_preview_chars is not None and content_preview_chars <= 0:
            raise OutputFormatError("content_preview_chars must be greater than 0.")
        self.content_preview_chars = content_preview_chars

    def format_response(self, response: BaseModel, as_json: bool = False) -> str:
        if as_json:
            return self.format_json(response)

        if isinstance(response, SearchResponse):
            return self.format_search(response)
        if isinstance(response, AnswerResponse):
            return self.format_answer(response)

        raise OutputFormatError(f"Unsupported response type: {type(response).__name__}")

    def format_json(self, response: BaseModel) -> str:
        return response.model_dump_json()

    def format_search(self, response: SearchResponse) -> str:
        lines = [
            f"Query: {response.query}",
            f"Results: {len(response.results)}",
        ]

        for index, result in enumerate(response.results, start=1):
            lines.extend(
                [
                    "",
                    f"{index}. source={_source_label(result)} page={_page_label(result.metadata)} score={_score_label(result.score)}",
                    _preview_content(result.content, self.content_preview_chars),
                ]
            )

        return "\n".join(lines)

    def format_answer(self, response: AnswerResponse) -> str:
        lines = [
            f"Query: {response.query}",
            f"Confidence: {response.confidence or 'n/a'}",
            "",
            response.answer,
            "",
            f"Sources: {len(response.sources)}",
        ]

        for index, source in enumerate(response.sources, start=1):
            lines.append(
                f"{index}. source={_source_label(source)} page={_page_label(source.metadata)} score={_score_label(source.score)}"
            )

        return "\n".join(lines)


def _source_label(result: SearchResult) -> str:
    if result.source:
        return result.source
    source = result.metadata.get("source")
    return source if isinstance(source, str) else "n/a"


def _page_label(metadata: dict[str, Any]) -> str:
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


def _preview_content(content: str, limit: int | None) -> str:
    if limit is None or len(content) <= limit:
        return content
    return f"{content[:limit].rstrip()}..."
