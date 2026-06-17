from dataclasses import asdict
import json
from typing import Any

from pydantic import BaseModel

from agentic_rag.agent.service import AgenticAskResult
from agentic_rag.core import (
    AnswerResponse,
    DeIngestResponse,
    IngestResponse,
    InspectResponse,
    OutputFormatError,
    SearchResponse,
    SearchResult,
)


_PUBLIC_METADATA_KEYS = (
    "source", "file_name", "file_type", "page", "page_count",
    "page_end", "page_index", "page_number", "page_start",
)


class OutputFormatter:
    def __init__(self, content_preview_chars: int | None = None) -> None:
        if content_preview_chars is not None and content_preview_chars <= 0:
            raise OutputFormatError("content_preview_chars must be greater than 0.")
        self.content_preview_chars = content_preview_chars

    def format_response(
        self,
        response: BaseModel | AgenticAskResult,
        as_json: bool = False,
        debug: bool = False,
    ) -> str:
        if isinstance(response, AgenticAskResult):
            return self.format_agentic_ask_result(response, as_json=as_json, debug=debug)

        if as_json:
            return self.format_json(response, debug=debug)

        if isinstance(response, SearchResponse):
            return self.format_search(response)
        if isinstance(response, AnswerResponse):
            return self.format_answer(response)
        if isinstance(response, IngestResponse):
            return self.format_ingest(response)
        if isinstance(response, DeIngestResponse):
            return self.format_de_ingest(response)
        if isinstance(response, InspectResponse):
            return self.format_inspect(response)

        raise OutputFormatError(f"Unsupported response type: {type(response).__name__}")

    def format_json(self, response: BaseModel, debug: bool = False) -> str:
        data = response.model_dump(mode="json")
        if not debug:
            data = _public_json_data(data)
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    def format_agentic_ask_result(
        self,
        response: AgenticAskResult,
        as_json: bool = False,
        debug: bool = False,
    ) -> str:
        if as_json:
            data = response.response.model_dump(mode="json")
            if not debug:
                data = _public_json_data(data)
            data["debug"] = asdict(response.debug)
            return json.dumps(data, ensure_ascii=False, separators=(",", ":"))

        lines = [
            self.format_answer(response.response),
            "",
            "Debug:",
            f"rewritten_query={response.debug.rewritten_query}",
            f"retrieval_queries={response.debug.retrieval_queries}",
            f"context_sufficient={response.debug.context_sufficient}",
            f"fallback_reason={response.debug.fallback_reason or 'n/a'}",
            f"selected_source_ids={response.debug.selected_source_ids}",
        ]
        return "\n".join(lines)

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

    def format_ingest(self, response: IngestResponse) -> str:
        lines = [
            f"Path: {response.path}",
            f"Loaded documents(pages): {response.loaded_documents}",
            f"Generated chunks: {response.generated_chunks}",
            f"Stored chunks: {response.stored_chunks}",
            f"Skipped: {len(response.skipped)}",
        ]

        for index, message in enumerate(response.skipped, start=1):
            lines.append(f"{index}. {message}")

        return "\n".join(lines)

    def format_de_ingest(self, response: DeIngestResponse) -> str:
        return "\n".join(
            [
                f"Source: {response.source}",
                f"Deleted chunks: {response.deleted_chunks}",
            ]
        )

    def format_inspect(self, response: InspectResponse) -> str:
        lines = [
            f"model_provider={response.model_provider}",
            f"chat_model_provider={response.chat_model_provider}",
            f"embedding_model_provider={response.embedding_model_provider}",
            f"ollama_base_url={response.ollama_base_url}",
            f"ollama_chat_model={response.ollama_chat_model}",
            f"ollama_embedding_model={response.ollama_embedding_model}",
            f"ollama_timeout_seconds={response.ollama_timeout_seconds}",
            f"ollama_think={response.ollama_think}",
            f"openai_compatible_base_url={response.openai_compatible_base_url}",
            f"openai_compatible_chat_model={response.openai_compatible_chat_model}",
            f"openai_compatible_embedding_model={response.openai_compatible_embedding_model}",
            f"openai_compatible_visual_model={response.openai_compatible_visual_model}",
            f"openai_compatible_timeout_seconds={response.openai_compatible_timeout_seconds}",
            f"document_load_strategy={response.document_load_strategy}",
            f"visual_model_provider={response.visual_model_provider}",
            f"visual_min_text_chars={response.visual_min_text_chars}",
            f"chroma_persist_dir={response.chroma_persist_dir}",
            f"chroma_collection={response.chroma_collection}",
            f"chroma_count={response.chroma_count}",
            f"search_strategy={response.search_strategy}",
            f"keyword_index_path={response.keyword_index_path}",
            f"keyword_source_count={response.keyword_source_count}",
            f"keyword_chunk_count={response.keyword_chunk_count}",
            f"agentic_engine={response.agentic_engine}",
        ]
        return "\n".join(lines)


def _source_label(result: SearchResult) -> str:
    if result.source:
        return result.source
    source = result.metadata.get("source")
    return source if isinstance(source, str) else "n/a"


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


def _preview_content(content: str, limit: int | None) -> str:
    if limit is None or len(content) <= limit:
        return content
    return f"{content[:limit].rstrip()}..."


def _public_json_data(value: Any) -> Any:
    if isinstance(value, list):
        return [_public_json_data(item) for item in value]
    if not isinstance(value, dict):
        return value

    data = {key: _public_json_data(item) for key, item in value.items()}
    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        data["metadata"] = {key: metadata[key] for key in _PUBLIC_METADATA_KEYS if key in metadata}
    return data
