from typing import Any

from agentic_rag.core import RAGPipelineError, SearchResult
from agentic_rag.models import ChatModel
from agentic_rag.rag.prompts import FALLBACK_ANSWER, RAG_SYSTEM_PROMPT, RAG_USER_PROMPT


class Generator:
    def __init__(self, chat_model: ChatModel) -> None:
        self.chat_model = chat_model

    def generate(self, query: str, results: list[SearchResult]) -> str:
        if not results:
            return FALLBACK_ANSWER

        context = format_context(results)
        response = self.chat_model.chat(
            [
                {"role": "system", "content": RAG_SYSTEM_PROMPT.strip()},
                {"role": "user", "content": RAG_USER_PROMPT.format(query=query, context=context).strip()},
            ]
        )
        answer = _extract_answer(response)
        return answer or FALLBACK_ANSWER


def format_context(results: list[SearchResult]) -> str:
    sections: list[str] = []
    for index, result in enumerate(results, start=1):
        source = result.source or _metadata_string(result.metadata, "source") or "unknown"
        page = _page_label(result.metadata)
        sections.append(
            "\n".join(
                [
                    f"[{index}] [source: {source} | page: {page}]",
                    result.content,
                ]
            )
        )
    return "\n\n".join(sections)


def _extract_answer(response: dict[str, Any]) -> str:
    message = response.get("message")
    if not isinstance(message, dict):
        raise RAGPipelineError("Chat response missing message object.")

    content = message.get("content")
    if not isinstance(content, str):
        raise RAGPipelineError("Chat response message missing content.")

    return content.strip()


def _metadata_string(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    return value if isinstance(value, str) else None


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
