from abc import ABC, abstractmethod
from typing import Any

from agentic_rag.core import ModelError


class ChatModel(ABC):
    @abstractmethod
    def chat(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        raise NotImplementedError


class EmbeddingModel(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


def extract_chat_content(response: dict[str, Any]) -> str:
    content = _extract_ollama_content(response)
    if content is not None:
        return content.strip()

    content = _extract_openai_compatible_content(response)
    if content is not None:
        return content.strip()

    raise ModelError("Chat response missing supported message content.")


def _extract_ollama_content(response: dict[str, Any]) -> str | None:
    message = response.get("message")
    if not isinstance(message, dict):
        return None

    content = message.get("content")
    return content if isinstance(content, str) else None


def _extract_openai_compatible_content(response: dict[str, Any]) -> str | None:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return None

    message = first_choice.get("message")
    if not isinstance(message, dict):
        return None

    content = message.get("content")
    return content if isinstance(content, str) else None
