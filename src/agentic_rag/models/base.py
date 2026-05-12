from abc import ABC, abstractmethod
from typing import Any


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
