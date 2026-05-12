from abc import ABC, abstractmethod

from agentic_rag.core import Chunk, SearchResult


class VectorStore(ABC):
    @abstractmethod
    def add_documents(self, chunks: list[Chunk]) -> None:
        raise NotImplementedError

    @abstractmethod
    def similarity_search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        raise NotImplementedError

    @abstractmethod
    def estimate_confidence(self, results: list[SearchResult]) -> str:
        raise NotImplementedError
