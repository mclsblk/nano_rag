from abc import ABC, abstractmethod

from agentic_rag.core import Chunk, SearchResult


class VectorStore(ABC):
    @abstractmethod
    def add_documents(self, chunks: list[Chunk]) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete_by_file(self, file_id: str, collection_id: str) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_by_ids(self, ids: list[str]) -> list[SearchResult]:
        raise NotImplementedError

    @abstractmethod
    def count_chunks(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def list_sources(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def similarity_search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        raise NotImplementedError
