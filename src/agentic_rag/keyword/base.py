from typing import Protocol

from agentic_rag.core import Chunk, Document, SearchResult


class KeywordStore(Protocol):
    def add_documents(self, documents: list[Document]) -> None:
        raise NotImplementedError

    def add_chunks(self, chunks: list[Chunk]) -> None:
        raise NotImplementedError

    def delete_by_file_id(self, file_id: str) -> int:
        raise NotImplementedError

    def keyword_search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        raise NotImplementedError

    def count_sources(self) -> int:
        raise NotImplementedError

    def count_chunks(self) -> int:
        raise NotImplementedError

    def list_sources(self) -> list[str]:
        raise NotImplementedError
