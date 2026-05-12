from agentic_rag.core import SearchResponse, SearchResult
from agentic_rag.vectorstore import VectorStore


class Retriever:
    def __init__(self, vectorstore: VectorStore) -> None:
        self.vectorstore = vectorstore

    def search(self, query: str, top_k: int = 5) -> SearchResponse:
        results = self.vectorstore.similarity_search(query, top_k=top_k)
        return SearchResponse(query=query, results=results)

    def estimate_confidence(self, results: list[SearchResult]) -> str:
        return self.vectorstore.estimate_confidence(results)
