from agentic_rag.core import AnswerResponse, SearchResponse
from agentic_rag.rag.generator import Generator
from agentic_rag.rag.retriever import Retriever


class RAGPipeline:
    def __init__(self, retriever: Retriever, generator: Generator) -> None:
        self.retriever = retriever
        self.generator = generator

    def search(self, query: str, top_k: int = 5) -> SearchResponse:
        return self.retriever.search(query, top_k=top_k)

    def ask(self, query: str, top_k: int = 5) -> AnswerResponse:
        search_response = self.search(query, top_k=top_k)
        answer = self.generator.generate(query, search_response.results)
        confidence = self.retriever.estimate_confidence(search_response.results)
        return AnswerResponse(
            query=query,
            answer=answer,
            sources=search_response.results,
            confidence=confidence,
        )
