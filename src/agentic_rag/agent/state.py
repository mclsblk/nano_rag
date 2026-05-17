from dataclasses import dataclass, field

from agentic_rag.core import SearchResult


@dataclass
class AgentState:
    query: str
    rewritten_query: str = ""
    retrieval_queries: list[str] = field(default_factory=list)
    results: list[SearchResult] = field(default_factory=list)
    answer: str = ""
    confidence: str | None = None
    context_sufficient: bool = False
    fallback_reason: str | None = None
    selected_source_ids: list[str] = field(default_factory=list)
