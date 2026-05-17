from dataclasses import dataclass

from agentic_rag.agent.policy import AgentPolicy
from agentic_rag.agent.query import MultiQueryGenerator, QueryRewriter
from agentic_rag.agent.retrieval import MultiQueryRetriever
from agentic_rag.agent.state import AgentState
from agentic_rag.core import AnswerResponse, SearchResult
from agentic_rag.rag import Generator, Retriever


GENERATION_FAILED = "generation_failed"
ANSWER_NOT_SUPPORTED = "answer_not_supported"


@dataclass(frozen=True)
class AgentDebugInfo:
    rewritten_query: str
    retrieval_queries: list[str]
    context_sufficient: bool
    fallback_reason: str | None
    selected_source_ids: list[str]


@dataclass(frozen=True)
class AgenticAskResult:
    response: AnswerResponse
    debug: AgentDebugInfo


class AgenticService:
    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        query_rewriter: QueryRewriter,
        multi_query_generator: MultiQueryGenerator,
        policy: AgentPolicy,
        multi_query_retriever: MultiQueryRetriever | None = None,
        multi_query_count: int = 3,
    ) -> None:
        if multi_query_count <= 0:
            raise ValueError("multi_query_count must be greater than 0.")

        self.retriever = retriever
        self.generator = generator
        self.query_rewriter = query_rewriter
        self.multi_query_generator = multi_query_generator
        self.policy = policy
        self.multi_query_retriever = multi_query_retriever or MultiQueryRetriever(retriever)
        self.multi_query_count = multi_query_count

    def ask(self, query: str, top_k: int = 5) -> AnswerResponse:
        return self.ask_with_debug(query, top_k=top_k).response

    def ask_with_debug(self, query: str, top_k: int = 5) -> AgenticAskResult:
        state = self.run(query, top_k=top_k)
        sources = self.policy.select_results_for_context(state.results) if state.context_sufficient else []
        response = AnswerResponse(
            query=query,
            answer=state.answer,
            sources=sources,
            confidence=state.confidence,
        )
        return AgenticAskResult(
            response=response,
            debug=AgentDebugInfo(
                rewritten_query=state.rewritten_query,
                retrieval_queries=state.retrieval_queries,
                context_sufficient=state.context_sufficient,
                fallback_reason=state.fallback_reason,
                selected_source_ids=state.selected_source_ids,
            ),
        )

    def run(self, query: str, top_k: int = 5) -> AgentState:
        rewritten_query = self.query_rewriter.rewrite(query)
        retrieval_queries = self.multi_query_generator.generate(
            query,
            rewritten_query=rewritten_query,
            count=self.multi_query_count,
        )
        results = self.multi_query_retriever.search(
            retrieval_queries,
            top_k=top_k,
            max_results=top_k,
        )
        selected_results = self.policy.select_results_for_context(results)
        fallback_reason = self.policy.fallback_reason(results)
        context_sufficient = fallback_reason is None
        answer, generation_failure_reason = self._generate_or_fallback(query, selected_results, context_sufficient)

        if generation_failure_reason is not None:
            context_sufficient = False
            fallback_reason = generation_failure_reason
        elif context_sufficient and _is_unsupported_answer(answer):
            answer = self.policy.fallback_answer()
            context_sufficient = False
            fallback_reason = ANSWER_NOT_SUPPORTED

        return AgentState(
            query=query,
            rewritten_query=rewritten_query,
            retrieval_queries=retrieval_queries,
            results=results,
            answer=answer,
            confidence=self.retriever.estimate_confidence(results),
            context_sufficient=context_sufficient,
            fallback_reason=fallback_reason,
            selected_source_ids=_source_ids(selected_results),
        )

    def _generate_or_fallback(
        self,
        query: str,
        selected_results: list[SearchResult],
        context_sufficient: bool,
    ) -> tuple[str, str | None]:
        if not context_sufficient:
            return self.policy.fallback_answer(), None

        try:
            return self.generator.generate(query, selected_results), None
        except Exception:
            return self.policy.fallback_answer(), GENERATION_FAILED


def _source_ids(results: list[SearchResult]) -> list[str]:
    return [result.id for result in results]


def _is_unsupported_answer(answer: str) -> bool:
    normalized = answer.strip().casefold()
    unsupported_markers = [
        "无法从当前知识库中确定",
        "无法确定",
        "无法回答",
        "不能确定",
        "不能回答",
        "无法判断",
        "cannot determine",
        "cannot answer",
        "not enough information",
        "insufficient information",
    ]
    return any(marker in normalized for marker in unsupported_markers)
