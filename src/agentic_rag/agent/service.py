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


def rewrite_query_step(state: AgentState, query_rewriter: QueryRewriter) -> AgentState:
    state.rewritten_query = query_rewriter.rewrite(state.query)
    return state


def generate_retrieval_queries_step(
    state: AgentState,
    multi_query_generator: MultiQueryGenerator,
    multi_query_count: int,
) -> AgentState:
    state.retrieval_queries = multi_query_generator.generate(
        state.query,
        rewritten_query=state.rewritten_query,
        count=multi_query_count,
    )
    return state


def retrieve_step(state: AgentState, multi_query_retriever: MultiQueryRetriever, top_k: int) -> AgentState:
    state.results = multi_query_retriever.search(
        state.retrieval_queries,
        top_k=top_k,
        max_results=top_k,
    )
    return state


def judge_context_step(state: AgentState, policy: AgentPolicy) -> AgentState:
    selected_results = policy.select_results_for_context(state.results)
    state.fallback_reason = policy.fallback_reason(state.results)
    state.context_sufficient = state.fallback_reason is None
    state.selected_source_ids = source_ids(selected_results)
    return state


def generate_answer_step(
    state: AgentState,
    policy: AgentPolicy,
    generator: Generator,
    retriever: Retriever,
) -> AgentState:
    selected_results = policy.select_results_for_context(state.results)
    answer, generation_failure_reason = generate_or_fallback(
        policy=policy,
        generator=generator,
        query=state.query,
        selected_results=selected_results,
        context_sufficient=state.context_sufficient,
    )

    if generation_failure_reason is not None:
        state.context_sufficient = False
        state.fallback_reason = generation_failure_reason
    elif state.context_sufficient and is_unsupported_answer(answer):
        answer = policy.fallback_answer()
        state.context_sufficient = False
        state.fallback_reason = ANSWER_NOT_SUPPORTED

    state.answer = answer
    state.confidence = retriever.estimate_confidence(state.results)
    return state


def generate_or_fallback(
    policy: AgentPolicy,
    generator: Generator,
    query: str,
    selected_results: list[SearchResult],
    context_sufficient: bool,
) -> tuple[str, str | None]:
    if not context_sufficient:
        return policy.fallback_answer(), None

    try:
        return generator.generate(query, selected_results), None
    except Exception:
        return policy.fallback_answer(), GENERATION_FAILED


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
        state = AgentState(query=query)
        rewrite_query_step(state, self.query_rewriter)
        generate_retrieval_queries_step(state, self.multi_query_generator, self.multi_query_count)
        retrieve_step(state, self.multi_query_retriever, top_k)
        judge_context_step(state, self.policy)
        generate_answer_step(state, self.policy, self.generator, self.retriever)
        return state


def source_ids(results: list[SearchResult]) -> list[str]:
    return [result.id for result in results]


def is_unsupported_answer(answer: str) -> bool:
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
