from typing import Any, TypedDict

from agentic_rag.agent.policy import AgentPolicy
from agentic_rag.agent.query import MultiQueryGenerator, QueryRewriter
from agentic_rag.agent.retrieval import MultiQueryRetriever
from agentic_rag.agent.service import ANSWER_NOT_SUPPORTED, AgenticService, _is_unsupported_answer, _source_ids
from agentic_rag.agent.state import AgentState
from agentic_rag.core import ConfigurationError
from agentic_rag.rag import Generator, Retriever


class _GraphState(TypedDict):
    agent_state: AgentState
    top_k: int


class LangGraphAgenticService(AgenticService):
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
        super().__init__(
            retriever=retriever,
            generator=generator,
            query_rewriter=query_rewriter,
            multi_query_generator=multi_query_generator,
            policy=policy,
            multi_query_retriever=multi_query_retriever,
            multi_query_count=multi_query_count,
        )
        self.graph = self._compile_graph()

    def run(self, query: str, top_k: int = 5) -> AgentState:
        result = self.graph.invoke({"agent_state": AgentState(query=query), "top_k": top_k})
        state = result.get("agent_state") if isinstance(result, dict) else None
        if not isinstance(state, AgentState):
            raise ConfigurationError("LangGraph agentic engine returned an invalid state.")
        return state

    def _compile_graph(self) -> Any:
        try:
            from langgraph.graph import END, StateGraph
        except ImportError as exc:
            raise ConfigurationError(
                "AGENTIC_ENGINE=langgraph requires the optional langgraph dependency. "
                "Install it with `python -m pip install -e '.[agentic]'`."
            ) from exc

        graph = StateGraph(_GraphState)
        graph.add_node("rewrite_query", self._rewrite_query)
        graph.add_node("generate_retrieval_queries", self._generate_retrieval_queries)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("judge_context", self._judge_context)
        graph.add_node("generate_or_fallback", self._generate_or_fallback_node)
        graph.add_node("build_response", self._build_response)

        graph.set_entry_point("rewrite_query")
        graph.add_edge("rewrite_query", "generate_retrieval_queries")
        graph.add_edge("generate_retrieval_queries", "retrieve")
        graph.add_edge("retrieve", "judge_context")
        graph.add_edge("judge_context", "generate_or_fallback")
        graph.add_edge("generate_or_fallback", "build_response")
        graph.add_edge("build_response", END)
        return graph.compile()

    def _rewrite_query(self, graph_state: _GraphState) -> _GraphState:
        state = graph_state["agent_state"]
        state.rewritten_query = self.query_rewriter.rewrite(state.query)
        return graph_state

    def _generate_retrieval_queries(self, graph_state: _GraphState) -> _GraphState:
        state = graph_state["agent_state"]
        state.retrieval_queries = self.multi_query_generator.generate(
            state.query,
            rewritten_query=state.rewritten_query,
            count=self.multi_query_count,
        )
        return graph_state

    def _retrieve(self, graph_state: _GraphState) -> _GraphState:
        state = graph_state["agent_state"]
        top_k = graph_state["top_k"]
        state.results = self.multi_query_retriever.search(
            state.retrieval_queries,
            top_k=top_k,
            max_results=top_k,
        )
        return graph_state

    def _judge_context(self, graph_state: _GraphState) -> _GraphState:
        state = graph_state["agent_state"]
        selected_results = self.policy.select_results_for_context(state.results)
        state.fallback_reason = self.policy.fallback_reason(state.results)
        state.context_sufficient = state.fallback_reason is None
        state.selected_source_ids = _source_ids(selected_results)
        return graph_state

    def _generate_or_fallback_node(self, graph_state: _GraphState) -> _GraphState:
        state = graph_state["agent_state"]
        selected_results = self.policy.select_results_for_context(state.results)
        answer, generation_failure_reason = self._generate_or_fallback(
            state.query,
            selected_results,
            state.context_sufficient,
        )

        if generation_failure_reason is not None:
            state.context_sufficient = False
            state.fallback_reason = generation_failure_reason
        elif state.context_sufficient and _is_unsupported_answer(answer):
            answer = self.policy.fallback_answer()
            state.context_sufficient = False
            state.fallback_reason = ANSWER_NOT_SUPPORTED

        state.answer = answer
        state.confidence = self.retriever.estimate_confidence(state.results)
        return graph_state

    def _build_response(self, graph_state: _GraphState) -> _GraphState:
        return graph_state
