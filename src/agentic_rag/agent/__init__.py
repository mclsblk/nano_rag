from agentic_rag.config import Settings
from agentic_rag.agent.policy import AgentPolicy
from agentic_rag.agent.graph import LangGraphAgenticService
from agentic_rag.agent.query import MultiQueryGenerator, QueryRewriter
from agentic_rag.agent.service import AgenticService
from agentic_rag.core import ConfigurationError
from agentic_rag.models.base import ChatModel
from agentic_rag.rag.context import ContextBuilder
from agentic_rag.rag.generator import Generator
from agentic_rag.rag.retriever import Retriever


def create_agentic_service(
    settings: Settings,
    retriever: Retriever,
    chat_model: ChatModel,
    engine: str | None = None,
) -> AgenticService:
    agentic = settings.agentic
    resolved_engine = (engine or agentic.engine).strip().lower().replace("-", "_")
    context_builder = ContextBuilder(max_chars=agentic.context_max_chars)
    generator = Generator(chat_model, context_builder=context_builder)

    service_kwargs = dict(
        retriever=retriever,
        generator=generator,
        query_rewriter=QueryRewriter(chat_model),
        multi_query_generator=MultiQueryGenerator(chat_model),
        policy=AgentPolicy.from_settings(settings),
        multi_query_count=agentic.multi_query_count,
    )

    if resolved_engine == "service":
        return AgenticService(**service_kwargs)
    if resolved_engine == "langgraph":
        return LangGraphAgenticService(**service_kwargs)

    raise ConfigurationError(
        f"Unsupported agentic engine: {engine or agentic.engine}. Supported engines: langgraph, service."
    )
