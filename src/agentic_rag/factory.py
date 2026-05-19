from agentic_rag.agent import AgentPolicy, AgenticService, LangGraphAgenticService, MultiQueryGenerator, QueryRewriter
from agentic_rag.config import Settings, load_settings
from agentic_rag.core import ConfigurationError
from agentic_rag.document import DocumentLoader, SemanticChunker, TextSplitter
from agentic_rag.keyword import SQLiteKeywordStore
from agentic_rag.models import (
    ChatModel,
    EmbeddingModel,
    OllamaChatModel,
    OllamaEmbeddingModel,
    OpenAICompatibleChatModel,
    OpenAICompatibleEmbeddingModel,
    OpenAICompatibleVisionModel,
    VisionModel,
)
from agentic_rag.output import OutputFormatter
from agentic_rag.rag import ContextBuilder, Generator, HybridRetriever, Indexer, KeywordRetriever, RAGPipeline, Retriever
from agentic_rag.vectorstore import ChromaVectorStore


def create_settings() -> Settings:
    return load_settings()


def create_chat_model(settings: Settings | None = None) -> ChatModel:
    resolved_settings = settings or create_settings()
    provider = _normalize_provider(resolved_settings.chat_model_provider)

    if provider == "ollama":
        return OllamaChatModel(resolved_settings)
    if provider == "openai_compatible":
        return OpenAICompatibleChatModel(resolved_settings)

    raise ConfigurationError(
        f"Unsupported chat model provider: {resolved_settings.chat_model_provider}. "
        "Supported providers: ollama, openai_compatible."
    )


def create_embedding_model(settings: Settings | None = None) -> EmbeddingModel:
    resolved_settings = settings or create_settings()
    provider = _normalize_provider(resolved_settings.embedding_model_provider)

    if provider == "ollama":
        return OllamaEmbeddingModel(resolved_settings)
    if provider == "openai_compatible":
        return OpenAICompatibleEmbeddingModel(resolved_settings)

    raise ConfigurationError(
        f"Unsupported embedding model provider: {resolved_settings.embedding_model_provider}. "
        "Supported providers: ollama, openai_compatible."
    )


def create_vision_model(settings: Settings | None = None) -> VisionModel:
    resolved_settings = settings or create_settings()
    provider = _normalize_provider(resolved_settings.visual_model_provider)

    if provider == "openai_compatible":
        return OpenAICompatibleVisionModel(resolved_settings)

    raise ConfigurationError(
        f"Unsupported visual model provider: {resolved_settings.visual_model_provider}. "
        "Supported providers: openai_compatible."
    )


def create_vectorstore(settings: Settings | None = None, embedding_model: EmbeddingModel | None = None) -> ChromaVectorStore:
    resolved_settings = settings or create_settings()
    resolved_embedding_model = embedding_model or create_embedding_model(resolved_settings)
    return ChromaVectorStore(embedding_model=resolved_embedding_model, settings=resolved_settings)


def create_keyword_store(settings: Settings | None = None) -> SQLiteKeywordStore:
    resolved_settings = settings or create_settings()
    return SQLiteKeywordStore(resolved_settings.keyword_index_path)


def create_chunker(settings: Settings | None = None, embedding_model: EmbeddingModel | None = None):
    resolved_settings = settings or create_settings()
    strategy = _normalize_provider(resolved_settings.chunk_strategy)

    if strategy == "character":
        return TextSplitter(
            chunk_size=resolved_settings.chunk_size_chars,
            chunk_overlap=resolved_settings.chunk_overlap_chars,
        )
    if strategy == "semantic":
        return SemanticChunker(
            chunk_size=resolved_settings.chunk_size_chars,
            chunk_overlap=resolved_settings.chunk_overlap_chars,
            chunk_min_chars=resolved_settings.chunk_min_chars,
            breakpoint_threshold=resolved_settings.semantic_breakpoint_threshold,
            page_merge_min_score=resolved_settings.semantic_page_merge_min_score,
            max_units_per_chunk=resolved_settings.semantic_max_units_per_chunk,
            embedding_model=embedding_model,
        )

    raise ConfigurationError(
        f"Unsupported chunk strategy: {resolved_settings.chunk_strategy}. "
        "Supported strategies: character, semantic."
    )


def create_indexer(settings: Settings | None = None, load_strategy: str | None = None) -> Indexer:
    resolved_settings = settings or create_settings()
    embedding_model = create_embedding_model(resolved_settings)
    resolved_load_strategy = _normalize_provider(load_strategy or resolved_settings.document_load_strategy)
    vision_model = _LazyVisionModel(resolved_settings) if resolved_load_strategy in {"auto", "visual"} else None
    return Indexer(
        loader=DocumentLoader(
            load_strategy=resolved_load_strategy,
            vision_model=vision_model,
            visual_min_text_chars=resolved_settings.visual_min_text_chars,
        ),
        splitter=create_chunker(resolved_settings, embedding_model=embedding_model),
        vectorstore=create_vectorstore(resolved_settings, embedding_model=embedding_model),
        keyword_store=create_keyword_store(resolved_settings),
    )


def create_retriever(settings: Settings | None = None):
    resolved_settings = settings or create_settings()
    strategy = _normalize_provider(resolved_settings.search_strategy)

    if strategy == "vector":
        return Retriever(create_vectorstore(resolved_settings))
    if strategy == "keyword":
        return KeywordRetriever(create_keyword_store(resolved_settings))
    if strategy == "hybrid":
        return HybridRetriever(
            vectorstore=create_vectorstore(resolved_settings),
            keyword_store=create_keyword_store(resolved_settings),
            vector_weight=resolved_settings.hybrid_vector_weight,
            candidate_multiplier=resolved_settings.hybrid_candidate_multiplier,
        )

    raise ConfigurationError(
        f"Unsupported search strategy: {resolved_settings.search_strategy}. "
        "Supported strategies: vector, keyword, hybrid."
    )


def create_pipeline(settings: Settings | None = None, require_gen: bool = False) -> RAGPipeline:
    resolved_settings = settings or create_settings()
    retriever = create_retriever(resolved_settings)
    generator = (
        Generator(
            create_chat_model(resolved_settings),
            context_builder=ContextBuilder(max_chars=resolved_settings.agentic_context_max_chars),
        )
        if require_gen
        else None
    )
    return RAGPipeline(retriever=retriever, generator=generator)


def create_agentic_service(settings: Settings | None = None, engine: str | None = None) -> AgenticService:
    resolved_settings = settings or create_settings()
    resolved_engine = _normalize_provider(engine or resolved_settings.agentic_engine)
    if resolved_engine not in {"langgraph", "service"}:
        raise ConfigurationError(
            f"Unsupported agentic engine: {engine or resolved_settings.agentic_engine}. "
            "Supported engines: langgraph, service."
        )
    if resolved_engine == "langgraph":
        _ensure_langgraph_available()

    retriever = create_retriever(resolved_settings)
    chat_model = create_chat_model(resolved_settings)
    context_builder = ContextBuilder(max_chars=resolved_settings.agentic_context_max_chars)
    generator = Generator(chat_model, context_builder=context_builder)

    service_kwargs = dict(
        retriever=retriever,
        generator=generator,
        query_rewriter=QueryRewriter(chat_model),
        multi_query_generator=MultiQueryGenerator(chat_model),
        policy=AgentPolicy.from_settings(resolved_settings),
        multi_query_count=resolved_settings.agentic_multi_query_count,
    )

    if resolved_engine == "service":
        return AgenticService(**service_kwargs)
    if resolved_engine == "langgraph":
        return LangGraphAgenticService(**service_kwargs)


def create_formatter(content_preview_chars: int | None = None) -> OutputFormatter:
    return OutputFormatter(content_preview_chars=content_preview_chars)


def _normalize_provider(provider: str) -> str:
    return provider.strip().lower().replace("-", "_")


def _ensure_langgraph_available() -> None:
    try:
        import langgraph  # noqa: F401
    except ImportError as exc:
        raise ConfigurationError(
            "AGENTIC_ENGINE=langgraph requires the optional langgraph dependency. "
            "Install it with `python -m pip install -e '.[agentic]'`."
        ) from exc


class _LazyVisionModel(VisionModel):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: VisionModel | None = None

    def extract_text(self, image_bytes: bytes, *, mime_type: str) -> str:
        if self._model is None:
            self._model = create_vision_model(self.settings)
        return self._model.extract_text(image_bytes, mime_type=mime_type)
