from agentic_rag.config import Settings, load_settings
from agentic_rag.core import ConfigurationError
from agentic_rag.document import DocumentLoader, TextSplitter
from agentic_rag.models import (
    ChatModel,
    EmbeddingModel,
    OllamaChatModel,
    OllamaEmbeddingModel,
    OpenAICompatibleChatModel,
    OpenAICompatibleEmbeddingModel,
)
from agentic_rag.output import OutputFormatter
from agentic_rag.rag import ContextBuilder, Generator, Indexer, RAGPipeline, Retriever
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


def create_vectorstore(settings: Settings | None = None) -> ChromaVectorStore:
    resolved_settings = settings or create_settings()
    embedding_model = create_embedding_model(resolved_settings)
    return ChromaVectorStore(embedding_model=embedding_model, settings=resolved_settings)


def create_indexer(settings: Settings | None = None) -> Indexer:
    resolved_settings = settings or create_settings()
    return Indexer(
        loader=DocumentLoader(),
        splitter=TextSplitter(),
        vectorstore=create_vectorstore(resolved_settings),
    )


def create_pipeline(settings: Settings | None = None, require_gen: bool = False) -> RAGPipeline:
    resolved_settings = settings or create_settings()
    vectorstore = create_vectorstore(resolved_settings)
    retriever = Retriever(vectorstore)
    generator = (
        Generator(
            create_chat_model(resolved_settings),
            context_builder=ContextBuilder(max_chars=resolved_settings.agentic_context_max_chars),
        )
        if require_gen
        else None
    )
    return RAGPipeline(retriever=retriever, generator=generator)


def create_formatter(content_preview_chars: int | None = None) -> OutputFormatter:
    return OutputFormatter(content_preview_chars=content_preview_chars)


def _normalize_provider(provider: str) -> str:
    return provider.strip().lower().replace("-", "_")
