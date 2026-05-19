from agentic_rag.agent import create_agentic_service as create_agentic_engine
from agentic_rag.agent.service import AgenticService
from agentic_rag.config import Settings, load_settings
from agentic_rag.document import create_document_chunker, create_document_loader
from agentic_rag.keyword import SQLiteKeywordStore
from agentic_rag.models import EmbeddingModel, create_chat_model, create_embedding_model, create_vision_model
from agentic_rag.output import OutputFormatter
from agentic_rag.rag import ContextBuilder, Generator, Indexer, RAGPipeline, create_retriever
from agentic_rag.vectorstore import ChromaVectorStore


def create_settings() -> Settings:
    return load_settings()


def create_vectorstore(settings: Settings | None = None, embedding_model: EmbeddingModel | None = None) -> ChromaVectorStore:
    resolved_settings = settings or create_settings()
    resolved_embedding_model = embedding_model or create_embedding_model(resolved_settings)
    return ChromaVectorStore(embedding_model=resolved_embedding_model, settings=resolved_settings)


def create_keyword_store(settings: Settings | None = None) -> SQLiteKeywordStore:
    resolved_settings = settings or create_settings()
    return SQLiteKeywordStore(resolved_settings.keyword.index_path)


def create_indexer(settings: Settings | None = None, load_strategy: str | None = None) -> Indexer:
    resolved_settings = settings or create_settings()
    embedding_model = create_embedding_model(resolved_settings)
    resolved_load_strategy = (load_strategy or resolved_settings.loader.load_strategy).strip().lower().replace("-", "_")
    vision_model = create_vision_model(resolved_settings) if resolved_load_strategy in {"auto", "visual"} else None
    return Indexer(
        loader=create_document_loader(resolved_settings, load_strategy=resolved_load_strategy, vision_model=vision_model),
        splitter=create_document_chunker(resolved_settings, embedding_model=embedding_model),
        vectorstore=create_vectorstore(resolved_settings, embedding_model=embedding_model),
        keyword_store=create_keyword_store(resolved_settings),
    )


def create_pipeline(settings: Settings | None = None, require_gen: bool = False) -> RAGPipeline:
    resolved_settings = settings or create_settings()
    retriever = create_retriever(
        resolved_settings,
        vectorstore_factory=lambda: create_vectorstore(resolved_settings),
        keyword_store_factory=lambda: create_keyword_store(resolved_settings),
    )
    generator = (
        Generator(
            create_chat_model(resolved_settings),
            context_builder=ContextBuilder(max_chars=resolved_settings.agentic.context_max_chars),
        )
        if require_gen
        else None
    )
    return RAGPipeline(retriever=retriever, generator=generator)


def create_agentic_service(settings: Settings | None = None, engine: str | None = None) -> AgenticService:
    resolved_settings = settings or create_settings()
    retriever = create_retriever(
        resolved_settings,
        vectorstore_factory=lambda: create_vectorstore(resolved_settings),
        keyword_store_factory=lambda: create_keyword_store(resolved_settings),
    )
    chat_model = create_chat_model(resolved_settings)
    return create_agentic_engine(
        settings=resolved_settings,
        retriever=retriever,
        chat_model=chat_model,
        engine=engine,
    )


def create_formatter(content_preview_chars: int | None = None) -> OutputFormatter:
    return OutputFormatter(content_preview_chars=content_preview_chars)
