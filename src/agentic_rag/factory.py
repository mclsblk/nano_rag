from agentic_rag.agent import create_agentic_service as create_agentic_engine
from agentic_rag.agent.service import AgenticService
from agentic_rag.config import Settings, load_settings
from agentic_rag.core import CollectionRecord
from agentic_rag.document import create_document_chunker, create_document_loader
from agentic_rag.file_sys import CollectionService, FileService, JobService, RegistryService, SystemStore, UploadService
from agentic_rag.keyword import SQLiteKeywordStore
from agentic_rag.models import EmbeddingModel, create_chat_model, create_embedding_model, create_vision_model
from agentic_rag.output import OutputFormatter
from agentic_rag.rag import ContextBuilder, Generator, Indexer, RAGPipeline, create_retriever
from agentic_rag.vectorstore import ChromaVectorStore


def create_settings() -> Settings:
    return load_settings()


def create_system_store(settings: Settings | None = None) -> SystemStore:
    resolved_settings = settings or create_settings()
    return SystemStore(resolved_settings.system.db_path)


def create_file_service(settings: Settings | None = None) -> FileService:
    resolved_settings = settings or create_settings()
    return FileService(create_system_store(resolved_settings), resolved_settings.system.file_storage_dir)


def create_upload_service(settings: Settings | None = None) -> UploadService:
    resolved_settings = settings or create_settings()
    return UploadService(resolved_settings.system.upload_dir, resolved_settings.system.max_upload_mb)


def create_collection_service(settings: Settings | None = None) -> CollectionService:
    resolved_settings = settings or create_settings()
    return CollectionService(create_system_store(resolved_settings), _keyword_collection_dir(resolved_settings))


def create_job_service(settings: Settings | None = None) -> JobService:
    return JobService(create_system_store(settings))


def create_vectorstore(
    settings: Settings | None = None,
    embedding_model: EmbeddingModel | None = None,
    collection_name: str | None = None,
) -> ChromaVectorStore:
    resolved_settings = settings or create_settings()
    resolved_embedding_model = embedding_model or create_embedding_model(resolved_settings)
    return ChromaVectorStore(
        embedding_model=resolved_embedding_model,
        settings=resolved_settings,
        collection_name=collection_name,
    )


def create_keyword_store(settings: Settings | None = None, path: str | None = None) -> SQLiteKeywordStore:
    resolved_settings = settings or create_settings()
    return SQLiteKeywordStore(path or resolved_settings.keyword.index_path)


def create_collection_indexer(
    collection: CollectionRecord,
    settings: Settings | None = None,
    load_strategy: str | None = None,
) -> Indexer:
    resolved_settings = settings or create_settings()
    embedding_model = create_embedding_model(resolved_settings)
    resolved_load_strategy = (load_strategy or resolved_settings.loader.load_strategy).strip().lower().replace("-", "_")
    vision_model = create_vision_model(resolved_settings) if resolved_load_strategy in {"auto", "visual"} else None
    return Indexer(
        loader=create_document_loader(resolved_settings, load_strategy=resolved_load_strategy, vision_model=vision_model),
        splitter=create_document_chunker(resolved_settings, embedding_model=embedding_model),
        vectorstore=create_vectorstore(
            resolved_settings,
            embedding_model=embedding_model,
            collection_name=collection.chroma_collection,
        ),
        keyword_store=create_keyword_store(resolved_settings, collection.keyword_index_path),
    )


def create_registry_service(settings: Settings | None = None) -> RegistryService:
    resolved_settings = settings or create_settings()
    store = create_system_store(resolved_settings)
    files = FileService(store, resolved_settings.system.file_storage_dir)
    collections = CollectionService(store, _keyword_collection_dir(resolved_settings))
    return RegistryService(
        store=store,
        files=files,
        collections=collections,
        indexer_factory=lambda collection, load_strategy: create_collection_indexer(
            collection,
            resolved_settings,
            load_strategy=load_strategy,
        ),
        vectorstore_factory=lambda collection: create_vectorstore(
            resolved_settings,
            collection_name=collection.chroma_collection,
        ),
        keyword_store_factory=lambda collection: create_keyword_store(resolved_settings, collection.keyword_index_path),
    )


def create_pipeline(
    settings: Settings | None = None,
    require_gen: bool = False,
    collection_id: str = "",
) -> RAGPipeline:
    resolved_settings = settings or create_settings()
    collection = create_collection_service(resolved_settings).get_collection(collection_id)
    retriever = create_retriever(
        resolved_settings,
        vectorstore_factory=lambda: create_vectorstore(
            resolved_settings,
            collection_name=collection.chroma_collection,
        ),
        keyword_store_factory=lambda: create_keyword_store(
            resolved_settings,
            collection.keyword_index_path,
        ),
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


def create_agentic_service(
    settings: Settings | None = None,
    engine: str | None = None,
    collection_id: str = "",
) -> AgenticService:
    resolved_settings = settings or create_settings()
    collection = create_collection_service(resolved_settings).get_collection(collection_id)
    retriever = create_retriever(
        resolved_settings,
        vectorstore_factory=lambda: create_vectorstore(
            resolved_settings,
            collection_name=collection.chroma_collection,
        ),
        keyword_store_factory=lambda: create_keyword_store(
            resolved_settings,
            collection.keyword_index_path,
        ),
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


def _keyword_collection_dir(settings: Settings):
    return settings.keyword.index_path.parent / "keyword"
