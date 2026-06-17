from agentic_rag.core import (
    AnswerResponse,
    CollectionListResponse,
    CollectionResponse,
    FileListResponse,
    FileResponse,
    InspectResponse,
    SearchResponse,
)
from agentic_rag.factory import (
    create_agentic_service,
    create_collection_service,
    create_file_service,
    create_keyword_store,
    create_pipeline,
    create_registry_service,
    create_settings,
    create_vectorstore,
)


class ApplicationService:
    def inspect_state(self) -> InspectResponse:
        settings = create_settings()
        vectorstore = create_vectorstore(settings)
        keyword_store = create_keyword_store(settings)
        file_service = create_file_service(settings)
        collection_service = create_collection_service(settings)
        registry = create_registry_service(settings)
        models = settings.models
        loader = settings.loader
        vectorstore_settings = settings.vectorstore
        keyword = settings.keyword
        search = settings.search
        agentic = settings.agentic

        return InspectResponse(
            model_provider=models.provider,
            chat_model_provider=models.chat_provider,
            embedding_model_provider=models.embedding_provider,
            ollama_base_url=models.ollama_base_url,
            ollama_chat_model=models.ollama_chat_model,
            ollama_embedding_model=models.ollama_embedding_model,
            ollama_timeout_seconds=models.ollama_timeout_seconds,
            ollama_think=models.ollama_think,
            openai_compatible_base_url=models.openai_compatible_base_url,
            openai_compatible_chat_model=models.openai_compatible_chat_model,
            openai_compatible_embedding_model=models.openai_compatible_embedding_model,
            openai_compatible_visual_model=models.openai_compatible_visual_model,
            openai_compatible_timeout_seconds=models.openai_compatible_timeout_seconds,
            document_load_strategy=loader.load_strategy,
            visual_model_provider=models.visual_provider,
            visual_min_text_chars=loader.visual_min_text_chars,
            chroma_persist_dir=str(vectorstore_settings.chroma_persist_dir),
            chroma_collection=vectorstore_settings.chroma_collection,
            chroma_count=vectorstore.count_chunks(),
            search_strategy=search.strategy,
            keyword_index_path=str(keyword.index_path),
            keyword_source_count=keyword_store.count_sources(),
            keyword_chunk_count=keyword_store.count_chunks(),
            file_count=file_service.count_active(),
            collection_count=collection_service.count(),
            registry_record_count=registry.count_records(),
            indexed_chunk_count=registry.count_indexed_chunks(),
            agentic_engine=agentic.engine,
        )

    def search(self, query: str, *, collection_id: str, top_k: int = 5) -> SearchResponse:
        return create_pipeline(collection_id=collection_id).search(query, top_k=top_k)

    def ask(
        self,
        query: str,
        *,
        collection_id: str,
        top_k: int = 5,
        agentic: bool = False,
        engine: str | None = None,
    ) -> AnswerResponse:
        if agentic:
            return create_agentic_service(engine=engine, collection_id=collection_id).ask(query, top_k=top_k)
        return create_pipeline(require_gen=True, collection_id=collection_id).ask(query, top_k=top_k)

    def list_files(self) -> FileListResponse:
        return create_file_service().list_files()

    def get_file(self, file_id: str) -> FileResponse:
        return FileResponse(file=create_file_service().get_file(file_id))

    def list_collections(self) -> CollectionListResponse:
        return create_collection_service().list_collections()

    def get_collection(self, collection_id: str) -> CollectionResponse:
        return CollectionResponse(collection=create_collection_service().get_collection(collection_id))
