from collections.abc import Callable

from agentic_rag.core import (
    AnswerResponse,
    CollectionDeleteResponse,
    CollectionListResponse,
    CollectionResponse,
    DeIngestResponse,
    FileDeleteResponse,
    FileListResponse,
    FileResponse,
    IngestResponse,
    InspectResponse,
    JobListResponse,
    JobResponse,
    RegistryError,
    SearchResponse,
)
from agentic_rag.factory import (
    create_agentic_service,
    create_collection_service,
    create_file_service,
    create_job_service,
    create_keyword_store,
    create_pipeline,
    create_registry_service,
    create_settings,
    create_system_store,
    create_upload_service,
    create_vectorstore,
)


class ApplicationService:
    def ready(self) -> dict[str, object]:
        checks = {
            "settings": _check(lambda: create_settings()),
            "system_db": _check(_check_system_db),
            "files": _check(lambda: create_file_service().count_active()),
            "collections": _check(lambda: create_collection_service().count()),
            "registry": _check(lambda: create_registry_service().count_records()),
            "jobs": _check(lambda: create_job_service().count_jobs()),
            "chroma": _check(lambda: create_vectorstore(create_settings()).count_chunks()),
            "keyword": _check(lambda: create_keyword_store(create_settings()).count_chunks()),
        }
        return {
            "status": "ok" if all(checks.values()) else "error",
            "checks": checks,
        }

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

    def import_file_path(self, path: str) -> FileResponse:
        return create_file_service().import_file(path)

    def upload_file(self, filename: str, stream) -> FileResponse:
        staged = create_upload_service().stage(filename, stream)
        return create_file_service().import_file(staged.path, original_name=staged.original_name)

    def delete_file(self, file_id: str) -> FileDeleteResponse:
        return create_registry_service().delete_file(file_id)

    def list_collections(self) -> CollectionListResponse:
        return create_collection_service().list_collections()

    def get_collection(self, collection_id: str) -> CollectionResponse:
        return CollectionResponse(collection=create_collection_service().get_collection(collection_id))

    def create_collection(self, name: str, description: str = "") -> CollectionResponse:
        return create_collection_service().create_collection(name, description=description)

    def delete_collection(self, collection_id: str) -> CollectionDeleteResponse:
        return create_registry_service().delete_collection(collection_id)

    def ingest_file(
        self,
        file_id: str,
        collection_id: str,
        *,
        loader: str | None = None,
    ) -> IngestResponse:
        return create_registry_service().ingest(file_id, collection_id, load_strategy=loader)

    def de_ingest_file(self, file_id: str, collection_id: str) -> DeIngestResponse:
        return create_registry_service().de_ingest(file_id, collection_id)

    def create_ingest_job(
        self,
        file_id: str,
        collection_id: str,
        *,
        loader: str | None = None,
    ) -> JobResponse:
        file = create_file_service().get_file(file_id)
        create_collection_service().get_collection(collection_id)
        record = create_registry_service().get_record_or_none(file_id, collection_id)
        if record is not None and record.index_status == "indexed":
            raise RegistryError(f"File is already indexed in collection: {file_id} + {collection_id}")
        return create_job_service().create_ingest_job(
            file_id=file_id,
            collection_id=collection_id,
            loader=loader,
            input_path=file.storage_path,
            upload_file_name=file.original_name,
        )

    def list_jobs(self) -> JobListResponse:
        return create_job_service().list_jobs()

    def get_job(self, job_id: str) -> JobResponse:
        return create_job_service().get_job(job_id)

    def run_ingest_job(self, job_id: str) -> None:
        jobs = create_job_service()
        job = jobs.mark_running(job_id).job
        try:
            response = create_registry_service().ingest(
                job.file_id,
                job.collection_id,
                load_strategy=job.loader,
            )
        except Exception as exc:
            jobs.mark_failed(job_id, str(exc))
            return
        jobs.mark_succeeded(
            job_id,
            loaded_documents=response.loaded_documents,
            generated_chunks=response.generated_chunks,
            stored_chunks=response.stored_chunks,
            skipped=response.skipped,
        )


def _check(operation: Callable[[], object]) -> bool:
    try:
        operation()
    except Exception:
        return False
    return True


def _check_system_db() -> None:
    with create_system_store(create_settings()).connect() as connection:
        connection.execute("SELECT 1").fetchone()
