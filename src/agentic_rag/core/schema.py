from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class CoreSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Document(CoreSchema):
    id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(CoreSchema):
    id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(CoreSchema):
    id: str
    content: str
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(CoreSchema):
    mode: Literal["search"] = "search"
    query: str
    results: list[SearchResult]


class AnswerResponse(CoreSchema):
    mode: Literal["ask"] = "ask"
    query: str
    answer: str
    sources: list[SearchResult]
    confidence: Literal["high", "medium", "low"] | None = None


class IngestResponse(CoreSchema):
    mode: Literal["ingest"] = "ingest"
    path: str
    file_id: str | None = None
    collection_id: str | None = None
    index_status: str | None = None
    loaded_documents: int = Field(ge=0)
    generated_chunks: int = Field(ge=0)
    stored_chunks: int = Field(ge=0)
    skipped: list[str] = Field(default_factory=list)


class DeIngestResponse(CoreSchema):
    mode: Literal["de_ingest"] = "de_ingest"
    source: str
    file_id: str | None = None
    collection_id: str | None = None
    index_status: str | None = None
    deleted_keyword_chunks: int | None = Field(default=None, ge=0)
    deleted_chunks: int = Field(ge=0)


class FileRecord(CoreSchema):
    file_id: str
    original_name: str
    content_hash: str
    storage_path: str
    file_type: str
    size_bytes: int = Field(ge=0)
    status: str
    created_at: str


class FileResponse(CoreSchema):
    mode: Literal["file"] = "file"
    file: FileRecord


class FileListResponse(CoreSchema):
    mode: Literal["file_list"] = "file_list"
    files: list[FileRecord]


class FileDeleteResponse(CoreSchema):
    mode: Literal["file_delete"] = "file_delete"
    file_id: str
    status: str


class CollectionRecord(CoreSchema):
    collection_id: str
    name: str
    description: str
    chroma_collection: str
    keyword_index_path: str
    created_at: str


class CollectionResponse(CoreSchema):
    mode: Literal["collection"] = "collection"
    collection: CollectionRecord


class CollectionListResponse(CoreSchema):
    mode: Literal["collection_list"] = "collection_list"
    collections: list[CollectionRecord]


class CollectionDeleteResponse(CoreSchema):
    mode: Literal["collection_delete"] = "collection_delete"
    collection_id: str
    status: str


class RegistryRecord(CoreSchema):
    file_id: str
    collection_id: str
    index_status: str
    indexed_chunk_count: int = Field(ge=0)
    indexed_at: str | None = None
    last_error: str | None = None


class RegistryListResponse(CoreSchema):
    mode: Literal["registry_list"] = "registry_list"
    records: list[RegistryRecord]


class JobRecord(CoreSchema):
    job_id: str
    job_type: str
    status: Literal["queued", "running", "succeeded", "failed"]
    file_id: str
    collection_id: str
    loader: str | None = None
    input_path: str | None = None
    upload_file_name: str | None = None
    loaded_documents: int = Field(default=0, ge=0)
    generated_chunks: int = Field(default=0, ge=0)
    stored_chunks: int = Field(default=0, ge=0)
    skipped: list[str] = Field(default_factory=list)
    error_message: str | None = None
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None


class JobResponse(CoreSchema):
    mode: Literal["job"] = "job"
    job: JobRecord


class JobListResponse(CoreSchema):
    mode: Literal["job_list"] = "job_list"
    jobs: list[JobRecord]


class InspectResponse(CoreSchema):
    mode: Literal["inspect"] = "inspect"
    model_provider: str
    chat_model_provider: str
    embedding_model_provider: str
    ollama_base_url: str = ""
    ollama_chat_model: str = ""
    ollama_embedding_model: str = ""
    ollama_timeout_seconds: float | None = None
    ollama_think: bool | None = None
    openai_compatible_base_url: str = ""
    openai_compatible_chat_model: str = ""
    openai_compatible_embedding_model: str = ""
    openai_compatible_visual_model: str = ""
    openai_compatible_timeout_seconds: float | None = None
    document_load_strategy: str = "text"
    visual_model_provider: str = "openai_compatible"
    visual_min_text_chars: int = Field(default=40, ge=0)
    chroma_persist_dir: str
    chroma_collection: str
    chroma_count: int = Field(ge=0)
    search_strategy: str = "hybrid"
    keyword_index_path: str = ""
    keyword_source_count: int = Field(default=0, ge=0)
    keyword_chunk_count: int = Field(default=0, ge=0)
    file_count: int = Field(default=0, ge=0)
    collection_count: int = Field(default=0, ge=0)
    registry_record_count: int = Field(default=0, ge=0)
    indexed_chunk_count: int = Field(default=0, ge=0)
    agentic_engine: str = "service"
