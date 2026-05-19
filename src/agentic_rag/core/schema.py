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
    loaded_documents: int = Field(ge=0)
    generated_chunks: int = Field(ge=0)
    stored_chunks: int = Field(ge=0)
    skipped: list[str] = Field(default_factory=list)


class DeIngestResponse(CoreSchema):
    mode: Literal["de_ingest"] = "de_ingest"
    source: str
    deleted_chunks: int = Field(ge=0)


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
    openai_compatible_timeout_seconds: float | None = None
    chroma_persist_dir: str
    chroma_collection: str
    chroma_count: int = Field(ge=0)
    search_strategy: str = "hybrid"
    keyword_index_path: str = ""
    keyword_source_count: int = Field(default=0, ge=0)
    keyword_chunk_count: int = Field(default=0, ge=0)
