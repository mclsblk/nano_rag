from dataclasses import dataclass
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True)
class ModelSettings:
    provider: str
    chat_provider: str
    embedding_provider: str
    visual_provider: str
    ollama_base_url: str
    ollama_chat_model: str
    ollama_embedding_model: str
    ollama_timeout_seconds: float
    ollama_think: bool
    openai_compatible_base_url: str
    openai_compatible_api_key: str
    openai_compatible_chat_model: str
    openai_compatible_embedding_model: str
    openai_compatible_visual_model: str
    openai_compatible_timeout_seconds: float


@dataclass(frozen=True)
class LoaderSettings:
    load_strategy: str
    visual_min_text_chars: int


@dataclass(frozen=True)
class VectorStoreSettings:
    chroma_persist_dir: Path
    chroma_collection: str


@dataclass(frozen=True)
class KeywordSettings:
    index_path: Path


@dataclass(frozen=True)
class SearchSettings:
    strategy: str
    hybrid_vector_weight: float
    hybrid_candidate_multiplier: int


@dataclass(frozen=True)
class ChunkingSettings:
    strategy: str
    size_chars: int
    overlap_chars: int
    min_chars: int
    semantic_breakpoint_threshold: float
    semantic_page_merge_min_score: float
    semantic_max_units_per_chunk: int


@dataclass(frozen=True)
class AgenticSettings:
    context_min_score: float
    context_min_chars: int
    context_max_chars: int
    multi_query_count: int
    engine: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    model_provider: str = "ollama"
    chat_model_provider: str = ""
    embedding_model_provider: str = ""
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_chat_model: str = "qwen3.5:0.8b"
    ollama_embedding_model: str = "qwen3-embedding:0.6b"
    ollama_timeout_seconds: float = 30.0
    ollama_think: bool = False
    openai_compatible_base_url: str = ""
    openai_compatible_api_key: str = ""
    openai_compatible_chat_model: str = ""
    openai_compatible_embedding_model: str = ""
    openai_compatible_visual_model: str = ""
    openai_compatible_timeout_seconds: float = 30.0
    document_load_strategy: str = "text"
    visual_model_provider: str = "openai_compatible"
    visual_min_text_chars: int = Field(default=40, ge=0)
    chroma_persist_dir: Path = Field(default=Path("./storage/chroma"))
    chroma_collection: str = "agentic_rag"
    search_strategy: str = "hybrid"
    keyword_index_path: Path = Field(default=Path("./storage/keyword.sqlite"))
    hybrid_vector_weight: float = Field(default=0.65, ge=0.0, le=1.0)
    hybrid_candidate_multiplier: int = Field(default=4, ge=1)
    chunk_strategy: str = "semantic"
    chunk_size_chars: int = Field(default=800, ge=1)
    chunk_overlap_chars: int = Field(default=120, ge=0)
    chunk_min_chars: int = Field(default=120, ge=0)
    semantic_breakpoint_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    semantic_page_merge_min_score: float = Field(default=0.55, ge=0.0, le=1.0)
    semantic_max_units_per_chunk: int = Field(default=12, ge=1)
    agentic_context_min_score: float = Field(default=0.45, ge=0.0, le=1.0)
    agentic_context_min_chars: int = Field(default=80, ge=0)
    agentic_context_max_chars: int = Field(default=4000, ge=1)
    agentic_multi_query_count: int = Field(default=3, ge=1)
    agentic_engine: str = "service"

    @model_validator(mode="after")
    def _default_specific_model_providers(self) -> "Settings":
        if not self.chat_model_provider.strip():
            self.chat_model_provider = self.model_provider
        if not self.embedding_model_provider.strip():
            self.embedding_model_provider = self.model_provider
        return self

    @property
    def models(self) -> ModelSettings:
        return ModelSettings(
            provider=self.model_provider,
            chat_provider=self.chat_model_provider,
            embedding_provider=self.embedding_model_provider,
            visual_provider=self.visual_model_provider,
            ollama_base_url=self.ollama_base_url,
            ollama_chat_model=self.ollama_chat_model,
            ollama_embedding_model=self.ollama_embedding_model,
            ollama_timeout_seconds=self.ollama_timeout_seconds,
            ollama_think=self.ollama_think,
            openai_compatible_base_url=self.openai_compatible_base_url,
            openai_compatible_api_key=self.openai_compatible_api_key,
            openai_compatible_chat_model=self.openai_compatible_chat_model,
            openai_compatible_embedding_model=self.openai_compatible_embedding_model,
            openai_compatible_visual_model=self.openai_compatible_visual_model,
            openai_compatible_timeout_seconds=self.openai_compatible_timeout_seconds,
        )

    @property
    def loader(self) -> LoaderSettings:
        return LoaderSettings(
            load_strategy=self.document_load_strategy,
            visual_min_text_chars=self.visual_min_text_chars,
        )

    @property
    def vectorstore(self) -> VectorStoreSettings:
        return VectorStoreSettings(
            chroma_persist_dir=self.chroma_persist_dir,
            chroma_collection=self.chroma_collection,
        )

    @property
    def keyword(self) -> KeywordSettings:
        return KeywordSettings(index_path=self.keyword_index_path)

    @property
    def search(self) -> SearchSettings:
        return SearchSettings(
            strategy=self.search_strategy,
            hybrid_vector_weight=self.hybrid_vector_weight,
            hybrid_candidate_multiplier=self.hybrid_candidate_multiplier,
        )

    @property
    def chunking(self) -> ChunkingSettings:
        return ChunkingSettings(
            strategy=self.chunk_strategy,
            size_chars=self.chunk_size_chars,
            overlap_chars=self.chunk_overlap_chars,
            min_chars=self.chunk_min_chars,
            semantic_breakpoint_threshold=self.semantic_breakpoint_threshold,
            semantic_page_merge_min_score=self.semantic_page_merge_min_score,
            semantic_max_units_per_chunk=self.semantic_max_units_per_chunk,
        )

    @property
    def agentic(self) -> AgenticSettings:
        return AgenticSettings(
            context_min_score=self.agentic_context_min_score,
            context_min_chars=self.agentic_context_min_chars,
            context_max_chars=self.agentic_context_max_chars,
            multi_query_count=self.agentic_multi_query_count,
            engine=self.agentic_engine,
        )


def load_settings() -> Settings:
    return Settings()
