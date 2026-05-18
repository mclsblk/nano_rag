from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class Settings(BaseModel):
    model_provider: str = "ollama"
    chat_model_provider: str = "ollama"
    embedding_model_provider: str = "ollama"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_chat_model: str = "qwen3.5:0.8b"
    ollama_embedding_model: str = "qwen3-embedding:0.6b"
    ollama_timeout_seconds: float = 30.0
    ollama_think: bool = False
    openai_compatible_base_url: str = ""
    openai_compatible_api_key: str = ""
    openai_compatible_chat_model: str = ""
    openai_compatible_embedding_model: str = ""
    openai_compatible_timeout_seconds: float = 30.0
    chroma_persist_dir: Path = Field(default=Path("./storage/chroma"))
    chroma_collection: str = "agentic_rag"
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


def load_settings() -> Settings:
    load_dotenv()

    import os

    model_provider = _env_string("MODEL_PROVIDER", Settings.model_fields["model_provider"].default)
    chat_model_provider = _env_string("CHAT_MODEL_PROVIDER", model_provider)
    embedding_model_provider = _env_string("EMBEDDING_MODEL_PROVIDER", model_provider)

    return Settings(
        model_provider=model_provider,
        chat_model_provider=chat_model_provider,
        embedding_model_provider=embedding_model_provider,
        ollama_base_url=_env_string("OLLAMA_BASE_URL", Settings.model_fields["ollama_base_url"].default),
        ollama_chat_model=_env_string("OLLAMA_CHAT_MODEL", Settings.model_fields["ollama_chat_model"].default),
        ollama_embedding_model=_env_string(
            "OLLAMA_EMBEDDING_MODEL",
            Settings.model_fields["ollama_embedding_model"].default,
        ),
        ollama_timeout_seconds=float(
            os.getenv(
                "OLLAMA_TIMEOUT_SECONDS",
                str(Settings.model_fields["ollama_timeout_seconds"].default),
            )
        ),
        ollama_think=_env_bool(
            "OLLAMA_THINK",
            Settings.model_fields["ollama_think"].default,
        ),
        openai_compatible_base_url=_env_string(
            "OPENAI_COMPATIBLE_BASE_URL",
            Settings.model_fields["openai_compatible_base_url"].default,
            allow_empty=True,
        ),
        openai_compatible_api_key=_env_string(
            "OPENAI_COMPATIBLE_API_KEY",
            Settings.model_fields["openai_compatible_api_key"].default,
            allow_empty=True,
        ),
        openai_compatible_chat_model=_env_string(
            "OPENAI_COMPATIBLE_CHAT_MODEL",
            Settings.model_fields["openai_compatible_chat_model"].default,
            allow_empty=True,
        ),
        openai_compatible_embedding_model=_env_string(
            "OPENAI_COMPATIBLE_EMBEDDING_MODEL",
            Settings.model_fields["openai_compatible_embedding_model"].default,
            allow_empty=True,
        ),
        openai_compatible_timeout_seconds=float(
            os.getenv(
                "OPENAI_COMPATIBLE_TIMEOUT_SECONDS",
                str(Settings.model_fields["openai_compatible_timeout_seconds"].default),
            )
        ),
        chroma_persist_dir=Path(
            os.getenv(
                "CHROMA_PERSIST_DIR",
                str(Settings.model_fields["chroma_persist_dir"].default),
            )
        ),
        chroma_collection=_env_string("CHROMA_COLLECTION", Settings.model_fields["chroma_collection"].default),
        chunk_strategy=_env_string("CHUNK_STRATEGY", Settings.model_fields["chunk_strategy"].default),
        chunk_size_chars=int(
            os.getenv(
                "CHUNK_SIZE_CHARS",
                str(Settings.model_fields["chunk_size_chars"].default),
            )
        ),
        chunk_overlap_chars=int(
            os.getenv(
                "CHUNK_OVERLAP_CHARS",
                str(Settings.model_fields["chunk_overlap_chars"].default),
            )
        ),
        chunk_min_chars=int(
            os.getenv(
                "CHUNK_MIN_CHARS",
                str(Settings.model_fields["chunk_min_chars"].default),
            )
        ),
        semantic_breakpoint_threshold=float(
            os.getenv(
                "SEMANTIC_BREAKPOINT_THRESHOLD",
                str(Settings.model_fields["semantic_breakpoint_threshold"].default),
            )
        ),
        semantic_page_merge_min_score=float(
            os.getenv(
                "SEMANTIC_PAGE_MERGE_MIN_SCORE",
                str(Settings.model_fields["semantic_page_merge_min_score"].default),
            )
        ),
        semantic_max_units_per_chunk=int(
            os.getenv(
                "SEMANTIC_MAX_UNITS_PER_CHUNK",
                str(Settings.model_fields["semantic_max_units_per_chunk"].default),
            )
        ),
        agentic_context_min_score=float(
            os.getenv(
                "AGENTIC_CONTEXT_MIN_SCORE",
                str(Settings.model_fields["agentic_context_min_score"].default),
            )
        ),
        agentic_context_min_chars=int(
            os.getenv(
                "AGENTIC_CONTEXT_MIN_CHARS",
                str(Settings.model_fields["agentic_context_min_chars"].default),
            )
        ),
        agentic_context_max_chars=int(
            os.getenv(
                "AGENTIC_CONTEXT_MAX_CHARS",
                str(Settings.model_fields["agentic_context_max_chars"].default),
            )
        ),
        agentic_multi_query_count=int(
            os.getenv(
                "AGENTIC_MULTI_QUERY_COUNT",
                str(Settings.model_fields["agentic_multi_query_count"].default),
            )
        ),
    )


def _env_string(name: str, default: str, allow_empty: bool = False) -> str:
    import os

    value = os.getenv(name)
    if value is None:
        return default

    value = value.strip()
    if not value and not allow_empty:
        return default
    return value


def _env_bool(name: str, default: bool) -> bool:
    import os

    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
