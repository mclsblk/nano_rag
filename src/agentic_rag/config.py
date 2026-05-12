from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class Settings(BaseModel):
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_chat_model: str = "qwen3.5:0.8b"
    ollama_embedding_model: str = "qwen3-embedding:0.6b"
    ollama_timeout_seconds: float = 30.0
    ollama_think: bool = False
    chroma_persist_dir: Path = Field(default=Path("./storage/chroma"))
    chroma_collection: str = "agentic_rag"


def load_settings() -> Settings:
    load_dotenv()

    import os

    return Settings(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", Settings.model_fields["ollama_base_url"].default),
        ollama_chat_model=os.getenv("OLLAMA_CHAT_MODEL", Settings.model_fields["ollama_chat_model"].default),
        ollama_embedding_model=os.getenv(
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
        chroma_persist_dir=Path(
            os.getenv(
                "CHROMA_PERSIST_DIR",
                str(Settings.model_fields["chroma_persist_dir"].default),
            )
        ),
        chroma_collection=os.getenv("CHROMA_COLLECTION", Settings.model_fields["chroma_collection"].default),
    )


def _env_bool(name: str, default: bool) -> bool:
    import os

    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
