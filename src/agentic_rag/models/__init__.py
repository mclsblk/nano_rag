from agentic_rag.config import Settings, load_settings
from agentic_rag.core import ConfigurationError
from agentic_rag.models.base import ChatModel, EmbeddingModel, VisionModel, extract_chat_content


def create_chat_model(settings: Settings | None = None) -> ChatModel:
    resolved_settings = settings or load_settings()
    model_settings = resolved_settings.models
    provider = model_settings.chat_provider.strip().lower().replace("-", "_")

    if provider == "ollama":
        from agentic_rag.models.ollama import OllamaChatModel

        return OllamaChatModel(resolved_settings)
    if provider == "openai_compatible":
        from agentic_rag.models.openai_compatible import OpenAICompatibleChatModel

        return OpenAICompatibleChatModel(resolved_settings)

    raise ConfigurationError(
        f"Unsupported chat model provider: {model_settings.chat_provider}. "
        "Supported providers: ollama, openai_compatible."
    )


def create_embedding_model(settings: Settings | None = None) -> EmbeddingModel:
    resolved_settings = settings or load_settings()
    model_settings = resolved_settings.models
    provider = model_settings.embedding_provider.strip().lower().replace("-", "_")

    if provider == "ollama":
        from agentic_rag.models.ollama import OllamaEmbeddingModel

        return OllamaEmbeddingModel(resolved_settings)
    if provider == "openai_compatible":
        from agentic_rag.models.openai_compatible import OpenAICompatibleEmbeddingModel

        return OpenAICompatibleEmbeddingModel(resolved_settings)

    raise ConfigurationError(
        f"Unsupported embedding model provider: {model_settings.embedding_provider}. "
        "Supported providers: ollama, openai_compatible."
    )


def create_vision_model(settings: Settings | None = None) -> VisionModel:
    resolved_settings = settings or load_settings()
    model_settings = resolved_settings.models
    provider = model_settings.visual_provider.strip().lower().replace("-", "_")

    if provider == "openai_compatible":
        from agentic_rag.models.openai_compatible import OpenAICompatibleVisionModel

        return OpenAICompatibleVisionModel(resolved_settings)

    raise ConfigurationError(
        f"Unsupported visual model provider: {model_settings.visual_provider}. "
        "Supported providers: openai_compatible."
    )
