from typing import Any

from openai import OpenAI, OpenAIError

from agentic_rag.config import Settings, load_settings
from agentic_rag.core import ConfigurationError, ModelError
from agentic_rag.models.base import ChatModel, EmbeddingModel, extract_chat_content


class OpenAICompatibleChatModel(ChatModel):
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        _require_base_url(self.settings)
        _require_model(
            self.settings.openai_compatible_chat_model,
            "OPENAI_COMPATIBLE_CHAT_MODEL",
        )
        self.client = _create_client(self.settings)

    def chat(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=self.settings.openai_compatible_chat_model,
                messages=messages,
                stream=False,
            )
        except OpenAIError as exc:
            raise ModelError(f"OpenAI-compatible chat request failed: {exc}") from exc

        data = _response_to_dict(response)
        extract_chat_content(data)
        return data


class OpenAICompatibleEmbeddingModel(EmbeddingModel):
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        _require_base_url(self.settings)
        _require_model(
            self.settings.openai_compatible_embedding_model,
            "OPENAI_COMPATIBLE_EMBEDDING_MODEL",
        )
        self.client = _create_client(self.settings)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        try:
            response = self.client.embeddings.create(
                model=self.settings.openai_compatible_embedding_model,
                input=texts,
            )
        except OpenAIError as exc:
            raise ModelError(f"OpenAI-compatible embeddings request failed: {exc}") from exc

        return _extract_embeddings(_response_to_dict(response), expected_count=len(texts))

    def embed_query(self, text: str) -> list[float]:
        try:
            response = self.client.embeddings.create(
                model=self.settings.openai_compatible_embedding_model,
                input=text,
            )
        except OpenAIError as exc:
            raise ModelError(f"OpenAI-compatible embeddings request failed: {exc}") from exc

        embeddings = _extract_embeddings(_response_to_dict(response), expected_count=1)
        return embeddings[0]


def _extract_embeddings(response: dict[str, Any], expected_count: int) -> list[list[float]]:
    data = response.get("data")
    if not isinstance(data, list):
        raise ModelError("OpenAI-compatible embedding response missing data list.")

    embeddings: list[tuple[int | None, list[float]]] = []
    for item in data:
        if not isinstance(item, dict):
            raise ModelError("OpenAI-compatible embedding response data item must be an object.")

        embedding = item.get("embedding")
        if not isinstance(embedding, list) or not all(isinstance(value, (int, float)) for value in embedding):
            raise ModelError("OpenAI-compatible embedding response missing numeric embedding list.")

        index = item.get("index")
        embeddings.append((index if isinstance(index, int) else None, [float(value) for value in embedding]))

    if len(embeddings) != expected_count:
        raise ModelError(
            f"OpenAI-compatible embedding response returned {len(embeddings)} embeddings; expected {expected_count}."
        )

    if all(index is not None for index, _ in embeddings):
        embeddings.sort(key=lambda indexed_embedding: indexed_embedding[0] or 0)

    return [embedding for _, embedding in embeddings]


def _require_base_url(settings: Settings) -> None:
    if not settings.openai_compatible_base_url.strip():
        raise ConfigurationError("OPENAI_COMPATIBLE_BASE_URL is required for openai_compatible provider.")


def _require_model(model_name: str, env_name: str) -> None:
    if not model_name.strip():
        raise ConfigurationError(f"{env_name} is required for openai_compatible provider.")


def _create_client(settings: Settings) -> OpenAI:
    return OpenAI(
        api_key=settings.openai_compatible_api_key.strip() or "unused",
        base_url=settings.openai_compatible_base_url.strip().rstrip("/"),
        timeout=settings.openai_compatible_timeout_seconds,
    )


def _response_to_dict(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response

    if hasattr(response, "model_dump"):
        try:
            data = response.model_dump(mode="json")
        except TypeError:
            data = response.model_dump()
    elif hasattr(response, "dict"):
        data = response.dict()
    else:
        raise ModelError("OpenAI-compatible SDK response cannot be converted to a dict.")

    if not isinstance(data, dict):
        raise ModelError("OpenAI-compatible SDK response must convert to a dict.")
    return data
