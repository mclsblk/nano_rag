from typing import Any

import requests

from agentic_rag.config import ModelSettings, Settings, load_settings
from agentic_rag.core import ModelError
from agentic_rag.models.base import ChatModel, EmbeddingModel


class OllamaChatModel(ChatModel):
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self.model_settings = self.settings.models

    def chat(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {
            "model": self.model_settings.ollama_chat_model,
            "messages": messages,
            "stream": False,
            "think": self.model_settings.ollama_think,
        }
        response = self._post("/api/chat", payload)
        if not isinstance(response.get("message"), dict):
            raise ModelError("Ollama chat response missing message object.")
        return response

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return _post_ollama(self.model_settings, path, payload)


class OllamaEmbeddingModel(EmbeddingModel):
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self.model_settings = self.settings.models

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        payload = {
            "model": self.model_settings.ollama_embedding_model,
            "prompt": text,
        }
        response = self._post("/api/embeddings", payload)
        embedding = response.get("embedding")
        if not isinstance(embedding, list) or not all(isinstance(value, (int, float)) for value in embedding):
            raise ModelError("Ollama embedding response missing numeric embedding list.")
        return [float(value) for value in embedding]

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return _post_ollama(self.model_settings, path, payload)


def _post_ollama(settings: ModelSettings, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    base_url = settings.ollama_base_url.rstrip("/")
    url = f"{base_url}{path}"
    try:
        with requests.Session() as session:
            session.trust_env = False
            response = session.post(
                url,
                json=payload,
                timeout=settings.ollama_timeout_seconds,
            )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise ModelError(f"Ollama request failed: {exc}") from exc
    except ValueError as exc:
        raise ModelError("Ollama response was not valid JSON.") from exc

    if not isinstance(data, dict):
        raise ModelError("Ollama response JSON must be an object.")
    return data
