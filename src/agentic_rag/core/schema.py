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
