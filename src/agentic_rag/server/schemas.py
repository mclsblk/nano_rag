from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ServerSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(ServerSchema):
    status: Literal["ok"] = "ok"


class SearchRequest(ServerSchema):
    query: str
    collection_id: str
    top_k: int = Field(default=5, ge=1)


class AskRequest(ServerSchema):
    query: str
    collection_id: str
    top_k: int = Field(default=5, ge=1)
    agentic: bool = False
    engine: str | None = None


class ErrorBody(ServerSchema):
    type: str
    message: str


class ErrorResponse(ServerSchema):
    error: ErrorBody


def error_response(error_type: str, message: str) -> dict[str, Any]:
    return ErrorResponse(error=ErrorBody(type=error_type, message=message)).model_dump(mode="json")
