from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal


IngestProgressStage = Literal[
    "load",
    "conflicts",
    "split",
    "keyword_documents",
    "keyword_chunks",
    "vector_chunks",
    "complete",
]


@dataclass(frozen=True)
class IngestProgressEvent:
    stage: IngestProgressStage
    message: str
    completed: int | None = None
    total: int | None = None


IngestProgressCallback = Callable[[IngestProgressEvent], None]


def report_ingest_progress(
    callback: IngestProgressCallback | None,
    stage: IngestProgressStage,
    message: str,
    *,
    completed: int | None = None,
    total: int | None = None,
) -> None:
    if callback is None:
        return
    callback(IngestProgressEvent(stage=stage, message=message, completed=completed, total=total))
