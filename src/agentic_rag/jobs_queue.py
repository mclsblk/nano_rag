from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from redis import Redis
from rq import Queue

from agentic_rag.config import QueueSettings, Settings, load_settings
from agentic_rag.factory import create_job_service


def create_redis(settings: Settings | None = None) -> Redis:
    resolved_settings = settings or load_settings()
    return Redis.from_url(resolved_settings.queue.redis_url)


def create_ingest_queue(settings: Settings | None = None) -> Queue:
    resolved_settings = settings or load_settings()
    return Queue(
        resolved_settings.queue.rq_queue_name,
        connection=create_redis(resolved_settings),
        default_timeout=resolved_settings.queue.ingest_job_timeout_seconds,
    )


def enqueue_ingest_job(job_id: str, settings: Settings | None = None) -> str:
    resolved_settings = settings or load_settings()
    queue = create_ingest_queue(resolved_settings)
    rq_job = queue.enqueue(
        "agentic_rag.worker.tasks.run_ingest_job",
        job_id,
        job_timeout=resolved_settings.queue.ingest_job_timeout_seconds,
    )
    create_job_service(resolved_settings).attach_queue_job(job_id, rq_job.id)
    return rq_job.id


@contextmanager
def ingest_lock(file_id: str, collection_id: str, settings: Settings | None = None) -> Iterator[None]:
    resolved_settings = settings or load_settings()
    queue_settings: QueueSettings = resolved_settings.queue
    redis = create_redis(resolved_settings)
    lock = redis.lock(
        f"ingest:{collection_id}:{file_id}",
        timeout=queue_settings.ingest_lock_timeout_seconds,
        blocking_timeout=queue_settings.ingest_lock_timeout_seconds,
    )
    with lock:
        yield
