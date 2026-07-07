from datetime import datetime, timezone
import json
import uuid
from typing import Any

from psycopg import errors

from agentic_rag.core import JobError, JobListResponse, JobRecord, JobResponse
from agentic_rag.file_sys.store import SystemStore


class JobService:
    def __init__(self, store: SystemStore) -> None:
        self.store = store

    def create_ingest_job(
        self,
        *,
        file_id: str,
        collection_id: str,
        loader: str | None = None,
        input_path: str | None = None,
        upload_file_name: str | None = None,
    ) -> JobResponse:
        if self.active_ingest_job_exists(file_id, collection_id):
            raise JobError(f"Active ingest job already exists: {file_id} + {collection_id}")

        now = _now()
        record = JobRecord(
            job_id=f"job_{uuid.uuid4().hex[:12]}",
            job_type="ingest",
            status="queued",
            file_id=file_id,
            collection_id=collection_id,
            loader=loader,
            input_path=input_path,
            upload_file_name=upload_file_name,
            created_at=now,
            updated_at=now,
        )
        try:
            with self.store.connect() as connection:
                connection.execute(
                    """
                    INSERT INTO jobs (
                        job_id, job_type, status, file_id, collection_id, loader,
                        input_path, upload_file_name, skipped, created_at, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                    """,
                    (
                        record.job_id,
                        record.job_type,
                        record.status,
                        record.file_id,
                        record.collection_id,
                        record.loader,
                        record.input_path,
                        record.upload_file_name,
                        json.dumps(record.skipped),
                        record.created_at,
                        record.updated_at,
                    ),
                )
        except errors.UniqueViolation as exc:
            raise JobError(f"Active ingest job already exists: {file_id} + {collection_id}") from exc
        return JobResponse(job=record)

    def get_job(self, job_id: str) -> JobResponse:
        with self.store.connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id = %s", (job_id,)).fetchone()
        if row is None:
            raise JobError(f"Job does not exist: {job_id}")
        return JobResponse(job=_job_record(row))

    def list_jobs(self) -> JobListResponse:
        with self.store.connect() as connection:
            rows = connection.execute("SELECT * FROM jobs ORDER BY created_at DESC, job_id DESC").fetchall()
        return JobListResponse(jobs=[_job_record(row) for row in rows])

    def active_ingest_job_exists(self, file_id: str, collection_id: str) -> bool:
        with self.store.connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count FROM jobs
                WHERE job_type = 'ingest'
                  AND file_id = %s
                  AND collection_id = %s
                  AND status IN ('queued', 'running')
                """,
                (file_id, collection_id),
            ).fetchone()
        return row is not None and int(row["count"]) > 0

    def count_jobs(self) -> int:
        with self.store.connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM jobs").fetchone()
        return int(row["count"]) if row is not None else 0

    def attach_queue_job(self, job_id: str, rq_job_id: str) -> JobResponse:
        with self.store.connect() as connection:
            connection.execute(
                "UPDATE jobs SET rq_job_id = %s, updated_at = %s WHERE job_id = %s",
                (rq_job_id, _now(), job_id),
            )
        return self.get_job(job_id)

    def mark_running(self, job_id: str) -> JobResponse:
        now = _now()
        with self.store.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET status = 'running',
                    updated_at = %s,
                    started_at = %s,
                    attempt_count = attempt_count + 1,
                    last_heartbeat_at = %s
                WHERE job_id = %s
                  AND status = 'queued'
                """,
                (now, now, now, job_id),
            )
            if cursor.rowcount != 1:
                raise JobError(f"Job is not queued or does not exist: {job_id}")
        return self.get_job(job_id)

    def mark_succeeded(
        self,
        job_id: str,
        *,
        loaded_documents: int,
        generated_chunks: int,
        stored_chunks: int,
        skipped: list[str],
    ) -> JobResponse:
        now = _now()
        with self.store.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET status = 'succeeded',
                    loaded_documents = %s,
                    generated_chunks = %s,
                    stored_chunks = %s,
                    skipped = %s::jsonb,
                    error_message = NULL,
                    updated_at = %s,
                    finished_at = %s
                WHERE job_id = %s
                  AND status = 'running'
                """,
                (
                    loaded_documents,
                    generated_chunks,
                    stored_chunks,
                    json.dumps(skipped),
                    now,
                    now,
                    job_id,
                ),
            )
            if cursor.rowcount != 1:
                raise JobError(f"Job is not running or does not exist: {job_id}")
        return self.get_job(job_id)

    def mark_failed(self, job_id: str, error_message: str) -> JobResponse:
        now = _now()
        with self.store.connect() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET status = 'failed',
                    error_message = %s,
                    updated_at = %s,
                    finished_at = %s
                WHERE job_id = %s
                  AND status IN ('queued', 'running')
                """,
                (error_message, now, now, job_id),
            )
        return self.get_job(job_id)

    def mark_interrupted_jobs(self) -> int:
        now = _now()
        with self.store.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET status = 'failed',
                    error_message = 'Job interrupted by service restart',
                    updated_at = %s,
                    finished_at = %s
                WHERE job_type = 'ingest'
                  AND status = 'running'
                """,
                (now, now),
            )
            return cursor.rowcount


def _job_record(row: Any) -> JobRecord:
    return JobRecord(
        job_id=str(row["job_id"]),
        job_type=str(row["job_type"]),
        status=str(row["status"]),
        file_id=str(row["file_id"]),
        collection_id=str(row["collection_id"]),
        loader=str(row["loader"]) if row["loader"] is not None else None,
        input_path=str(row["input_path"]) if row["input_path"] is not None else None,
        upload_file_name=str(row["upload_file_name"]) if row["upload_file_name"] is not None else None,
        loaded_documents=int(row["loaded_documents"]),
        generated_chunks=int(row["generated_chunks"]),
        stored_chunks=int(row["stored_chunks"]),
        skipped=_json_list(row["skipped"]),
        error_message=str(row["error_message"]) if row["error_message"] is not None else None,
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        started_at=str(row["started_at"]) if row["started_at"] is not None else None,
        finished_at=str(row["finished_at"]) if row["finished_at"] is not None else None,
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    loaded = json.loads(str(value))
    return [str(item) for item in loaded] if isinstance(loaded, list) else []
