from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
import sqlite3

from agentic_rag.core import FileDeleteResponse, FileListResponse, FileRecord, FileResponse, RegistryError
from agentic_rag.file_sys.store import SystemStore


class FileService:
    def __init__(self, store: SystemStore, storage_dir: str | Path) -> None:
        self.store = store
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def import_file(self, path: str | Path) -> FileResponse:
        source_path = Path(path)
        if not source_path.is_file():
            raise RegistryError(f"File path does not exist or is not a file: {source_path}")

        content_hash = _sha256_file(source_path)
        file_id = f"sha256_{content_hash}"
        existing = self.get_file_or_none(file_id)
        if existing is not None and existing.status == "active":
            return FileResponse(file=existing)

        target_dir = self.storage_dir / content_hash[:2] / file_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / source_path.name
        shutil.copy2(source_path, target_path)

        record = FileRecord(
            file_id=file_id,
            original_name=source_path.name,
            content_hash=content_hash,
            storage_path=str(target_path),
            file_type=source_path.suffix.lower(),
            size_bytes=target_path.stat().st_size,
            status="active",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with self.store.connect() as connection:
            connection.execute(
                """
                INSERT INTO files (
                    file_id, original_name, content_hash, storage_path,
                    file_type, size_bytes, status, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(file_id) DO UPDATE SET
                    original_name = excluded.original_name,
                    storage_path = excluded.storage_path,
                    file_type = excluded.file_type,
                    size_bytes = excluded.size_bytes,
                    status = excluded.status,
                    created_at = excluded.created_at
                """,
                (
                    record.file_id,
                    record.original_name,
                    record.content_hash,
                    record.storage_path,
                    record.file_type,
                    record.size_bytes,
                    record.status,
                    record.created_at,
                ),
            )
        return FileResponse(file=record)

    def list_files(self) -> FileListResponse:
        with self.store.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM files WHERE status = 'active' ORDER BY created_at, file_id"
            ).fetchall()
        return FileListResponse(files=[_file_record(row) for row in rows])

    def get_file(self, file_id: str) -> FileRecord:
        record = self.get_file_or_none(file_id)
        if record is None or record.status != "active":
            raise RegistryError(f"File does not exist: {file_id}")
        return record

    def get_file_or_none(self, file_id: str) -> FileRecord | None:
        with self.store.connect() as connection:
            row = connection.execute("SELECT * FROM files WHERE file_id = ?", (file_id,)).fetchone()
        return _file_record(row) if row is not None else None

    def delete_file(self, file_id: str) -> FileDeleteResponse:
        record = self.get_file(file_id)
        storage_path = Path(record.storage_path)
        shutil.rmtree(storage_path.parent)
        with self.store.connect() as connection:
            connection.execute("UPDATE files SET status = 'deleted' WHERE file_id = ?", (file_id,))
        return FileDeleteResponse(file_id=file_id, status="deleted")

    def count_active(self) -> int:
        with self.store.connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM files WHERE status = 'active'"
            ).fetchone()
        return int(row["count"]) if row is not None else 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(row: sqlite3.Row) -> FileRecord:
    return FileRecord(
        file_id=str(row["file_id"]),
        original_name=str(row["original_name"]),
        content_hash=str(row["content_hash"]),
        storage_path=str(row["storage_path"]),
        file_type=str(row["file_type"]),
        size_bytes=int(row["size_bytes"]),
        status=str(row["status"]),
        created_at=str(row["created_at"]),
    )
