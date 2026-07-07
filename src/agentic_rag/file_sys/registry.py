from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from agentic_rag.file_sys.collections import CollectionService
from agentic_rag.core import (
    CollectionDeleteResponse,
    CollectionRecord,
    DeIngestResponse,
    FileDeleteResponse,
    IngestResponse,
    RegistryError,
    RegistryListResponse,
    RegistryRecord,
)
from agentic_rag.file_sys.files import FileService
from agentic_rag.file_sys.store import SystemStore


class RegistryService:
    def __init__(
        self,
        store: SystemStore,
        files: FileService,
        collections: CollectionService,
        indexer_factory: Callable[[CollectionRecord, str | None], Any],
        vectorstore_factory: Callable[[CollectionRecord], Any],
        keyword_store_factory: Callable[[CollectionRecord], Any],
    ) -> None:
        self.store = store
        self.files = files
        self.collections = collections
        self.indexer_factory = indexer_factory
        self.vectorstore_factory = vectorstore_factory
        self.keyword_store_factory = keyword_store_factory

    def ingest(
        self,
        file_id: str,
        collection_id: str,
        *,
        load_strategy: str | None = None,
    ) -> IngestResponse:
        file = self.files.get_file(file_id)
        collection = self.collections.get_collection(collection_id)
        record = self.get_record_or_none(file_id, collection_id)
        if record is not None and record.index_status == "indexed":
            raise RegistryError(f"File is already indexed in collection: {file_id} + {collection_id}")

        self._upsert_record(
            file_id=file_id,
            collection_id=collection_id,
            index_status="indexing",
            indexed_chunk_count=0,
            indexed_at=None,
            last_error=None,
        )
        try:
            response = self.indexer_factory(collection, load_strategy).ingest_file(
                file.storage_path,
                file_id=file_id,
                collection_id=collection_id,
                source=f"{file.file_id}/{file.original_name}",
            )
        except Exception as exc:
            self._cleanup_partial_index(collection, file_id, collection_id)
            self._upsert_record(
                file_id=file_id,
                collection_id=collection_id,
                index_status="failed",
                indexed_chunk_count=0,
                indexed_at=None,
                last_error=str(exc),
            )
            raise

        vectorstore = self.vectorstore_factory(collection)
        if hasattr(vectorstore, "mark_file_indexed"):
            vectorstore.mark_file_indexed(file_id, collection_id)

        self._upsert_record(
            file_id=file_id,
            collection_id=collection_id,
            index_status="indexed",
            indexed_chunk_count=response.stored_chunks,
            indexed_at=datetime.now(timezone.utc).isoformat(),
            last_error=None,
        )
        return response.model_copy(
            update={
                "file_id": file_id,
                "collection_id": collection_id,
                "index_status": "indexed",
            }
        )

    def de_ingest(self, file_id: str, collection_id: str) -> DeIngestResponse:
        collection = self.collections.get_collection(collection_id)
        record = self.get_record_or_none(file_id, collection_id)
        if record is None or record.index_status != "indexed":
            raise RegistryError(f"Active registry record does not exist: {file_id} + {collection_id}")

        deleted_chunks = self.vectorstore_factory(collection).delete_by_file(file_id, collection_id)
        deleted_keyword_chunks = self.keyword_store_factory(collection).delete_by_file_id(file_id)
        self._upsert_record(
            file_id=file_id,
            collection_id=collection_id,
            index_status="de_ingested",
            indexed_chunk_count=0,
            indexed_at=None,
            last_error=None,
        )
        return DeIngestResponse(
            source=file_id,
            file_id=file_id,
            collection_id=collection_id,
            index_status="de_ingested",
            deleted_chunks=deleted_chunks,
            deleted_keyword_chunks=deleted_keyword_chunks,
        )

    def _cleanup_partial_index(self, collection: CollectionRecord, file_id: str, collection_id: str) -> None:
        try:
            self.vectorstore_factory(collection).delete_by_file(file_id, collection_id)
        except Exception:
            pass

    def delete_file(self, file_id: str) -> FileDeleteResponse:
        records = self.active_records_for_file(file_id)
        if records:
            collection_ids = ", ".join(record.collection_id for record in records)
            raise RegistryError(
                f"File still has active indexed registry records: {file_id}. De-ingest from: {collection_ids}"
            )
        with self.store.connect() as connection:
            connection.execute("DELETE FROM registry WHERE file_id = %s", (file_id,))
        return self.files.delete_file(file_id)

    def delete_collection(self, collection_id: str) -> CollectionDeleteResponse:
        records = self.active_records_for_collection(collection_id)
        if records:
            file_ids = ", ".join(record.file_id for record in records)
            raise RegistryError(
                f"Collection still has active indexed registry records: {collection_id}. De-ingest files: {file_ids}"
            )
        with self.store.connect() as connection:
            connection.execute("DELETE FROM registry WHERE collection_id = %s", (collection_id,))
        return self.collections.delete_collection(collection_id)

    def list_records(self) -> RegistryListResponse:
        with self.store.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM registry ORDER BY collection_id, file_id"
            ).fetchall()
        return RegistryListResponse(records=[_registry_record(row) for row in rows])

    def get_record_or_none(self, file_id: str, collection_id: str) -> RegistryRecord | None:
        with self.store.connect() as connection:
            row = connection.execute(
                "SELECT * FROM registry WHERE file_id = %s AND collection_id = %s",
                (file_id, collection_id),
            ).fetchone()
        return _registry_record(row) if row is not None else None

    def active_records_for_file(self, file_id: str) -> list[RegistryRecord]:
        return self._active_records("file_id", file_id)

    def active_records_for_collection(self, collection_id: str) -> list[RegistryRecord]:
        return self._active_records("collection_id", collection_id)

    def count_records(self) -> int:
        with self.store.connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM registry").fetchone()
        return int(row["count"]) if row is not None else 0

    def count_indexed_chunks(self) -> int:
        with self.store.connect() as connection:
            row = connection.execute(
                """
                SELECT COALESCE(SUM(indexed_chunk_count), 0) AS count
                FROM registry
                WHERE index_status = 'indexed'
                """
            ).fetchone()
        return int(row["count"]) if row is not None else 0

    def _active_records(self, column: str, value: str) -> list[RegistryRecord]:
        if column not in {"file_id", "collection_id"}:
            raise RegistryError(f"Unsupported registry lookup column: {column}")
        with self.store.connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM registry WHERE {column} = %s AND index_status IN ('indexing', 'indexed')",
                (value,),
            ).fetchall()
        return [_registry_record(row) for row in rows]

    def _upsert_record(
        self,
        *,
        file_id: str,
        collection_id: str,
        index_status: str,
        indexed_chunk_count: int,
        indexed_at: str | None,
        last_error: str | None,
    ) -> None:
        with self.store.connect() as connection:
            connection.execute(
                """
                INSERT INTO registry (
                    file_id, collection_id, index_status,
                    indexed_chunk_count, indexed_at, last_error
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT(file_id, collection_id) DO UPDATE SET
                    index_status = excluded.index_status,
                    indexed_chunk_count = excluded.indexed_chunk_count,
                    indexed_at = excluded.indexed_at,
                    last_error = excluded.last_error
                """,
                (file_id, collection_id, index_status, indexed_chunk_count, indexed_at, last_error),
            )


def _registry_record(row: Any) -> RegistryRecord:
    return RegistryRecord(
        file_id=str(row["file_id"]),
        collection_id=str(row["collection_id"]),
        index_status=str(row["index_status"]),
        indexed_chunk_count=int(row["indexed_chunk_count"]),
        indexed_at=str(row["indexed_at"]) if row["indexed_at"] is not None else None,
        last_error=str(row["last_error"]) if row["last_error"] is not None else None,
    )
