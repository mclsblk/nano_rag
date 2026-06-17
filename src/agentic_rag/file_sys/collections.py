from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import uuid

from agentic_rag.core import (
    CollectionDeleteResponse,
    CollectionListResponse,
    CollectionRecord,
    CollectionResponse,
    RegistryError,
)
from agentic_rag.file_sys.store import SystemStore


class CollectionService:
    def __init__(self, store: SystemStore, keyword_index_dir: str | Path) -> None:
        self.store = store
        self.keyword_index_dir = Path(keyword_index_dir)
        self.keyword_index_dir.mkdir(parents=True, exist_ok=True)

    def create_collection(self, name: str, description: str = "") -> CollectionResponse:
        collection_id = f"col_{uuid.uuid4().hex[:12]}"
        record = CollectionRecord(
            collection_id=collection_id,
            name=name,
            description=description,
            chroma_collection=collection_id,
            keyword_index_path=str(self.keyword_index_dir / f"{collection_id}.sqlite"),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with self.store.connect() as connection:
            connection.execute(
                """
                INSERT INTO collections (
                    collection_id, name, description, chroma_collection,
                    keyword_index_path, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.collection_id,
                    record.name,
                    record.description,
                    record.chroma_collection,
                    record.keyword_index_path,
                    record.created_at,
                ),
            )
        return CollectionResponse(collection=record)

    def list_collections(self) -> CollectionListResponse:
        with self.store.connect() as connection:
            rows = connection.execute("SELECT * FROM collections ORDER BY created_at, collection_id").fetchall()
        return CollectionListResponse(collections=[_collection_record(row) for row in rows])

    def get_collection(self, collection_id: str) -> CollectionRecord:
        with self.store.connect() as connection:
            row = connection.execute(
                "SELECT * FROM collections WHERE collection_id = ?",
                (collection_id,),
            ).fetchone()
        if row is None:
            raise RegistryError(f"Collection does not exist: {collection_id}")
        return _collection_record(row)

    def delete_collection(self, collection_id: str) -> CollectionDeleteResponse:
        self.get_collection(collection_id)
        with self.store.connect() as connection:
            connection.execute("DELETE FROM collections WHERE collection_id = ?", (collection_id,))
        return CollectionDeleteResponse(collection_id=collection_id, status="deleted")

    def count(self) -> int:
        with self.store.connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM collections").fetchone()
        return int(row["count"]) if row is not None else 0


def _collection_record(row: sqlite3.Row) -> CollectionRecord:
    return CollectionRecord(
        collection_id=str(row["collection_id"]),
        name=str(row["name"]),
        description=str(row["description"]),
        chroma_collection=str(row["chroma_collection"]),
        keyword_index_path=str(row["keyword_index_path"]),
        created_at=str(row["created_at"]),
    )
