from pathlib import Path
import sqlite3


class SystemStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path))
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS files (
                    file_id TEXT PRIMARY KEY,
                    original_name TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE,
                    storage_path TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS collections (
                    collection_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    chroma_collection TEXT NOT NULL,
                    keyword_index_path TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS registry (
                    file_id TEXT NOT NULL,
                    collection_id TEXT NOT NULL,
                    index_status TEXT NOT NULL,
                    indexed_chunk_count INTEGER NOT NULL DEFAULT 0,
                    indexed_at TEXT,
                    last_error TEXT,
                    PRIMARY KEY (file_id, collection_id),
                    FOREIGN KEY (file_id) REFERENCES files(file_id),
                    FOREIGN KEY (collection_id) REFERENCES collections(collection_id)
                );
                """
            )


__all__ = ["SystemStore"]
