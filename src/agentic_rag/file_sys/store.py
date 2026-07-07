from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from agentic_rag.postgres.schema import schema_sql


_INITIALIZED: set[tuple[str, int]] = set()


class SystemStore:
    def __init__(self, database_url: str, embedding_dimension: int = 1024) -> None:
        self.database_url = database_url
        self.embedding_dimension = embedding_dimension
        key = (database_url, embedding_dimension)
        if key not in _INITIALIZED:
            self.initialize()
            _INITIALIZED.add(key)

    @contextmanager
    def connect(self) -> Iterator[psycopg.Connection]:
        with psycopg.connect(self.database_url, row_factory=dict_row) as connection:
            connection.execute("SET lock_timeout = '10s'")
            register_vector(connection)
            yield connection

    def initialize(self) -> None:
        with psycopg.connect(self.database_url, autocommit=True) as connection:
            connection.execute(schema_sql(self.embedding_dimension))


__all__ = ["SystemStore"]
