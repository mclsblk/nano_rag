from __future__ import annotations

import os

from alembic import op


revision = "0001_postgres_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    embedding_dimension = int(os.environ.get("EMBEDDING_DIMENSION", "1024"))
    op.execute(
        f"""
        CREATE EXTENSION IF NOT EXISTS vector;

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
            file_id TEXT NOT NULL REFERENCES files(file_id),
            collection_id TEXT NOT NULL REFERENCES collections(collection_id),
            index_status TEXT NOT NULL,
            indexed_chunk_count INTEGER NOT NULL DEFAULT 0,
            indexed_at TEXT,
            last_error TEXT,
            PRIMARY KEY (file_id, collection_id)
        );

        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            job_type TEXT NOT NULL,
            status TEXT NOT NULL,
            file_id TEXT NOT NULL REFERENCES files(file_id),
            collection_id TEXT NOT NULL REFERENCES collections(collection_id),
            loader TEXT,
            input_path TEXT,
            upload_file_name TEXT,
            rq_job_id TEXT,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            loaded_documents INTEGER NOT NULL DEFAULT 0,
            generated_chunks INTEGER NOT NULL DEFAULT 0,
            stored_chunks INTEGER NOT NULL DEFAULT 0,
            skipped JSONB NOT NULL DEFAULT '[]'::jsonb,
            error_message TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            last_heartbeat_at TEXT
        );

        CREATE UNIQUE INDEX IF NOT EXISTS uniq_active_ingest_job
            ON jobs (file_id, collection_id)
            WHERE job_type = 'ingest' AND status IN ('queued', 'running');

        CREATE TABLE IF NOT EXISTS chunk_sources (
            source TEXT PRIMARY KEY,
            file_id TEXT,
            collection_id TEXT,
            metadata_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL REFERENCES chunk_sources(source) ON DELETE CASCADE,
            file_id TEXT,
            collection_id TEXT,
            content TEXT NOT NULL,
            metadata_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            keyword_text TEXT NOT NULL DEFAULT '',
            search_vector TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', keyword_text)) STORED,
            embedding vector({embedding_dimension}),
            chunk_index INTEGER,
            page_start INTEGER,
            page_end INTEGER,
            index_status TEXT NOT NULL DEFAULT 'indexing',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_file_collection
            ON chunks (file_id, collection_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_collection_status
            ON chunks (collection_id, index_status);
        CREATE INDEX IF NOT EXISTS idx_chunks_search_vector
            ON chunks USING GIN (search_vector);
        CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
            ON chunks USING hnsw (embedding vector_cosine_ops);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP TABLE IF EXISTS chunks;
        DROP TABLE IF EXISTS chunk_sources;
        DROP TABLE IF EXISTS jobs;
        DROP TABLE IF EXISTS registry;
        DROP TABLE IF EXISTS collections;
        DROP TABLE IF EXISTS files;
        """
    )
