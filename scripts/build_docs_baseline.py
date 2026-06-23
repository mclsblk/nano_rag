from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag.factory import (
    create_collection_service,
    create_file_service,
    create_registry_service,
)


SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and ingest the docs baseline collection.")
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--collection-name", default="docs-baseline-20260620")
    parser.add_argument(
        "--description",
        default="Baseline collection for docs/ materials, created on 2026-06-20.",
    )
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    if not docs_dir.is_dir():
        raise SystemExit(f"Docs directory does not exist: {docs_dir}")

    collection_service = create_collection_service()
    existing = [
        collection
        for collection in collection_service.list_collections().collections
        if collection.name == args.collection_name
    ]
    if existing:
        collection = existing[0]
        created = False
    else:
        collection = collection_service.create_collection(
            args.collection_name,
            description=args.description,
        ).collection
        created = True

    file_service = create_file_service()
    registry_service = create_registry_service()
    paths = sorted(
        path
        for path in docs_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )

    results = []
    for path in paths:
        file_record = file_service.import_file(path).file
        record = registry_service.get_record_or_none(file_record.file_id, collection.collection_id)
        if record is not None and record.index_status == "indexed" and record.indexed_chunk_count == 0:
            registry_service.de_ingest(file_record.file_id, collection.collection_id)
        try:
            ingest = registry_service.ingest(file_record.file_id, collection.collection_id)
            results.append(
                {
                    "path": str(path),
                    "file_id": file_record.file_id,
                    "status": "ingested",
                    "loaded_documents": ingest.loaded_documents,
                    "generated_chunks": ingest.generated_chunks,
                    "stored_chunks": ingest.stored_chunks,
                    "skipped": ingest.skipped,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "path": str(path),
                    "file_id": file_record.file_id,
                    "status": "error",
                    "error": str(exc),
                }
            )

    print(
        json.dumps(
            {
                "collection_created": created,
                "collection_id": collection.collection_id,
                "collection_name": collection.name,
                "keyword_index_path": collection.keyword_index_path,
                "file_count": len(paths),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
