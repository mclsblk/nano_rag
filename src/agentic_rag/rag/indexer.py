from pathlib import Path
from typing import Protocol

from agentic_rag.core import Chunk, Document, IngestResponse
from agentic_rag.document import DocumentLoader
from agentic_rag.keyword import KeywordStore
from agentic_rag.vectorstore import VectorStore


class DocumentChunker(Protocol):
    def split_documents(self, documents: list[Document]) -> list[Chunk]:
        raise NotImplementedError


class Indexer:
    def __init__(
        self,
        loader: DocumentLoader,
        splitter: DocumentChunker,
        vectorstore: VectorStore,
        keyword_store: KeywordStore | None = None,
    ) -> None:
        self.loader = loader
        self.splitter = splitter
        self.vectorstore = vectorstore
        self.keyword_store = keyword_store

    def ingest_file(
        self,
        path: str | Path,
        *,
        file_id: str,
        collection_id: str,
        source: str,
    ) -> IngestResponse:
        load_report = self.loader.load_with_report(path)
        documents = [
            _registry_document(
                document,
                file_id=file_id,
                collection_id=collection_id,
                source=source,
                index=index,
            )
            for index, document in enumerate(load_report.documents, start=1)
        ]
        chunks = [
            _registry_chunk(chunk, file_id=file_id, collection_id=collection_id, index=index)
            for index, chunk in enumerate(self.splitter.split_documents(documents), start=1)
        ]

        if self.keyword_store is not None:
            self.keyword_store.add_documents(documents)
            self.keyword_store.add_chunks(chunks)

        if chunks:
            for i in range(0, len(chunks), 50):
                self.vectorstore.add_documents(chunks[i : i + 50])

        return IngestResponse(
            path=str(path),
            file_id=file_id,
            collection_id=collection_id,
            index_status="indexed",
            loaded_documents=len(documents),
            generated_chunks=len(chunks),
            stored_chunks=len(chunks),
            skipped=load_report.skipped,
        )


def _registry_document(
    document: Document,
    *,
    file_id: str,
    collection_id: str,
    source: str,
    index: int,
) -> Document:
    metadata = dict(document.metadata)
    metadata.update(
        {
            "file_id": file_id,
            "collection_id": collection_id,
            "source": source,
        }
    )
    return document.model_copy(
        update={
            "id": f"{collection_id}:{file_id}:doc:{index:04d}",
            "metadata": metadata,
        }
    )


def _registry_chunk(chunk: Chunk, *, file_id: str, collection_id: str, index: int) -> Chunk:
    metadata = dict(chunk.metadata)
    metadata.update(
        {
            "file_id": file_id,
            "collection_id": collection_id,
            "chunk_index": index,
        }
    )
    return chunk.model_copy(
        update={
            "id": f"{collection_id}:{file_id}:chunk:{index:04d}",
            "metadata": metadata,
        }
    )
