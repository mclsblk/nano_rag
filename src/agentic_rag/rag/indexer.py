from pathlib import Path
from typing import Protocol

from agentic_rag.core import Chunk, DeIngestResponse, Document, IngestResponse, SourceConflictError
from agentic_rag.document import DocumentLoader
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
    ) -> None:
        self.loader = loader
        self.splitter = splitter
        self.vectorstore = vectorstore

    def ingest(self, path: str | Path) -> int:
        return self.ingest_with_report(path).stored_chunks

    def ingest_with_report(self, path: str | Path) -> IngestResponse:
        load_report = self.loader.load_with_report(path)
        documents = load_report.documents
        sources = _document_sources(documents)
        conflicts = [source for source in sources if self.vectorstore.source_exists(source)]
        if conflicts:
            raise SourceConflictError(_source_conflict_message(conflicts))

        chunks = self.splitter.split_documents(documents)
        stored_chunks = 0

        if chunks:
            stored_chunks = len(chunks)
            for i in range(0, len(chunks), 50):
                i_end = min(i + 50, len(chunks))
                self.vectorstore.add_documents(chunks[i:i_end])

        return IngestResponse(
            path=str(path),
            loaded_documents=len(documents),
            generated_chunks=len(chunks),
            stored_chunks=stored_chunks,
            skipped=load_report.skipped,
        )

    def de_ingest(self, source: str) -> DeIngestResponse:
        deleted_chunks = self.vectorstore.delete_by_source(source)
        return DeIngestResponse(source=source, deleted_chunks=deleted_chunks)


def _document_sources(documents) -> list[str]:
    sources: set[str] = set()
    for document in documents:
        source = document.metadata.get("source")
        if isinstance(source, str) and source:
            sources.add(source)
    return sorted(sources)


def _source_conflict_message(sources: list[str]) -> str:
    if len(sources) == 1:
        source = sources[0]
        return f"Source already exists: {source}. Run `rag de-ingest {source}` before ingesting again."

    source_list = ", ".join(sources)
    return f"Sources already exist: {source_list}. Run `rag de-ingest <source>` for each source before ingesting again."
