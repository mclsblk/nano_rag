from pathlib import Path
from typing import Protocol

from agentic_rag.core import AgenticRAGError, Chunk, DeIngestResponse, Document, IngestResponse, SourceConflictError
from agentic_rag.document import DocumentLoader
from agentic_rag.keyword import KeywordStore
from agentic_rag.rag.progress import IngestProgressCallback, report_ingest_progress
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

    def ingest(self, path: str | Path) -> int:
        return self.ingest_with_report(path).stored_chunks

    def ingest_with_report(self, path: str | Path, progress: IngestProgressCallback | None = None) -> IngestResponse:
        report_ingest_progress(progress, "load", "Loading documents")
        load_report = self.loader.load_with_report(path)
        documents = load_report.documents
        report_ingest_progress(
            progress,
            "load",
            "Loaded documents",
            completed=len(documents),
            total=len(documents),
        )

        report_ingest_progress(progress, "conflicts", "Checking source conflicts")
        sources = _document_sources(documents)
        conflicts = self._source_conflicts(sources)
        if conflicts:
            raise SourceConflictError(_source_conflict_message(conflicts))
        report_ingest_progress(
            progress,
            "conflicts",
            "Checked source conflicts",
            completed=len(sources),
            total=len(sources),
        )

        report_ingest_progress(progress, "split", "Splitting documents")
        chunks = self.splitter.split_documents(documents)
        report_ingest_progress(progress, "split", "Generated chunks", completed=len(chunks), total=len(chunks))
        stored_chunks = 0
        should_cleanup = False

        try:
            if self.keyword_store is not None:
                report_ingest_progress(progress, "keyword_documents", "Writing keyword source records")
                self.keyword_store.add_documents(documents)
                should_cleanup = bool(sources)
                report_ingest_progress(
                    progress,
                    "keyword_documents",
                    "Wrote keyword source records",
                    completed=len(documents),
                    total=len(documents),
                )
                report_ingest_progress(
                    progress,
                    "keyword_chunks",
                    "Writing keyword chunks",
                    completed=0,
                    total=len(chunks),
                )
                self.keyword_store.add_chunks(chunks)
                report_ingest_progress(
                    progress,
                    "keyword_chunks",
                    "Wrote keyword chunks",
                    completed=len(chunks),
                    total=len(chunks),
                )

            if chunks:
                stored_chunks = len(chunks)
                report_ingest_progress(
                    progress,
                    "vector_chunks",
                    "Writing vector chunks",
                    completed=0,
                    total=len(chunks),
                )
                for i in range(0, len(chunks), 50):
                    i_end = min(i + 50, len(chunks))
                    self.vectorstore.add_documents(chunks[i:i_end])
                    should_cleanup = True
                    report_ingest_progress(
                        progress,
                        "vector_chunks",
                        "Writing vector chunks",
                        completed=i_end,
                        total=len(chunks),
                    )
        except Exception as exc:
            cleanup_errors = self._cleanup_sources(sources) if should_cleanup else []
            if cleanup_errors:
                raise AgenticRAGError(_cleanup_error_message(exc, sources, cleanup_errors)) from exc
            raise

        report_ingest_progress(progress, "complete", "Ingest complete", completed=stored_chunks, total=stored_chunks)
        return IngestResponse(
            path=str(path),
            loaded_documents=len(documents),
            generated_chunks=len(chunks),
            stored_chunks=stored_chunks,
            skipped=load_report.skipped,
        )

    def de_ingest(self, source: str) -> DeIngestResponse:
        deleted_chunks = self.vectorstore.delete_by_source(source)
        if self.keyword_store is not None:
            self.keyword_store.delete_by_source(source)
        return DeIngestResponse(source=source, deleted_chunks=deleted_chunks)

    def _source_conflicts(self, sources: list[str]) -> list[str]:
        conflicts: set[str] = set()
        for source in sources:
            if self.vectorstore.source_exists(source):
                conflicts.add(source)
            if self.keyword_store is not None and self.keyword_store.source_exists(source):
                conflicts.add(source)
        return sorted(conflicts)

    def _cleanup_sources(self, sources: list[str]) -> list[str]:
        errors: list[str] = []
        for source in sources:
            if self.keyword_store is not None:
                try:
                    self.keyword_store.delete_by_source(source)
                except Exception as exc:
                    errors.append(f"SQLite keyword store cleanup failed for {source}: {exc}")
            try:
                self.vectorstore.delete_by_source(source)
            except Exception as exc:
                errors.append(f"Chroma cleanup failed for {source}: {exc}")
        return errors


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


def _cleanup_error_message(exc: Exception, sources: list[str], cleanup_errors: list[str]) -> str:
    source_hint = ", ".join(sources) if sources else "<source>"
    cleanup_hint = "; ".join(cleanup_errors)
    return (
        f"Ingest failed: {exc}. Best-effort cleanup also failed: {cleanup_hint}. "
        f"Run `rag de-ingest {source_hint}` manually before ingesting again."
    )
