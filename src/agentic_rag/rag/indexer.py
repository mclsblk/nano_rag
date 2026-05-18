from pathlib import Path

from agentic_rag.core import IngestResponse
from agentic_rag.document import DocumentLoader, TextSplitter
from agentic_rag.vectorstore import VectorStore


class Indexer:
    def __init__(
        self,
        loader: DocumentLoader,
        splitter: TextSplitter,
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
