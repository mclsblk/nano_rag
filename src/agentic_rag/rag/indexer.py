from pathlib import Path

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
        documents = self.loader.load(path)
        chunks = self.splitter.split_documents(documents)
        if not chunks:
            return 0

        self.vectorstore.add_documents(chunks)
        return len(chunks)
