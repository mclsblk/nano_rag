from agentic_rag.core import Chunk, Document, DocumentError


class TextSplitter:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        if chunk_size <= 0:
            raise DocumentError("chunk_size must be greater than 0.")
        if chunk_overlap < 0:
            raise DocumentError("chunk_overlap must be greater than or equal to 0.")
        if chunk_overlap >= chunk_size:
            raise DocumentError("chunk_overlap must be smaller than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self.split_document(document))
        return chunks

    def split_document(self, document: Document) -> list[Chunk]:
        content = document.content
        if not content:
            return []

        chunks: list[Chunk] = []
        step = self.chunk_size - self.chunk_overlap
        start = 0
        index = 1

        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            chunk_content = content[start:end]
            if chunk_content.strip():
                chunks.append(
                    Chunk(
                        id=f"{document.id}:chunk:{index:04d}",
                        content=chunk_content,
                        metadata=dict(document.metadata),
                    )
                )
                index += 1

            if end == len(content):
                break
            start += step

        return chunks
