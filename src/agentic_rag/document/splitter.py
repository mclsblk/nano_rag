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
        start = 0
        index = 1

        while start < len(content):
            end = _end_index_after_words(content, start, self.chunk_size)
            if end <= start:
                break

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

            next_start = _start_index_for_overlap(content, end, self.chunk_overlap)
            start = next_start if next_start > start else end

        return chunks


def _end_index_after_words(content: str, start: int, max_words: int) -> int:
    words_seen = 0
    for index in range(start, len(content)):
        if _is_word_char(content[index]):
            words_seen += 1
            if words_seen == max_words:
                return _next_word_start_or_end(content, index + 1)
    return len(content)


def _next_word_start_or_end(content: str, start: int) -> int:
    for index in range(start, len(content)):
        if _is_word_char(content[index]):
            return index
    return len(content)


def _start_index_for_overlap(content: str, end: int, overlap_words: int) -> int:
    if overlap_words <= 0:
        return end

    words_seen = 0
    for index in range(end - 1, -1, -1):
        if _is_word_char(content[index]):
            words_seen += 1
            if words_seen == overlap_words:
                return index
    return 0


def _is_word_char(character: str) -> bool:
    return character.isalnum()
