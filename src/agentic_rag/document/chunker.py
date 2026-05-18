from dataclasses import dataclass
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from agentic_rag.core import Chunk, Document, DocumentError
from agentic_rag.document.builder import BuiltSection, DocumentBuilder, TextBlock
from agentic_rag.models import EmbeddingModel


@dataclass(frozen=True)
class _TextUnit:
    content: str
    start_char: int
    end_char: int
    block_index: int = 0


@dataclass(frozen=True)
class _DraftChunk:
    content: str
    metadata: dict[str, Any]
    start_char: int
    end_char: int
    page_start: int | None = None
    page_end: int | None = None
    page_count: int | None = None
    hard_break_key: str | None = None
    semantic_score: float = 1.0


class SemanticChunker:
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        chunk_min_chars: int = 120,
        breakpoint_threshold: float = 0.35,
        page_merge_min_score: float = 0.55,
        max_units_per_chunk: int = 12,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise DocumentError("chunk_size must be greater than 0.")
        if chunk_overlap < 0:
            raise DocumentError("chunk_overlap must be greater than or equal to 0.")
        if chunk_overlap >= chunk_size:
            raise DocumentError("chunk_overlap must be smaller than chunk_size.")
        if chunk_min_chars < 0:
            raise DocumentError("chunk_min_chars must be greater than or equal to 0.")
        if not 0.0 <= breakpoint_threshold <= 1.0:
            raise DocumentError("breakpoint_threshold must be between 0.0 and 1.0.")
        if not 0.0 <= page_merge_min_score <= 1.0:
            raise DocumentError("page_merge_min_score must be between 0.0 and 1.0.")
        if max_units_per_chunk <= 0:
            raise DocumentError("max_units_per_chunk must be greater than 0.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_min_chars = chunk_min_chars
        self.breakpoint_threshold = breakpoint_threshold
        self.page_merge_min_score = page_merge_min_score
        self.max_units_per_chunk = max_units_per_chunk
        self.embedding_model = embedding_model
        self.builder = DocumentBuilder(block_min_chars=chunk_min_chars, block_max_chars=chunk_size)

    def split_documents(self, documents: list[Document]) -> list[Chunk]:
        sections = self.builder.build_documents(documents)
        chunks: list[Chunk] = []
        index = 0

        while index < len(sections):
            source = sections[index].source
            source_sections: list[BuiltSection] = []
            while index < len(sections) and sections[index].source == source:
                source_sections.append(sections[index])
                index += 1

            if all(section.page_number is not None for section in source_sections):
                source_sections = sorted(source_sections, key=lambda section: section.page_number or 0)

            drafts: list[_DraftChunk] = []
            for section in source_sections:
                drafts.extend(self._split_section_to_drafts(section))

            chunks.extend(self._finalize_chunks(self._merge_pdf_page_boundaries(drafts), source))

        return chunks

    def _split_section_to_drafts(self, section: BuiltSection) -> list[_DraftChunk]:
        units = self._blocks_to_units(section.blocks)
        if not units:
            return []

        similarities = self._unit_boundary_scores(units)
        breakpoint_cutoff = _adaptive_breakpoint_cutoff(similarities, self.breakpoint_threshold)
        drafts: list[_DraftChunk] = []
        current_units: list[_TextUnit] = []
        current_scores: list[float] = []

        for index, unit in enumerate(units):
            if not current_units:
                current_units = [unit]
                current_scores = []
                continue

            similarity = similarities[index - 1]
            would_exceed_size = sum(_word_count(item.content) for item in [*current_units, unit]) > self.chunk_size
            would_exceed_units = len(current_units) >= self.max_units_per_chunk
            should_break = similarity <= breakpoint_cutoff or would_exceed_size or would_exceed_units

            if should_break:
                drafts.append(self._units_to_draft(section, current_units, current_scores))
                current_units = [unit]
                current_scores = []
            else:
                current_units.append(unit)
                current_scores.append(similarity)

        if current_units:
            drafts.append(self._units_to_draft(section, current_units, current_scores))

        return self._merge_small_chunks(drafts)

    def _blocks_to_units(self, blocks: list[TextBlock]) -> list[_TextUnit]:
        units: list[_TextUnit] = []
        for block_index, block in enumerate(blocks):
            if _word_count(block.content) <= self.chunk_size:
                units.append(
                    _TextUnit(
                        content=block.content,
                        start_char=block.start_char,
                        end_char=block.end_char,
                        block_index=block_index,
                    )
                )
                continue

            cursor = 0
            while cursor < len(block.content):
                unit_end = _end_index_after_words(block.content, cursor, self.chunk_size)
                if unit_end <= cursor:
                    break

                unit_content = block.content[cursor:unit_end]
                unit_start = block.start_char + cursor
                absolute_unit_end = block.start_char + unit_end
                if unit_content.strip():
                    leading_trim = len(unit_content) - len(unit_content.lstrip())
                    trailing_trim = len(unit_content) - len(unit_content.rstrip())
                    units.append(
                        _TextUnit(
                            content=unit_content.strip(),
                            start_char=unit_start + leading_trim,
                            end_char=absolute_unit_end - trailing_trim,
                            block_index=block_index,
                        )
                    )
                cursor = unit_end

        return units

    def _units_to_draft(
        self,
        section: BuiltSection,
        units: list[_TextUnit],
        scores: list[float],
    ) -> _DraftChunk:
        content = units[0].content
        previous_block_index = units[0].block_index
        for unit in units[1:]:
            separator = "" if unit.block_index == previous_block_index else "\n\n"
            content = f"{content.rstrip()}{separator}{unit.content.lstrip()}"
            previous_block_index = unit.block_index

        return _DraftChunk(
            content=content.strip(),
            metadata=_scalar_metadata(section.metadata),
            start_char=units[0].start_char,
            end_char=units[-1].end_char,
            page_start=section.page_number,
            page_end=section.page_number,
            page_count=_int_metadata(section.metadata.get("page_count")),
            hard_break_key=section.hard_break_key,
            semantic_score=_average(scores),
        )

    def _merge_small_chunks(self, drafts: list[_DraftChunk]) -> list[_DraftChunk]:
        if self.chunk_min_chars <= 0 or len(drafts) <= 1:
            return drafts

        merged: list[_DraftChunk] = []
        index = 0
        while index < len(drafts):
            draft = drafts[index]
            if _word_count(draft.content) >= self.chunk_min_chars:
                merged.append(draft)
                index += 1
                continue

            if index + 1 < len(drafts) and _combined_length(draft, drafts[index + 1]) <= self.chunk_size:
                merged.append(self._merge_drafts(draft, drafts[index + 1]))
                index += 2
                continue

            if merged and _combined_length(merged[-1], draft) <= self.chunk_size:
                merged[-1] = self._merge_drafts(merged[-1], draft)
            else:
                merged.append(draft)
            index += 1

        return merged

    def _merge_pdf_page_boundaries(self, drafts: list[_DraftChunk]) -> list[_DraftChunk]:
        if len(drafts) <= 1:
            return drafts

        merged: list[_DraftChunk] = []
        for draft in drafts:
            if not merged:
                merged.append(draft)
                continue

            previous = merged[-1]
            if previous.page_end is None or draft.page_start is None:
                merged.append(draft)
                continue
            if draft.page_start != previous.page_end + 1:
                merged.append(draft)
                continue
            if draft.hard_break_key is not None and draft.hard_break_key != previous.hard_break_key:
                merged.append(draft)
                continue
            if _combined_length(previous, draft) > self.chunk_size:
                merged.append(draft)
                continue

            score = self._semantic_similarity(previous.content, draft.content)
            if score < self.page_merge_min_score:
                merged.append(draft)
                continue

            merged[-1] = self._merge_drafts(previous, draft, semantic_score=score)

        return merged

    def _merge_drafts(
        self,
        first: _DraftChunk,
        second: _DraftChunk,
        semantic_score: float | None = None,
    ) -> _DraftChunk:
        metadata = dict(first.metadata)
        metadata.update(second.metadata)

        return _DraftChunk(
            content=f"{first.content.rstrip()}\n\n{second.content.lstrip()}".strip(),
            metadata=metadata,
            start_char=first.start_char,
            end_char=second.end_char,
            page_start=first.page_start,
            page_end=second.page_end,
            page_count=first.page_count or second.page_count,
            hard_break_key=first.hard_break_key or second.hard_break_key,
            semantic_score=semantic_score if semantic_score is not None else _average([first.semantic_score, second.semantic_score]),
        )

    def _finalize_chunks(self, drafts: list[_DraftChunk], source: str) -> list[Chunk]:
        chunk_count = len(drafts)
        chunks: list[Chunk] = []

        for index, draft in enumerate(drafts, start=1):
            metadata = dict(draft.metadata)
            metadata.update(
                {
                    "chunk_strategy": "semantic",
                    "chunk_index": index,
                    "chunk_count": chunk_count,
                    "chunk_start_char": draft.start_char,
                    "chunk_end_char": draft.end_char,
                    "semantic_score": draft.semantic_score,
                }
            )

            if draft.page_start is not None and draft.page_end is not None:
                metadata["page_start"] = draft.page_start
                metadata["page_end"] = draft.page_end
                metadata["page_number"] = draft.page_start
            if draft.page_count is not None:
                metadata["page_count"] = draft.page_count
            if draft.hard_break_key is not None:
                metadata["hard_break_key"] = draft.hard_break_key

            chunks.append(
                Chunk(
                    id=f"{source}:chunk:{index:04d}",
                    content=draft.content,
                    metadata=_scalar_metadata(metadata),
                )
            )

        return chunks

    def _semantic_similarity(self, first: str, second: str) -> float:
        first = first.strip()
        second = second.strip()
        if not first or not second:
            return 0.0
        if first == second:
            return 1.0

        if self.embedding_model is not None:
            embeddings = self.embedding_model.embed_documents([first, second])
            return _cosine_score(embeddings[0], embeddings[1])

        try:
            matrix = TfidfVectorizer(analyzer="char", ngram_range=(2, 4)).fit_transform([first, second])
            score = float(cosine_similarity(matrix[0], matrix[1])[0][0])
        except ValueError:
            score = _character_jaccard(first, second)

        return max(0.0, min(1.0, score))

    def _unit_boundary_scores(self, units: list[_TextUnit]) -> list[float]:
        if len(units) <= 1:
            return []

        if self.embedding_model is not None:
            embeddings = self.embedding_model.embed_documents([unit.content for unit in units])
            return [
                _cosine_score(
                    _mean_embedding(embeddings[max(0, index - 1) : index + 1]),
                    _mean_embedding(embeddings[index + 1 : min(len(units), index + 3)]),
                )
                for index in range(len(units) - 1)
            ]

        try:
            matrix = TfidfVectorizer(analyzer="char", ngram_range=(2, 4)).fit_transform(
                [unit.content for unit in units]
            )
        except ValueError:
            return [_character_jaccard(units[index].content, units[index + 1].content) for index in range(len(units) - 1)]

        scores: list[float] = []
        for index in range(len(units) - 1):
            left_start = max(0, index - 1)
            right_end = min(len(units), index + 3)
            left_vector = matrix[left_start : index + 1].mean(axis=0).A
            right_vector = matrix[index + 1 : right_end].mean(axis=0).A
            score = float(cosine_similarity(left_vector, right_vector)[0][0])
            scores.append(max(0.0, min(1.0, score)))

        return scores


def _int_metadata(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _scalar_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    scalar_metadata: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            scalar_metadata[key] = value
    return scalar_metadata


def _average(scores: list[float]) -> float:
    if not scores:
        return 1.0
    return sum(scores) / len(scores)


def _adaptive_breakpoint_cutoff(scores: list[float], default_threshold: float) -> float:
    if len(scores) < 3:
        return default_threshold

    sorted_scores = sorted(scores)
    if sorted_scores[-1] - sorted_scores[0] < 0.08:
        return default_threshold

    percentile_index = int((len(sorted_scores) - 1) * 0.25)
    return max(default_threshold, sorted_scores[percentile_index])


def _combined_length(first: _DraftChunk, second: _DraftChunk) -> int:
    return _word_count(first.content) + _word_count(second.content)


def _character_jaccard(first: str, second: str) -> float:
    first_chars = set(first)
    second_chars = set(second)
    if not first_chars or not second_chars:
        return 0.0
    return len(first_chars & second_chars) / len(first_chars | second_chars)


def _mean_embedding(embeddings: list[list[float]]) -> list[float]:
    if not embeddings:
        return []

    dimension = len(embeddings[0])
    return [sum(embedding[index] for embedding in embeddings) / len(embeddings) for index in range(dimension)]


def _cosine_score(first: list[float], second: list[float]) -> float:
    if not first or not second or len(first) != len(second):
        return 0.0

    dot_product = sum(left * right for left, right in zip(first, second))
    first_norm = sum(value * value for value in first) ** 0.5
    second_norm = sum(value * value for value in second) ** 0.5
    if first_norm == 0.0 or second_norm == 0.0:
        return 0.0

    score = dot_product / (first_norm * second_norm)
    return max(0.0, min(1.0, score))


def _word_count(text: str) -> int:
    return sum(1 for character in text if _is_word_char(character))


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


def _is_word_char(character: str) -> bool:
    return character.isalnum()
