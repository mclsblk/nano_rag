from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re

from agentic_rag.core import Document


_NUMBER_PATTERN = r"(?:\d+|[一二三四五六七八九十百千万〇零两]+)"
_DECORATIVE_LINE_RE = re.compile(r"^\s*(?:[-_=*#~·•—–—|+]{3,}|(?:[<《【\[]\s*[-_=*#~·•—–—|+]{2,}\s*[>》】\]])+)\s*$")
_PDF_PAGE_MARKER_RE = re.compile(
    rf"^\s*(?:第\s*{_NUMBER_PATTERN}\s*页|Page\s*\d+(?:\s*/\s*\d+)?|\d+\s*/\s*\d+|[-–—]\s*\d{{1,4}}\s*[-–—])\s*$",
    re.IGNORECASE,
)
_PURE_NUMBER_LINE_RE = re.compile(rf"^[（(]?\s*{_NUMBER_PATTERN}\s*[）)]?$")
_SPACES_RE = re.compile(r"[ \t\f\v]+")
_EDGE_SEPARATOR_RE = re.compile(r"[\s\-–—_/|:：,，.。]+")
_EDGE_BOUNDARY_NUMBER_RE = re.compile(r"(^|[\s\-–—_/|:：])\d{1,4}(?=$|[\s\-–—_/|:：])")
_ENGLISH_PAGE_RE = re.compile(r"\bpage\s*\d+(?:\s*/\s*\d+)?\b", re.IGNORECASE)
_CHINESE_PAGE_RE = re.compile(rf"第\s*{_NUMBER_PATTERN}\s*页")


@dataclass(frozen=True)
class _PdfPage:
    document: Document
    source: str
    page_number: int


@dataclass(frozen=True)
class _PdfEdgePatterns:
    top: frozenset[str] = frozenset()
    bottom: frozenset[str] = frozenset()


class _DocumentCleaner:
    def __init__(
        self,
        edge_lines: int = 3,
        repeat_threshold: float = 0.6,
        min_pages: int = 3,
        max_line_chars: int = 120,
    ) -> None:
        self.edge_lines = edge_lines
        self.repeat_threshold = repeat_threshold
        self.min_pages = min_pages
        self.max_line_chars = max_line_chars

    def clean_documents(self, documents: list[Document]) -> list[Document]:
        pdf_edge_patterns_by_source = {
            source: _detect_repeated_pdf_edges(
                pages,
                edge_lines=self.edge_lines,
                repeat_threshold=self.repeat_threshold,
                min_pages=self.min_pages,
                max_line_chars=self.max_line_chars,
            )
            for source, pages in _group_pdf_pages_by_source(documents).items()
        }

        cleaned_documents: list[Document] = []
        for document in documents:
            source = _document_source(document)
            edge_patterns = pdf_edge_patterns_by_source.get(source, _PdfEdgePatterns())
            cleaned_content = _clean_document_lines(
                document,
                edge_patterns=edge_patterns,
                edge_lines=self.edge_lines,
            )
            cleaned_documents.append(document.model_copy(update={"content": cleaned_content}))

        return cleaned_documents


def _group_pdf_pages_by_source(documents: list[Document]) -> dict[str, list[_PdfPage]]:
    grouped: dict[str, list[_PdfPage]] = defaultdict(list)
    for document in documents:
        if document.metadata.get("file_type") != ".pdf":
            continue

        source = _document_source(document)
        page_number = _page_number(document)
        if not source or page_number is None:
            continue

        grouped[source].append(_PdfPage(document=document, source=source, page_number=page_number))

    return {
        source: sorted(pages, key=lambda page: page.page_number)
        for source, pages in grouped.items()
    }


def _detect_repeated_pdf_edges(
    pages: list[_PdfPage],
    edge_lines: int,
    repeat_threshold: float,
    min_pages: int,
    max_line_chars: int,
) -> _PdfEdgePatterns:
    if len(pages) < min_pages or edge_lines <= 0:
        return _PdfEdgePatterns()

    top_counts: Counter[str] = Counter()
    bottom_counts: Counter[str] = Counter()

    for page in pages:
        lines = _candidate_edge_lines(page.document.content, max_line_chars=max_line_chars)
        top_counts.update(_edge_keys_for_page(lines[:edge_lines]))
        bottom_counts.update(_edge_keys_for_page(lines[-edge_lines:]))

    minimum_matches = max(1, math.ceil(len(pages) * repeat_threshold))
    return _PdfEdgePatterns(
        top=frozenset(key for key, count in top_counts.items() if count >= minimum_matches),
        bottom=frozenset(key for key, count in bottom_counts.items() if count >= minimum_matches),
    )


def _normalize_edge_line_key(line: str) -> str:
    cleaned_line = _clean_line(line)
    if not cleaned_line or _is_decorative_line(cleaned_line):
        return ""
    if _is_pdf_page_marker(cleaned_line):
        return "<page-marker>"

    key = cleaned_line.casefold()
    key = _ENGLISH_PAGE_RE.sub(" page <num> ", key)
    key = _CHINESE_PAGE_RE.sub(" page <num> ", key)
    key = _EDGE_BOUNDARY_NUMBER_RE.sub(lambda match: f"{match.group(1)}<num>", key)
    key = _EDGE_SEPARATOR_RE.sub(" ", key).strip()

    if not key:
        return ""
    if _word_count(key) <= 1 and "<num>" not in key:
        return ""
    return key


def _clean_document_lines(
    document: Document,
    edge_patterns: _PdfEdgePatterns,
    edge_lines: int = 3,
) -> str:
    is_pdf = document.metadata.get("file_type") == ".pdf"
    raw_lines = document.content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    normalized_lines = [_clean_line(line) for line in raw_lines]
    nonempty_indexes = [index for index, line in enumerate(normalized_lines) if line]
    top_indexes = set(nonempty_indexes[:edge_lines]) if is_pdf and edge_lines > 0 else set()
    bottom_indexes = set(nonempty_indexes[-edge_lines:]) if is_pdf and edge_lines > 0 else set()

    filtered_lines: list[str] = []
    for index, raw_line in enumerate(raw_lines):
        normalized_line = normalized_lines[index]
        if not normalized_line:
            filtered_lines.append(raw_line)
            continue
        if _is_decorative_line(normalized_line):
            continue
        if is_pdf and _is_pdf_page_marker(normalized_line):
            continue
        if _is_short_number_line(normalized_line):
            continue
        if is_pdf and _is_repeated_edge_line(index, normalized_line, top_indexes, bottom_indexes, edge_patterns):
            continue
        filtered_lines.append(raw_line)

    return "\n".join(filtered_lines)


def _is_decorative_line(line: str) -> bool:
    return bool(_DECORATIVE_LINE_RE.match(line))


def _is_pdf_page_marker(line: str) -> bool:
    return bool(_PDF_PAGE_MARKER_RE.match(line))


def _is_short_number_line(line: str) -> bool:
    return len(line) <= 4 and bool(_PURE_NUMBER_LINE_RE.match(line))


def _candidate_edge_lines(content: str, max_line_chars: int) -> list[str]:
    lines: list[str] = []
    for line in content.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        cleaned_line = _clean_line(line)
        if not cleaned_line or len(cleaned_line) > max_line_chars:
            continue
        lines.append(cleaned_line)
    return lines


def _edge_keys_for_page(lines: list[str]) -> set[str]:
    return {key for line in lines if (key := _normalize_edge_line_key(line))}


def _is_repeated_edge_line(
    index: int,
    line: str,
    top_indexes: set[int],
    bottom_indexes: set[int],
    edge_patterns: _PdfEdgePatterns,
) -> bool:
    key = _normalize_edge_line_key(line)
    if not key:
        return False
    if index in top_indexes and key in edge_patterns.top:
        return True
    if index in bottom_indexes and key in edge_patterns.bottom:
        return True
    return False


def _clean_line(line: str) -> str:
    return _SPACES_RE.sub(" ", line.strip())


def _document_source(document: Document) -> str:
    source = document.metadata.get("source")
    return source if isinstance(source, str) and source else document.id


def _page_number(document: Document) -> int | None:
    value = document.metadata.get("page_number")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _word_count(text: str) -> int:
    return sum(1 for character in text if character.isalnum())
