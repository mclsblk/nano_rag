from dataclasses import dataclass
import re
from typing import Any

from agentic_rag.core import Document


@dataclass(frozen=True)
class TextBlock:
    content: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class BuiltSection:
    source: str
    blocks: list[TextBlock]
    metadata: dict[str, Any]
    start_char: int
    end_char: int
    page_number: int | None = None
    hard_break_key: str | None = None


_NUMBER_PATTERN = r"(?:\d+|[一二三四五六七八九十百千万〇零两]+)"
_BLANK_LINE_RE = re.compile(r"\n\s*\n+")
_HARD_BREAK_RE = re.compile(
    rf"""
    ^\s*
    (?:
        (?:问题|问|题)\s*[:：.\-、]?\s*{_NUMBER_PATTERN}
        |
        第\s*{_NUMBER_PATTERN}\s*(?:题|问|小题)
        |
        Q(?:uestion)?\s*[:：.\-、]?\s*\d+
    )
    (?=\s|[、，,.:：。)\]】）-]|$)
    """,
    re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)
_HARD_BREAK_KEY_CLEAN_RE = re.compile(r"[\s:：.\-、，,。)\]】）]+")
_DECORATIVE_LINE_RE = re.compile(r"^\s*(?:[-_=*#~·•—–—|+]{3,}|(?:[<《【\[]\s*[-_=*#~·•—–—|+]{2,}\s*[>》】\]])+)\s*$")
_PDF_PAGE_MARKER_RE = re.compile(
    rf"^\s*(?:第\s*{_NUMBER_PATTERN}\s*页|Page\s*\d+(?:\s*/\s*\d+)?|\d+\s*/\s*\d+|[-–—]\s*\d{{1,4}}\s*[-–—])\s*$",
    re.IGNORECASE,
)
_SPACES_RE = re.compile(r"[ \t\f\v]+")


class DocumentBuilder:
    def __init__(
        self,
        block_min_chars: int = 120,
        block_max_chars: int = 800,
    ) -> None:
        self.block_min_chars = block_min_chars
        self.block_max_chars = block_max_chars

    def build_documents(self, documents: list[Document]) -> list[BuiltSection]:
        sections: list[BuiltSection] = []
        for document in documents:
            sections.extend(self._build_sections(document))
        return sections

    def _build_sections(self, document: Document) -> list[BuiltSection]:
        content = document.content.replace("\r\n", "\n").replace("\r", "\n")
        sections: list[BuiltSection] = []

        for start_char, end_char, hard_break_key in self._split_hard_break_sections(content):
            blocks = self._build_blocks(
                document=document,
                content=content,
                start_char=start_char,
                end_char=end_char,
            )
            if not blocks:
                continue

            sections.append(
                BuiltSection(
                    source=_document_source(document),
                    blocks=blocks,
                    metadata=document.metadata,
                    start_char=start_char,
                    end_char=end_char,
                    page_number=_page_number(document),
                    hard_break_key=hard_break_key,
                )
            )

        return sections

    def _split_hard_break_sections(self, content: str) -> list[tuple[int, int, str | None]]:
        matches = list(_HARD_BREAK_RE.finditer(content))
        if not matches:
            return [(0, len(content), None)]

        sections: list[tuple[int, int, str | None]] = []
        if matches[0].start() > 0:
            sections.append((0, matches[0].start(), None))

        for index, match in enumerate(matches):
            next_start = matches[index + 1].start() if index + 1 < len(matches) else len(content)
            hard_break_key = _HARD_BREAK_KEY_CLEAN_RE.sub("", match.group(0))
            sections.append((match.start(), next_start, hard_break_key))

        return sections

    def _build_blocks(
        self,
        document: Document,
        content: str,
        start_char: int,
        end_char: int,
    ) -> list[TextBlock]:
        blocks: list[TextBlock] = []
        cursor = start_char

        for match in _BLANK_LINE_RE.finditer(content, start_char, end_char):
            block = self._clean_block(document, content, cursor, match.start())
            if block is not None:
                blocks.append(block)
            cursor = match.end()

        block = self._clean_block(document, content, cursor, end_char)
        if block is not None:
            blocks.append(block)

        return self._merge_blocks(blocks)

    def _clean_block(
        self,
        document: Document,
        content: str,
        start_char: int,
        end_char: int,
    ) -> TextBlock | None:
        raw_block = content[start_char:end_char]
        is_pdf = document.metadata.get("file_type") == ".pdf"
        lines: list[str] = []

        for line in raw_block.split("\n"):
            cleaned_line = _SPACES_RE.sub(" ", line.strip())
            if not cleaned_line:
                continue
            if _DECORATIVE_LINE_RE.match(cleaned_line):
                continue
            if is_pdf and _PDF_PAGE_MARKER_RE.match(cleaned_line):
                continue
            lines.append(cleaned_line)

        if not lines:
            return None

        if is_pdf:
            cleaned_content = lines[0]
            for line in lines[1:]:
                if cleaned_content.endswith("-") and line[:1].islower():
                    cleaned_content = f"{cleaned_content[:-1]}{line}"
                elif cleaned_content and line and _is_cjk(cleaned_content[-1]) and _is_cjk(line[0]):
                    cleaned_content = f"{cleaned_content}{line}"
                else:
                    cleaned_content = f"{cleaned_content} {line}"
        else:
            cleaned_content = "\n".join(lines)

        leading_trim = len(raw_block) - len(raw_block.lstrip())
        trailing_trim = len(raw_block) - len(raw_block.rstrip())

        return TextBlock(
            content=cleaned_content.strip(),
            start_char=start_char + leading_trim,
            end_char=end_char - trailing_trim,
        )

    def _merge_blocks(self, blocks: list[TextBlock]) -> list[TextBlock]:
        if self.block_min_chars <= 0 or len(blocks) <= 1:
            return blocks

        merged: list[TextBlock] = []
        index = 0
        while index < len(blocks):
            block = blocks[index]

            if _word_count(block.content) >= self.block_min_chars:
                merged.append(block)
                index += 1
                continue

            if (
                index + 1 < len(blocks)
                and _word_count(block.content) + _word_count(blocks[index + 1].content) <= self.block_max_chars
            ):
                next_block = blocks[index + 1]
                merged.append(
                    TextBlock(
                        content=f"{block.content.rstrip()}\n\n{next_block.content.lstrip()}".strip(),
                        start_char=block.start_char,
                        end_char=next_block.end_char,
                    )
                )
                index += 2
                continue

            if merged and _word_count(merged[-1].content) + _word_count(block.content) <= self.block_max_chars:
                previous = merged[-1]
                merged[-1] = TextBlock(
                    content=f"{previous.content.rstrip()}\n\n{block.content.lstrip()}".strip(),
                    start_char=previous.start_char,
                    end_char=block.end_char,
                )
            else:
                merged.append(block)
            index += 1

        return merged


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


def _is_cjk(character: str) -> bool:
    return "\u4e00" <= character <= "\u9fff"


def _word_count(text: str) -> int:
    return sum(1 for character in text if character.isalnum())
