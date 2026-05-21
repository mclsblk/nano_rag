import re
from typing import Any

from agentic_rag.core import Chunk


_CJK_NUMBER = r"零〇一二三四五六七八九十百千万两"
_QUESTION_PREFIX_RE = re.compile(rf"(?:问题|问|题)\s*[:：.\-、]?\s*([0-9{_CJK_NUMBER}]+)", re.IGNORECASE)
_QUESTION_ORDER_RE = re.compile(rf"第\s*([0-9{_CJK_NUMBER}]+)\s*(?:题|问|小题)", re.IGNORECASE)
_QUESTION_EN_RE = re.compile(r"(?<![A-Za-z0-9])Q(?:uestion)?\s*[:：.\-、]?\s*(\d+)(?![A-Za-z0-9])", re.IGNORECASE)
_VERSION_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]*\d+(?:[._-]\d+){1,})(?![A-Za-z0-9])")
_ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def build_keyword_text(chunk: Chunk) -> str:
    parts = [chunk.content]
    for key in _searchable_metadata_keys(chunk.metadata):
        value = chunk.metadata.get(key)
        if isinstance(value, (str, int, float, bool)):
            parts.append(str(value))

    tokens = tokenize(" ".join(parts))
    return " ".join(tokens)


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []

    for token in _special_tokens(text):
        tokens.append(token)

    cursor = 0
    while cursor < len(text):
        character = text[cursor]
        if _is_cjk(character):
            end = cursor + 1
            while end < len(text) and _is_cjk(text[end]):
                end += 1
            tokens.extend(_cjk_ngrams(text[cursor:end]))
            cursor = end
            continue

        if character.isascii() and character.isalnum():
            match = _ASCII_TOKEN_RE.match(text, cursor)
            if match is not None:
                tokens.append(match.group(0).casefold())
                cursor = match.end()
                continue

        cursor += 1

    return _dedupe(tokens)


def _special_tokens(text: str) -> list[str]:
    tokens: list[str] = []

    for match in _QUESTION_PREFIX_RE.finditer(text):
        number = _compact(match.group(1))
        if number:
            tokens.append(f"问题{number}")

    for match in _QUESTION_ORDER_RE.finditer(text):
        number = _compact(match.group(1))
        if number:
            tokens.append(f"第{number}题")

    for match in _QUESTION_EN_RE.finditer(text):
        number = match.group(1)
        tokens.extend([f"q{number}", f"question{number}"])

    for match in _VERSION_RE.finditer(text):
        raw = match.group(1).casefold()
        normalized = re.sub(r"[._-]+", "_", raw)
        compact = re.sub(r"[._-]+", "", raw)
        tokens.extend([normalized, compact])

    return tokens


def _searchable_metadata_keys(metadata: dict[str, Any]) -> list[str]:
    preferred = [
        "source",
        "file_name",
        "file_type",
        "hard_break_key",
        "chunk_index",
        "page_start",
        "page_end",
        "page_number",
    ]
    keys = [key for key in preferred if key in metadata]
    extras = sorted(key for key in metadata if key not in set(preferred) and isinstance(metadata.get(key), str))
    return [*keys, *extras]


def _cjk_ngrams(span: str) -> list[str]:
    tokens: list[str] = []
    if len(span) >= 2:
        tokens.extend(span[index : index + 2] for index in range(len(span) - 1))
    if len(span) >= 3:
        tokens.extend(span[index : index + 3] for index in range(len(span) - 2))
    return tokens


def _is_cjk(character: str) -> bool:
    return "\u4e00" <= character <= "\u9fff"


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _dedupe(tokens: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        cleaned = token.strip().casefold()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return deduped
