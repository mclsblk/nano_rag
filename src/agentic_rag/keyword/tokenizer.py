from dataclasses import dataclass
import re

from agentic_rag.core import Chunk
from agentic_rag.keyword.query_vocab import QUERY_IGNORED_TOKENS, QUERY_NOISE_SUBSTRINGS, QUERY_OPTIONAL_TOKENS


_CJK_NUMBER = r"零〇一二三四五六七八九十百千万两"
_QUESTION_PREFIX_RE = re.compile(rf"(?:问题|问|题)\s*[:：.\-、]?\s*([0-9{_CJK_NUMBER}]+)", re.IGNORECASE)
_QUESTION_ORDER_RE = re.compile(rf"第\s*([0-9{_CJK_NUMBER}]+)\s*(?:题|问|小题)", re.IGNORECASE)
_QUESTION_EN_RE = re.compile(r"(?<![A-Za-z0-9])Q(?:uestion)?\s*[:：.\-、]?\s*(\d+)(?![A-Za-z0-9])", re.IGNORECASE)
_VERSION_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]*\d+(?:[._-]\d+){1,})(?![A-Za-z0-9])")
_ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_QUESTION_TOKEN_RE = re.compile(rf"(?:问题[0-9{_CJK_NUMBER}]+|第[0-9{_CJK_NUMBER}]+题|q\d+|question\d+)")
_VERSION_TOKEN_RE = re.compile(r"[a-z]*\d+(?:_\d+)+|[a-z]*\d{2,}")


@dataclass(frozen=True)
class KeywordQuery:
    raw: str
    required_tokens: tuple[str, ...]
    optional_tokens: tuple[str, ...]
    ignored_tokens: tuple[str, ...]


def build_keyword_text(chunk: Chunk) -> str:
    tokens = tokenize(chunk.content, dedupe=False)
    return " ".join(tokens)


def analyze_query(query: str) -> KeywordQuery:
    required_tokens: list[str] = []
    optional_tokens: list[str] = []
    ignored_tokens: list[str] = []

    for token in _query_tokens(query):
        if _is_required_query_token(token):
            required_tokens.append(token)
        elif _is_ignored_query_token(token):
            ignored_tokens.append(token)
        else:
            optional_tokens.append(token)

    return KeywordQuery(
        raw=query,
        required_tokens=tuple(required_tokens),
        optional_tokens=tuple(optional_tokens),
        ignored_tokens=tuple(ignored_tokens),
    )


def _query_tokens(query: str) -> list[str]:
    tokens = [
        *_special_tokens(query),
        *(match.group(0).casefold() for match in _ASCII_TOKEN_RE.finditer(query)),
        *_known_query_terms(query),
    ]
    if not any(_is_required_query_token(token) for token in tokens):
        tokens.extend(token for token in tokenize(query) if not _is_ignored_query_token(token))
    return _dedupe(tokens)


def tokenize(text: str, *, dedupe: bool = True) -> list[str]:
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

    return _dedupe(tokens) if dedupe else _clean_tokens(tokens)


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


def _known_query_terms(query: str) -> list[str]:
    folded = query.casefold()
    terms = QUERY_OPTIONAL_TOKENS | QUERY_IGNORED_TOKENS
    matches = [(folded.find(term), -len(term), term) for term in terms if term in folded]
    return [term for _, _, term in sorted(matches)]


def _is_required_query_token(token: str) -> bool:
    if _is_question_token(token):
        return True
    if _is_version_token(token):
        return True
    return _is_ascii_query_token(token)


def _is_ascii_query_token(token: str) -> bool:
    if not token.isascii() or not token.isalnum():
        return False
    return len(token) >= 2 and any(character.isalpha() for character in token)


def _is_question_token(token: str) -> bool:
    return bool(_QUESTION_TOKEN_RE.fullmatch(token))


def _is_version_token(token: str) -> bool:
    return bool(_VERSION_TOKEN_RE.fullmatch(token))


def _is_ignored_query_token(token: str) -> bool:
    if token in QUERY_OPTIONAL_TOKENS:
        return False
    if token in QUERY_IGNORED_TOKENS:
        return True
    if token.isdecimal() and len(token) == 1:
        return True
    if token.endswith("的") and len(token) > 2:
        return True
    return any(noise in token for noise in QUERY_NOISE_SUBSTRINGS)


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
    for cleaned in _clean_tokens(tokens):
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return deduped


def _clean_tokens(tokens: list[str]) -> list[str]:
    return [cleaned for token in tokens if (cleaned := token.strip().casefold())]
