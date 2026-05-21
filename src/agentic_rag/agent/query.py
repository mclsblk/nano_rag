import re

from agentic_rag.models import ChatModel, extract_chat_content


QUERY_REWRITE_SYSTEM_PROMPT = """
你是一个检索 query 改写器。

规则：
1. 只输出一个适合知识库检索的 query。
2. 不解释。
3. 不增加用户问题中不存在的事实。
4. 保留用户问题的原始语言。
"""

QUERY_REWRITE_USER_PROMPT = """
原始问题：
{query}

请输出改写后的检索 query。
"""

MULTI_QUERY_SYSTEM_PROMPT = """
你是一个多 query 检索规划器。

规则：
1. 每行只输出一个检索 query。
2. 不解释，不编号。
3. 不增加用户问题中不存在的事实。
4. 查询之间应覆盖不同表达、同义词或关键短语。
5. 保留用户问题的原始语言。
"""

MULTI_QUERY_USER_PROMPT = """
原始问题：
{query}

已改写 query：
{rewritten_query}

还需要 {additional_count} 个补充检索 query。
"""


class QueryRewriter:
    def __init__(self, chat_model: ChatModel) -> None:
        self.chat_model = chat_model

    def rewrite(self, query: str) -> str:
        original_query = query.strip()
        if not original_query:
            return query

        try:
            response = self.chat_model.chat(
                [
                    {"role": "system", "content": QUERY_REWRITE_SYSTEM_PROMPT.strip()},
                    {"role": "user", "content": QUERY_REWRITE_USER_PROMPT.format(query=original_query).strip()},
                ]
            )
            rewritten_query = _clean_query(extract_chat_content(response))
        except Exception:
            return original_query

        return rewritten_query or original_query


class MultiQueryGenerator:
    def __init__(self, chat_model: ChatModel) -> None:
        self.chat_model = chat_model

    def generate(self, query: str, rewritten_query: str = "", count: int = 3) -> list[str]:
        target_count = max(1, count)
        queries = _dedupe_queries([query, rewritten_query])
        additional_count = target_count - len(queries)

        if additional_count > 0:
            try:
                response = self.chat_model.chat(
                    [
                        {"role": "system", "content": MULTI_QUERY_SYSTEM_PROMPT.strip()},
                        {
                            "role": "user",
                            "content": MULTI_QUERY_USER_PROMPT.format(
                                query=query.strip(),
                                rewritten_query=rewritten_query.strip() or query.strip(),
                                additional_count=additional_count,
                            ).strip(),
                        },
                    ]
                )
                queries = _dedupe_queries([*queries, *_parse_queries(extract_chat_content(response))])
            except Exception:
                pass

        return queries[:target_count]


def _parse_queries(text: str) -> list[str]:
    return [_clean_query(line) for line in text.splitlines()]


def _clean_query(query: str) -> str:
    cleaned = query.strip()
    cleaned = re.sub(r"^(?:[-*]|\d+[.)、])\s*", "", cleaned)
    return cleaned.strip("\"'“”‘’` ")


def _dedupe_queries(queries: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()

    for query in queries:
        cleaned = _clean_query(query)
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)

    return deduped
