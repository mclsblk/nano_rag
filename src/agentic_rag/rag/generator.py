from agentic_rag.core import SearchResult
from agentic_rag.models import ChatModel, extract_chat_content
from agentic_rag.rag.context import ContextBuilder
from agentic_rag.rag.prompts import FALLBACK_ANSWER, RAG_SYSTEM_PROMPT, RAG_USER_PROMPT


class Generator:
    def __init__(self, chat_model: ChatModel, context_builder: ContextBuilder | None = None) -> None:
        self.chat_model = chat_model
        self.context_builder = context_builder or ContextBuilder()

    def generate(self, query: str, results: list[SearchResult]) -> str:
        if not results:
            return FALLBACK_ANSWER

        context = self.context_builder.build(results)
        response = self.chat_model.chat(
            [
                {"role": "system", "content": RAG_SYSTEM_PROMPT.strip()},
                {"role": "user", "content": RAG_USER_PROMPT.format(query=query, context=context).strip()},
            ]
        )
        answer = extract_chat_content(response)
        return answer or FALLBACK_ANSWER


def format_context(results: list[SearchResult]) -> str:
    return ContextBuilder().build(results)
