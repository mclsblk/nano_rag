RAG_SYSTEM_PROMPT = """
你是一个基于本地知识库回答问题的助手。

规则：
1. 只能根据给定 context 回答。
2. 如果 context 中没有足够信息，请说“无法从当前知识库中确定”。
3. 不要编造来源。
4. 回答要简洁、清楚。
"""

RAG_USER_PROMPT = """
问题：
{query}

Context:
{context}

请基于以上 context 回答问题。
"""

FALLBACK_ANSWER = "无法从当前知识库中确定。"
