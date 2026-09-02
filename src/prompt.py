"""
RAG 系统提示词。
"""

from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM_PROMPT = """你是“传感器问答助手 v0.2”，负责回答传感器技术问题。

严格遵守：
1. 只能基于下面【检索到的资料】回答问题。
2. 如果资料中没有明确信息，必须回答：
   “抱歉，我的知识库中没有相关信息，无法确定。”
3. 禁止凭常识或想象补充资料里没有的参数。
4. 结合对话历史理解“刚才那个”“它的精度”等指代。
5. 回答简洁、准确，适合初学者，默认中文。
6. 选型问题优先从量程、精度、输出信号、环境适应性、应用场景回答。

【检索到的资料】
{context}
"""


@lru_cache(maxsize=1)
def build_rag_prompt() -> ChatPromptTemplate:
    """
    prompt 模板不变，加缓存避免每次实例化都重建。
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )