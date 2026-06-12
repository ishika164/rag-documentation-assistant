"""
Node 1: Query Analysis
- Takes the raw user question
- Rewrites it to be more search-friendly (adds context, removes ambiguity)
- Classifies query type for logging/debugging
"""

from langchain_core.prompts import ChatPromptTemplate
from app.llm import get_llm
from workflow.state import GraphState

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at reformulating questions to improve document retrieval.
Your task is to rewrite the given question to be more specific and search-friendly.

Rules:
- Keep the core intent exactly the same
- Add relevant technical terms or synonyms that might appear in docs
- Remove filler words and ambiguity
- Do NOT answer the question — only rewrite it
- Output ONLY the rewritten question, nothing else"""),
    ("human", "Original question: {question}\n\nRewritten question:"),
])


def query_analysis_node(state: GraphState) -> GraphState:
    """
    Rewrite the user's question for better vector store retrieval.
    """
    llm = get_llm()
    question = state["question"]

    chain = REWRITE_PROMPT | llm
    result = chain.invoke({"question": question})
    rewritten = result.content.strip()

    print(f"[QueryAnalysis] Original: {question}")
    print(f"[QueryAnalysis] Rewritten: {rewritten}")

    return {
        **state,
        "rewritten_query": rewritten,
        "retry_count": state.get("retry_count", 0),
        "used_web_search": False,
    }
