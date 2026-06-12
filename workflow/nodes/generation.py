"""
Node 4: Generation
- Takes relevant docs + question
- Generates a grounded answer with citations
- Can work with both vector store docs and web search results
"""

from langchain_core.prompts import ChatPromptTemplate
from app.llm import get_llm
from workflow.state import GraphState

GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful technical documentation assistant.
Answer the user's question using ONLY the provided context.

Rules:
- Be accurate and precise
- If the context partially answers the question, say so clearly
- Cite your sources using [Source: filename] inline in your answer
- Do not make up information not present in the context
- Structure your answer clearly with paragraphs or bullet points as appropriate
- If multiple sources say different things, mention both"""),
    ("human", """Context documents:
{context}

Question: {question}

Answer:"""),
])


def _format_context(docs: list) -> str:
    """Format doc list into a readable context block for the prompt."""
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(f"[Document {i}] Source: {doc['source']}\n{doc['content']}")
    return "\n\n---\n\n".join(parts)


def generation_node(state: GraphState) -> GraphState:
    """
    Generate final answer from relevant docs (vector store or web).
    """
    llm = get_llm()
    question = state["question"]

    # Prefer vector store results; fall back to web results
    docs = state.get("relevant_docs") or []
    web_results = state.get("web_results") or []

    # Combine: local docs first, then web results
    all_docs = docs + web_results

    if not all_docs:
        return {
            **state,
            "answer": "I could not find relevant information to answer your question. Please try rephrasing or ingesting more relevant documents.",
            "sources": [],
        }

    context = _format_context(all_docs)
    sources = list({doc["source"] for doc in all_docs})

    chain = GENERATION_PROMPT | llm
    result = chain.invoke({"context": context, "question": question})
    answer = result.content.strip()

    print(f"[Generation] Answer generated ({len(answer)} chars), sources: {sources}")

    return {
        **state,
        "answer": answer,
        "sources": sources,
    }
