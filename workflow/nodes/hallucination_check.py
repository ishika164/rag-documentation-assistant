"""
Bonus Node: Hallucination Check (Self-RAG inspired)
- Verifies the generated answer is actually grounded in the retrieved context
- If answer contains information not in the context, flags it
- Routes back to generation if hallucination detected (up to 1 retry)
"""

from langchain_core.prompts import ChatPromptTemplate
from app.llm import get_llm
from workflow.state import GraphState

HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a hallucination detector. Your job is to verify if an answer 
is fully supported by the given context documents.

Respond with ONLY one word:
- "grounded" if every claim in the answer is supported by the context
- "hallucinated" if the answer contains information NOT present in the context

Be strict. If even one claim cannot be traced back to the context, respond "hallucinated"."""),
    ("human", """Context documents:
{context}

Generated answer:
{answer}

Is the answer grounded in the context? (grounded/hallucinated):"""),
])


def hallucination_check_node(state: GraphState) -> GraphState:
    """
    Check if the generated answer is supported by retrieved documents.
    """
    llm = get_llm()
    answer = state.get("answer", "")
    relevant_docs = state.get("relevant_docs", [])

    if not answer or not relevant_docs:
        print("[HallucinationCheck] Skipping — no answer or docs to check")
        return {**state, "hallucination_detected": False}

    context = "\n\n".join([doc["content"][:500] for doc in relevant_docs])

    chain = HALLUCINATION_PROMPT | llm
    result = chain.invoke({"context": context, "answer": answer})
    verdict = result.content.strip().lower()

    hallucinated = "hallucinated" in verdict
    print(f"[HallucinationCheck] Verdict: {verdict} → hallucinated={hallucinated}")

    return {
        **state,
        "hallucination_detected": hallucinated,
    }