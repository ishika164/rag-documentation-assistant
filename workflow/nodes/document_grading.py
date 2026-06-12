"""
Node 3: Document Grading (self-corrective component)
- Uses LLM to evaluate each retrieved chunk against the question
- Filters out irrelevant chunks
- Sets all_irrelevant=True if nothing passes, triggering retry/fallback
"""

from langchain_core.prompts import ChatPromptTemplate
from app.llm import get_llm
from workflow.state import GraphState

GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a relevance grader. Given a question and a document chunk,
decide if the chunk contains information useful for answering the question.

Respond with ONLY one word:
- "relevant" if the chunk helps answer the question
- "irrelevant" if the chunk does not help at all

Be strict — partial relevance counts as relevant. Only mark irrelevant if the
chunk is completely off-topic."""),
    ("human", """Question: {question}

Document chunk:
{chunk}

Grade (relevant/irrelevant):"""),
])


def document_grading_node(state: GraphState) -> GraphState:
    """
    Grade each retrieved document chunk. Keep only relevant ones.
    """
    llm = get_llm()
    question = state["question"]
    retrieved_docs = state.get("retrieved_docs", [])

    if not retrieved_docs:
        print("[Grading] No docs to grade — marking all_irrelevant=True")
        return {**state, "relevant_docs": [], "all_irrelevant": True}

    chain = GRADE_PROMPT | llm
    relevant_docs = []

    for doc in retrieved_docs:
        result = chain.invoke({
            "question": question,
            "chunk": doc["content"][:1500],  # truncate very long chunks for grading
        })
        grade = result.content.strip().lower()
        is_relevant = "relevant" in grade and "irrelevant" not in grade

        print(f"[Grading] Score={doc['score']:.3f} | Grade={grade} | Source={doc['source'][:50]}")

        if is_relevant:
            relevant_docs.append(doc)

    all_irrelevant = len(relevant_docs) == 0

    print(f"[Grading] {len(relevant_docs)}/{len(retrieved_docs)} chunks passed")

    return {
        **state,
        "relevant_docs": relevant_docs,
        "all_irrelevant": all_irrelevant,
    }
