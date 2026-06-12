from langchain_core.prompts import ChatPromptTemplate
from app.llm import get_llm
from workflow.state import GraphState

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at reformulating questions to improve document retrieval.

Your task is to:
1. Classify the question type as one of: conceptual, how-to, troubleshooting, api-reference
2. Rewrite the question to be more specific and search-friendly

Rules for rewriting:
- Keep the core intent exactly the same
- Add relevant technical terms or synonyms
- Remove filler words and ambiguity
- Do NOT answer the question

Respond in exactly this format:
TYPE: <question_type>
QUERY: <rewritten_question>"""),
    ("human", "Original question: {question}"),
])


def query_analysis_node(state: GraphState) -> GraphState:
    llm = get_llm()
    question = state["question"]

    # On retry, use existing rewritten query as base
    base = state.get("rewritten_query") or question

    chain = REWRITE_PROMPT | llm
    result = chain.invoke({"question": base})
    output = result.content.strip()

    # Parse TYPE and QUERY from response
    query_type = "how-to"
    rewritten = question

    for line in output.split("\n"):
        if line.startswith("TYPE:"):
            query_type = line.replace("TYPE:", "").strip()
        elif line.startswith("QUERY:"):
            rewritten = line.replace("QUERY:", "").strip()

    print(f"[QueryAnalysis] Type: {query_type}")
    print(f"[QueryAnalysis] Original: {question}")
    print(f"[QueryAnalysis] Rewritten: {rewritten}")

    return {
        **state,
        "rewritten_query": rewritten,
        "query_type": query_type,
        "retry_count": state.get("retry_count", 0),
        "used_web_search": state.get("used_web_search", False),
    }