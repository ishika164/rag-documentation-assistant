from typing import TypedDict, List, Optional


class GraphState(TypedDict):
    # Input
    question: str

    # Query Analysis output
    rewritten_query: str
    query_type: str

    # Retrieval output
    retrieved_docs: List[dict]

    # Document Grading output
    relevant_docs: List[dict]
    all_irrelevant: bool

    # Generation output
    answer: str
    sources: List[str]

    # Hallucination check
    hallucination_detected: bool
    hallucination_retries: int

    # Web search output
    web_results: List[dict]

    # Control flow
    retry_count: int
    used_web_search: bool

    # Chat history for conversation memory
    chat_history: List[dict]

    # Feedback
    feedback: Optional[dict]