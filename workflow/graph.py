from langgraph.graph import StateGraph, END

from workflow.state import GraphState
from workflow.nodes.query_analysis import query_analysis_node
from workflow.nodes.retrieval import retrieval_node
from workflow.nodes.document_grading import document_grading_node
from workflow.nodes.generation import generation_node
from workflow.nodes.web_search import web_search_node
from workflow.nodes.hallucination_check import hallucination_check_node
from app.config import MAX_RETRIES


def route_after_grading(state: GraphState) -> str:
    if not state.get("all_irrelevant", False):
        print("[Router] Relevant docs found → generation")
        return "generate"

    retry_count = state.get("retry_count", 0)
    if retry_count < MAX_RETRIES:
        print(f"[Router] No relevant docs, retry {retry_count + 1}/{MAX_RETRIES}")
        return "retry"
    else:
        print(f"[Router] Retries exhausted → web search")
        return "web_search"


def route_after_hallucination_check(state: GraphState) -> str:
    """
    If hallucination detected and we haven't retried yet → regenerate.
    Otherwise → end.
    """
    if state.get("hallucination_detected") and state.get("hallucination_retries", 0) < 1:
        print("[Router] Hallucination detected → regenerating")
        return "regenerate"
    print("[Router] Answer is grounded → end")
    return "end"


def increment_retry(state: GraphState) -> GraphState:
    return {
        **state,
        "retry_count": state.get("retry_count", 0) + 1,
        "rewritten_query": "",
    }


def increment_hallucination_retry(state: GraphState) -> GraphState:
    return {
        **state,
        "hallucination_retries": state.get("hallucination_retries", 0) + 1,
        "hallucination_detected": False,
    }


def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("query_analysis", query_analysis_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("document_grading", document_grading_node)
    graph.add_node("generation", generation_node)
    graph.add_node("hallucination_check", hallucination_check_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("increment_retry", increment_retry)
    graph.add_node("increment_hallucination_retry", increment_hallucination_retry)

    # Entry point
    graph.set_entry_point("query_analysis")

    # Linear flow
    graph.add_edge("query_analysis", "retrieval")
    graph.add_edge("retrieval", "document_grading")

    # Conditional: after grading
    graph.add_conditional_edges(
        "document_grading",
        route_after_grading,
        {
            "generate": "generation",
            "retry": "increment_retry",
            "web_search": "web_search",
        },
    )

    # Retry loop
    graph.add_edge("increment_retry", "query_analysis")

    # Web search → generation
    graph.add_edge("web_search", "generation")

    # Generation → hallucination check
    graph.add_edge("generation", "hallucination_check")

    # Conditional: after hallucination check
    graph.add_conditional_edges(
        "hallucination_check",
        route_after_hallucination_check,
        {
            "regenerate": "increment_hallucination_retry",
            "end": END,
        },
    )

    # Hallucination retry → regenerate
    graph.add_edge("increment_hallucination_retry", "generation")

    return graph.compile()


rag_graph = build_graph()


def run_query(question: str, chat_history: list = None) -> dict:
    initial_state = GraphState(
        question=question,
        rewritten_query="",
        query_type="",
        retrieved_docs=[],
        relevant_docs=[],
        all_irrelevant=False,
        answer="",
        sources=[],
        hallucination_detected=False,
        hallucination_retries=0,
        web_results=[],
        retry_count=0,
        used_web_search=False,
        chat_history=chat_history or [],
        feedback=None,
    )

    result = rag_graph.invoke(initial_state)
    return result