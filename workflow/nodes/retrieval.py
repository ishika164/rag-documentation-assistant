"""
Node 2: Retrieval
- Uses the (possibly rewritten) query to search ChromaDB
- Returns top-k chunks with source metadata
"""

from app.vectorstore import similarity_search
from workflow.state import GraphState


def retrieval_node(state: GraphState) -> GraphState:
    """
    Search the vector store using the rewritten query.
    Falls back to original question if rewrite is empty.
    """
    query = state.get("rewritten_query") or state["question"]

    retrieved = similarity_search(query)

    print(f"[Retrieval] Query: {query}")
    print(f"[Retrieval] Found {len(retrieved)} chunks")
    for doc in retrieved:
        print(f"  - [{doc['score']:.3f}] {doc['source'][:60]}")

    return {
        **state,
        "retrieved_docs": retrieved,
    }
