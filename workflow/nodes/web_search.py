"""
Bonus Node: Web Search Fallback
- Triggered when vector store has no relevant docs AND retries are exhausted
- Uses Tavily API to search the web
- Formats results in the same dict structure as vector store docs
"""

from tavily import TavilyClient
from app.config import TAVILY_API_KEY
from workflow.state import GraphState


def web_search_node(state: GraphState) -> GraphState:
    """
    Search the web for an answer when local docs fail.
    """
    query = state.get("rewritten_query") or state["question"]

    print(f"[WebSearch] Falling back to web search for: {query}")

    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(
            query=query,
            max_results=3,
            search_depth="basic",
        )

        web_results = []
        for r in response.get("results", []):
            web_results.append({
                "content": r.get("content", ""),
                "source": r.get("url", "web"),
                "score": r.get("score", 0.0),
            })

        print(f"[WebSearch] Got {len(web_results)} results")

        return {
            **state,
            "web_results": web_results,
            "used_web_search": True,
            # Treat web results as relevant_docs for generation node
            "relevant_docs": web_results,
            "all_irrelevant": len(web_results) == 0,
        }

    except Exception as e:
        print(f"[WebSearch] Error: {e}")
        return {
            **state,
            "web_results": [],
            "used_web_search": True,
            "all_irrelevant": True,
        }
