import os
from typing import List, Dict
from langchain_chroma import Chroma
from langchain_core.documents import Document
from app.llm import get_embeddings
from app.config import CHROMA_PERSIST_DIR, COLLECTION_NAME, TOP_K


def get_vectorstore() -> Chroma:
    """Return (or create) the ChromaDB vector store."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )


def add_documents(docs: List[Document]) -> int:
    """
    Add a list of LangChain Document objects to the vector store.
    Returns the number of chunks added.
    """
    vs = get_vectorstore()
    vs.add_documents(docs)
    return len(docs)


def similarity_search(query: str, k: int = TOP_K) -> List[Dict]:
    """
    Search the vector store and return top-k chunks with metadata.
    Returns a list of dicts: {content, source, score}
    """
    vs = get_vectorstore()
    results = vs.similarity_search_with_relevance_scores(query, k=k)

    docs_out = []
    for doc, score in results:
        docs_out.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "score": round(score, 4),
        })
    return docs_out


def list_documents() -> List[Dict]:
    """
    Return a list of unique source documents currently indexed.
    """
    vs = get_vectorstore()
    collection = vs._collection
    all_meta = collection.get(include=["metadatas"])

    seen = {}
    for meta in all_meta["metadatas"]:
        src = meta.get("source", "unknown")
        if src not in seen:
            seen[src] = {
                "source": src,
                "chunk_count": 1,
            }
        else:
            seen[src]["chunk_count"] += 1

    return list(seen.values())


def get_chunk_count() -> int:
    vs = get_vectorstore()
    return vs._collection.count()
