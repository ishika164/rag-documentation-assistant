"""
Run this script once to ingest the sample documents into ChromaDB.

Usage:
    python ingest_docs.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingestion.pipeline import ingest_file
from app.vectorstore import get_chunk_count, list_documents

DOCS = [
    "docs/fastapi_docs.md",
    "docs/langchain_docs.md",
    "docs/langgraph_docs.md",
]


def main():
    print("=" * 50)
    print("RAG Assistant — Document Ingestion")
    print("=" * 50)

    total_chunks = 0
    for doc_path in DOCS:
        if not os.path.exists(doc_path):
            print(f"⚠️  Skipping (not found): {doc_path}")
            continue
        try:
            chunks = ingest_file(doc_path)
            total_chunks += chunks
            print(f"✅ {doc_path} → {chunks} chunks")
        except Exception as e:
            print(f"❌ {doc_path} → Error: {e}")

    print("\n" + "=" * 50)
    print(f"Total chunks in store: {get_chunk_count()}")
    print("\nIndexed documents:")
    for doc in list_documents():
        print(f"  • {doc['source']} ({doc['chunk_count']} chunks)")
    print("=" * 50)
    print("\nDone! Run the API with: uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()
