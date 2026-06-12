"""
Ingestion pipeline: load → chunk → embed → store

Chunking strategy:
- RecursiveCharacterTextSplitter with size=800, overlap=150
- Why 800? Technical docs have dense information. Too small = lost context.
  Too large = noisy retrieval. 800 chars (~200 tokens) is a good balance.
- Why 150 overlap? Prevents losing context at chunk boundaries (e.g., a
  function signature on one chunk, its description on the next).
- RecursiveCharacterTextSplitter tries to split on paragraphs → sentences →
  words in that order, so it preserves semantic units where possible.
"""

import os
import requests
from typing import List
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.vectorstore import add_documents
from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def _get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # priority order
    )


def ingest_file(file_path: str, source_name: str = None) -> int:
    """
    Load a local text/markdown file, chunk it, and add to vector store.
    Returns number of chunks added.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    text = path.read_text(encoding="utf-8")
    source = source_name or path.name

    return _chunk_and_store(text, source)


def ingest_url(url: str) -> int:
    """
    Fetch a URL, extract text content, chunk it, and add to vector store.
    Returns number of chunks added.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    # Basic HTML stripping - good enough for docs pages
    text = response.text
    if "<html" in text.lower():
        from html.parser import HTMLParser

        class TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.parts = []
                self._skip = False

            def handle_starttag(self, tag, attrs):
                if tag in ("script", "style", "nav", "footer"):
                    self._skip = True

            def handle_endtag(self, tag):
                if tag in ("script", "style", "nav", "footer"):
                    self._skip = False

            def handle_data(self, data):
                if not self._skip and data.strip():
                    self.parts.append(data.strip())

        extractor = TextExtractor()
        extractor.feed(text)
        text = "\n".join(extractor.parts)

    return _chunk_and_store(text, source=url)


def ingest_text(text: str, source_name: str) -> int:
    """
    Directly ingest raw text string. Used for uploaded file content.
    """
    return _chunk_and_store(text, source=source_name)


def _chunk_and_store(text: str, source: str) -> int:
    splitter = _get_splitter()
    chunks = splitter.split_text(text)

    docs = [
        Document(
            page_content=chunk,
            metadata={"source": source},
        )
        for chunk in chunks
        if chunk.strip()  # skip empty chunks
    ]

    if not docs:
        raise ValueError(f"No content extracted from source: {source}")

    return add_documents(docs)
