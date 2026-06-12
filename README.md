# RAG Technical Documentation Assistant

A self-corrective RAG system built with LangGraph, FastAPI, Groq LLM, ChromaDB, and Streamlit.

---

## Architecture
User Question

│

▼

┌─────────────────┐

│  Query Analysis │  ← Rewrites query + classifies type

└────────┬────────┘

│

┌────────▼────────┐

│    Retrieval    │  ← Searches ChromaDB vector store (top-5 chunks)

└────────┬────────┘

│

┌────────▼────────┐

│ Doc Grading     │  ← LLM grades each chunk: relevant/irrelevant

└────────┬────────┘

│

┌────┴──────────────────────┐

│ all irrelevant?           │ relevant docs found

│                           │

│ retries left?    ┌────────▼────────┐

│ → re-retrieve    │   Generation    │ ← Answer with citations

│                  └────────┬────────┘

│ retries done?             │

▼              ┌────────────▼────────────┐

┌──────────┐       │  Hallucination Check    │ ← Verifies answer is grounded

│Web Search│       └────────────┬────────────┘

└────┬─────┘                    │

│              grounded → END

│              hallucinated → regenerate (max 1 retry)

▼

┌──────────┐

│Generation│

└──────────┘

### Why This Architecture?

- **Query rewriting + classification** improves retrieval by adding technical synonyms and classifying query type (conceptual, how-to, troubleshooting, api-reference).
- **Document grading** is the self-corrective core. Vector similarity scores alone are unreliable; LLM grading catches false positives.
- **Retry loop** before web search: a rewritten query often succeeds on a second attempt.
- **Web search fallback** (Tavily) ensures the system works even when local docs don't have the answer.
- **Hallucination check** verifies every generated answer is grounded in retrieved context.
- **Conversation memory** via session_id maintains chat history across follow-up questions.

---

## Chunking & Embedding Strategy

### Chunking
- **Method**: `RecursiveCharacterTextSplitter`
- **Chunk size**: 800 characters (~200 tokens)
- **Overlap**: 150 characters

**Why 800 chars?** Technical docs have dense information. Too small loses context; too large makes retrieval noisy.

**Why RecursiveCharacterTextSplitter?** Splits on paragraphs → sentences → words in priority order, preserving semantic units.

**Why 150 overlap?** Prevents losing context at chunk boundaries.

### Embeddings
- **Model**: `all-MiniLM-L6-v2` via sentence-transformers
- Runs completely locally — no API key needed, no quota limits

---

## Project Structure
rag-assistant/

├── app/

│   ├── config.py          # All settings

│   ├── llm.py             # Groq LLM + HuggingFace Embeddings

│   ├── main.py            # FastAPI application

│   └── vectorstore.py     # ChromaDB wrapper

├── workflow/

│   ├── state.py           # LangGraph state schema

│   ├── graph.py           # Graph assembly + routing logic

│   └── nodes/

│       ├── query_analysis.py      # Node 1: Query rewriting + classification

│       ├── retrieval.py           # Node 2: Vector search

│       ├── document_grading.py    # Node 3: LLM grading

│       ├── generation.py          # Node 4: Answer generation

│       ├── hallucination_check.py # Bonus: Verify answer is grounded

│       └── web_search.py          # Bonus: Tavily fallback

├── ingestion/

│   └── pipeline.py        # Load → chunk → embed → store

├── docs/

│   ├── fastapi_docs.md

│   ├── langchain_docs.md

│   └── langgraph_docs.md

├── ingest_docs.py         # Run once to populate vector store

├── streamlit_app.py       # Streamlit UI

├── requirements.txt

└── .env.example

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/ishika164/rag-documentation-assistant.git
cd rag-documentation-assistant
pip install -r requirements.txt
```

### 2. Get API keys

**Groq (free):**
1. Go to https://console.groq.com
2. Sign up and create an API key

**Tavily (free — 1000 searches/month):**
1. Go to https://app.tavily.com
2. Sign up and get your API key

### 3. Set up environment

```bash
cp .env.example .env
```

Edit `.env`:
GROQ_API_KEY=gsk_xxxxxxxxxxxx

TAVILY_API_KEY=tvly-xxxxxxxxxxxx

### 4. Ingest sample documents

```bash
python ingest_docs.py
```

### 5. Start the FastAPI server

```bash
uvicorn app.main:app --reload
```

API docs: http://localhost:8000/docs

### 6. Start the Streamlit UI (separate terminal)

```bash
streamlit run streamlit_app.py
```

UI: http://localhost:8501

---

## API Reference

### POST /query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I create a path parameter in FastAPI?", "session_id": "user_1"}'
```

Response:
```json
{
  "question": "How do I create a path parameter in FastAPI?",
  "answer": "You can declare path parameters using curly braces... [Source: fastapi_docs.md]",
  "sources": ["fastapi_docs.md"],
  "used_web_search": false,
  "retry_count": 0,
  "hallucination_detected": false,
  "session_id": "user_1"
}
```

### POST /ingest

```bash
# From file
curl -X POST http://localhost:8000/ingest -F "file=@my_docs.md"

# From URL
curl -X POST http://localhost:8000/ingest -F "url=https://docs.python.org/3/"
```

### GET /documents

```bash
curl http://localhost:8000/documents
```

### POST /feedback

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"question": "...", "answer": "...", "rating": "thumbs_up", "comment": "Great answer"}'
```

---

## Design Decisions & Tradeoffs

**Groq over Gemini:** Groq free tier has no daily quota limits and very fast inference. Gemini free tier gets throttled heavily under load.

**Local embeddings over API embeddings:** sentence-transformers runs entirely locally — no API quota, no auth issues, works offline. Slight quality tradeoff vs OpenAI embeddings but sufficient for this use case.

**LLM grading over threshold filtering:** Cosine similarity alone is unreliable — a chunk can score high while being off-topic. LLM grading catches false positives at the cost of extra API calls.

**Hallucination check node:** Self-RAG inspired verification after generation. If answer contains claims not in context, triggers regeneration (max 1 retry).

**Conversation memory via session_id:** Maintains last 5 turns per session for follow-up questions.

**ChromaDB over FAISS:** Persistent, metadata support, easy `/documents` endpoint. FAISS is faster at scale but needs manual metadata management.

---

## What I Would Improve With More Time

1. Hybrid search (dense + BM25) for better retrieval of exact technical terms
2. Persistent feedback in SQLite instead of JSON file
3. Async LLM grading — grade all chunks in parallel to reduce latency
4. Evaluation metrics to measure retrieval and answer quality
5. Docker container for easy deployment

---

## Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| LLM | Groq (llama-3.1-8b-instant) | Free tier, fast, no quota limits |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Local, free, no API needed |
| Workflow | LangGraph StateGraph | Cyclic graph, self-correction support |
| Vector Store | ChromaDB | Local, persistent, simple |
| API | FastAPI | Async, auto-docs, type-safe |
| UI | Streamlit | Fast to build, good for demos |
| Web Search | Tavily | AI-optimized, 1000 free searches/month |
