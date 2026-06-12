# RAG Technical Documentation Assistant

A **Retrieval-Augmented Generation (RAG)** system built with LangGraph, FastAPI, and Google Gemini that answers questions about technical documentation using a self-corrective workflow.

---

## Architecture

```
User Question
     │
     ▼
┌─────────────────┐
│  Query Analysis │  ← Rewrites query for better retrieval
└────────┬────────┘
         │
┌────────▼────────┐
│    Retrieval    │  ← Searches ChromaDB vector store
└────────┬────────┘
         │
┌────────▼────────┐
│ Doc Grading     │  ← LLM grades each chunk: relevant/irrelevant
└────────┬────────┘
         │
    ┌────┴─────────────────────┐
    │ all irrelevant?          │ relevant docs found
    │                          │
    │ retries left?    ┌───────▼────────┐
    │ → re-retrieve    │   Generation   │ ← Answer with citations
    │                  └────────────────┘
    │ retries done?
    ▼
┌────────────────┐
│  Web Search    │  ← Tavily fallback
└────────┬───────┘
         │
┌────────▼────────┐
│   Generation    │
└─────────────────┘
```

### Why This Architecture?

- **Query rewriting** improves retrieval by adding technical synonyms and removing ambiguity — especially useful when users ask vague questions.
- **Document grading** is the self-corrective core. Vector similarity scores alone are unreliable; LLM grading catches false positives (high-similarity but off-topic chunks).
- **Retry loop** before web search: a rewritten query often succeeds on a second attempt without needing external search.
- **Web search fallback** ensures the system is still useful for questions about topics not in the corpus.
- **ChromaDB** was chosen over FAISS for its built-in persistence and metadata support, which simplifies the `/documents` endpoint.

---

## Chunking & Embedding Strategy

### Chunking
- **Method**: `RecursiveCharacterTextSplitter`
- **Chunk size**: 800 characters (~200 tokens)
- **Overlap**: 150 characters

**Why 800 chars?** Technical documentation has dense, information-rich paragraphs. Too small (< 300) loses context (e.g., a function signature separated from its description). Too large (> 1500) makes retrieval noisy — the chunk matches too many queries. 800 is a practical balance.

**Why RecursiveCharacterTextSplitter?** It tries to split on `\n\n` (paragraphs) first, then `\n` (lines), then sentences, then words. This preserves semantic units in technical docs much better than fixed character splitting.

**Why 150 overlap?** Prevents information loss at boundaries. Without overlap, a code example might start on one chunk while its explanation ends on the previous one.

### Embeddings
- **Model**: `models/embedding-001` (Google Generative AI)
- Free tier, no additional cost beyond Gemini API access

---

## Project Structure

```
rag-assistant/
├── app/
│   ├── config.py          # All settings in one place
│   ├── llm.py             # Gemini LLM + Embeddings singletons
│   ├── main.py            # FastAPI application
│   └── vectorstore.py     # ChromaDB wrapper
├── workflow/
│   ├── state.py           # LangGraph state schema
│   ├── graph.py           # Graph assembly + routing logic
│   └── nodes/
│       ├── query_analysis.py   # Node 1: Query rewriting
│       ├── retrieval.py        # Node 2: Vector search
│       ├── document_grading.py # Node 3: LLM grading
│       ├── generation.py       # Node 4: Answer generation
│       └── web_search.py       # Bonus: Tavily fallback
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
```

---

## Setup

### 1. Clone and install

```bash
git clone <your-repo-url>
cd rag-assistant
pip install -r requirements.txt
```

### 2. Get API keys

**Google Gemini (free):**
1. Go to https://aistudio.google.com/app/apikey
2. Create an API key

**Tavily (free tier — 1000 searches/month):**
1. Go to https://tavily.com
2. Sign up and get your API key

### 3. Set up environment

```bash
cp .env.example .env
# Edit .env and add your keys:
# GOOGLE_API_KEY=your_key_here
# TAVILY_API_KEY=your_key_here
```

### 4. Ingest sample documents

```bash
python ingest_docs.py
```

### 5. Start the FastAPI server

```bash
uvicorn app.main:app --reload
```

API docs available at: http://localhost:8000/docs

### 6. Start the Streamlit UI (optional, separate terminal)

```bash
streamlit run streamlit_app.py
```

---

## API Reference

### POST /query

Submit a question and get an answer.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I create a FastAPI route with path parameters?"}'
```

**Response:**
```json
{
  "question": "How do I create a FastAPI route with path parameters?",
  "answer": "You can declare path parameters using curly braces in the route path... [Source: fastapi_docs.md]",
  "sources": ["fastapi_docs.md"],
  "used_web_search": false,
  "retry_count": 0
}
```

### POST /ingest

Ingest a URL:
```bash
curl -X POST http://localhost:8000/ingest \
  -F "url=https://docs.python.org/3/library/asyncio.html"
```

Ingest a file:
```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@my_docs.md"
```

**Response:**
```json
{
  "status": "success",
  "source": "my_docs.md",
  "chunks_added": 12
}
```

### GET /documents

```bash
curl http://localhost:8000/documents
```

**Response:**
```json
{
  "total_documents": 3,
  "total_chunks": 47,
  "documents": [
    {"source": "fastapi_docs.md", "chunk_count": 18},
    {"source": "langchain_docs.md", "chunk_count": 15},
    {"source": "langgraph_docs.md", "chunk_count": 14}
  ]
}
```

### POST /feedback

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I install FastAPI?",
    "answer": "pip install fastapi",
    "rating": "thumbs_up",
    "comment": "Clear and correct"
  }'
```

---

## Design Decisions & Tradeoffs

### State schema
All data flows between nodes as a single `GraphState` TypedDict. Alternatives like passing individual variables would work but make the graph harder to extend. The tradeoff is that the state object grows large — acceptable for a prototype.

### LLM grading vs. threshold-based filtering
I chose LLM grading over a simple similarity score threshold because semantic relevance is hard to capture with cosine similarity alone. A document can have high embedding similarity to a question (shared vocabulary) while being completely off-topic. The cost is extra LLM calls per retrieval. With a larger corpus, this could be optimized by grading only chunks below a certain score.

### ChromaDB vs. FAISS
ChromaDB: persistent, has metadata support, easy to query for the `/documents` list endpoint. FAISS is faster for large-scale similarity search but requires manual metadata management and serialization. For a prototype with < 10k chunks, ChromaDB is the pragmatic choice.

### Retry limit = 2
Each retry adds 1-2 LLM calls (query rewrite + grading). More than 2 retries adds latency with diminishing returns. The web search fallback is a better solution when the local corpus genuinely doesn't have the answer.

---

## What I Would Improve With More Time

1. **Hallucination check node** (bonus feature from the spec): Add a Self-RAG inspired node that verifies whether the generated answer is actually grounded in the retrieved context. Helps catch cases where the LLM "fills in" missing information.

2. **Conversation memory**: Maintain chat history in a session so follow-up questions like "tell me more about that" work correctly. Would use LangGraph's checkpointing with thread IDs.

3. **Persistent feedback to database**: Currently feedback is logged to a JSON file. A proper SQLite or Postgres store with session tracking would make quality monitoring actionable.

4. **Hybrid search**: Combine dense vector search (current) with BM25 keyword search. Keyword search performs better for exact technical terms like function names or error codes.

5. **Async retrieval**: Parallelize the LLM grading calls across chunks using `asyncio.gather` to reduce latency.

---

## Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| LLM | Google Gemini 1.5 Flash | Free tier, fast, capable |
| Embeddings | Google embedding-001 | Same provider, no extra cost |
| Workflow | LangGraph StateGraph | Cyclic graph support, self-correction |
| Vector Store | ChromaDB | Local, persistent, simple |
| API | FastAPI | Async, auto-docs, type-safe |
| UI | Streamlit | Fast to build, good for demos |
| Web Search | Tavily | AI-optimized, generous free tier |
