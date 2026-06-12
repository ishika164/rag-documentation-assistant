"""
FastAPI Application — RAG Documentation Assistant

Endpoints:
  POST /query       - Submit a question, get answer + sources
  POST /ingest      - Ingest documents (file upload or URL)
  GET  /documents   - List all indexed documents
  POST /feedback    - Submit thumbs up/down on an answer
"""

import json
import os
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from workflow.graph import run_query
from ingestion.pipeline import ingest_file, ingest_url, ingest_text
from app.vectorstore import list_documents, get_chunk_count

app = FastAPI(
    title="RAG Documentation Assistant",
    description="Ask questions about your technical documentation using a self-corrective RAG pipeline.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_feedback_store = []
_sessions: dict = {}


class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "question": "How do I create a FastAPI route with path parameters?",
                "session_id": "user_123"
            }
        }


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    used_web_search: bool
    retry_count: int
    hallucination_detected: bool
    session_id: str


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: str
    comment: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "question": "How do I install FastAPI?",
                "answer": "pip install fastapi",
                "rating": "thumbs_up",
                "comment": "Perfect answer!"
            }
        }


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "RAG Assistant is running"}


@app.post("/query", response_model=QueryResponse, tags=["Query"])
def query(request: QueryRequest):
    """
    Submit a natural language question.
    Pass session_id to maintain conversation memory across follow-up questions.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    session_id = request.session_id or "default"
    chat_history = _sessions.get(session_id, [])

    result = run_query(request.question, chat_history=chat_history)

    chat_history.append({"role": "user", "content": request.question})
    chat_history.append({"role": "assistant", "content": result["answer"]})
    _sessions[session_id] = chat_history[-10:]

    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        sources=result["sources"],
        used_web_search=result.get("used_web_search", False),
        retry_count=result.get("retry_count", 0),
        hallucination_detected=result.get("hallucination_detected", False),
        session_id=session_id,
    )


@app.post("/ingest", tags=["Ingestion"])
async def ingest(
    url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    """
    Ingest a document into the vector store.
    Provide either a URL or a file upload (text or markdown).
    """
    if not url and not file:
        raise HTTPException(
            status_code=400,
            detail="Provide either a 'url' or a 'file' to ingest"
        )

    try:
        if url:
            chunks_added = ingest_url(url.strip())
            return {
                "status": "success",
                "source": url,
                "chunks_added": chunks_added,
            }

        if file:
            content = await file.read()
            text = content.decode("utf-8")
            chunks_added = ingest_text(text, source_name=file.filename)
            return {
                "status": "success",
                "source": file.filename,
                "chunks_added": chunks_added,
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", tags=["Documents"])
def documents():
    """
    List all documents currently indexed in the vector store.
    """
    docs = list_documents()
    total_chunks = get_chunk_count()
    return {
        "total_documents": len(docs),
        "total_chunks": total_chunks,
        "documents": docs,
    }


@app.post("/feedback", tags=["Feedback"])
def feedback(request: FeedbackRequest):
    """
    Submit feedback (thumbs up/down) on a generated answer.
    """
    if request.rating not in ("thumbs_up", "thumbs_down"):
        raise HTTPException(
            status_code=400,
            detail="Rating must be 'thumbs_up' or 'thumbs_down'"
        )

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": request.question,
        "answer": request.answer,
        "rating": request.rating,
        "comment": request.comment,
    }
    _feedback_store.append(entry)

    with open("feedback_log.json", "a") as f:
        f.write(json.dumps(entry) + "\n")

    return {"status": "recorded", "total_feedback": len(_feedback_store)}


@app.get("/feedback/summary", tags=["Feedback"])
def feedback_summary():
    """View feedback statistics."""
    ups = sum(1 for f in _feedback_store if f["rating"] == "thumbs_up")
    downs = sum(1 for f in _feedback_store if f["rating"] == "thumbs_down")
    return {
        "total": len(_feedback_store),
        "thumbs_up": ups,
        "thumbs_down": downs,
        "approval_rate": round(ups / len(_feedback_store), 2) if _feedback_store else None,
    }