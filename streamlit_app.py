"""
Streamlit UI — RAG Documentation Assistant
Connects to the FastAPI backend running at localhost:8000
"""

import streamlit as st
import requests
import json

API_BASE = "http://localhost:8000"

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Docs Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 RAG Technical Documentation Assistant")
st.caption("Powered by LangGraph + Gemini + ChromaDB")

# ──────────────────────────────────────────────
# Sidebar: Ingestion + Document list
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("📥 Ingest Documents")

    ingest_tab, url_tab = st.tabs(["Upload File", "From URL"])

    with ingest_tab:
        uploaded = st.file_uploader("Upload .txt or .md file", type=["txt", "md"])
        if st.button("Ingest File") and uploaded:
            with st.spinner("Ingesting..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/ingest",
                        files={"file": (uploaded.name, uploaded.getvalue(), "text/plain")},
                    )
                    data = resp.json()
                    if resp.status_code == 200:
                        st.success(f"✅ Added {data['chunks_added']} chunks from {data['source']}")
                    else:
                        st.error(data.get("detail", "Ingestion failed"))
                except Exception as e:
                    st.error(f"Error: {e}")

    with url_tab:
        url_input = st.text_input("Document URL")
        if st.button("Ingest URL") and url_input:
            with st.spinner("Fetching and ingesting..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/ingest",
                        data={"url": url_input},
                    )
                    data = resp.json()
                    if resp.status_code == 200:
                        st.success(f"✅ Added {data['chunks_added']} chunks from URL")
                    else:
                        st.error(data.get("detail", "Ingestion failed"))
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()
    st.header("📄 Indexed Documents")
    if st.button("Refresh List"):
        try:
            resp = requests.get(f"{API_BASE}/documents")
            data = resp.json()
            st.metric("Total Chunks", data["total_chunks"])
            st.metric("Documents", data["total_documents"])
            for doc in data["documents"]:
                st.text(f"• {doc['source'][:40]}... ({doc['chunk_count']} chunks)")
        except Exception as e:
            st.error(f"Could not reach API: {e}")

# ──────────────────────────────────────────────
# Main: Chat interface
# ──────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_response" not in st.session_state:
    st.session_state.last_response = None

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources"):
                for s in msg["sources"]:
                    st.code(s)
        if msg.get("meta"):
            cols = st.columns(3)
            cols[0].caption(f"🔁 Retries: {msg['meta']['retry_count']}")
            cols[1].caption(f"🌐 Web: {'Yes' if msg['meta']['used_web_search'] else 'No'}")

# Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/query",
                    json={"question": prompt},
                    timeout=60,
                )
                data = resp.json()

                if resp.status_code == 200:
                    answer = data["answer"]
                    sources = data["sources"]
                    meta = {
                        "retry_count": data["retry_count"],
                        "used_web_search": data["used_web_search"],
                    }

                    st.markdown(answer)

                    if sources:
                        with st.expander("📎 Sources"):
                            for s in sources:
                                st.code(s)

                    cols = st.columns(3)
                    cols[0].caption(f"🔁 Retries: {meta['retry_count']}")
                    cols[1].caption(f"🌐 Web: {'Yes' if meta['used_web_search'] else 'No'}")

                    # Feedback buttons
                    st.divider()
                    feedback_cols = st.columns([1, 1, 4])
                    if feedback_cols[0].button("👍"):
                        requests.post(f"{API_BASE}/feedback", json={
                            "question": prompt,
                            "answer": answer,
                            "rating": "thumbs_up",
                        })
                        st.toast("Thanks for the feedback!")
                    if feedback_cols[1].button("👎"):
                        requests.post(f"{API_BASE}/feedback", json={
                            "question": prompt,
                            "answer": answer,
                            "rating": "thumbs_down",
                        })
                        st.toast("Thanks for the feedback!")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "meta": meta,
                    })

                else:
                    st.error(data.get("detail", "API error"))

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Make sure FastAPI is running: `uvicorn app.main:app`")
            except Exception as e:
                st.error(f"Error: {e}")
