import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# LLM settings
LLM_MODEL = "gemini-2.0-flash-lite"    # fast + free tier
EMBEDDING_MODEL = "models/gemini-embedding-001" # Google embedding model

# Vector store
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "rag_docs"

# Retrieval settings
TOP_K = 5           # how many chunks to retrieve
CHUNK_SIZE = 800    # chars per chunk
CHUNK_OVERLAP = 150 # overlap between chunks to preserve context

# Workflow settings
MAX_RETRIES = 2     # max query rewrites before giving up
