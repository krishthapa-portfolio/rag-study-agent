"""
Central config. Loads from .env so nothing sensitive is hardcoded.
"""
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
APP_ENV = os.getenv("APP_ENV", "development")
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))

# Embedding model used across ingestion + query (must match at both ends)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Default chunking values — the test harness (scripts/test_chunk_sizes.py)
# is what you use to empirically pick the real values per doc type.
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
