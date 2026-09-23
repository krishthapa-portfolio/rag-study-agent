"""
BM25 keyword search over one class's chunks.

BM25 is a classic keyword-matching algorithm (not embedding-based) — it's
strong at exact terms, acronyms, and names (e.g. "RDF", "CPOE", "Dr. Harris")
that semantic search can sometimes blur past. We combine both (see hybrid.py)
because each catches things the other misses.
"""
import re
from rank_bm25 import BM25Okapi

from app.vectorstore.chroma_client import get_all_chunks


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def build_bm25_index(class_name: str):
    data = get_all_chunks(class_name)
    if not data["documents"]:
        return None

    tokenized_corpus = [_tokenize(doc) for doc in data["documents"]]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, data["ids"], data["documents"], data["metadatas"]


def bm25_search(class_name: str, query: str, n_results: int = 10):
    index_data = build_bm25_index(class_name)
    if index_data is None:
        return []

    bm25, ids, documents, metadatas = index_data
    scores = bm25.get_scores(_tokenize(query))

    ranked = sorted(
        zip(ids, documents, metadatas, scores), key=lambda x: x[3], reverse=True
    )
    return ranked[:n_results]