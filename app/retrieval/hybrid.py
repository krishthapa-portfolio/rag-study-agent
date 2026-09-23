"""
Hybrid retrieval: BM25 (keyword) + semantic (embedding) search, combined
via Reciprocal Rank Fusion (RRF).
"""
from app.vectorstore.chroma_client import query_class
from app.retrieval.bm25_index import bm25_search

RRF_K = 60


def _semantic_ranked(class_name: str, query: str, n_results: int):
    result = query_class(class_name, query, n_results=n_results)
    if not result["ids"] or not result["ids"][0]:
        return []
    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    return list(zip(ids, documents, metadatas))


def _bm25_ranked(class_name: str, query: str, n_results: int):
    results = bm25_search(class_name, query, n_results=n_results)
    return [(r[0], r[1], r[2]) for r in results]


def hybrid_search(
    class_name: str,
    query: str,
    n_results: int = 5,
    candidate_pool: int = 15,
) -> list[dict]:
    semantic_results = _semantic_ranked(class_name, query, candidate_pool)
    bm25_results = _bm25_ranked(class_name, query, candidate_pool)

    fused: dict[str, dict] = {}

    for rank, (chunk_id, document, metadata) in enumerate(semantic_results, start=1):
        entry = fused.setdefault(
            chunk_id, {"rrf_score": 0.0, "document": document, "metadata": metadata}
        )
        entry["rrf_score"] += 1.0 / (RRF_K + rank)

    for rank, (chunk_id, document, metadata) in enumerate(bm25_results, start=1):
        entry = fused.setdefault(
            chunk_id, {"rrf_score": 0.0, "document": document, "metadata": metadata}
        )
        entry["rrf_score"] += 1.0 / (RRF_K + rank)

    ranked = sorted(fused.items(), key=lambda x: x[1]["rrf_score"], reverse=True)

    return [
        {
            "id": chunk_id,
            "document": data["document"],
            "metadata": data["metadata"],
            "rrf_score": round(data["rrf_score"], 5),
        }
        for chunk_id, data in ranked[:n_results]
    ]