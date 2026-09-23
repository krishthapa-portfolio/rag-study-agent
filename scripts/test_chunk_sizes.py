"""
Chunk size test harness — THIS is the script you run yourself to pick
real chunk sizes empirically, per the plan we agreed on.

What it does:
  1. Ingests one document at several chunk sizes into separate temp
     ChromaDB collections (so they don't pollute your real class data).
  2. Runs a handful of test questions you provide against each size.
  3. Prints the top-match similarity score and preview text per size,
     so you can visually compare which chunk size retrieves the most
     relevant, complete context.

This is a starting point, not the final eval system — once LangFuse is
wired in (Phase 5), this same comparison gets proper precision/groundedness
metrics instead of eyeballing similarity scores.

Usage:
    python scripts/test_chunk_sizes.py --file ./data/uploads/example.pdf \
        --questions "What is validity in data quality?" "What is a knowledge graph?"
"""
import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb
from chromadb.utils import embedding_functions

from app.ingestion.parsers import parse_document
from app.ingestion.security import scan_for_injection, mask_pii
from app.ingestion.chunker import chunk_text, token_count
from app.config import EMBEDDING_MODEL

CHUNK_SIZES_TO_TEST = [250, 500, 1000]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--questions", nargs="+", required=True)
    parser.add_argument("--chunk-sizes", nargs="+", type=int, default=CHUNK_SIZES_TO_TEST)
    args = parser.parse_args()

    units = parse_document(args.file)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    client = chromadb.EphemeralClient()  # in-memory, thrown away after this run

    for size in args.chunk_sizes:
        collection_name = f"test_{size}_{uuid.uuid4().hex[:6]}"
        collection = client.create_collection(collection_name, embedding_function=embedding_fn)

        all_chunks = []
        for raw_text, unit_num in units:
            if scan_for_injection(raw_text).is_suspicious:
                continue
            masked, _ = mask_pii(raw_text)
            chunks = chunk_text(masked, chunk_size=size, overlap=min(50, size // 10))
            all_chunks.extend(chunks)

        if not all_chunks:
            print(f"\n=== chunk_size={size} ===\nNo chunks produced, skipping.")
            continue

        ids = [f"c{i}" for i in range(len(all_chunks))]
        collection.add(documents=all_chunks, ids=ids)

        avg_tokens = sum(token_count(c) for c in all_chunks) / len(all_chunks)
        print(f"\n=== chunk_size={size} ===")
        print(f"Total chunks: {len(all_chunks)} | Avg tokens/chunk: {avg_tokens:.0f}")

        for question in args.questions:
            result = collection.query(query_texts=[question], n_results=1)
            top_doc = result["documents"][0][0] if result["documents"][0] else "(no result)"
            top_dist = result["distances"][0][0] if result["distances"][0] else None
            preview = top_doc[:200].replace("\n", " ")
            print(f'\n  Q: "{question}"')
            print(f"  distance: {top_dist:.4f}" if top_dist is not None else "  no match")
            print(f"  top chunk preview: {preview}...")


if __name__ == "__main__":
    main()
