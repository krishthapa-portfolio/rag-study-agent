"""
The ingestion pipeline: parse -> scan for injection -> mask PII -> chunk
-> tag metadata -> embed & store.

Security is not a separate pass tacked on afterward — it runs inline,
before anything is chunked or embedded, so nothing unsafe ever makes it
into the vector store.
"""
import uuid
from pathlib import Path
from datetime import datetime, timezone

from app.ingestion.parsers import parse_document
from app.ingestion.security import scan_for_injection, mask_pii
from app.ingestion.chunker import chunk_text
from app.vectorstore.chroma_client import add_chunks
from app.utils.logging_config import audit_log
from app.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


def ingest_document(
    file_path: str,
    class_name: str,
    doc_type: str = "unspecified",  # e.g. "slides", "notes", "reading"
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict:
    doc_name = Path(file_path).name
    audit_log("ingestion_start", doc_name=doc_name, class_name=class_name)

    units = parse_document(file_path)  # list of (text, page_or_slide_num)

    total_chunks_added = 0
    total_pii_masked = 0
    injection_flags = []

    for raw_text, unit_num in units:
        # 1. Prompt injection scan — flag, don't silently drop, so a human
        #    can review what tripped it. The flagged unit is still masked
        #    for PII and skipped from ingestion rather than embedded blindly.
        scan_result = scan_for_injection(raw_text)
        if scan_result.is_suspicious:
            injection_flags.append(
                {"unit": unit_num, "patterns": scan_result.matched_patterns}
            )
            audit_log(
                "ingestion_blocked_injection",
                doc_name=doc_name,
                unit=unit_num,
                patterns=scan_result.matched_patterns,
            )
            continue  # do not ingest this unit

        # 2. PII masking
        masked_text, pii_counts = mask_pii(raw_text)
        if pii_counts:
            total_pii_masked += sum(pii_counts.values())
            audit_log(
                "pii_masked", doc_name=doc_name, unit=unit_num, counts=pii_counts
            )

        # 3. Chunk
        chunks = chunk_text(masked_text, chunk_size=chunk_size, overlap=chunk_overlap)

        # 4. Metadata tagging + store
        metadatas = [
            {
                "class_name": class_name,
                "doc_name": doc_name,
                "doc_type": doc_type,
                "page_or_slide": unit_num,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "chunk_size_used": chunk_size,
            }
            for _ in chunks
        ]
        ids = [f"{doc_name}-{unit_num}-{i}-{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]

        if chunks:
            add_chunks(class_name, chunks, metadatas, ids)
            total_chunks_added += len(chunks)

    result = {
        "doc_name": doc_name,
        "class_name": class_name,
        "units_processed": len(units),
        "chunks_added": total_chunks_added,
        "pii_instances_masked": total_pii_masked,
        "injection_flags": injection_flags,
    }
    audit_log("ingestion_complete", **result)
    return result
