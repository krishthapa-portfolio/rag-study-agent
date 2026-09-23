"""
FastAPI entrypoint.

Phase 1 scope: document upload + ingestion endpoints only.
/query (retrieval + LangGraph agent) lands in Phase 3-4 — see README roadmap.
"""
import shutil
import tempfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from pydantic import BaseModel

from app.ingestion.pipeline import ingest_document
from app.vectorstore.chroma_client import list_classes, get_collection, _client
from app.retrieval.hybrid import hybrid_search
from app.agent.graph import ask as agent_ask
from app.utils.rate_limit import enforce_rate_limit
from app.utils.logging_config import audit_log

app = FastAPI(title="Student RAG Study Agent", version="0.1.0")

SUPPORTED_EXTENSIONS = (".pdf", ".pptx", ".md", ".txt")


class IngestResponse(BaseModel):
    doc_name: str
    class_name: str
    units_processed: int
    chunks_added: int
    pii_instances_masked: int
    injection_flags: list


class BatchIngestResponse(BaseModel):
    class_name: str
    results: List[IngestResponse]
    failed: List[dict]  # [{"filename": ..., "error": ...}]


def _already_ingested(class_name: str, doc_name: str) -> bool:
    """
    Duplicate check: does this class's collection already have chunks
    tagged with this exact doc_name? Prevents accidentally doubling up
    a document if you upload the same file twice.
    """
    try:
        collection = get_collection(class_name)
        existing = collection.get(where={"doc_name": doc_name}, limit=1)
        return len(existing.get("ids", [])) > 0
    except Exception:
        return False


def _save_and_ingest(file: UploadFile, class_name: str, doc_type: str) -> dict:
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    if _already_ingested(class_name, file.filename):
        raise HTTPException(
            status_code=409,
            detail=f"'{file.filename}' has already been ingested into class '{class_name}'. "
            f"Delete the class first if you want to re-ingest it.",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = ingest_document(tmp_path, class_name=class_name, doc_type=doc_type)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return result


@app.post("/upload", response_model=IngestResponse)
async def upload_document(
    file: UploadFile = File(...),
    class_name: str = Form(...),
    doc_type: str = Form("unspecified"),
    _rate_limit=Depends(enforce_rate_limit),
):
    """
    Accepts ONE PDF, PPTX, or MD/TXT file, ingests it into the given class's
    namespace. Rejects (409) if this exact filename was already ingested
    into this class, to avoid silent duplicates.
    """
    return _save_and_ingest(file, class_name, doc_type)


@app.post("/upload-batch", response_model=BatchIngestResponse)
async def upload_documents_batch(
    files: List[UploadFile] = File(...),
    class_name: str = Form(...),
    doc_type: str = Form("unspecified"),
    _rate_limit=Depends(enforce_rate_limit),
):
    """
    Accepts MULTIPLE files in one request, all ingested into the same
    class namespace. In Swagger's /docs UI, the file picker for this
    endpoint lets you select several files at once (ctrl/cmd-click, or
    drag-select) instead of uploading one at a time.

    A file that fails (bad type, or already ingested) is skipped and
    reported in `failed`, without blocking the rest of the batch.
    """
    results = []
    failed = []
    for file in files:
        try:
            result = _save_and_ingest(file, class_name, doc_type)
            results.append(result)
        except HTTPException as e:
            failed.append({"filename": file.filename, "error": e.detail})
        except Exception as e:
            failed.append({"filename": file.filename, "error": str(e)})

    return {"class_name": class_name, "results": results, "failed": failed}


@app.get("/classes")
async def get_classes(_rate_limit=Depends(enforce_rate_limit)):
    """Lists every class namespace that has ingested documents so far."""
    return {"classes": list_classes()}


@app.delete("/classes/{class_name}")
async def delete_class(class_name: str, _rate_limit=Depends(enforce_rate_limit)):
    """
    Deletes an entire class namespace (all its ingested chunks). Use this
    to clear out a test/mistaken class, or to reset a class before
    re-ingesting its documents from scratch.
    """
    from app.vectorstore.chroma_client import _collection_name

    try:
        _client.delete_collection(_collection_name(class_name))
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not delete '{class_name}': {e}")

    audit_log("class_deleted", class_name=class_name)
    return {"deleted": class_name}


@app.get("/health")
async def health():
    return {"status": "ok"}

class SearchResult(BaseModel):
    id: str
    document: str
    metadata: dict
    rrf_score: float


class SearchResponse(BaseModel):
    class_name: str
    query: str
    results: List[SearchResult]


@app.get("/search", response_model=SearchResponse)
async def search(
    class_name: str,
    query: str,
    n_results: int = 5,
    _rate_limit=Depends(enforce_rate_limit),
):
    """
    Phase 3 — hybrid retrieval only, NO LLM involved yet.
    """
    results = hybrid_search(class_name, query, n_results=n_results)
    audit_log(
        "search_query",
        class_name=class_name,
        query=query,
        n_results_returned=len(results),
    )
    return {"class_name": class_name, "query": query, "results": results}

class AskRequest(BaseModel):
    class_name: str
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list
    retrieval_attempts: int


@app.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    _rate_limit=Depends(enforce_rate_limit),
):
    """
    Phase 4 — the full agent: retrieval -> sufficiency check -> retry
    loop -> synthesis with citations. This is the endpoint that actually
    answers a question.
    """
    result = await agent_ask(request.class_name, request.question)
    audit_log(
        "agent_answer",
        class_name=request.class_name,
        question=request.question,
        retrieval_attempts=result["retrieval_attempts"],
    )
    return result