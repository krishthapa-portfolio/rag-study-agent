"""
Per-class namespace management over ChromaDB.

Each class gets its own ChromaDB *collection* (Chroma's namespace unit).
This is what enforces retrieval isolation: a query scoped to "INFO4670"
physically cannot return chunks stored under "Database Systems", because
they live in different collections entirely — it's not just a metadata
filter that could be forgotten, it's structural.
"""
import chromadb
from chromadb.utils import embedding_functions

from app.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL

_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)


def _collection_name(class_name: str) -> str:
    # Chroma collection names are restricted (alnum + a few chars),
    # so normalize the class name into something safe.
    safe = "".join(c if c.isalnum() else "_" for c in class_name.lower())
    return f"class_{safe}"


def get_collection(class_name: str):
    return _client.get_or_create_collection(
        name=_collection_name(class_name),
        embedding_function=_embedding_fn,
        metadata={"class_name": class_name},
    )


def add_chunks(
    class_name: str,
    chunks: list[str],
    metadatas: list[dict],
    ids: list[str],
):
    collection = get_collection(class_name)
    collection.add(documents=chunks, metadatas=metadatas, ids=ids)


def query_class(class_name: str, query_text: str, n_results: int = 5):
    collection = get_collection(class_name)
    return collection.query(query_texts=[query_text], n_results=n_results)


def list_classes() -> list[str]:
    return [c.metadata.get("class_name", c.name) for c in _client.list_collections()]



def get_all_chunks(class_name: str) -> dict:
    collection = get_collection(class_name)
    count = collection.count()
    if count == 0:
        return {"ids": [], "documents": [], "metadatas": []}
    result = collection.get(limit=count, include=["documents", "metadatas"])
    return {
        "ids": result["ids"],
        "documents": result["documents"],
        "metadatas": result["metadatas"],
    }