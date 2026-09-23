"""
CLI to bulk-ingest a folder of documents into the vector store.

Usage:
    python scripts/run_ingestion.py --folder ./data/uploads/info4670 --class-name "INFO 4670" --doc-type slides
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.pipeline import ingest_document

SUPPORTED = {".pdf", ".pptx", ".md", ".txt"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Folder of documents to ingest")
    parser.add_argument("--class-name", required=True, help="e.g. 'INFO 4670'")
    parser.add_argument("--doc-type", default="unspecified", help="slides | notes | reading")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    args = parser.parse_args()

    folder = Path(args.folder)
    files = [f for f in folder.iterdir() if f.suffix.lower() in SUPPORTED]

    if not files:
        print(f"No supported files found in {folder}")
        return

    print(f"Ingesting {len(files)} files into class '{args.class_name}'...")
    for f in files:
        result = ingest_document(
            str(f),
            class_name=args.class_name,
            doc_type=args.doc_type,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        print(
            f"  {result['doc_name']}: {result['chunks_added']} chunks added, "
            f"{result['pii_instances_masked']} PII instances masked, "
            f"{len(result['injection_flags'])} injection flags"
        )

    print("Done.")


if __name__ == "__main__":
    main()
