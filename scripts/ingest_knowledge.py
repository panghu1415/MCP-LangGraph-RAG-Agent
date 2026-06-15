from __future__ import annotations

import argparse
from pathlib import Path

from config import settings
from rag.vector_store import ingest_chunks_to_milvus, load_knowledge_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest travel knowledge into Milvus with DashScope Embedding.")
    parser.add_argument("--file", type=Path, default=settings.knowledge_file, help="Knowledge markdown/txt file.")
    parser.add_argument("--drop-old", action="store_true", help="Drop old Milvus collection before ingesting.")
    args = parser.parse_args()

    chunks = load_knowledge_chunks(args.file)
    count = ingest_chunks_to_milvus(chunks, source=str(args.file), drop_old=args.drop_old)
    print(f"Ingested {count} chunks into Milvus collection '{settings.milvus_collection}'.")


if __name__ == "__main__":
    main()
