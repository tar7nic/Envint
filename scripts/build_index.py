"""Index every document in data/corpus into Qdrant.  Run: python -m scripts.build_index"""
import sys
import time

from app.config import get_settings
from app.ingestion.chunker import load_corpus
from app.ingestion.indexer import Indexer


def main() -> int:
    s = get_settings()
    indexer = Indexer()
    indexer.ensure_collection()
    chunks, failed = load_corpus(s.corpus_dir)
    t0 = time.perf_counter()
    stats = indexer.upsert_chunks(chunks)
    print(f"chunks={stats.total} embedded={stats.embedded} skipped={stats.skipped} "
          f"time={time.perf_counter() - t0:.2f}s")
    for name, err in failed:
        print(f"SKIPPED {name}: {err}")
    return 0


if __name__ == "__main__":
    sys.exit(main())