"""
Baseline retriever: embed every chunk with a sentence-transformers model,
index in a local Chroma collection, expose retrieve(query, k).

This is deliberately the simplest possible retriever — no hybrid search,
no reranking, no query rewriting. It exists as the control group: every
later retrieval improvement (Phase 4) gets measured against this baseline
using the same golden set, so "hybrid retrieval improved recall@5 from
X to Y" is an actual number, not a claim.
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

ROOT = Path(__file__).parent.parent.parent
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"
PERSIST_DIR = str(ROOT / "data" / "processed" / "chroma_baseline")
COLLECTION_NAME = "compliance_baseline"

# Small, fast, good-enough-for-baseline embedding model. Swap out in
# Phase 4 if a domain-tuned or larger model measurably improves recall —
# "measurably" being the point of having this baseline at all.
_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _embedding_fn():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=_EMBEDDING_MODEL
    )


def build_index() -> chromadb.Collection:
    """Rebuild the index from scratch from chunks.jsonl. Always deletes
    any existing collection first — running this after re-chunking should
    never leave stale entries from a previous corpus version behind."""
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    ef = _embedding_fn()

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # didn't exist yet — fine

    collection = client.create_collection(COLLECTION_NAME, embedding_function=ef)

    chunks = load_chunks()
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[
            {
                "source_file": c["source_file"],
                "section": c["section"] or "",
                "start_page": c["start_page"],
                "end_page": c["end_page"],
            }
            for c in chunks
        ],
    )
    print(f"Indexed {len(chunks)} chunks into '{COLLECTION_NAME}'")
    return collection


def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    return client.get_collection(COLLECTION_NAME, embedding_function=_embedding_fn())


def retrieve(query: str, k: int = 5, collection: chromadb.Collection | None = None) -> list[dict]:
    """Return the top-k chunks for a query, each with its citation metadata."""
    if collection is None:
        collection = get_collection()

    results = collection.query(query_texts=[query], n_results=k)

    hits = []
    for doc, meta, distance, chunk_id in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        results["ids"][0],
    ):
        hits.append(
            {
                "chunk_id": chunk_id,
                "text": doc,
                "source_file": meta["source_file"],
                "section": meta["section"] or None,
                "distance": distance,
            }
        )
    return hits


if __name__ == "__main__":
    build_index()

    print("\nSmoke test:")
    hits = retrieve("What TDS rate applies to dividend payments?", k=3)
    for h in hits:
        print(f"  [{h['section']}] {h['chunk_id']}  (distance={h['distance']:.4f})")
        print(f"    {h['text'][:120]}...")