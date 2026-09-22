"""
Hybrid retriever: BM25 (lexical) + dense (the Phase 2 baseline) combined
via Reciprocal Rank Fusion (RRF).

Motivation, grounded in the actual Phase 2 baseline failures: dense
embeddings blur between semantically similar but distinct provisions
(SSF Sections 4/5/8/14 all discuss employer contribution obligations in
similar language — the baseline confused them). BM25 anchors on exact
term and number overlap, which is exactly the signal dense embeddings
under-weight. RRF combines both rankings without needing to tune a
weight between two very differently-scaled similarity metrics.
"""

from __future__ import annotations

import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.retrieval.baseline import CHUNKS_PATH, get_collection, load_chunks

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


class HybridRetriever:
    """Builds the BM25 index once (corpus is small — ~360 chunks, this
    is fast) and reuses it across queries, alongside the existing
    persisted Chroma collection for the dense side."""

    def __init__(self, chunks_path: Path = CHUNKS_PATH):
        self.chunks = load_chunks(chunks_path)
        self._by_id = {c["chunk_id"]: c for c in self.chunks}
        tokenized_corpus = [_tokenize(c["text"]) for c in self.chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._dense_collection = get_collection()

    def _bm25_ranking(self, query: str, candidate_k: int) -> list[str]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i]["chunk_id"] for i in ranked_indices[:candidate_k]]

    def _dense_ranking(self, query: str, candidate_k: int) -> list[str]:
        results = self._dense_collection.query(query_texts=[query], n_results=candidate_k)
        return results["ids"][0]

    def retrieve(self, query: str, k: int = 5, candidate_k: int = 20, rrf_k: int = 60) -> list[dict]:
        """
        candidate_k: how many results each individual retriever
        contributes before fusion — wider than k so RRF has enough
        signal to work with.
        rrf_k: RRF's smoothing constant (60 is the standard default from
        the original paper) — controls how much weight lower ranks still
        get; not tuned for this corpus specifically.
        """
        bm25_ids = self._bm25_ranking(query, candidate_k)
        dense_ids = self._dense_ranking(query, candidate_k)

        rrf_scores: dict[str, float] = {}
        for rank, chunk_id in enumerate(bm25_ids, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
        for rank, chunk_id in enumerate(dense_ids, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)

        top_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:k]

        hits = []
        for chunk_id in top_ids:
            c = self._by_id[chunk_id]
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "text": c["text"],
                    "source_file": c["source_file"],
                    "section": c["section"] or None,
                    "rrf_score": rrf_scores[chunk_id],
                }
            )
        return hits


_singleton: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _singleton
    if _singleton is None:
        _singleton = HybridRetriever()
    return _singleton


def retrieve(query: str, k: int = 5) -> list[dict]:
    return get_retriever().retrieve(query, k=k)


if __name__ == "__main__":
    hits = retrieve("What TDS rate applies to dividend payments?", k=3)
    for h in hits:
        print(f"  [{h['section']}] {h['chunk_id']}  (rrf={h['rrf_score']:.4f})")
        print(f"    {h['text'][:120]}...")