"""
Reranking layer: cross-encoder reranks a wider candidate pool from
hybrid retrieval down to the final top-k.

Cross-encoders jointly encode (query, document) pairs rather than
embedding them independently, letting them pick up fine-grained
discrimination that neither BM25 term-overlap nor bi-encoder cosine
similarity can. This directly targets the gs_012/gs_013 failure —
SSF Sections 4/5/8/9/14 all use near-identical "contribution/deposit/
employer" vocabulary, which persisted through hybrid retrieval (see
tests/eval/test_retrieval_hybrid.py's docstring for that finding).
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from sentence_transformers import CrossEncoder

from langsmith import traceable

from src.retrieval.hybrid import retrieve as hybrid_retrieve

_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(_RERANKER_MODEL)
    return _model


@traceable(run_type="retriever", name="hybrid_reranked_retrieve")
def retrieve(query: str, k: int = 5, candidate_k: int = 15) -> list[dict]:
    """
    Pull `candidate_k` results from hybrid retrieval (wider than k, so
    the reranker has real signal to work with — reranking a pool of 5
    down to 5 does nothing), score each with the cross-encoder, return
    the top-k by that score.
    """
    candidates = hybrid_retrieve(query, k=candidate_k)
    if not candidates:
        return []

    model = _get_model()
    pairs = [[query, c["text"]] for c in candidates]
    scores = model.predict(pairs)

    for c, score in zip(candidates, scores):
        c["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:k]


if __name__ == "__main__":
    hits = retrieve("What TDS rate applies to dividend payments?", k=3)
    for h in hits:
        print(f"  [{h['section']}] {h['chunk_id']}  (rerank={h['rerank_score']:.4f})")
        print(f"    {h['text'][:120]}...")