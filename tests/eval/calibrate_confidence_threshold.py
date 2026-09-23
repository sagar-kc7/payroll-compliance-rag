"""
Calibration: what do top-1 rerank scores actually look like for
in-scope questions (the 25-entry golden set) vs. genuinely
out-of-scope ones? No Groq calls — retrieval and reranking run
entirely locally, free to run repeatedly.

The output of this script is what decides the confidence threshold in
src/pipeline.py — not a guessed number.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from src.retrieval.reranker import retrieve

ROOT = Path(__file__).parent.parent.parent
GOLDEN_SET_PATH = ROOT / "data" / "golden_set" / "golden_set.jsonl"

# Deliberately unrelated to Nepali payroll/tax law — a well-behaved
# system should retrieve nothing genuinely relevant to these, and the
# reranker should reflect that with low scores.
OUT_OF_SCOPE_QUESTIONS = [
    "What is the capital gains tax rate in the United States?",
    "Who is the current president of Nepal?",
    "What's the weather like in Kathmandu today?",
    "How do I bake a chocolate cake?",
    "What is the boiling point of water?",
]


def load_golden_questions() -> list[str]:
    with GOLDEN_SET_PATH.open(encoding="utf-8") as f:
        return [json.loads(line)["question"] for line in f]


def top_score(question: str) -> float:
    results = retrieve(question, k=5)
    return results[0]["rerank_score"] if results else float("-inf")


if __name__ == "__main__":
    in_scope_scores = [top_score(q) for q in load_golden_questions()]
    out_of_scope_scores = [top_score(q) for q in OUT_OF_SCOPE_QUESTIONS]

    print("=== In-scope (golden set, 25 questions) ===")
    print(f"  min: {min(in_scope_scores):.3f}")
    print(f"  max: {max(in_scope_scores):.3f}")
    print(f"  mean: {statistics.mean(in_scope_scores):.3f}")
    print(f"  median: {statistics.median(in_scope_scores):.3f}")

    print("\n=== Out-of-scope (5 questions) ===")
    for q, s in zip(OUT_OF_SCOPE_QUESTIONS, out_of_scope_scores):
        print(f"  {s:7.3f}  {q}")
    print(f"  mean: {statistics.mean(out_of_scope_scores):.3f}")

    print("\n=== Separation ===")
    print(f"  lowest in-scope score: {min(in_scope_scores):.3f}")
    print(f"  highest out-of-scope score: {max(out_of_scope_scores):.3f}")
    if min(in_scope_scores) > max(out_of_scope_scores):
        print("  Clean separation — a threshold between these two values is well-justified.")
    else:
        print("  OVERLAP — no clean threshold exists; gating on score alone may not be reliable.")