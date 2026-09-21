"""
Basic answer generation: question + retrieved chunks -> grounded answer.

Deliberately simple for now — no refusal/confidence gating, no PII
handling, no injection defense. Those are Phase 5 (safety layer). This
exists so the generator and pipeline eval layers have something to
measure; don't mistake it for the production version.
"""

from __future__ import annotations

import os

from langchain_groq import ChatGroq

_GENERATION_MODEL = "openai/gpt-oss-120b"

_SYSTEM_PROMPT = """You are a compliance assistant answering questions about \
Nepali payroll and tax law, using ONLY the provided source text.

Rules:
- Answer using only the information in the provided context.
- If the context does not contain enough information to answer, say so \
explicitly rather than guessing.
- Be concise and specific — cite section numbers when the context includes them.
"""


def format_context(chunks: list[dict]) -> str:
    """Turn retrieved chunks into a labeled context block the model can cite from."""
    parts = []
    for c in chunks:
        label = f"{c.get('source_file', '?')} :: {c.get('section', '?')}"
        parts.append(f"[{label}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def generate_answer(question: str, context_chunks: list[dict]) -> str:
    """Generate an answer grounded in the given context chunks."""
    model = ChatGroq(
        model=_GENERATION_MODEL,
        temperature=0,
        api_key=os.environ["GROQ_API_KEY"],
    )

    context_block = format_context(context_chunks)
    prompt = f"Context:\n{context_block}\n\nQuestion: {question}\n\nAnswer:"

    response = model.invoke(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )
    return response.content


if __name__ == "__main__":
    # Smoke test with a hand-built context chunk, no retriever involved —
    # isolates "does generation work at all" from "does retrieval work."
    fake_chunk = {
        "source_file": "income_tax_act_2058.pdf",
        "section": "Section 88",
        "text": (
            "88. Withholding of tax in making payment for investment return "
            "and service charge: (1) In making payment by a resident person "
            "for interest, natural resource, rent, royalty, service charge "
            "having source in Nepal... the person shall withhold tax at the "
            "rate of Fifteen percent of the total amount of payment."
        ),
    }
    answer = generate_answer(
        "What TDS rate applies to royalty payments under Section 88?",
        [fake_chunk],
    )
    print(answer)