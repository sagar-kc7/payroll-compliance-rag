"""
Streamlit demo UI for the Nepal Compliance RAG project.

Calls src.pipeline.answer_question() DIRECTLY, not over HTTP — this
keeps a single-process, single-service deployment story rather than
running Streamlit + FastAPI as two separate services, which would
double the retrieval-model memory footprint (a real, already-flagged
concern given free-tier deployment RAM limits). The FastAPI service
(src/api/main.py) still exists separately for programmatic /ask,
/extract API access; this app is the human-facing demo, and it's the
same real pipeline function either way — nothing about going direct
vs. through HTTP changes what's actually being exercised.

Run:
    PYTHONPATH=. streamlit run src/ui/streamlit_app.py
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from src.pipeline import answer_question

st.set_page_config(page_title="Nepal Compliance RAG", page_icon="⚖️", layout="centered")

st.title("Nepal Compliance RAG")
st.caption(
    "Grounded Q&A over Nepali payroll & tax law, with real, measured evaluation "
    "results, not vibes."
)

st.subheader("Ask a question")
st.caption(
    "Grounded in the Income Tax Act 2058, Income Tax Rules 2059, and the SSF Act. "
    "Not legal or tax advice."
)

EXAMPLES = {
    "TDS on dividends": "What TDS rate applies to dividend payments under Section 88?",
    "Out-of-scope question": "What is the capital gains tax rate in the United States?",
}

if "question" not in st.session_state:
    st.session_state.question = ""

cols = st.columns(len(EXAMPLES))
for col, (label, q) in zip(cols, EXAMPLES.items()):
    if col.button(label):
        st.session_state.question = q

question = st.text_area(
    "Your question",
    key="question",
    height=100,
    placeholder="e.g. What TDS rate applies to dividend payments under Section 88?",
)

if st.button("Ask", type="primary"):
    if not question.strip():
        st.warning("Type a question first.")
    else:
        with st.spinner("Retrieving and generating an answer..."):
            result = answer_question(question)

        if result.escalated:
            st.warning(f"**Escalated — not enough confidence to answer.**\n\n{result.answer}")
            reason_line = f"reason: {result.escalation_reason}"
            if result.confidence_score is not None:
                reason_line = f"confidence score: {result.confidence_score:.3f} · " + reason_line
            st.caption(reason_line)
        else:
            st.success(result.answer)
            if result.confidence_score is not None:
                st.caption(f"confidence score: {result.confidence_score:.3f}")
            if result.citations:
                st.write("**Citations:**")
                for c in result.citations:
                    st.code(c, language=None)

st.divider()

st.subheader("About this project")
st.markdown(
    "Built to demonstrate what a portfolio RAG chatbot usually doesn't: evaluation "
    "discipline, structured extraction with validation, safety controls, and measured "
    "cost/latency. Every number below is from a real, documented test run."
)

st.table(
    {
        "Capability": [
            "Retrieval (baseline -> hybrid -> reranked)",
            "Generator faithfulness / relevancy",
            "End-to-end pipeline correctness",
            "Salary slip extraction accuracy (via /extract API)",
            "PII redaction",
            "Prompt-injection resistance",
            "Confidence gate",
        ],
        "Result": [
            "88.0% -> 92.0% -> 92.0% recall@5",
            "0.983 / 0.936",
            "0.800, 88.0% citation hit rate",
            "100% (8-slip labelled set)",
            "4/4 tests passed (Presidio + spacy)",
            "Resisted 2 real adversarial attempts",
            "Calibrated threshold, real score separation",
        ],
    }
)

st.markdown(
    "Full evaluation methodology, honest scope notes, and real bugs found and fixed "
    "along the way: [GitHub repo](https://github.com/sagar-kc7/payroll-compliance-rag)"
)

st.caption("Built by Sagar K.C.")