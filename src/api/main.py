"""
FastAPI service wrapping src/pipeline.py (RAG Q&A) and
src/extraction/extractor.py (salary slip extraction).

Models (embedding, cross-encoder reranker) are preloaded once at
startup via the lifespan context manager, not per-request — real,
measured cost of NOT doing this: ~35s cold start on the first query
per process (confirmed via LangSmith traces during Phase 6). Every
subsequent request in this running server reuses the warm models.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.extraction.extractor import extract_salary_slip
from src.pipeline import answer_question
from src.retrieval.reranker import retrieve


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Warming up retrieval models (embedding + cross-encoder)...")
    retrieve("warmup query", k=1)
    print("Models warm. Ready to serve.")
    yield


app = FastAPI(
    title="Nepal Compliance RAG API",
    description="Grounded Q&A over Nepali payroll/tax law, plus salary slip extraction.",
    lifespan=lifespan,
)


class HealthResponse(BaseModel):
    status: str


class AskRequest(BaseModel):
    question: str
    k: int = 5


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    confidence_score: float | None
    escalated: bool
    escalation_reason: str | None = None


class ExtractRequest(BaseModel):
    slip_text: str


class ExtractResponse(BaseModel):
    pay_period: str
    employee_name: str | None
    basic_salary: float
    allowances: dict[str, float]
    gross_salary: float
    ssf_employee_contribution: float
    ssf_employer_contribution: float
    tds_deducted: float
    other_deductions: dict[str, float]
    net_pay: float
    arithmetic_warnings: list[str]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        result = answer_question(req.question, k=req.k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return AskResponse(
        answer=result.answer,
        citations=result.citations,
        confidence_score=result.confidence_score,
        escalated=result.escalated,
        escalation_reason=result.escalation_reason,
    )


@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest) -> ExtractResponse:
    """
    KNOWN LIMITATION: extract_salary_slip's LangSmith trace currently
    includes the full, unredacted slip text (see its @traceable
    decorator in src/extraction/extractor.py). Acceptable for this
    portfolio demo — every test so far has used synthetic data — but a
    real production deployment handling actual user slips would need
    trace-level redaction, or tracing disabled on this endpoint, before
    processing real PII. Not solved here; flagged rather than silently
    shipped as if it were.
    """
    try:
        result = extract_salary_slip(req.slip_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return ExtractResponse(
        pay_period=result.pay_period,
        employee_name=result.employee_name,
        basic_salary=result.basic_salary,
        allowances=result.allowances,
        gross_salary=result.gross_salary,
        ssf_employee_contribution=result.ssf_employee_contribution,
        ssf_employer_contribution=result.ssf_employer_contribution,
        tds_deducted=result.tds_deducted,
        other_deductions=result.other_deductions,
        net_pay=result.net_pay,
        arithmetic_warnings=result.arithmetic_warnings,
    )