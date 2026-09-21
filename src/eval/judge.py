"""
Custom DeepEval judge model, wrapping Groq via langchain-groq.

DeepEval's judged metrics (Faithfulness, AnswerRelevancy, GEval, etc.)
need a model implementing DeepEvalBaseLLM. Using Groq here — same
free/cheap model family as the rest of this project — avoids any
OpenAI cost entirely, and keeps the scarce AWS credits reserved for the
Phase 6 cost-comparison demo rather than spent on eval judging now.

Usage:
    from src.eval.judge import GroqJudge
    metric = FaithfulnessMetric(model=GroqJudge())
"""

from __future__ import annotations

import os

from deepeval.models import DeepEvalBaseLLM
from langchain_groq import ChatGroq


class GroqJudge(DeepEvalBaseLLM):
    """
    Note: DeepEval's metrics often ask the judge for structured
    (JSON-shaped) output internally. Larger instruction-tuned models
    handle this more reliably than smaller ones — if a metric run
    throws JSON-parsing errors, that's usually the model struggling
    with the expected output shape, not a bug in this wrapper. Try
    bumping to a larger Groq model before assuming something's broken.
    """

    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        self.model_name = model_name
        self._model = ChatGroq(
            model=model_name,
            temperature=0,
            api_key=os.environ["GROQ_API_KEY"],
        )

    def load_model(self):
        return self._model

    def generate(self, prompt: str) -> str:
        return self._model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        response = await self._model.ainvoke(prompt)
        return response.content

    def get_model_name(self) -> str:
        return self.model_name