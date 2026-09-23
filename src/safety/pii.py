"""
PII detection and redaction using Presidio.

Applied to raw slip text / free-form user input before it's logged,
stored, or (in Phase 6) sent to LangSmith tracing — NOT applied to the
Act/Rules corpus, which is public statute text with no PII to protect.

Explicitly configured to use spacy's medium English model (en_core_web_md)
rather than relying on Presidio's default, which pulls the much larger
en_core_web_lg (~800MB) if left unspecified — worth being deliberate
about that rather than surprising anyone with an unexpected download.

Setup (one-time):
    python -m spacy download en_core_web_md
"""

from __future__ import annotations

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

_NLP_CONFIGURATION = {
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_md"}],
}

# PERSON/PHONE_NUMBER/EMAIL_ADDRESS are Presidio built-ins (English/
# US-centric pattern quality, but reasonable for names/emails/phones
# generally). NEPAL_PAN is custom — Presidio has no built-in for it.
_ENTITIES = ["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "NEPAL_PAN"]

_analyzer: AnalyzerEngine | None = None
_anonymizer: AnonymizerEngine | None = None


def _register_custom_recognizers(analyzer: AnalyzerEngine) -> None:
    """
    Nepal PAN (Permanent Account Number) is a bare 9-digit number with
    no distinguishing format — no Presidio built-in exists for it, and
    this pattern WILL also match some unrelated 9-digit numbers.
    Deliberate: for redaction, over-flagging (false positive) is the
    safe failure mode, under-flagging (false negative, a real PAN
    getting through) is not.
    """
    pan_pattern = Pattern(name="nepal_pan_pattern", regex=r"\b\d{9}\b", score=0.5)
    pan_recognizer = PatternRecognizer(supported_entity="NEPAL_PAN", patterns=[pan_pattern])
    analyzer.registry.add_recognizer(pan_recognizer)


def _get_analyzer() -> AnalyzerEngine:
    global _analyzer
    if _analyzer is None:
        provider = NlpEngineProvider(nlp_configuration=_NLP_CONFIGURATION)
        nlp_engine = provider.create_engine()
        _analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
        _register_custom_recognizers(_analyzer)
    return _analyzer


def _get_anonymizer() -> AnonymizerEngine:
    global _anonymizer
    if _anonymizer is None:
        _anonymizer = AnonymizerEngine()
    return _anonymizer


def detect_pii(text: str) -> list[dict]:
    """Return detected PII entities without modifying the text — for
    logging/flagging (e.g. 'this input contained 2 PII entities')."""
    results = _get_analyzer().analyze(text=text, entities=_ENTITIES, language="en")
    return [
        {
            "entity_type": r.entity_type,
            "start": r.start,
            "end": r.end,
            "score": r.score,
            "text": text[r.start : r.end],
        }
        for r in results
    ]


def redact_pii(text: str) -> str:
    """Return text with detected PII entities replaced by <ENTITY_TYPE> placeholders."""
    analyzer_results = _get_analyzer().analyze(text=text, entities=_ENTITIES, language="en")
    anonymized = _get_anonymizer().anonymize(text=text, analyzer_results=analyzer_results)
    return anonymized.text


if __name__ == "__main__":
    sample = """
    Employee: Ramesh Shrestha
    PAN: 123456789
    Phone: 9841234567
    Email: ramesh.shrestha@example.com

    Basic Salary: Rs. 40,000
    """

    print("=== Detected entities ===")
    for e in detect_pii(sample):
        print(f"  {e['entity_type']}: {e['text']!r} (score={e['score']:.2f})")

    print("\n=== Redacted text ===")
    print(redact_pii(sample))