"""
Load raw compliance PDFs into page-level documents with metadata.

This is intentionally the *loader* only — not the chunker. Keep them
separate: the loader's job is faithful extraction + provenance, the
chunker's job (src/ingestion/chunker.py, next step) is deciding split
boundaries. Mixing them makes it hard to swap chunking strategies later
without re-touching extraction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf as fitz  # PyMuPDF (fitz is the legacy import name)


@dataclass
class RawPage:
    """One page of a source PDF, with enough metadata to cite it later."""

    source_file: str
    page_number: int  # 1-indexed, matches what a human would cite
    text: str
    detected_clause: str | None = field(default=None)


# Matches "Section 87", "Sec. 87", "दफा" numbers are Devanagari digits and
# need separate handling — see NOTE below.
_SECTION_PATTERN = re.compile(r"\b(?:Section|Sec\.?)\s+(\d+[A-Za-z]?)\b")


def _detect_clause(text: str) -> str | None:
    """
    Best-effort clause/section detector for citation metadata.

    NOTE: this only catches English "Section N" patterns. If your corpus
    includes Nepali-script PDFs (see data/SOURCES.md caveat), this will
    silently return None for those pages — that's a real gap, not an
    edge case, given IRD's directives are Devanagari-only. Decide
    explicitly whether v1 scopes to English-translated sources or whether
    you add a Devanagari दफा (\u0926\u092b\u093e) pattern here before
    treating this function as done.
    """
    match = _SECTION_PATTERN.search(text)
    return f"Section {match.group(1)}" if match else None


def load_pdf(path: Path) -> list[RawPage]:
    """
    Extract one RawPage per PDF page, in reading order.

    Clause detection carries state across pages: a section spanning many
    pages (common — definitions sections, schedules) only restates
    "Section N" on its first page. Without carry-forward, every
    continuation page loses its citation. We track the last section seen
    and apply it to any page that doesn't introduce a new one, and reset
    that state per-document so section numbers never leak across files.
    """
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    pages: list[RawPage] = []
    doc = fitz.open(path)
    running_clause: str | None = None
    try:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue  # skip blank/scanned pages rather than fabricate content

            new_clause = _detect_clause(text)
            if new_clause:
                running_clause = new_clause

            pages.append(
                RawPage(
                    source_file=path.name,
                    page_number=i,
                    text=text,
                    detected_clause=running_clause,
                )
            )
    finally:
        doc.close()

    return pages


def load_corpus(raw_dir: Path) -> list[RawPage]:
    """Load every PDF in raw_dir. Fails loudly on an empty directory —
    an empty corpus should never silently produce an empty index."""
    pdfs = sorted(raw_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"No PDFs found in {raw_dir}. Populate it per data/SOURCES.md first."
        )

    all_pages: list[RawPage] = []
    for pdf_path in pdfs:
        pages = load_pdf(pdf_path)
        print(f"  {pdf_path.name}: {len(pages)} pages extracted")
        all_pages.extend(pages)

    return all_pages


if __name__ == "__main__":
    corpus_dir = Path(__file__).parent.parent.parent / "data" / "raw_pdfs"
    pages = load_corpus(corpus_dir)
    print(f"\nTotal: {len(pages)} pages across corpus")

    with_clause = sum(1 for p in pages if p.detected_clause)
    print(f"Clause metadata detected on {with_clause}/{len(pages)} pages")
    if with_clause < len(pages) * 0.5:
        print(
            "WARNING: clause detection hit rate is low. Check whether your "
            "corpus is Nepali-script (expected, see docstring) or whether "
            "the English pattern needs work."
        )
