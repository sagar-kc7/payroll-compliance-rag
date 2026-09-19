"""
Load raw compliance PDFs into page-level documents with metadata.

This is intentionally the *loader* only — not the chunker. Keep them
separate: the loader's job is faithful extraction + provenance, the
chunker's job (src/ingestion/chunker.py) is deciding split boundaries.
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


# Real headings look like "4.  Computation and rate of tax:" — bare
# number, period, 1-4 spaces, Capitalized Title, colon.
_HEADING_PATTERN = re.compile(
    r"(?:^|\n)(\d{1,3})\.\s{1,4}([A-Z][A-Za-z,\-\s]{2,100}):",
)


def _heading_word(source_file: str) -> str:
    """
    Acts and Rules number their provisions under different labels
    ("Section N" vs "Rule N"). Citing a Rules document as "Section" would
    be a wrong, checkable citation — infer the right word from filename.
    """
    return "Rule" if "rule" in source_file.lower() else "Section"


def load_pdf(path: Path) -> list[RawPage]:
    """
    Extract one RawPage per PDF page, in reading order.

    Clause detection requires the matched number to strictly exceed the
    current running section number. This rejects nested numbered items
    inside quoted amendment text, or Schedule items that restart
    numbering from 1, which match the same "N. Title:" shape as a real
    heading but use small/reused numbers — a real Act's section numbers
    only increase as you read forward.

    KNOWN LIMITATION: Schedules (which often contain the actual tax rate
    tables) will NOT get their own distinct section labels once the main
    numbered body has passed a higher number than the Schedule's own item
    numbers. Those pages keep citing the last real Section instead. This
    is a real correctness gap for any golden-set question sourced from a
    Schedule.
    """
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    heading_word = _heading_word(path.name)
    pages: list[RawPage] = []
    doc = fitz.open(path)
    running_clause: str | None = None
    running_num: int | None = None
    try:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue

            for match in _HEADING_PATTERN.finditer(text):
                num = int(match.group(1))
                if running_num is None or num > running_num:
                    running_num = num
                    running_clause = f"{heading_word} {num}"
                    break

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
