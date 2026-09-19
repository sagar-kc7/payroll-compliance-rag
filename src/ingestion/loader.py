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


# Real Section/Rule headings look like "4.  Computation and rate of tax:"
_HEADING_PATTERN = re.compile(
    r"(?:^|\n)(\d{1,3})\.\s{1,4}([A-Z][A-Za-z,\-\s]{2,100}):",
)

# Real Schedule headings are a distinctive two-line shape:
#   "Schedule-1"
#   "(Relating to Section 4)"
# Cross-references and amendment-history mentions of "Schedule-1" elsewhere
# in the document do NOT have that second line immediately following, so
# requiring both together is what keeps this from false-triggering on
# body-text mentions (verified against a real one at physical page 463 —
# "Schedule-1 / (70) / Amendment to Schedule-1:" — which does not match).
_SCHEDULE_BOUNDARY_PATTERN = re.compile(
    r"(?:^|\n)\s*Schedule[\s\-]?(\d{1,2})\s*\n\s*\(Relating to Section",
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

    Two independent labeling modes, switched by whether a real Schedule
    boundary has been seen yet:

    - Main-body mode: pages are labeled "Section N" / "Rule N", using a
      monotonic-increasing constraint (a real Act's section numbers only
      go up) to reject false matches from nested/quoted numbering inside
      amendment text.
    - Schedule mode: once a genuine "Schedule-N (Relating to Section M)"
      boundary is detected, pages switch to "Schedule N Item M" labels,
      with their OWN monotonic item counter that resets at each new
      Schedule boundary — schedule items restart numbering from 1, which
      would otherwise collide with the main-body counter.

    Once in schedule mode, we stay there for the rest of the document —
    Nepali Acts structurally put Schedules at the end, after all
    substantive Sections, so there's no real case of "returning" to
    main-body content after the first Schedule starts.

    KNOWN LIMITATION: any page after schedules end that isn't itself a
    numbered item (e.g. a later "Amendments" appendix, like the one seen
    around physical page 463) will just carry forward the last real
    schedule-item label rather than getting its own — not solved here,
    flag if a golden-set question ends up sourced from that region.
    """
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    heading_word = _heading_word(path.name)
    pages: list[RawPage] = []
    doc = fitz.open(path)

    running_clause: str | None = None
    running_num: int | None = None          # main-body Section/Rule counter
    in_schedule = False
    schedule_num: int | None = None
    schedule_item_num: int | None = None    # resets at each new Schedule

    try:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue

            # Check for a new Schedule boundary first — it takes priority
            # over ordinary heading matches and flips the mode permanently.
            sched_match = _SCHEDULE_BOUNDARY_PATTERN.search(text)
            if sched_match:
                in_schedule = True
                schedule_num = int(sched_match.group(1))
                schedule_item_num = None
                running_clause = f"Schedule {schedule_num}"

            if in_schedule:
                for match in _HEADING_PATTERN.finditer(text):
                    num = int(match.group(1))
                    if schedule_item_num is None or num > schedule_item_num:
                        schedule_item_num = num
                        running_clause = f"Schedule {schedule_num} Item {num}"
                        break
            else:
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

    schedule_pages = sum(1 for p in pages if p.detected_clause and "Schedule" in p.detected_clause)
    print(f"Pages labeled as Schedule content: {schedule_pages}")