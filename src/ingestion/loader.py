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
    """
    One segment of a source PDF, with enough metadata to cite it later.

    Despite the name, this is NOT always a whole physical page: if a page
    contains more than one heading (e.g. Section 87 and Section 88 both
    start on the same PDF page), that page is split into multiple RawPage
    entries sharing the same page_number, each carrying only the text
    that actually belongs to its own section. See load_pdf's docstring.
    """

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
# body-text mentions (verified against one at physical page 463 —
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


def _split_page_by_headings(
    text: str,
    heading_word: str,
    state: dict,
) -> list[tuple[str, str | None]]:
    """
    Split one page's text at every real heading boundary found on it,
    returning [(segment_text, label), ...] in reading order.

    `state` is the document-level running state (running_num, in_schedule,
    schedule_num, schedule_item_num, running_clause) — read at the start
    of this page and MUTATED in place so the next page continues from
    wherever this page left off. This is what lets a single page contain
    a schedule boundary AND a schedule item heading AND have both take
    effect at the right position, instead of only the page-level winner.

    A heading match that fails its monotonic check (see load_pdf) is
    simply not accepted as a split point — its text stays part of
    whatever segment was already running, exactly as before.
    """
    events: list[tuple[int, str]] = []  # (position, kind) where kind is "schedule" or "heading"

    for m in _SCHEDULE_BOUNDARY_PATTERN.finditer(text):
        events.append((m.start(), "schedule", int(m.group(1))))
    for m in _HEADING_PATTERN.finditer(text):
        events.append((m.start(), "heading", int(m.group(1))))

    events.sort(key=lambda e: e[0])

    accepted: list[tuple[int, str]] = []  # (position, new_label)
    for pos, kind, num in events:
        if kind == "schedule":
            state["in_schedule"] = True
            state["schedule_num"] = num
            state["schedule_item_num"] = None
            label = f"Schedule {num}"
            accepted.append((pos, label))
        else:  # heading
            if state["in_schedule"]:
                current = state["schedule_item_num"]
                if current is None or num > current:
                    state["schedule_item_num"] = num
                    label = f"Schedule {state['schedule_num']} Item {num}"
                    accepted.append((pos, label))
            else:
                current = state["running_num"]
                if current is None or num > current:
                    state["running_num"] = num
                    label = f"{heading_word} {num}"
                    accepted.append((pos, label))

    if not accepted:
        return [(text, state["running_clause"])]

    segments: list[tuple[str, str | None]] = []
    cursor = 0
    current_label = state["running_clause"]
    for pos, label in accepted:
        seg_text = text[cursor:pos].strip()
        if seg_text:
            segments.append((seg_text, current_label))
        current_label = label
        cursor = pos
    tail = text[cursor:].strip()
    if tail:
        segments.append((tail, current_label))

    state["running_clause"] = current_label
    return segments


def load_pdf(path: Path) -> list[RawPage]:
    """
    Extract RawPage segments in reading order, one or more per physical
    page depending on how many real headings it contains.

    Two independent labeling modes:
    - Main-body: "Section N" / "Rule N", monotonic-increasing (rejects
      false matches from nested/quoted numbering in amendment text).
    - Schedule: once a genuine "Schedule-N (Relating to Section M)"
      boundary is seen, switches to "Schedule N Item M" with its OWN
      monotonic counter, reset at each new Schedule boundary. Stays in
      schedule mode for the rest of the document (Nepali Acts put
      Schedules at the end).

    KNOWN LIMITATION (unchanged from before): content after schedules
    end that isn't itself a numbered item — e.g. a later "Amendments"
    appendix — carries forward the last real label rather than getting
    its own. Not solved here.
    """
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    heading_word = _heading_word(path.name)
    pages: list[RawPage] = []
    doc = fitz.open(path)
    # income_tax_act_2058.pdf pages 190-626 are consolidated amendment
    # Ordinances/Financial Acts (e.g. "Financial Ordinance, 2059" at page
    # 190) — a structurally distinct document with no Section/Rule/Schedule
    # heading format we model. Their content is already folded into the
    # sections above; citing this range would surface superseded historical
    # amendment language as if it were current law. Excluded deliberately —
    # see data/SOURCES.md. Verified pages 605-626 are still this same
    # amendment-instrument content, not a hidden later Schedule.
    max_page = 189 if path.name == "income_tax_act_2058.pdf" else None

    state = {
        "running_clause": None,
        "running_num": None,
        "in_schedule": False,
        "schedule_num": None,
        "schedule_item_num": None,
    }

    try:
        for i, page in enumerate(doc, start=1):
            if max_page and i > max_page:
                break
            text = page.get_text("text").strip()
            if not text:
                continue

            for seg_text, label in _split_page_by_headings(text, heading_word, state):
                pages.append(
                    RawPage(
                        source_file=path.name,
                        page_number=i,
                        text=seg_text,
                        detected_clause=label,
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
        print(f"  {pdf_path.name}: {len(pages)} segments extracted")
        all_pages.extend(pages)

    return all_pages


if __name__ == "__main__":
    corpus_dir = Path(__file__).parent.parent.parent / "data" / "raw_pdfs"
    pages = load_corpus(corpus_dir)
    print(f"\nTotal: {len(pages)} segments across corpus")

    with_clause = sum(1 for p in pages if p.detected_clause)
    print(f"Segments with clause metadata: {with_clause}/{len(pages)}")

    schedule_segments = sum(1 for p in pages if p.detected_clause and "Schedule" in p.detected_clause)
    print(f"Segments labeled as Schedule content: {schedule_segments}")