"""
Structure-aware chunking for legal/statutory text.

Strategy, in order of priority:
1. Group consecutive pages that share the same detected_clause (section)
   into one "section block" — a section shouldn't be split just because
   it crosses a page boundary in the source PDF.
2. If a section block is small enough to fit one chunk, keep it whole.
   Splitting "Section 2(g): definition of incapacitated person" away from
   its own clause label would hurt retrieval for no reason.
3. If a section block is too large (the Income Tax Act's Section 2 —
   definitions — runs for pages), split on legal sub-structure in this
   priority order: sub-clause markers like "(a)", "(b)" > paragraph
   breaks > sentence breaks. This keeps each chunk's boundary somewhere
   a human reading the Act would also treat as a natural break.

Every chunk keeps its section label and source page range, so citation
in Phase 5 (grounding) can point back to something checkable.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.loader import RawPage, load_corpus

_ENCODING = tiktoken.get_encoding("cl100k_base")

MAX_CHUNK_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_file: str
    section: str | None
    start_page: int
    end_page: int
    token_count: int


def _count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def _group_into_section_blocks(pages: list[RawPage]) -> list[list[RawPage]]:
    """
    Group consecutive pages sharing the same detected_clause into blocks.

    Pages before the first detected section (None) form their own block —
    that's front matter (preamble, title page) and is legitimately
    citation-less; don't force it under a fabricated label.
    """
    if not pages:
        return []

    blocks: list[list[RawPage]] = [[pages[0]]]
    for page in pages[1:]:
        if page.detected_clause == blocks[-1][-1].detected_clause:
            blocks[-1].append(page)
        else:
            blocks.append([page])
    return blocks


# Splits on a sub-clause marker at the start of a line: "(a)", "(1)", "(aj)"
# — this is the Act's own numbering scheme, seen directly in your loader
# output (e.g. "(am)  \"Investment insurance\" means...").
_SUBCLAUSE_SPLITTER = RecursiveCharacterTextSplitter(
    separators=[
        r"\n\([a-z]+\)\s",   # (a), (aj), (am) — regex-aware split below
        r"\n\(\d+\)\s",      # (1), (2)
        "\n\n",
        "\n",
        ". ",
        " ",
    ],
    is_separator_regex=True,
    chunk_size=MAX_CHUNK_TOKENS,
    chunk_overlap=CHUNK_OVERLAP_TOKENS,
    length_function=_count_tokens,
)


def _chunk_section_block(pages: list[RawPage], index_in_doc: int) -> list[Chunk]:
    full_text = "\n\n".join(p.text for p in pages)
    section = pages[0].detected_clause
    source_file = pages[0].source_file
    start_page = pages[0].page_number
    end_page = pages[-1].page_number

    total_tokens = _count_tokens(full_text)

    if total_tokens <= MAX_CHUNK_TOKENS:
        return [
            Chunk(
                chunk_id=f"{source_file}::{section or 'preamble'}::{index_in_doc}",
                text=full_text,
                source_file=source_file,
                section=section,
                start_page=start_page,
                end_page=end_page,
                token_count=total_tokens,
            )
        ]

    # Oversized block: split on legal sub-structure, but every resulting
    # chunk still carries the SAME section label and page range — a
    # sub-clause split doesn't change what section it's part of.
    pieces = _SUBCLAUSE_SPLITTER.split_text(full_text)
    return [
        Chunk(
            chunk_id=f"{source_file}::{section or 'preamble'}::{index_in_doc}.{i}",
            text=piece,
            source_file=source_file,
            section=section,
            start_page=start_page,
            end_page=end_page,
            token_count=_count_tokens(piece),
        )
        for i, piece in enumerate(pieces)
    ]


def chunk_document(pages: list[RawPage]) -> list[Chunk]:
    """Chunk one document's pages (already loaded via loader.load_pdf)."""
    blocks = _group_into_section_blocks(pages)
    chunks: list[Chunk] = []
    for i, block in enumerate(blocks):
        chunks.extend(_chunk_section_block(block, i))
    return chunks


def chunk_corpus(pages: list[RawPage]) -> list[Chunk]:
    """Chunk a full multi-document corpus, grouping by source_file first
    so section blocks never merge across different PDFs."""
    by_file: dict[str, list[RawPage]] = {}
    for page in pages:
        by_file.setdefault(page.source_file, []).append(page)

    all_chunks: list[Chunk] = []
    for source_file, doc_pages in by_file.items():
        doc_chunks = chunk_document(doc_pages)
        print(f"  {source_file}: {len(doc_pages)} pages -> {len(doc_chunks)} chunks")
        all_chunks.extend(doc_chunks)
    return all_chunks


def save_chunks(chunks: list[Chunk], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    root = Path(__file__).parent.parent.parent
    pages = load_corpus(root / "data" / "raw_pdfs")
    print(f"\nLoaded {len(pages)} pages, chunking...\n")

    chunks = chunk_corpus(pages)

    token_counts = [c.token_count for c in chunks]
    with_section = sum(1 for c in chunks if c.section)

    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Avg tokens/chunk: {sum(token_counts) / len(token_counts):.0f}")
    print(f"Max tokens in a chunk: {max(token_counts)}")
    print(f"Chunks with section citation: {with_section}/{len(chunks)}")

    out_path = root / "data" / "processed" / "chunks.jsonl"
    save_chunks(chunks, out_path)
    print(f"\nSaved to {out_path}")