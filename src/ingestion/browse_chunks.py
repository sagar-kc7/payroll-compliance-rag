"""
Browse data/processed/chunks.jsonl while writing golden-set questions.

Usage:
    python -m src.ingestion.browse_chunks --list-sections
    python -m src.ingestion.browse_chunks --section "Section 4"
    python -m src.ingestion.browse_chunks --keyword "tax rate"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_chunks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def list_sections(chunks: list[dict]) -> None:
    seen: dict[str, str] = {}
    for c in chunks:
        if c["section"]:
            key = f"{c['source_file']} :: {c['section']}"
            seen.setdefault(key, c["section"])
    for key in sorted(seen):
        print(key)
    print(f"\n{len(seen)} distinct sections/rules/schedules across corpus")


def show_section(chunks: list[dict], section: str) -> None:
    matches = [c for c in chunks if c["section"] == section]
    if not matches:
        print(f"No chunks found with section == {section!r}. "
              f"Run --list-sections to see exact labels.")
        return
    for c in matches:
        print(f"--- {c['chunk_id']} (page {c['start_page']}-{c['end_page']}, "
              f"{c['token_count']} tokens) ---")
        print(c["text"])
        print()


def search_keyword(chunks: list[dict], keyword: str) -> None:
    keyword_lower = keyword.lower()
    matches = [c for c in chunks if keyword_lower in c["text"].lower()]
    print(f"{len(matches)} chunk(s) contain {keyword!r}\n")
    for c in matches:
        idx = c["text"].lower().find(keyword_lower)
        start = max(0, idx - 80)
        end = min(len(c["text"]), idx + len(keyword) + 200)
        snippet = c["text"][start:end].replace("\n", " ")
        print(f"[{c['chunk_id']}] ...{snippet}...")
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-sections", action="store_true")
    parser.add_argument("--section", type=str)
    parser.add_argument("--keyword", type=str)
    args = parser.parse_args()

    chunks_path = Path(__file__).parent.parent.parent / "data" / "processed" / "chunks.jsonl"
    chunks = load_chunks(chunks_path)

    if args.list_sections:
        list_sections(chunks)
    elif args.section:
        show_section(chunks, args.section)
    elif args.keyword:
        search_keyword(chunks, args.keyword)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()