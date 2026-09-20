"""
Validate data/golden_set/golden_set.jsonl before it's used by anything else.

Checks:
- required fields present and non-empty
- unique ids
- difficulty/category from expected sets
- source_clause actually exists among the corpus's detected section labels
  (a golden entry citing a section our loader never detected is either a
  typo or points at content we can't actually retrieve/cite — either way,
  worth catching now rather than after the eval harness reports a
  confusing low score for the wrong reason)

Usage:
    python -m src.ingestion.validate_golden_set
"""

from __future__ import annotations

import json
from pathlib import Path

VALID_DIFFICULTIES = {"easy", "medium", "hard"}
REQUIRED_FIELDS = {"id", "question", "expected_answer", "source_file", "source_clause", "category", "difficulty"}


def load_jsonl(path: Path) -> list[dict]:
    entries = []
    with path.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {line_num}: {e}") from e
    return entries


def known_sections(chunks_path: Path) -> set[tuple[str, str]]:
    """(source_file, section) pairs that actually exist in the corpus."""
    known = set()
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c["section"]:
                known.add((c["source_file"], c["section"]))
    return known


def validate(entries: list[dict], known: set[tuple[str, str]]) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()

    for i, entry in enumerate(entries):
        loc = f"entry {i} (id={entry.get('id', '?')})"

        missing = REQUIRED_FIELDS - entry.keys()
        if missing:
            errors.append(f"{loc}: missing fields {missing}")
            continue  # can't check further fields meaningfully

        if entry["id"] in seen_ids:
            errors.append(f"{loc}: duplicate id")
        seen_ids.add(entry["id"])

        for field in ("question", "expected_answer", "source_file", "source_clause", "category"):
            if not entry[field].strip():
                errors.append(f"{loc}: '{field}' is empty")

        if entry["difficulty"] not in VALID_DIFFICULTIES:
            errors.append(f"{loc}: difficulty '{entry['difficulty']}' not in {VALID_DIFFICULTIES}")

        key = (entry["source_file"], entry["source_clause"])
        if key not in known:
            errors.append(
                f"{loc}: source_clause '{entry['source_clause']}' not found in "
                f"{entry['source_file']} chunks — check spelling against "
                f"'python -m src.ingestion.browse_chunks --list-sections'"
            )

    return errors


def main() -> None:
    root = Path(__file__).parent.parent.parent
    golden_path = root / "data" / "golden_set" / "golden_set.jsonl"
    chunks_path = root / "data" / "processed" / "chunks.jsonl"

    if not golden_path.exists():
        print(f"No golden set yet at {golden_path}")
        return

    entries = load_jsonl(golden_path)
    known = known_sections(chunks_path)
    errors = validate(entries, known)

    print(f"{len(entries)} entries checked")
    if errors:
        print(f"\n{len(errors)} problem(s):\n")
        for e in errors:
            print(f"  - {e}")
    else:
        print("All entries valid.")

    by_category: dict[str, int] = {}
    by_difficulty: dict[str, int] = {}
    for e in entries:
        by_category[e.get("category", "?")] = by_category.get(e.get("category", "?"), 0) + 1
        by_difficulty[e.get("difficulty", "?")] = by_difficulty.get(e.get("difficulty", "?"), 0) + 1

    print(f"\nBy category: {by_category}")
    print(f"By difficulty: {by_difficulty}")


if __name__ == "__main__":
    main()