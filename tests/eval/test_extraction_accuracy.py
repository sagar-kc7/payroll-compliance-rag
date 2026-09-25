"""
Field-level accuracy eval for salary slip extraction, against 8
hand-verified synthetic slips (data/golden_set/salary_slips.jsonl).

Unlike the RAG golden set, expected values here are self-constructed
test data (not claims about external facts) — verified by hand-checking
each slip's arithmetic before writing it, same rigor, different reason.

Compares 8 fields per slip: basic_salary, allowances_total (vs
sum(allowances.values())), gross_salary, ssf_employee_contribution,
ssf_employer_contribution, tds_deducted, other_deductions_total (vs
sum(other_deductions.values())), net_pay. allowances/other_deductions
are compared by TOTAL, not key-by-key — the schema deliberately allows
open-ended naming, so exact key matching would be testing prompt
phrasing, not extraction correctness.

Run standalone:
    PYTHONPATH=. python -m tests.eval.test_extraction_accuracy

Run under pytest:
    pytest tests/eval/test_extraction_accuracy.py -m eval
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.extraction.extractor import extract_salary_slip

ROOT = Path(__file__).parent.parent.parent
GOLDEN_PATH = ROOT / "data" / "golden_set" / "salary_slips.jsonl"

TOLERANCE = 1.0  # rupees — allows for rounding, not for real errors


def load_golden() -> list[dict]:
    with GOLDEN_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def evaluate_extraction() -> dict:
    golden = load_golden()

    field_results: dict[str, list[bool]] = {}
    per_slip = []

    for entry in golden:
        result = extract_salary_slip(entry["slip_text"])
        expected = entry["expected"]

        actual = {
            "basic_salary": result.basic_salary,
            "allowances_total": sum(result.allowances.values()),
            "gross_salary": result.gross_salary,
            "ssf_employee_contribution": result.ssf_employee_contribution,
            "ssf_employer_contribution": result.ssf_employer_contribution,
            "tds_deducted": result.tds_deducted,
            "other_deductions_total": sum(result.other_deductions.values()),
            "net_pay": result.net_pay,
        }

        field_matches = {}
        for field, expected_value in expected.items():
            actual_value = actual[field]
            matches = abs(actual_value - expected_value) <= TOLERANCE
            field_matches[field] = matches
            field_results.setdefault(field, []).append(matches)

        per_slip.append(
            {
                "id": entry["id"],
                "all_correct": all(field_matches.values()),
                "field_matches": field_matches,
                "expected": expected,
                "actual": actual,
            }
        )

    per_field_accuracy = {
        field: sum(matches) / len(matches) for field, matches in field_results.items()
    }
    total_checks = sum(len(m) for m in field_results.values())
    total_correct = sum(sum(m) for m in field_results.values())
    overall_accuracy = total_correct / total_checks
    perfect_slips = sum(1 for s in per_slip if s["all_correct"])

    return {
        "total_slips": len(golden),
        "overall_field_accuracy": overall_accuracy,
        "perfect_slip_rate": perfect_slips / len(golden),
        "per_field_accuracy": per_field_accuracy,
        "per_slip": per_slip,
    }

@pytest.mark.llm
@pytest.mark.eval
def test_extraction_field_accuracy():
    """
    First run — no threshold yet. Run standalone, record real numbers
    per field, THEN add a real assertion.
    """
    result = evaluate_extraction()
    print(f"\noverall field accuracy: {result['overall_field_accuracy']:.2%}")
    print(f"perfect-slip rate: {result['perfect_slip_rate']:.2%}")


if __name__ == "__main__":
    result = evaluate_extraction()

    print(f"Slips evaluated: {result['total_slips']}")
    print(f"Overall field accuracy: {result['overall_field_accuracy']:.2%}")
    print(f"Perfect-slip rate: {result['perfect_slip_rate']:.2%}\n")

    print("Per-field accuracy:")
    for field, acc in sorted(result["per_field_accuracy"].items()):
        print(f"  {field}: {acc:.2%}")

    errors = [s for s in result["per_slip"] if not s["all_correct"]]
    if errors:
        print(f"\n{len(errors)} slip(s) with at least one field mismatch:\n")
        for s in errors:
            print(f"  [{s['id']}]")
            for field, ok in s["field_matches"].items():
                if not ok:
                    print(f"    {field}: expected {s['expected'][field]}, got {s['actual'][field]}")
            print()