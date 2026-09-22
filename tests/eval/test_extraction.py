"""
Extraction tests: does the arithmetic cross-check actually fire when it
should, and does extraction survive messier, more realistic formatting
than the clean smoke-test sample?

Costs real Groq tokens (3 extraction calls) — run deliberately, not on
every save.

Run standalone:
    PYTHONPATH=. python -m tests.eval.test_extraction

Run under pytest:
    pytest tests/eval/test_extraction.py -m eval
"""

from __future__ import annotations

import pytest

from src.extraction.extractor import extract_salary_slip

# Deliberately internally CONSISTENT — same as the extractor's own smoke
# test sample. Included here too as a regression check: if this ever
# starts producing warnings, something changed (prompt, model, schema),
# not the input.
CONSISTENT_SLIP = """
ABC Trading Pvt. Ltd.
Salary Slip - Shrawan 2082

Employee: Ramesh Shrestha

Basic Salary: Rs. 40,000
Dashain Allowance: Rs. 5,000
Transport Allowance: Rs. 3,000
Gross Salary: Rs. 48,000

Deductions:
SSF Employee Contribution (11%): Rs. 5,280
TDS: Rs. 2,000

Employer SSF Contribution (20%): Rs. 9,600

Net Pay: Rs. 40,720
"""

# Deliberately INCONSISTENT — net_pay is stated as 45,000, but
# 48,000 - 5,280 - 2,000 = 40,720, not 45,000. A slip like this in the
# real world would mean either a typo on the slip itself or a
# miscalculation by the employer — exactly the kind of thing the
# arithmetic check exists to catch, not paper over.
INCONSISTENT_SLIP = """
XYZ Pvt. Ltd.
Salary Slip - Bhadra 2082

Employee: Sita Gurung

Basic Salary: Rs. 40,000
Dashain Allowance: Rs. 5,000
Transport Allowance: Rs. 3,000
Gross Salary: Rs. 48,000

Deductions:
SSF Employee Contribution: Rs. 5,280
TDS: Rs. 2,000

Employer SSF Contribution: Rs. 9,600

Net Pay: Rs. 45,000
"""

# Messier, more realistic formatting: no explicit "Gross Salary" line
# (has to be inferred/computed), inconsistent labeling style, an
# allowance folded into a parenthetical rather than its own line. Not
# testing a specific assertion here beyond "doesn't crash, produces a
# plausible result" — this is a robustness check, not a correctness
# check (we have no independently-verified ground truth for this one).
MESSY_SLIP = """
Kathmandu Traders — Payslip for Ashwin, 2082

Name: Bikash Tamang

Pay Details:
Basic: NPR 35000
Allowances: dashain bonus 4500, transport 2500, meal 1200

SSF (emp 11%): 4477
SSF (co 20%): 8140
TDS deducted this month: 1500

Take home: 35223
"""


@pytest.mark.eval
def test_consistent_slip_has_no_warnings():
    result = extract_salary_slip(CONSISTENT_SLIP)
    assert result.arithmetic_warnings == [], (
        f"Expected no warnings on a hand-verified consistent slip, got: "
        f"{result.arithmetic_warnings}"
    )


@pytest.mark.eval
def test_inconsistent_slip_triggers_warning():
    result = extract_salary_slip(INCONSISTENT_SLIP)
    assert result.arithmetic_warnings, (
        "Expected a net_pay arithmetic warning on a deliberately "
        "inconsistent slip, got none"
    )
    assert any("net_pay" in w for w in result.arithmetic_warnings), (
        f"Expected the warning to mention net_pay specifically, got: "
        f"{result.arithmetic_warnings}"
    )


@pytest.mark.eval
def test_messy_formatting_extracts_without_crashing():
    result = extract_salary_slip(MESSY_SLIP)
    assert result.basic_salary > 0
    assert len(result.allowances) >= 1
    print(f"\nMessy-slip extraction: {result.model_dump_json(indent=2)}")
    if result.arithmetic_warnings:
        print(f"Warnings (informational, not asserted): {result.arithmetic_warnings}")


if __name__ == "__main__":
    print("=== Consistent slip ===")
    r1 = extract_salary_slip(CONSISTENT_SLIP)
    print(f"Warnings: {r1.arithmetic_warnings or 'none'}\n")

    print("=== Inconsistent slip ===")
    r2 = extract_salary_slip(INCONSISTENT_SLIP)
    print(f"Warnings: {r2.arithmetic_warnings or 'none'}\n")

    print("=== Messy slip ===")
    r3 = extract_salary_slip(MESSY_SLIP)
    print(r3.model_dump_json(indent=2))
    print(f"Warnings: {r3.arithmetic_warnings or 'none'}")