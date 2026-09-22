"""
Salary slip extraction schema.

Every field is grounded in something we've actually read in the corpus,
not a guessed payslip format:
- gross/allowances/basic: Section 5 (income headings), Section 8
  (taxable employment payments)
- retirement_contribution: Section 8(f) / Section 64 (deductible
  retirement contribution)
- ssf_*_contribution: SSF Act Section 7 — the Act itself doesn't fix a
  rate (Ministry sets it by notice), so this schema captures whatever
  value is on the slip rather than hardcoding a percentage
- tds_deducted: Section 87 (employer withholding at Schedule-1 rates)

allowances and other_deductions are open dicts, not fixed fields —
employers vary in what they offer (dashain, transport, housing, etc.),
and forcing a fixed field set would mean silently dropping whatever
doesn't fit.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, PrivateAttr, model_validator


class SalarySlip(BaseModel):
    pay_period: str = Field(
        ..., description="Pay period this slip covers, e.g. 'Shrawan 2082' or '2025-07'"
    )
    employee_name: str | None = Field(None, description="Employee name if present on the slip")

    basic_salary: float = Field(..., ge=0, description="Base/basic salary before allowances")
    allowances: dict[str, float] = Field(
        default_factory=dict,
        description="Named allowances and amounts, e.g. {'dashain': 5000, 'transport': 2000}",
    )
    gross_salary: float = Field(
        ..., ge=0, description="Total gross salary = basic_salary + sum(allowances)"
    )

    ssf_employee_contribution: float = Field(
        0, ge=0, description="Employee's SSF contribution, deducted from gross (SSF Act Section 7)"
    )
    ssf_employer_contribution: float = Field(
        0, ge=0, description="Employer's SSF contribution — does NOT reduce net pay, informational only"
    )
    tds_deducted: float = Field(
        0, ge=0, description="Tax deducted at source by the employer (Income Tax Act Section 87)"
    )
    other_deductions: dict[str, float] = Field(
        default_factory=dict, description="Any other named deductions, e.g. {'loan_repayment': 3000}"
    )

    net_pay: float = Field(..., ge=0, description="Take-home pay after all employee-side deductions")

    _arithmetic_warnings: list[str] = PrivateAttr(default_factory=list)

    @model_validator(mode="after")
    def check_arithmetic(self) -> "SalarySlip":
        """
        Cross-checks gross and net against their components. This is a
        business-rule check, not a type constraint — deliberately
        collected as warnings, not a raised ValidationError, because a
        genuinely malformed source slip (real-world OCR noise, rounding
        conventions) shouldn't make extraction fail outright. The caller
        decides what to do with self.arithmetic_warnings.
        """
        warnings: list[str] = []

        expected_gross = self.basic_salary + sum(self.allowances.values())
        if abs(expected_gross - self.gross_salary) > 1.0:
            warnings.append(
                f"gross_salary ({self.gross_salary}) does not match "
                f"basic_salary + allowances ({expected_gross})"
            )

        expected_net = (
            self.gross_salary
            - self.ssf_employee_contribution
            - self.tds_deducted
            - sum(self.other_deductions.values())
        )
        if abs(expected_net - self.net_pay) > 1.0:
            warnings.append(
                f"net_pay ({self.net_pay}) does not match gross minus deductions "
                f"({expected_net})"
            )

        self._arithmetic_warnings = warnings
        return self

    @property
    def arithmetic_warnings(self) -> list[str]:
        return self._arithmetic_warnings