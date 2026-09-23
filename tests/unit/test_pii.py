"""
Tests for PII detection/redaction. No Groq calls — entirely local,
free to run as often as you like.
"""

from __future__ import annotations

from src.safety.pii import detect_pii, redact_pii


def test_detects_name_email_phone_and_pan():
    text = "Employee: Ramesh Shrestha\nPAN: 123456789\nPhone: 9841234567\nEmail: ramesh@example.com"
    entities = {e["entity_type"] for e in detect_pii(text)}
    assert "PERSON" in entities
    assert "EMAIL_ADDRESS" in entities
    assert "PHONE_NUMBER" in entities
    assert "NEPAL_PAN" in entities


def test_redaction_removes_pii_from_output():
    text = "Employee: Ramesh Shrestha, PAN: 123456789"
    redacted = redact_pii(text)
    assert "Ramesh Shrestha" not in redacted
    assert "123456789" not in redacted


def test_no_false_positives_on_clean_statute_text():
    """
    The Act/Rules corpus should never trigger PII redaction — it's
    public statute text. Using a real sentence from Section 5 (seen
    during Phase 2) as the test input, not an invented one.
    """
    text = (
        "The taxable income of any person in any income year shall be "
        "equal to the amount computed by subtracting the amount, if any, "
        "claimed pursuant to Section 12 or 63 from the grand total amount "
        "of assessable income of each of the following income headings "
        "in that income year: Business, Employment, and Investment."
    )
    entities = detect_pii(text)
    assert entities == [], f"Expected no PII in clean statute text, got: {entities}"


def test_pan_pattern_does_not_flag_currency_amounts():
    """
    Known false-positive risk: the PAN pattern is a bare 9-digit regex.
    A large currency figure formatted without commas/decimals could
    coincidentally be 9 digits. Confirms typical salary figures (well
    under 9 digits even for a high earner) don't trigger it.
    """
    text = "Gross Salary: Rs. 105000\nNet Pay: Rs. 87250"
    entities = detect_pii(text)
    pan_hits = [e for e in entities if e["entity_type"] == "NEPAL_PAN"]
    assert pan_hits == [], f"Currency amounts incorrectly flagged as PAN: {pan_hits}"


if __name__ == "__main__":
    test_detects_name_email_phone_and_pan()
    print("test_detects_name_email_phone_and_pan: PASSED")
    test_redaction_removes_pii_from_output()
    print("test_redaction_removes_pii_from_output: PASSED")
    test_no_false_positives_on_clean_statute_text()
    print("test_no_false_positives_on_clean_statute_text: PASSED")
    test_pan_pattern_does_not_flag_currency_amounts()
    print("test_pan_pattern_does_not_flag_currency_amounts: PASSED")