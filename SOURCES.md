# Corpus sources (verified, live as of Sept 2026)

Download these manually into `data/raw_pdfs/`

## Primary statute
- **Income Tax Act 2058 (unofficial English translation)**
  https://www.ird.gov.np/public/pdf/379988550.pdf
  Save as: `income_tax_act_2058.pdf`

## TDS / withholding
- **TDS & TCS rate schedule** (cross-check the year against IRD's current
  circular list at https://ird.gov.np — the one below is FY 2077/78, verify
  a newer one exists before treating it as current-year ground truth)
  https://www.ird.gov.np/public/pdf/1747549774.pdf
  Save as: `tds_rates_2077_78.pdf`

## Directives
- **Income Tax Directive 2066 (with 3rd amendment 2077)** — linked from
  https://ird.gov.np/public/pdf/1657274501.pdf (this PDF itself is a notice
  pointing to the directive; get the actual directive from the URL it cites:
  ird.gov.np/Content/Tax,LawsRules/Directives/IncomeTax/...)

## Social Security Fund
- SSF is on a **separate portal** (ssf.gov.np, not ird.gov.np). Governing
  instruments:
  - Social Security Act, 2018 (2075)
  - Social Security Fund Regulations, 2075
  - Social Security Scheme Operation Directive, 2075
  - SSF registration guideline: https://ssf.gov.np/uploads/content/1542636187.pdf
  Search ssf.gov.np's own document section for current PDFs — I could not
  verify a stable direct link to the Act itself in this session.

## What this means for scope, honestly
Two of these sources are Nepali-script only, and I have not personally
verified a current-year (2082/83) TDS schedule PDF — only a 2077/78 one and
secondhand summaries citing 2082/83 rates. Before you build the golden set:

1. Confirm you're comfortable with a corpus that's partly Nepali script
   (your chunker will need language-aware handling — flag if it doesn't
   already), OR restrict v1 to English-translated sources only and note
   that as a documented scope limitation, not a bug.
2. Get the current-year TDS/SSF rate sheet yourself directly from
   ird.gov.np and ssf.gov.np — don't trust the 2077/78 PDF as current, and
   don't trust the summarized rate tables above as primary source; they're
   third-party blog restatements, not the government PDF itself.

This is exactly the kind of "don't hallucinate a source" discipline the
Phase 5 grounding work is meant to catch — good if it shows up in your
corpus decisions.

## Excluded content
- income_tax_act_2058.pdf pages 190-626 (of 626 total) are excluded at
  load time. This range is a sequence of consolidated amendment
  Ordinances/Financial Acts (starts at page 190: "Financial Ordinance,
  2059"), not the Act itself — their changes are already reflected in
  the Sections above. Retaining them risked citing superseded historical
  amendment language as current law. See src/ingestion/loader.py for
  the exact cutoff logic.
