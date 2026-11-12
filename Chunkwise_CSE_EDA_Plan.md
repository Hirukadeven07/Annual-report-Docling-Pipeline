# Chunkwise CSE: Exploratory Data Analysis Plan

This document is the plan for the EDA notebook. The EDA has two stages:

- **Part A: Screening.** Decides which reports can be used and how they should be chunked.
- **Part B: Exploratory analysis.** Informs the structure-aware chunker, the filtering rules and the section-specific prompts.

Image figures (charts and infographics) are out of scope. Only narrative text and tables are analysed.

---

## Setup

### Folder structure

```
reports/{sector}/{company}/{year}.pdf
```

### Libraries

- `docling` for layout-aware extraction (headings, text blocks, tables)
- `pymupdf` (fitz) for the text-layer check and embedded bookmarks
- `pandas`, `numpy` for analysis
- `matplotlib`, `seaborn`, `plotly` for plots
- `rapidfuzz` for fuzzy matching of section headings

### Master metadata table

Build one row per report. Every later result is joined back to this table.

| Column | Description |
|---|---|
| `report_id` | Unique ID, for example `JKH_2024` |
| `company` | Company name |
| `ticker` | CSE ticker |
| `gics_group` | GICS industry group |
| `year` | Financial year |
| `file_path` | Path to the PDF |
| `file_size_mb` | File size |
| `page_count` | Number of pages |

---

# Part A: Screening

Each report passes through five checks. The results feed a single screening table and a tier for each report.

## A1. Text-based PDF check

**Goal:** Confirm the PDF has a real text layer, because OCR is out of scope.

**Steps**
1. For each page, count extractable characters using PyMuPDF.
2. Flag a page as image-only if it has fewer than a threshold of characters (for example 50) and is covered by an image.
3. Compute the share of text pages per report.

**Classification**
- **Pass:** at least 95% of pages are text-based.
- **Partial:** some image-only pages, for example a scanned auditor's report or signature pages. Record the page ranges.
- **Fail:** mostly image-only. Exclude the report.

**Outputs**
- Columns: `text_page_ratio`, `image_only_pages`, `text_status`
- Plot: share of pass, partial and fail reports, by sector and year

## A2. Heading hierarchy check

**Goal:** Find the best structure source for each report. Docling does not read the PDF's table of contents. It infers headings with its layout model, and its headings are often flat. Check each source separately.

**Steps**
1. **Embedded bookmarks:** read `doc.get_toc()` with PyMuPDF. Record the number of entries and levels.
2. **Printed contents page:** detect a "Contents" page in the first few pages. Check whether its listed page numbers match the actual pages, since printed and PDF page numbers often differ by an offset.
3. **Docling headings:** count the `section_header` items Docling detects and check whether they agree with the bookmarks or the contents page.

**Classification (`structure_source`)**
- `bookmarks`: usable embedded hierarchy
- `contents_page`: printed contents page aligned to real pages
- `docling_only`: only Docling headings available
- `none`: no reliable structure, so fallback chunking rules apply

**Outputs**
- Columns: `has_bookmarks`, `bookmark_levels`, `has_contents_page`, `page_offset`, `docling_header_count`, `structure_source`
- Plot: distribution of `structure_source` by sector and year

## A3. Target section coverage

**Goal:** Check that the sections the pipeline will chunk can be found in each report.

**Synonym map**

| Target section | Common variants |
|---|---|
| Chairman's message | Chairman's Review, Chairperson's Statement, Chairman's Letter |
| Management discussion | Management Discussion and Analysis, CEO's Review, Managing Director's Review |
| Income statement | Statement of Profit or Loss, Income Statement, Statement of Comprehensive Income |
| Balance sheet | Statement of Financial Position, Balance Sheet |
| Cash flow statement | Statement of Cash Flows |
| Changes in equity | Statement of Changes in Equity |
| Notes | Notes to the Financial Statements |
| Auditor's report | Independent Auditor's Report |
| Risk | Risk Management, Enterprise Risk Management |
| Governance | Corporate Governance, Corporate Governance Report |
| Sustainability | Sustainability Report, Corporate Social Responsibility, ESG Report |

**Steps**
1. Collect headings from the structure source chosen in A2.
2. Fuzzy-match each heading against the synonym map, using a score threshold, for example 85.
3. Record the start and end page of each matched section.

**Outputs**
- Columns: one found/not found flag per target, page ranges, `section_coverage` (for example 9/11)
- Heatmap: coverage of each target section by sector
- List of unmatched headings, which you review to extend the synonym map

## A4. Table extractability

**Goal:** Confirm tables come out as structured, machine-readable data. The main trap is a financial statement pasted as an image inside an otherwise text-based PDF.

**Steps** (for each table Docling returns)
1. Record the number of rows and columns, and the share of empty cells.
2. Compute the **numeric parse rate**: the share of value cells that parse as numbers. Handle accounting formats:
   - `(1,234)` for negatives
   - `-` or `–` for nil
   - unit headers such as `Rs. '000` or `LKR Mn`
3. Check that rows have a consistent column count, to catch merged or split columns.
4. Flag target financial-statement sections from A3 that contain **zero tables**, since this suggests the statement is an image.

**Outputs**
- Columns: `table_count`, `mean_numeric_parse_rate`, `empty_cell_ratio`, `image_statement_flag`
- Plot: numeric parse rate distribution by sector

## A5. Extraction correctness

**Goal:** Measure whether the extracted content is actually right.

### Automatic accounting checks

| Check | Rule |
|---|---|
| Balance sheet identity | Total assets = total equity + total liabilities |
| Subtotals | Line items sum to their stated subtotals |
| Cross-statement | Profit for the year matches across the income statement and the statement of changes in equity |
| Highlights | Current-year figures match the financial highlights page |

Allow a small tolerance for rounding.

### Manual ground truth

1. Select a stratified sample of 2 to 3 reports per sector, covering different years.
2. From each PDF, record by hand about 20 key figures (revenue, profit, total assets, EPS, dividend per share and similar) and the start page of each target section.
3. Compare them with the extracted values.

**Metrics**
- Section detection recall
- Table cell accuracy
- Numeric match rate

**Outputs**
- Columns: `accounting_checks_passed`, `ground_truth_match_rate` (sample only)
- Table: error types found, such as wrong column alignment, lost negatives or unit mismatch

## Screening table and tiers

Combine A1 to A5 into one table, then assign each report a tier:

| Tier | Criteria | Pipeline treatment |
|---|---|---|
| A | Text-based, reliable structure, good table extraction | Main structure-aware pipeline |
| B | Text-based, but weak structure or partial table issues | Fallback chunking rules |
| C | Scanned, or key statements are images | Excluded and counted |

**Outputs**
- Tier counts by sector and year
- The final dataset size, stated in the report with the exclusion reasons

---

# Part B: Exploratory Analysis

## B1. Corpus overview

**Steps**
1. Count reports by sector, year and company.
2. Build a company-by-year grid to find missing reports.
3. Plot distributions of page count and file size.

**Outputs**
- Bar charts: reports per sector and per year
- Heatmap: company-by-year availability
- Box plots: page count by sector

## B2. Report and section size

Measure size in pages and words.

**Steps**
1. Compute word counts per report.
2. Compute word and page counts per target section.
3. Compute the share of each report taken up by target sections, compared with low-value content such as photo spreads, corporate profiles and AGM notices.

**Outputs**
- Box plots: report size and section size by sector
- Stacked bar: target vs non-target content per report

## B3. Table characteristics

**Steps**
1. Measure table sizes in rows and columns.
2. Detect tables that continue across pages, for example matching headers on consecutive pages, so they can be stitched back together.
3. Compute table density per section and per sector.

**Outputs**
- Histogram: table sizes
- Count and share of multi-page tables
- Heatmap: tables per section by sector

## B4. Fixed-size chunking damage

**Goal:** Show with your own data why structure-aware chunking is needed.

**Steps**
1. Simulate naive fixed-size chunking at a few chunk sizes.
2. Count how many tables and paragraphs are cut across a chunk boundary.
3. Count how many table rows end up separated from their column headers.

**Outputs**
- Bar chart: share of tables and paragraphs split, by chunk size
- One or two concrete examples of a split table, for the report

## B5. Boilerplate and repetition

**Steps**
1. **Within a company across years:** measure text similarity between the same section in consecutive years, since governance and risk text is often reused.
2. **Across companies:** find standard passages repeated across many reports, for example accounting policy notes.
3. Measure the share of numbers vs narrative text in each section.

**Outputs**
- Heatmap: year-to-year similarity per section
- List of high-frequency boilerplate passages, used as candidate filter rules
- Bar chart: numeric density per section

## B6. Vocabulary and terminology

**Steps**
1. Find the most frequent financial terms and phrases per sector, using TF-IDF to surface sector-specific terms.
2. Compare sectors, for example banks (NPL ratio, capital adequacy, deposits) against manufacturing (inventory, cost of sales, exports).

**Outputs**
- Top terms per sector
- Word clouds or bar charts
- A term list that feeds the section-specific prompts and the coverage checklist

## B7. Consistency over time

**Steps**
1. For each company, compare structure across years: section order, heading names, bookmark presence.
2. Check whether older reports have weaker structure or more extraction issues.

**Outputs**
- Line chart: share of Tier A reports by year
- Table: companies whose layout changed significantly between years

---

# Summary: how the EDA informs the pipeline

| EDA finding | Pipeline decision |
|---|---|
| Text-based screening | Which reports enter the dataset |
| Structure source | Which heading source the chunker uses, and when fallback rules apply |
| Section coverage and synonym map | Section detection rules |
| Table extractability | Code-based figure extraction, and which reports are excluded |
| Extraction correctness | Docling configuration and number parsing rules |
| Multi-page tables | Table stitching before chunking |
| Fixed-size chunking damage | Evidence for the structure-aware approach |
| Boilerplate | Low-value chunk filtering |
| Vocabulary | Section-specific prompts and the coverage checklist |
| Consistency over time | Fallback rules for older layouts |

# Suggested notebook structure

1. Setup and metadata table
2. A1 Text-based check
3. A2 Heading hierarchy check
4. A3 Target section coverage
5. A4 Table extractability
6. A5 Extraction correctness
7. Screening table and tiers
8. B1 Corpus overview
9. B2 Report and section size
10. B3 Table characteristics
11. B4 Fixed-size chunking damage
12. B5 Boilerplate and repetition
13. B6 Vocabulary and terminology
14. B7 Consistency over time
15. Summary of findings
