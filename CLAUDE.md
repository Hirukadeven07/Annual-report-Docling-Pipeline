# CLAUDE.md

Guidance for Claude Code when working in this repository. Read this file fully before making changes.

## 1. Project

**Annual Report Docling Pipeline** turns Colombo Stock Exchange (CSE) annual reports from PDF into structured JSON and markdown with Docling. It is one component of **Chunkwise CSE**, a Data Science Group Project. Its JSON output is the input for the group's chunking work, and later for summarization work in separate repositories.

Extraction errors carry into everything built on this output, so correct tables and figures matter more than speed.

### Goals

1. Extract every annual report into JSON, markdown and pictures, keeping text, headings, tables and page numbers.
2. Keep extraction practical: conversion runs on a Colab GPU, at roughly 10 minutes per report instead of about 1 hour on a local CPU.
3. Make it easy to check that a report was extracted correctly.

### Non-goals

- No chunking, language model inference or model evaluation in this repository.
- No user interface, web app or database.

## 2. Constraints and conventions

- Language: Python 3.11+.
- Code style: clean code with **docstrings on every function, method and class**, and **no inline `#` comments**. Names and structure should explain intent; the docstring says what the function does, not how.
- Docstring format (Google style): a one-line summary in the imperative mood, then `Args:`, `Returns:` and `Raises:` sections only when they add information. Keep them short. Add a one-line module docstring at the top of each file. Include type hints on all function signatures.
- Deliver **complete, working files**, not partial snippets. When editing, keep the file runnable.
- Prefer small, focused modules and functions. Iterate in small steps; keep each step runnable.
- Standard library plus well-known packages only: `docling`, `pyyaml`, `typer`, `pytest`. The Colab notebook also uses what Colab provides. The EDA (`docling_pipeline.eda` and its notebook) may also use the `eda` extra: `pandas`, `numpy`, `pymupdf`, `rapidfuzz`, `scikit-learn`, `matplotlib`, `seaborn`.
- Every stage must be **resumable and idempotent**: re-running skips finished work unless forced.
- Runs are **reproducible**: a frozen copy of the config is saved with every run.
- Never hand-edit extracted output. Fix the pipeline and extract again.

## 3. Project structure

```
annual-report-docling-pipeline/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── .gitignore
├── configs/
│   └── extraction.yaml
├── notebooks/
│   ├── chunkwise_docling_gpu.ipynb   Docling extraction on a Colab GPU
│   └── chunkwise_eda.ipynb           exploratory data analysis (Chunkwise_CSE_EDA_Plan.md)
├── data/
│   ├── extracted/               JSON, markdown and pictures, same subfolders as reports_dir
│   ├── parsed/                  flat copy of the markdown, <company>_<year>.md
│   ├── eda/                     EDA feature caches and result tables (git-ignored)
│   └── reference/               hand-maintained EDA inputs: company_metadata.csv, ground_truth.csv
├── runs/
│   └── <run_id>/
│       ├── config.yaml          frozen copy of the config used
│       └── extraction_log.csv
├── src/
│   └── docling_pipeline/
│       ├── __init__.py
│       ├── cli.py               entry point
│       ├── pipeline.py          picks the engine and runs it
│       ├── config.py            YAML loading and validation
│       ├── schemas.py           the extraction log record
│       ├── stages/
│       │   ├── parse.py         local Docling conversion and report discovery
│       │   └── colab_import.py  imports the notebook's output
│       └── eda/                 helpers for the EDA notebook: PDF and Docling features, screening, analysis
└── tests/
    ├── test_colab_import.py
    ├── test_config.py
    ├── test_eda.py
    └── test_parse.py
```

The input PDFs live outside the repo, in `data.reports_dir` (named `<company>_<year>.pdf`, in any subfolders). `data/extracted/`, `data/parsed/*.md` and `runs/` may be large and are git-ignored, except for `runs/<run_id>/config.yaml`. `data/reports/` is an old local copy of the PDFs and is git-ignored too. `data/eda/` holds rebuildable EDA caches and is git-ignored; `data/reference/` is edited by hand and tracked.

## 4. Pipeline

```
PDFs ──copy──▶ Google Drive ──notebook on Colab GPU──▶ Drive output ──docling-pipeline run──▶ data/extracted + data/parsed
```

1. The PDFs stay in `data.reports_dir`, the `Annual Reports` folder on this PC. Drive for Desktop syncs it, and the notebook reads it from `Othercomputers/My laptop/Lectures/Year 2/DSGP/Annual Reports`.
2. `notebooks/chunkwise_docling_gpu.ipynb` converts them on a Colab GPU and writes to `My Drive/chunkwise-cse/extracted_<layout model>`. It holds its own Docling settings and skips `Interim` folders.
3. `docling-pipeline run` copies finished reports from Drive into the project. With `engine: docling` it converts locally instead.

### Resumability rules

- The notebook skips reports that already have their JSON and `_pictures.json`, so after a disconnect run all again.
- The import skips a report if its markdown exists in `data/parsed`, unless `--force`.
- Reports with no finished Colab output are logged as `missing`, and reruns pick them up.

### CLI

```
docling-pipeline run --config configs/extraction.yaml
docling-pipeline run --config configs/extraction.yaml --force
docling-pipeline run --config configs/extraction.yaml --force --reports johnkeells_2024
```

## 5. Configuration

`configs/extraction.yaml`:

```yaml
run_id: colab_extraction
data:
  reports_dir: C:/Users/hiruk/Desktop/Lectures/Year 2/DSGP/Annual Reports
  parsed_dir: data/parsed
  extracted_dir: data/extracted
parse:
  engine: colab
  import_dir: G:/My Drive/chunkwise-cse/extracted_egret_large
  ocr: false
  table_mode: accurate
  exclude_dirs: [Interim]
```

The config is validated on load. Unknown keys are errors. A frozen copy is written to `runs/<run_id>/config.yaml`; if a different config is already frozen there, the run stops, so a changed config needs a new `run_id`.

`parse.engine` is `colab` or `docling`. `import_dir` is required for `colab`, and points at the notebook's output folder as Google Drive for Desktop shows it on this machine. Under `colab`, `ocr` and `table_mode` are not applied, because the notebook holds its own Docling settings.

`reports_dir` is searched recursively. Each report is named by its lowercased filename stem (`johnkeells_2024`), so stems must be unique across subfolders. Folders listed in `parse.exclude_dirs` (case-insensitive, optional) are skipped, which keeps interim reports out.

## 6. Output formats

For each report, `data/extracted/<same subfolder as the PDF>/` holds:

| File | What it holds |
|---|---|
| `<report>.json` | Docling JSON: text, headings, tables, pictures, page numbers |
| `<report>.md` | Markdown version, also copied to `data/parsed/<report>.md` |
| `<report>_pictures.json` | Every picture: page, type, title, image file, and values read from chart labels |
| `<report>_pictures/` | A PNG of every picture |

`runs/<run_id>/extraction_log.csv` has `report, source, pages, chars, seconds, status, low_text`. `status` is `ok`, `skipped`, `missing`, `partial_success`, `failed in colab: ...` or `failed: ...`.

## 7. Acceptance checks

- The notebook and import handle every text-based PDF, and a failure on one report does not stop the others.
- A report without all of its files counts as `missing`. Files are matched by lowercased stem.
- Logs mark "low text, possibly a scan" when characters per page are under about 200.
- **Done when**: each text-based PDF has markdown in which financial tables appear as markdown tables, and key figures match the PDF. Check by eye until an automated check exists.

## 8. Open decisions (do not assume)

- **Checking extraction quality**: the planned direction is an HTML view beside the original PDF pages, plus an automated comparison of key figures with hand-checked values. The EDA notebook runs automatic accounting checks and compares against `data/reference/ground_truth.csv` once it is filled in; the HTML view is not built yet.
- **Finish line for tuning**: the target for correct figures and tables is not set yet.
- **Document extraction tooling**: Docling is the current choice. Final tool choice for the production pipeline is still open.

## 9. Testing

- Tests use stub converters and temporary folders, so they need neither Docling models nor a GPU.
- Cover report discovery, the skip and force rules, failure paths and the extraction log.
- The notebook is not covered by automated tests. Check its changes on a real Colab GPU.

## 10. Common pitfalls

- Colab kernels cannot see files on this PC, even when the notebook is opened from VS Code. Drive carries files both ways.
- Markdown from local Docling and from the notebook use different settings. Do not mix them: import with `--force` to replace all of it.
- Docling output differs between layouts, so check more than one company's reports.
- Scanned PDFs produce almost no text. Replace them with text-based reports or enable OCR for that file only.
- `parse.import_dir` assumes Drive is on `G:`. Change it if Drive uses another letter.
