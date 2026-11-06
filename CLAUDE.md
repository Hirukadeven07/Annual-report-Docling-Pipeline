# CLAUDE.md

Guidance for Claude Code when working in this repository. Read this file fully before making changes.

## 1. Project

**Chunkwise CSE** is a DSGP (Data Science Group Project) system that summarizes Colombo Stock Exchange (CSE) annual reports into concise bullet points using **small language models (SLMs)**. The project's main contribution is a **custom structure-aware chunking approach**, because SLMs struggle with long documents and small context windows. Fine-tuning an SLM is a later, optional add-on.

This repository contains the **benchmarking pipeline**: it measures how SLMs behave on CSE annual reports, first with **no chunking** (the baseline), and later with chunking strategies, so that the improvement from structure-aware chunking can be measured fairly.

The owner of this repository is responsible for the SLM component. Other group members own other components (for example, chunking). Keep the code modular so components can be swapped and compared.

### Goals

1. Benchmark several SLMs on CSE annual-report tasks and shortlist 2 to 3 models.
2. Establish a **baseline**: whole report in one prompt, truncated to the context budget when it does not fit.
3. Provide the same pipeline, reports, gold data, prompts and scorers for later experiments, so a chunking strategy can be compared with the baseline by changing one config value.
4. Produce results usable as EDA and as evidence in the project report (accuracy, hallucination, coverage, speed, memory, behaviour vs report length).

### Non-goals (for now)

- No user interface, web app or database in this repository.
- No fine-tuning code.
- No decision yet on the final chunking approach or on cloud storage. Do not hard-code assumptions about either (see section 12).

## 2. Constraints and conventions

- Language: Python 3.11+.
- Code style: clean code with **docstrings on every function, method and class**, and **no inline `#` comments**. Names and structure should explain intent; the docstring says what the function does, not how.
- Docstring format (Google style): a one-line summary in the imperative mood, then `Args:`, `Returns:` and `Raises:` sections only when they add information. Keep them short. Add a one-line module docstring at the top of each file. Include type hints on all function signatures.
- Deliver **complete, working files**, not partial snippets. When editing, keep the file runnable.
- Prefer small, focused modules and functions. Iterate in small steps; keep each step runnable.
- Standard library plus well-known packages only: `docling`, `ollama`, `pandas`, `numpy`, `matplotlib`, `psutil`, `tiktoken`, `rouge-score`, `bert-score` (optional), `pyyaml`, `typer` (or `argparse`), `pytest`.
- Every stage must be **resumable and idempotent**: re-running skips finished work unless forced.
- Runs are **reproducible**: temperature 0, fixed seed, and a frozen copy of the config saved with every run.
- Never modify gold files or parsed reports programmatically. They are inputs.
- Local models only for inference in this repository (Ollama). Keep the backend behind an interface so others can be added.

## 3. Project structure

```
slm-bench/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── baseline.yaml
│   └── chunked.yaml
├── notebooks/
│   └── chunkwise_docling_gpu.ipynb   Docling extraction on a Colab GPU
├── data/
│   ├── reports/                 input PDFs, named <company>_<year>.pdf, in any subfolders
│   ├── extracted/               Colab output (JSON, markdown, pictures), same subfolders as reports/
│   ├── parsed/                  Docling markdown, <company>_<year>.md
│   └── gold/
│       ├── TEMPLATE.json
│       └── <company>_<year>.json
├── runs/
│   └── <run_id>/
│       ├── config.yaml          frozen copy of the config used
│       ├── prepared/            model-ready inputs (units) per model and report
│       ├── raw_outputs.csv      one row per model x report x prompt x run
│       ├── scored.csv           raw rows plus metrics
│       ├── fact_level.csv       one row per gold fact per facts run
│       ├── summary/             per-model tables, ranking, analysis tables
│       └── plots/
├── src/
│   └── slm_bench/
│       ├── __init__.py
│       ├── cli.py               entry point
│       ├── pipeline.py          stage orchestration
│       ├── config.py            YAML loading and validation
│       ├── schemas.py           dataclasses for Unit, RawOutput, GoldFile
│       ├── stages/
│       │   ├── parse.py         local Docling conversion and report discovery
│       │   ├── colab_import.py  imports the notebook's output
│       │   ├── prepare.py
│       │   ├── infer.py
│       │   ├── score.py
│       │   └── report.py
│       ├── strategies/
│       │   ├── base.py          Strategy interface
│       │   ├── whole_report.py  baseline
│       │   ├── fixed_chunks.py  later: naive fixed-size chunking
│       │   └── structure_aware.py  later: custom chunking (not decided)
│       ├── backends/
│       │   ├── base.py          Backend interface
│       │   ├── ollama.py
│       │   └── fake.py          deterministic backend for dry runs and tests
│       ├── metrics/
│       │   ├── numbers.py       number extraction and normalization
│       │   ├── facts.py         fact accuracy, JSON validity
│       │   ├── grounding.py     hallucinated-number rate
│       │   ├── coverage.py      fact and risk coverage
│       │   └── text_sim.py      ROUGE-L, BERTScore
│       └── prompts.py           prompt registry
└── tests/
    ├── test_numbers.py
    ├── test_facts.py
    ├── test_grounding.py
    ├── test_strategies.py
    ├── test_pipeline_dry_run.py
    └── fixtures/
```

`data/reports/*.pdf`, `data/extracted/`, `data/parsed/*.md` and `runs/` may be large and are git-ignored except for `runs/<run_id>/config.yaml` and `summary/` for runs worth keeping.

## 4. Pipeline

The pipeline is a fixed sequence of stages. Each stage reads artifacts from the previous one, writes its own artifacts, and skips work already done.

```
PDFs ──parse──▶ markdown ──prepare──▶ units ──infer──▶ raw outputs ──score──▶ scored ──report──▶ tables + plots
        │                    │                   │                     │                  │
     Docling          strategy plug-in      backend plug-in        metric modules       ranking, analysis
```

| Stage | Reads | Writes | Notes |
|---|---|---|---|
| `parse` | Colab output in `parse.import_dir`, or `data/reports/*.pdf` | `data/extracted/`, `data/parsed/*.md`, `runs/<id>/extraction_log.csv` | Docling only. With `engine: colab` it imports what the notebook produced; with `engine: docling` it converts locally, about 1 hour per report on a CPU. Flags scans (very little text per page). |
| `prepare` | `data/parsed/*.md`, config (strategy, models) | `runs/<id>/prepared/` | Applies the configured strategy per model's context budget. Records token counts and truncation. |
| `infer` | prepared units, prompts, models | `runs/<id>/raw_outputs.csv` | Calls the backend, measures latency, time to first token, tokens/sec, peak memory. Appends row by row. |
| `score` | `raw_outputs.csv`, `data/gold`, parsed text | `scored.csv`, `fact_level.csv` | Computes all metrics. Missing gold files skip gold-based metrics with a warning. |
| `report` | `scored.csv`, `fact_level.csv` | `summary/*.csv`, `plots/*.png` | Per-model table, ranking, analysis tables, plots. |

### Resumability rules

- `parse`: skip a report if its markdown exists, unless `--force`. Reports with no finished Colab output are logged as `missing`, and reruns pick them up.
- `prepare`: skip a (model, report) pair if its prepared file exists and the strategy params are unchanged.
- `infer`: the key is `(model, report, prompt, run)`. Skip keys already present **without** an error. Rows with an error are retried and appended; `score` keeps the **last** row per key.
- `score` and `report` are cheap and always recompute from their inputs.

### CLI

```
slm-bench run --config configs/baseline.yaml
slm-bench run --config configs/baseline.yaml --stage score
slm-bench run --config configs/baseline.yaml --stage infer --models llama3.2:3b --reports acme_2024 --runs 1
slm-bench run --config configs/baseline.yaml --dry-run
slm-bench tokens --config configs/baseline.yaml
slm-bench init-gold --config configs/baseline.yaml
slm-bench compare runs/baseline_colab runs/chunked
```

`--dry-run` swaps the backend for `fake` and writes to a separate run folder so real results are never polluted.

`compare` aligns two runs on (model, report, prompt) and reports metric deltas, with the baseline as reference.

## 5. Strategy interface (the plug-in point)

The baseline and the future chunking approaches differ only in how a report becomes model input. Design for **multiple units per report from the start**.

```
Strategy.prepare(report_text, model_ctx, params) -> list[Unit]

Unit:
    report        str
    unit_id       str          "whole" for the baseline, "c001", "c002", ... for chunks
    text          str
    section       str | None   section label, when the strategy knows it
    pages         tuple | None
    tokens        int
    meta          dict
```

- `whole_report`: returns one unit. Counts tokens with `tiktoken` `cl100k_base`, and if the report exceeds the budget `int((ctx - prompt_reserve) * token_safety)` it truncates to the budget and records `truncated=true`, `report_tokens`, `tokens_fed`, `chars_fed`.
- `fixed_chunks` (later): naive fixed-size windows with overlap.
- `structure_aware` (later): section-aware chunking that never splits tables or paragraphs. **Not decided.** Do not implement or assume details until the group decides.

When a strategy returns several units, `infer` runs each unit and a later **assemble** step combines unit outputs into one report-level output before scoring. How units are merged is an open decision (section 12). Until then, the baseline path (one unit per report) must work end to end, and the code must not assume exactly one unit.

## 6. Backend interface

```
Backend.generate(model, prompt_text, options) -> Generation

Generation:
    text              str
    prompt_tokens     int | None
    output_tokens     int
    ttft_s            float | None
    latency_s         float
    tok_s             float | None
    peak_mem_mb       float
    error             str
```

- `ollama`: streaming chat, `options` = `num_ctx`, `temperature`, `seed`, `num_predict`. Does a warm-up call per model before timing, and unloads the model afterwards. Peak memory is the summed resident size of all processes whose name contains `ollama`, sampled every 0.25 s in a thread.
- `fake`: deterministic output built from the gold file, with one deliberately wrong fact and one invented number, so scoring and reporting can be tested without a model.

## 7. Configuration

`configs/baseline.yaml`:

```yaml
run_id: baseline_colab
data:
  reports_dir: data/reports
  parsed_dir: data/parsed
  gold_dir: data/gold
  extracted_dir: data/extracted
parse:
  engine: colab
  import_dir: G:/My Drive/chunkwise-cse/extracted_egret_large
  ocr: false
  table_mode: accurate
  exclude_dirs: [Interim]
prepare:
  strategy: whole_report
  params:
    prompt_reserve: 1536
    token_safety: 0.85
models:
  - name: llama3.2:3b
    num_ctx: 32768
  - name: qwen2.5:3b
    num_ctx: 32768
  - name: phi4
    num_ctx: 16384
  - name: gemma2:2b
    num_ctx: 8192
inference:
  backend: ollama
  runs: 3
  seed: 42
  temperature: 0.0
  max_new_tokens: 1024
  keep_alive: 10m
prompts: [summary, facts, risks]
scoring:
  bertscore: false
ranking_weights:
  accuracy: 0.40
  hallucination: 0.25
  json: 0.15
  efficiency: 0.20
analysis:
  length_bins: [0, 8000, 16000, 32000, 64000]
```

The config is validated on load. Unknown keys are errors. A frozen copy is written to `runs/<run_id>/config.yaml`; if a different config is already frozen there, the run stops. Changing the strategy or its params means a new `run_id`.

`parse.engine` is `colab` or `docling`. `import_dir` is required for `colab`, and points at the notebook's output folder as Google Drive for Desktop shows it on this machine. Under `colab`, `ocr` and `table_mode` are not applied, because the notebook holds its own Docling settings.

`parse` searches `reports_dir` recursively. Each report is named by its lowercased filename stem (`johnkeells_2024`), so stems must be unique across subfolders. Folders listed in `parse.exclude_dirs` (case-insensitive, optional) are skipped, which keeps interim reports out of the annual-report benchmark.

## 8. Prompts

Three fixed prompts, identical for every model. Stored in `prompts.py`. Any change creates a new prompt version, recorded in the config and the run folder, because results are only comparable under identical prompts.

Input layout: the report text first, then the task last:

```
Annual report:

<document text>

---

Task: <instruction>
```

Putting the task last keeps the instruction in context if a runtime trims the start of the prompt.

| Prompt | Instruction (summary) | Output |
|---|---|---|
| `summary` | Summarize into 5 to 8 concise bullets covering financial performance, dividends, key risks and outlook. Quote figures exactly as printed. | Bullets |
| `facts` | Return a JSON object with exactly the keys below. Copy numbers exactly as printed, in the report's own units, with no unit conversion or currency symbols. `auditor_opinion` is one of unmodified, qualified, adverse, disclaimer. Use null when not found. JSON only. | JSON |
| `risks` | List the key risks and the outlook as concise bullets. | Bullets |

Fact keys: `revenue`, `net_profit`, `eps`, `dividend_per_share`, `total_assets`, `total_equity`, `auditor_opinion`.

## 9. Data formats

### Gold file `data/gold/<company>_<year>.json`

```json
{
  "company": "Example PLC",
  "year": 2024,
  "facts": {
    "revenue": "12,430",
    "net_profit": "1,820",
    "eps": "4.15",
    "dividend_per_share": "2.50",
    "total_assets": "58,900",
    "total_equity": "21,300",
    "auditor_opinion": "unmodified"
  },
  "table_keys": ["total_assets", "total_equity"],
  "risks": [["currency", "exchange rate"], "supplier concentration"],
  "reference_summary": ["...", "...", "...", "...", "..."]
}
```

- `facts`: values **exactly as printed** in the report, in the report's own units. `null` means the report does not state it and the fact is skipped.
- `table_keys`: facts whose value comes from a table rather than narrative text.
- `risks`: keywords to find in output. An inner list means any synonym counts.
- `reference_summary`: five bullets used for ROUGE-L and BERTScore.
- The gold set is small and must be verified by hand. Wrong gold data invalidates every comparison.

### `raw_outputs.csv`

`model, report, unit_id, prompt, run, ctx, report_tokens, tokens_fed, truncated, chars_fed, prompt_eval_count, eval_count, latency_s, ttft_s, tok_s, peak_mem_mb, error, output`

### `scored.csv`

All raw columns except `output`, plus: `failed, looping, halluc_rate, n_numbers` for every row; `json_strict, json_lenient, fact_acc, answered_rate` for `facts`; `coverage_facts, coverage_risks, coverage, rouge_l` (and `bertscore_f1` when enabled) for `summary`; `coverage_risks` for `risks`.

### `fact_level.csv`

`model, report, run, key, answered, correct, is_table, found_in_text, rel_pos, in_window, report_tokens, tokens_fed, truncated`

`rel_pos` is the relative offset of the first occurrence of the gold value in the markdown. `in_window` says whether that occurrence lies inside the text the model actually saw. Both are approximate because the first occurrence may not be the relevant one.

## 10. Metrics (exact definitions)

Number handling (`metrics/numbers.py`):
- Extract numbers with a regex for digits with optional thousands commas and decimals. Strip commas. Compare **absolute values**; sign and parentheses are ignored.
- Compare equal with `math.isclose` (tight tolerance). **No unit conversion**: "12.4 bn" does not match "12,430". This is deliberate.

Metrics:
- **fact_acc**: correct facts divided by gold facts present (non-null, in the fact keys). A missing or null answer counts as wrong. A failed run counts as all wrong.
- **auditor_opinion** is compared by canonical class: unmodified (also "unqualified", "clean"), qualified, adverse, disclaimer.
- **json_strict**: the whole output parses as a JSON object. **json_lenient**: the first `{...}` block parses.
- **halluc_rate**: share of numbers in the output (single-digit integers ignored) that do not appear in the **text the model actually saw** (`text[:chars_fed]`). Numbers the model computes itself count as ungrounded. `None` when the output has no countable numbers.
- **coverage_facts**: share of gold facts present in a summary (numbers by value, opinion by keyword). **coverage_risks**: share of gold risk keywords found. **coverage**: mean of the two available parts.
- **rouge_l**: ROUGE-L F-measure against the joined reference summary. Secondary metric.
- **bertscore_f1**: optional, `distilbert-base-uncased`. Secondary metric.
- **failed**: error text present or empty output. **looping**: at least 10 non-empty lines with under 50% unique.
- **Speed and memory**: `tok_s` (from eval counts and durations), `ttft_s`, `latency_s`, `peak_mem_mb`.

Report-stage analyses:
1. Per-model table: mean of each metric, plus run-to-run standard deviation of `fact_acc` averaged over (model, report, prompt) groups (a determinism check).
2. Ranking: `100 * (w_acc * acc + w_hall * (1 - halluc) + w_json * json_strict + w_eff * eff)` where `eff = 0.5 * scale(tok_s) + 0.5 * scale(-peak_mem)` with min-max scaling across models. Weights come from the config.
3. Fact accuracy vs report length (token bins).
4. Facts inside vs outside the fed window (truncation effect).
5. Table vs narrative facts, restricted to in-window facts.
6. Position effect (start, middle, end of the report), restricted to in-window facts.
7. If `manual_ratings.csv` exists (`model, report, prompt, faithfulness, coverage`, scores 1 to 5, model names hidden when rating), add per-model manual means.
8. Plots: fact accuracy by model, hallucination by model, accuracy vs length, speed vs accuracy.

## 11. Stage specifications and acceptance checks

### parse
- `notebooks/chunkwise_docling_gpu.ipynb` converts the PDFs on a Colab GPU (table structure on, accurate mode, OCR off) and writes to Google Drive. Google Drive for Desktop makes that output visible on this machine.
- `engine: colab` copies each finished report's JSON, markdown, `_pictures.json` and images to `data/extracted/`, in the same subfolder its PDF has under `data/reports`, and copies the markdown to `data/parsed/<report>.md`. Files are matched by lowercased stem. A report without all of its files counts as `missing`.
- `engine: docling` converts locally instead, with the settings in `parse`.
- Logs pages, characters, seconds, status per report. Marks "low text, possibly a scan" when characters per page are under about 200.
- Continues after a per-file failure and records it.
- **Done when**: each text-based PDF has a markdown file in which financial tables appear as markdown tables (check two or three by eye).

### prepare
- Loads the strategy by name, applies it per model using that model's `num_ctx`, writes units and token stats.
- Tokenization uses `cl100k_base` as an approximation. The budget includes `prompt_reserve` and `token_safety`.
- `slm-bench tokens` prints each report's token count and which models can take it whole.
- **Done when**: baseline units are a prefix of the report text, with correct `truncated`, `tokens_fed` and `chars_fed`.

### infer
- Warm-up call per model (loads the model so load time is not in the first measurement); if the model is not available, print the `ollama pull` hint and skip it.
- Streaming call; first non-empty chunk gives `ttft_s`.
- Appends and flushes each row immediately. Wraps each call so one failure does not stop the run.
- Unloads each model when finished.
- **Done when**: a one-model, one-report, one-run smoke test produces a complete row with output, timings and memory, and the check `prompt_eval_count < ctx` holds (nothing silently trimmed).

### score
- Deduplicates on `(model, report, prompt, run)` keeping the last row.
- Uses `fed_text = parsed_text[:chars_fed]` as the grounding source.
- Skips gold-based metrics with a warning when there is no gold file.
- **Done when**: the fake backend run produces the expected scores (see tests).

### report
- Writes the tables and plots listed in section 10 into `summary/` and `plots/`.
- Tolerates missing columns or empty groups without crashing.

## 12. Open decisions (do not assume)

- **Custom chunking approach**: not decided. Another group member covers chunking. Keep the `Strategy` interface stable so it can be added later.
- **How per-chunk outputs are merged** into a report-level output (for example per-chunk fact extraction to JSON, then a merge prompt per output mode). The intended direction is: domain knowledge lives in the pipeline (chunking plus section-specific prompts), and figures are pulled from tables by code rather than by the SLM. Not specified yet.
- **Document extraction tooling**: Docling is the current choice for parsing. Final tool choice for the production pipeline is still open.
- **Storage**: whether documents and chunks live on AWS, and whether a vector or graph database is needed, is undecided. This benchmark repository uses local files only.
- **Model shortlist**: decided by the baseline results.
- Fine-tuning is optional and later. Do not add training code.

## 13. Testing

- Unit tests for number extraction, value matching, JSON parsing, opinion canonicalization, hallucination rate, coverage and ROUGE wrappers.
- Strategy tests: `whole_report` returns one unit, truncates exactly at the budget, and the unit text is a prefix of the source.
- End-to-end dry run with the `fake` backend on two tiny synthetic reports (one longer than a small context so truncation is exercised, one with a null gold fact and one without a gold file). Assert the expected metric values, not just that the code runs.
- Failure paths: a row with an error, an empty output, a looping output, malformed JSON, and a duplicate key after a retry.
- `tiktoken` downloads its vocabulary on first use. Tests that need it should be skipped or use a stub when offline.

## 14. Common pitfalls

- Ollama silently uses a small default `num_ctx`. Always pass it explicitly and verify with `prompt_eval_count`.
- Token counts from `cl100k_base` are approximate for Llama, Qwen, Phi and Gemma tokenizers. Keep the safety factor.
- Docling output differs between layouts. Markdown tables use more tokens than plain text, so always count tokens on the parsed markdown.
- Markdown from local Docling and from the notebook use different settings. Do not mix them in one run: import with `--force` to replace all of it.
- Scanned PDFs produce almost no text. Replace them with text-based reports or enable OCR for that file only.
- Do not report only ROUGE. Accuracy, hallucinated numbers and coverage are the headline metrics.
- Never overwrite a finished run folder. Use a new `run_id`.
- Rerunning `infer` after fixing an error appends duplicate keys on purpose. `score` handles it.

## 15. Suggested build order

1. `config.py`, `schemas.py`, `cli.py` skeleton, and the `parse` stage. Inspect the Docling output.
2. `strategies/whole_report.py`, `prepare` stage, `tokens` command.
3. `backends/fake.py` and `infer` against it; then `backends/ollama.py` and a smoke test with one model.
4. Gold files for two or three reports (`init-gold` creates stubs); then `metrics/` and the `score` stage with tests.
5. `report` stage: tables, ranking, plots.
6. Full baseline run on all models and reports. Freeze the run folder.
7. Add `compare` and the `fixed_chunks` strategy. Add `structure_aware` once the group decides on it.
