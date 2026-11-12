# Annual Report Docling Pipeline

Extraction pipeline that turns Colombo Stock Exchange annual reports from PDF into structured JSON and markdown with Docling, part of the Chunkwise CSE project. Its output is the input for later chunking and summarization work.

## Setup

```bash
pip install -e ".[dev]"
```

Run the tests with `pytest`.

## Extracting reports on a Colab GPU

Docling takes about an hour per report on a CPU and roughly ten minutes on a Colab GPU, so extraction runs in `notebooks/chunkwise_docling_gpu.ipynb` and the project imports the results through Google Drive.

You need the Google Colab extension for VS Code and Google Drive for Desktop, signed in to the same Google account as Colab.

1. Keep the PDFs in `Desktop/Lectures/Year 2/DSGP/Annual Reports`, in their company sub-folders, and let Drive for Desktop finish syncing. Colab reads them from `Othercomputers/My laptop/Lectures/Year 2/DSGP/Annual Reports`.
2. Open the notebook in VS Code and choose **Select Kernel → Colab → New Colab Server → GPU**.
3. Choose **Run All** and approve the Google Drive prompt. Interim reports are skipped. If the session disconnects, run all again, and finished reports are skipped.
4. Wait for Drive for Desktop to sync, and check that `parse.import_dir` in `configs/extraction.yaml` uses the right drive letter.
5. Import the results:

```bash
docling-pipeline run --config configs/extraction.yaml --force
```

Each report's JSON, markdown and pictures land in `data/extracted/`, sorted like the `Annual Reports` folder, and its markdown is copied to `data/parsed/`. `--force` replaces markdown from earlier local conversions, and it replaces only reports Colab has finished. Without it, reports that already have markdown are skipped. Reports Colab has not finished are logged as `missing` in `runs/colab_extraction/extraction_log.csv`.

To convert locally instead, set `parse.engine` to `docling`.

## Exploratory data analysis

`notebooks/chunkwise_eda.ipynb` follows `Chunkwise_CSE_EDA_Plan.md`. Part A screens every report (text layer, structure source, section coverage, table extractability, accounting checks) and gives it tier A, B or C. Part B explores report and section size, tables, fixed-size chunking damage, boilerplate, sector vocabulary and changes over time.

```bash
pip install -e ".[dev,eda]"
```

Open the notebook locally (not on Colab, which cannot see the PDFs on this PC) and run all cells. The first run reads every PDF with PyMuPDF and caches its features in `data/eda/cache`; later runs only add new reports. Docling-based steps use the JSON in `data/extracted` or in `parse.import_dir`, and fall back to the PDF text layer when neither is there. Extra Docling folders can be listed in the `EDA_DOCLING_DIRS` environment variable.

Result tables go to `data/eda/`. Two files in `data/reference/` are edited by hand and kept in git:

- `company_metadata.csv`: ticker and GICS industry group per company. Missing companies are added with a sector guessed from the name (`sector_source = name_rule`); correct them in place.
- `ground_truth.csv`: a stratified sample of reports with blank key figures and section start pages. Fill them in from the PDFs and rerun step A5 to measure extraction accuracy.
