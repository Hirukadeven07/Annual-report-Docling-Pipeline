# Annual Report Docling Pipeline

Extraction pipeline that turns Colombo Stock Exchange annual reports from PDF into structured JSON and markdown with Docling, part of the Chunkwise CSE project. Its output is the input for chunking and for the small language model benchmark, which will move to its own repository.

## Extracting reports on a Colab GPU

Docling takes about an hour per report on a CPU and roughly ten minutes on a Colab GPU, so extraction runs in `notebooks/chunkwise_docling_gpu.ipynb` and the project imports the results through Google Drive.

You need the Google Colab extension for VS Code and Google Drive for Desktop, signed in to the same Google account as Colab.

1. Copy `data/reports` to `My Drive/chunkwise-cse/reports`, keeping the company sub-folders.
2. Open the notebook in VS Code and choose **Select Kernel → Colab → New Colab Server → GPU**.
3. Choose **Run All** and approve the Google Drive prompt. Interim reports are skipped. If the session disconnects, run all again, and finished reports are skipped.
4. Wait for Drive for Desktop to sync, and check that `parse.import_dir` in `configs/baseline.yaml` uses the right drive letter.
5. Import the results:

```bash
slm-bench run --config configs/baseline.yaml --stage parse --force
```

Each report's JSON, markdown and pictures land in `data/extracted/`, sorted like `data/reports`, and its markdown is copied to `data/parsed/`. `--force` replaces markdown from earlier local conversions, and it replaces only reports Colab has finished. Without it, reports that already have markdown are skipped. Reports Colab has not finished are logged as `missing` in `runs/baseline_colab/extraction_log.csv`.

To convert locally instead, set `parse.engine` to `docling`.
