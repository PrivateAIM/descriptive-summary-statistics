# Data Summary Report

Federated exploratory data analysis and descriptive summary statistics for
privacy-preserving medical research, built on
[FLAME](https://github.com/PrivateAIM/python-sdk-patterns).

Each participating node (hospital) computes descriptive, comparative, and
inferential statistics on its own data — numeric, categorical, and temporal
summaries, dimensionality reduction (PCA/MCA), and association screening. A
central aggregator combines the per-node results into federated (cross-node)
statistics, without any node sharing raw patient data. Every run produces
PDF reports (per-node and federated) plus machine-readable JSON summaries.

## How it works

The project has two faces:

- **A local implementation** (`data_report/`, `generate_reports/`) — a
  regular installable Python package, used to develop and test the analysis
  pipeline entirely on one machine.
- **A Hub entry point** (`hub_entrypoint_10.py`) — a single self-contained
  file that packages the same logic for deployment on the real FLAME Hub
  platform.

## Local implementation

The local implementation runs via the FLAME SDK's own sandbox tooling
(`StarModelTester`), which simulates the full federated execution model —
every node's analyzer, then the aggregator — locally, without needing a real
distributed network.

**Install:**

```bash
pip install -e .
```

Requires Python >= 3.10.

**Run:**

```bash
dr-analyze
```

This runs the full pipeline against the dataset configured in
`data_report/cli.py::analyze_main`, and writes all outputs to `results/`:

- `results/local_results/<node>/` — per-node CSVs, figures, PDF reports
  (`local_report_<node>_short.pdf` / `_full.pdf`), and a `summary.json`.
- `results/federated_results/` — the same, pooled across all nodes
  (`global_report_short.pdf` / `_full.pdf`).

## Genomic and imaging analysis

Genomic (FASTQ/BAM/VCF/FASTA) and imaging (DICOM/NIfTI/raster) data analysis
are also implemented, as standalone modules separate from the federated
pipeline above — they are not wired into `dr-analyze`. Each expects its own
data to already be present locally (`genomic data/`, `imaging data/`):

```bash
python run_genomic_analysis.py
python run_imaging_analysis.py
```

Each writes a CSV summary plus a full-detail JSON backup to
`results/genomic_analysis/` / `results/imaging_analysis/`.

## Hub entry point

`hub_entrypoint_10.py` is the file used as the Hub entry point when
deploying to the FLAME Hub platform. When selecting a master image for the
deployment, use the custom-made **reportstats** image.

## Decoding results

After a run on the Hub platform, results come back as a base64-encoded
`.tar.gz` archive rather than files written directly to disk. `decode_results.py`
decodes and extracts it:

```bash
python decode_results.py results.tar.gz.b64 --output my_folder
```

## Project layout

- `data_report/` — the installable package: `analyze.py`
  (`DataReportAnalyzer` / `DataReportAggregator`), `cli.py` (`dr-analyze`
  entry point), `statistical_analysis/`, `generate_figures/`, `get_data/`.
- `generate_reports/` — PDF report builders (`generate_local_report.py`,
  `generate_global_report.py`), the JSON summary builder
  (`generate_json_summary.py`), and shared `report_utils.py`.
- `data/` — input datasets, one subdirectory per dataset, each with one
  subdirectory per node.
- `hub_entrypoint_10.py` — the Hub deployment entry point (see above).
- `decode_results.py` — decodes a Hub run's base64-encoded output archive.

## Testing

```bash
pytest
```

## Further reading

- [`LOCAL_PIPELINE_README.md`](LOCAL_PIPELINE_README.md) — a detailed,
  module-by-module reference for how the local implementation turns a raw
  CSV into the tables, plots, and PDF reports.
