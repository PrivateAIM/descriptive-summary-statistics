# Local Pipeline — Code Review Reference

This document describes everything that happens on a **local node** during a federated
data-report run. It is intended as a reading guide: follow the sections in order to
understand how a raw CSV becomes the tables, plots, and PDF reports stored in
`results/local_results/nodeN/`.

---

## 1. Architecture Context

The pipeline is built on the **FLAME Star topology**:

```
Node 0 (DataReportAnalyzer)  ─┐
Node 1 (DataReportAnalyzer)  ─┼──► Aggregator (DataReportAggregator)
Node 2 (DataReportAnalyzer)  ─┘
```

- **`DataReportAnalyzer`** (`data_report/analyze.py`) runs independently on every node.
  It receives only the raw CSV bytes for that node, runs all local statistics and plots,
  and returns a JSON-serialisable result dictionary. No raw data leaves the node.
- **`DataReportAggregator`** (`data_report/analyze.py`) runs on the central coordinator.
  It receives the per-node result dictionaries, federates the statistics, writes all
  outputs, and generates PDFs. It never sees the raw patient rows.

Both classes inherit from `flame.star.StarAnalyzer` / `StarAggregator`. In the
single-iteration mode used here (`simple_analysis=True`), each analyzer runs once
and returns its result; the aggregator then runs once.

---

## 2. Entry Point

**`data_report/cli.py` → `analyze_main()`**

```
load_dataset(data/datasetN/)
      │
      ▼
StarModelTester(
    data_splits   = node CSVs as bytes,
    analyzer      = DataReportAnalyzer,
    aggregator    = DataReportAggregator,
    output_type   = "pickle",
)
      │
      ▼  (after all analyzers finish)
generate_global_report()    ← federated PDF (short + full)
generate_local_report()     ← per-node PDF  (short + full)
generate_global_json_summary()
generate_local_json_summary()
```

`load_dataset` (`data_report/get_data/load_data.py`) scans the dataset directory for
per-node subdirectories, reads each CSV file as raw bytes, and wraps it in the list
format `[{filename: bytes}]` that FLAME expects.

---

## 3. Local Analysis — Step by Step

Everything in this section happens inside
`DataReportAnalyzer.analysis_method(data, aggregator_results)`.

### 3.1 CSV Parsing

```python
df = pd.read_csv(BytesIO(file_bytes), sep=None, engine="python")
```

The separator is auto-detected. After loading:

1. Blank strings, `"NULL"`, `"null"`, `"NaN"` → `pd.NA`.
2. Column names are lower-cased and spaces replaced with underscores.
3. A **date-conversion loop** iterates over every non-numeric column and tries
   `pd.to_datetime(col, format="mixed")`. A column is converted to `datetime64`
   when more than 60 % of its non-null values parse successfully. This runs
   **before** identifier detection so that high-uniqueness date columns are not
   mistaken for ID columns.

### 3.2 Identifier Detection

`detect_id_column(series, column_name)` (`compute_statistics.py`) flags a column as
an identifier when it satisfies all of:

- Uniqueness ratio > 90 % of non-null values.
- Not a datetime column (those are already converted).
- Not a known non-ID name (e.g. `"diagnosis"`, `"status"`).
- Does not look purely numeric (numeric IDs are rejected unless the column name
  explicitly contains `"id"`, `"code"`, or `"number"`).

Identifier columns are excluded from every downstream analysis step (statistics,
PCA/MCA, inferential screening). The first identifier is renamed to
`patient_id` and used for temporal patient-level analysis.

### 3.3 Column-Type Detection

`detect_column_types(df)` (`compute_statistics.py`) classifies every non-ID column
into one of four buckets:

| Bucket | Rule |
|--------|------|
| `temporal` | `datetime64` dtype after step 3.1 |
| `binary` | ≤ 2 unique non-null values AND one of: numeric `{0,1}`, boolean dtype, semantic pair (`yes/no`, `true/false`, `t/f`, `ja/nein`), or a numeric column containing only `{0,1}` with a medical keyword in its name |
| `numeric` | `select_dtypes("number")`, minus columns classified as binary |
| `categorical` | everything else, plus binary columns (binary is a subset of categorical for MCA/association purposes) |

Columns where every value is null are excluded from all four buckets.

`detect_quasi_numeric_categorical_columns(df, categorical_columns)` separately flags
categorical columns where most (≥ 50%) but not all values parse as numeric — e.g. a lab
result column recorded as `["12.5", "8.1", "<5", "300.2"]`, kept at object dtype (hence
categorical) solely because of the censored value `"<5"`. Written to
`categorical/quasi_numeric_columns.csv` and surfaced as a report notice rather than
silently coercing the unparseable values to NaN.

### 3.4 Reduction Gating

`should_apply_reductions(df, column_types)` returns boolean flags controlling whether
each heavy analysis step runs. Thresholds are conservative to keep results
interpretable:

| Flag | Threshold |
|------|-----------|
| `pca` | ≥ 10 numeric columns |
| `mca` | ≥ 8 low-cardinality categorical columns |

> **Removed (2026-07):** a WGCNA/TOM-style variable-clustering step (similarity →
> soft-thresholded adjacency → Topological Overlap Measure → hierarchical clustering)
> used to run here unconditionally and feed `screen_associations`'s pair selection
> (§7.3). It was removed because TOM was designed for gene co-expression networks and
> has no literature backing when applied to generic clinical variables or used to
> gate which pairs get a significance test. See `CLUSTERING_REMOVAL_README.md` for
> the full rationale and what replaced it.

---

## 4. Dimensionality Reduction

### 4.1 PCA

**File:** `data_report/generate_figures/pca_plots.py`

Runs on numeric columns only (after dropping ID columns). Internally uses
`sklearn.decomposition.PCA`. Outputs:

| Plot | Description |
|---|---|
| `pca_explained_variance.png` | Scree plot of per-component explained variance |
| `pca_overview.png` | 2×2 panel: variance bar, PC1–PC2 scatter, PC1–PC3 scatter, biplot thumbnail |
| `pca_scatter_2d.png` | PC1 vs PC2 scatter coloured by outcome (if detected) |
| `pca_scatter_3d.html` | Interactive 3-D Plotly scatter |
| `pca_scatter_matrix.png` | Pairwise scatter matrix of first 4 PCs |
| `pca_loadings_pc1_pc2.png` | **Biplot** — arrows show each variable's loading vector on PC1/PC2. Labels use quadrant-aware alignment (left/right, top/bottom based on arrow direction) with a 1.08× offset and white background box to avoid overlap. |
| `pca_loadings_pc1_pc3.png` | Same for PC1/PC3 |

The biplot truncates to the top 20 features by loading magnitude when there are
more than 20 numeric columns.

### 4.2 MCA

**File:** `data_report/generate_figures/mca_plots.py`

Categorical analog of PCA, using Multiple Correspondence Analysis via the `prince`
library. Runs on low-cardinality (≤ 30 levels) categorical columns. Binary numeric
columns (0/1 coded) are cast to `category` dtype before being passed to MCA.

Outputs:

| Plot | Description |
|---|---|
| `mca_explained_inertia.png` | Per-dimension explained inertia (analog of variance) |
| `mca_overview.png` | Panel summary |
| `mca_row_scatter_2d.png` | Patients projected into the first two MCA dimensions |
| `mca_column_map_batch_XX.png` | Category-level coordinates (one batch per ≤12 categories) |
| `mca_scatter_matrix.png` | Pairwise scatter of first 4 MCA dimensions |

---

## 5. Descriptive Statistics

**File:** `data_report/statistical_analysis/local/compute_statistics.py`

### 5.1 Numeric Statistics (`compute_numeric_statistics`)

For every numeric column (after removing IDs):

| Statistic | Notes |
|---|---|
| count, mean, median, mode | Standard |
| min, max, variance, std\_dev | Sample variance (ddof=1) |
| IQR | Q3 − Q1 |
| skewness, kurtosis | `scipy.stats.skew/kurtosis` |
| outliers | IQR fence: values outside Q1−1.5·IQR or Q3+1.5·IQR |
| missing\_values | Count of NaN |
| frequency / relative\_frequency | Mode frequency and its proportion |

`compute_age_histogram(df)` separately bins the `age` column into fixed 0-100
(step-5) bins for the age distribution plot. `count_out_of_range_ages(df)` counts
parseable age values outside that range (negative, or a typo like 999) — these are
silently excluded from the histogram bins otherwise, with no indication why the
plotted total doesn't match the patient count; the count is surfaced as a report
notice instead.

### 5.2 Categorical Statistics (`compute_categorical_statistics`)

For every categorical column (including binary): value counts, relative frequencies,
missing count, entropy.

### 5.3 Temporal Statistics (`compute_temporal_statistics`)

For every datetime column: min/max date, range in days, monthly activity counts
(number of events per calendar month), missingness.

---

## 6. Data Quality

**File:** `data_report/statistical_analysis/local/data_quality.py`

- `compute_missing_by_column(df)` — per-column missing count and rate.
- `compute_total_missing(df)` — overall missing cell count.

**Visual outputs** (generated in the analyzer while `df` is available):

| Plot | Tool | When rendered |
|---|---|---|
| `missingno_bar.png` | `missingno.bar` | Always |
| `missingno_heatmap.png` | `missingno.heatmap` | When ≥ 2 columns have missing values (shows nullity correlation between columns — if glucose is NaN, is HbA1c also NaN?) |
| `missing_values_by_column_XX.png` | Custom horizontal bar chart | Always, paginated per 30 columns |

---

## 7. Inferential Analysis

**File:** `data_report/statistical_analysis/local/inferential_analysis.py`

### 7.1 Outcome Column Detection

`detect_outcome_column(df, column_types, keyword_groups)` searches categorical
columns for an auto-detected outcome variable using priority-ordered keyword groups
defined in `data_report/config.py`:

```
Priority 1: death, died, deceased, mortality, survival
Priority 2: icu, intensive_care, ventilation, intubation
Priority 3: readmission, admission
Priority 4: outcome, status, diagnosis, condition, label, pasc
Priority 5: complication, adverse
```

A candidate column is accepted only when it has 2–5 distinct non-null levels and
every level has ≥ 20 observations (to support stable group comparisons).

### 7.2 Outcome-Driven Group Comparisons

For every numeric column vs. the detected outcome:

- **2 groups:** `compare_two_groups` — selects Student's t-test, Welch's t-test,
  or Mann-Whitney U based on distribution diagnostics (Shapiro-Wilk normality,
  skewness, outlier presence, equal-variance Levene test).
- **3+ groups:** `one_way_group_comparison` — selects one-way ANOVA or Kruskal-Wallis.
  If significant (p < 0.05), a post-hoc test runs:
  - Tukey HSD (ANOVA path, equal variance)
  - Games-Howell (ANOVA path, unequal variance)
  - Dunn with BH correction (Kruskal-Wallis path)

Effect sizes: Hedges' g (t-tests), rank-biserial correlation (Mann-Whitney), η²
(ANOVA), ε² (Kruskal-Wallis).

Plots generated per significant comparison: `{col}_vs_{outcome}_oneway.png`,
`posthoc_{col}_vs_{outcome}.png` (heatmap of post-hoc p-values).

### 7.3 Association Screening (`screen_associations`)

Tests **every pair** of the relevant type directly (`itertools.combinations`) rather
than restricting to a data-driven subset — the Benjamini-Hochberg correction below is
specifically designed to stay valid regardless of how many tests are run, so no
pre-filter is statistically necessary. (Before 2026-07 this instead tested only pairs
within the same WGCNA/TOM cluster; that clustering step was removed, see
`docs/CLUSTERING_REMOVAL_README.md`.)

Three pair types:

| Type | Pairs selected | Test |
|---|---|---|
| `num-num` | Every pair of numeric columns | Pearson or Spearman (auto-selected by skew/outlier heuristics) |
| `cat-cat` | Every pair of low-cardinality (≤ 6 levels) categorical columns | Chi-square; Fisher exact for 2×2 with sparse expected counts |
| `num-cat` | Every numeric column vs. every binary categorical column | Student's t / Welch's t / Mann-Whitney (same auto-selection as group comparisons) |

`num-cat` pairs are additionally skipped (with a logged warning, not an error) when
either group has fewer than `_MIN_GROUP_SIZE_FOR_COMPARISON` (2) observations —
`compare_two_groups`' `ddof=1` variance/effect-size math is undefined for a group of
size 1, which previously surfaced as a silent `NaN` effect size instead of the pair
just being left out.

**Multiple testing correction:** Benjamini-Hochberg FDR over all p-values collected
in a single pass. A pair is marked `significant=True` only when:
- Adjusted p-value < 0.05 **AND**
- Effect size ≥ small-effect threshold (|r| ≥ 0.2, Cramér's V ≥ 0.1, Hedges' g ≥ 0.2,
  rank-biserial ≥ 0.2).

Outputs: `association_screening.csv` (all pairs), `significant_associations.csv`
(filtered), `association_screening.png` (bubble plot).

---

## 8. What Gets Serialised and Sent to the Aggregator

The analyzer returns a single Python dictionary containing **only statistics, not
raw data**:

```python
{
    "node_id":                str,
    "n_rows":                 int,
    "n_cols":                 int,
    "n_analytical_cols":      int,
    "column_types":           dict[str, list[str]],
    "numeric_statistics":     dict[col → {mean, std, min, ...}],
    "categorical_statistics": dict[col → {counts, entropy, ...}],
    "temporal_statistics":    dict[col → {min_date, max_date, monthly_counts, ...}],
    "missing_by_col":         dict[col → {count, rate}],
    "total_missing":          int,
    "missing_values_percentage": float,
    "n_duplicates":           int,
    "age_hist":               list,
    "age_edges":              list,
    "sex_counts":             dict[str → int],
    "means":                  dict[col → float],
}
```

(Before 2026-07 this also included `cluster_results`/`clusters` keys holding
`ClusterResult` objects with similarity/TOM matrices; removed along with clustering.)

---

## 9. Output Directory Structure

```
results/
├── data_report.pkl                   ← pickled aggregated result
├── federated_results/
│   ├── global_report_short.pdf
│   ├── global_report_full.pdf
│   ├── summary.json
│   ├── overview/
│   ├── numeric/
│   ├── categorical/
│   └── temporal/
└── local_results/
    └── nodeN/
        ├── local_report_nodeN_short.pdf
        ├── local_report_nodeN_full.pdf
        ├── summary.json
        ├── overview/
        │   ├── data_type_distribution.png
        │   ├── missing_values_by_column_XX.png
        │   ├── missingno_bar.png
        │   └── missingno_heatmap.png          ← only if ≥2 columns have NaN
        ├── numeric/
        │   ├── age_distribution.png
        │   ├── numeric_summary.csv
        │   ├── numeric_histograms_XX.png       ← batched, 6 columns per image
        │   └── numeric_boxplots_XX.png         ← batched, 6 columns per image
        ├── categorical/
        │   ├── sex_distribution.png
        │   ├── categorical_distributions_XX.png ← batched, 6 columns per image
        │   └── categorical_summary.csv
        ├── temporal/
        │   ├── temporal_summary.csv
        │   ├── {feature}_activity.png          ← one per column (used by short mode)
        │   └── temporal_activity_batch_XX.png  ← batched, 6 columns per image (full mode)
        ├── pca/
        │   ├── pca_overview.png
        │   ├── pca_explained_variance.png
        │   ├── pca_scatter_2d.png
        │   ├── pca_scatter_3d.html
        │   ├── pca_scatter_matrix.png
        │   ├── pca_loadings_pc1_pc2.png
        │   ├── pca_loadings_pc1_pc3.png
        │   └── excluded_columns.csv            ← only if a numeric column was entirely missing
        ├── mca/
        │   ├── mca_overview.png
        │   ├── mca_explained_inertia.png
        │   ├── mca_row_scatter_2d.png
        │   ├── mca_column_map_batch_XX.png
        │   ├── mca_scatter_matrix.png
        │   └── excluded_columns.csv            ← only if a categorical column was entirely missing
        ├── inferential/
        │   ├── association_screening.csv
        │   ├── association_screening.png
        │   ├── significant_associations.csv
        │   ├── comparisons_by_{outcome}.csv
        │   └── comparisons/
        │       ├── {col}_vs_{outcome}_oneway.png
        │       └── posthoc_{col}_vs_{outcome}.png
        └── comparison/
            ├── numeric_comparison.csv
            └── column_availability.png
```

---

## 10. Report Generation

**Files:** `generate_reports/generate_local_report.py`, `generate_reports/report_utils.py`,
`generate_reports/section_definitions.py`

PDFs are generated in two modes:

- **Short:** One overview plot + a summary table per section. PCA/MCA show only the
  overview panel. Temporal line charts show only a top-N-by-activity subset.
- **Full:** All plots in every section and subsection, including all batched
  numeric/categorical/temporal distribution images, all PCA biplot variants, and all
  post-hoc heatmaps.

`section_definitions.py` holds the declarative configuration (directory names,
CSV filenames, short-mode plot filenames) shared between the local and global report
builders.

---

## 11. Key Module Map

| Module | Responsibility |
|---|---|
| `data_report/cli.py` | Entry point, orchestrates the full run |
| `data_report/analyze.py` | `DataReportAnalyzer` (local) + `DataReportAggregator` (central) |
| `data_report/config.py` | Outcome keyword groups, PASC label config |
| `data_report/statistical_analysis/local/compute_statistics.py` | Column-type detection, descriptive statistics |
| `data_report/statistical_analysis/local/inferential_analysis.py` | Association screening, group comparisons, post-hoc tests, outcome detection |
| `data_report/statistical_analysis/local/data_quality.py` | Missing value counts |
| `data_report/generate_figures/pca_plots.py` | PCA outputs including biplot |
| `data_report/generate_figures/mca_plots.py` | MCA outputs |
| `data_report/generate_figures/local_descriptive_plots.py` | Numeric, categorical, and temporal distribution plots (batched) |
| `data_report/generate_figures/inferential_plots.py` | Association bubble plots, group comparison plots, post-hoc heatmaps |
| `data_report/generate_figures/data_quality_plots.py` | Missingno bar and heatmap |
| `data_report/generate_figures/primitives.py` | Low-level plot helpers (histogram, boxplot, violin, bar, scatter, heatmap) |
| `data_report/comparison/utils.py` | Column availability classification (common\_all / common\_partial / unique) |
| `data_report/get_data/load_data.py` | Reads node CSVs as bytes |
| `data_report/get_data/generate_synthetic_data.py` | Synthetic hospital datasets for testing |
| `generate_reports/generate_local_report.py` | Per-node PDF builder |
| `generate_reports/generate_global_report.py` | Federated PDF builder |
| `generate_reports/generate_json_summary.py` | Machine-readable JSON summaries |
| `generate_reports/section_definitions.py` | Declarative section config (paths, titles, short-mode plots) |
