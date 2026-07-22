# Graphics Module Refactoring — Final Report

## Overview

This document records every change made during the 20-step graphics/visualization
refactoring, why each change was made, how it was implemented, and how the code
works after the refactoring.

---

## Before vs. After: High-Level Summary

| Metric | Before | After |
|---|---|---|
| Visualization files | 1 (`generate_plots.py`) | 9 dedicated modules |
| Shared style/palette | None — hardcoded colours per function | `style.py` — single source of truth |
| Axis-level primitives | None — every plot called `plt.figure()` + `savefig` inline | `primitives.py` — 12 reusable axis-level functions |
| `analyze.py` lines | ~1,400 | ~1,262 (−138 after deletions + additions) |
| Dead code | `save_cluster_outputs` duplicate (~67 lines) | Deleted |
| Overwrite bug | Pie + bar both saved to same path | Fixed |
| Disabled plot | `missing_values_by_column` savefig commented out | Enabled |
| Auto-binning | `bins=30` hardcoded everywhere | Freedman-Diaconis (Sturges fallback) |
| PCA/MCA location | `statistical_analysis/local/` | `generate_figures/` |
| Threshold gating | Ad-hoc `>= 2` checks scattered | `should_apply_reductions()` single function |

---

## Step-by-Step Change Log

### Step 1 — Remove dead code in `analyze.py`

**What:** Deleted the duplicate `save_cluster_outputs` function (~67 lines) from `analyze.py`
and the unused `from scipy.cluster.hierarchy import dendrogram, leaves_list` import.

**Why:** The file already imported `save_cluster_outputs` from
`generate_figures/clustering_plots.py` and used that import throughout. The duplicate
inline definition was never called — it was dead code left over from an earlier refactor.
Dead code misleads readers into thinking the function matters and blocks future cleanup.

**How:** Identified via grep that `cp_save_cluster_outputs` (the aliased import) was the
only call site. Removed the duplicate function body and the unused import.

---

### Step 2 — Fix silent overwrite bug in `_save_local_node_results`

**What:** Replaced a pie chart + bar chart pair that both saved to the same
`output_path` variable. The bar chart silently overwrote the pie.

**Why:** The bug produced only one image where two were intended, and the surviving
image (bar chart) was less informative than the pie for 3-category compositional data.
The decision was made to keep only the pie chart since a bar chart of 3 categories
adds nothing over the pie.

**How:** Deleted the bar chart entirely. Renamed the output file from
`column_distribution.png` → `column_availability.png` (more descriptive).
Added a dynamic title showing the node count: `Column Availability (N nodes)`.

---

### Step 3 — Enable the disabled missing-values-by-column plot

**What:** Uncommented `plt.savefig(overview_dir / "missing_values_by_column.png", dpi=200)`
which had been disabled.

**Why:** The plot was fully implemented and correct — it had been accidentally commented
out, silently producing no output despite the code running.

---

### Step 4 — Create `generate_figures/style.py` and `__init__.py`

**What:** Created two new files:
- `generate_figures/__init__.py` (empty, marks the directory as a package)
- `generate_figures/style.py` — global palette, DPI constant, named themes, `set_palette()`, `set_theme()`

**Why:** Every plot module was defining its own colours as hardcoded hex strings.
Changing the colour scheme required editing dozens of functions. A single style
module lets callers call `style.set_theme("colorblind")` once before report
generation and have it propagate to every plot.

**How:** `PALETTE` is a module-level list. All primitives access it as
`style.PALETTE[0]` at **call time** (not at import time), so `set_theme()` called
before plotting propagates automatically. Named themes include `default`,
`colorblind` (Wong 2011), and `grayscale`.

```python
from data_report.generate_figures import style
style.set_theme("colorblind")   # one call — affects every plot
```

---

### Step 5 — Create `generate_figures/primitives.py`

**What:** Created the axis-level rendering library: 12 public functions +
4 private helpers.

**Why:** Every plot in the codebase repeated the same pattern: create figure,
set title, set labels, set grid, savefig, close. Extracting these into primitives
eliminates duplication and enforces a consistent visual style.

**Design contract:**
- Every function takes a `matplotlib.Axes` as its first argument
- Mutates the axes in place — never creates a figure, never calls `savefig`
- Reads `style.PALETTE` at call time, not at import time

**Functions:**

| Function | Purpose |
|---|---|
| `make_subplots(n, ncols, width, height)` | Grid of n axes, unused axes hidden |
| `save_fig(fig, path, dpi)` | `tight_layout` + `savefig` + `close` |
| `histogram(ax, values, ...)` | Auto-binned histogram (Freedman-Diaconis) |
| `histogram_from_bins(ax, edges, counts, ...)` | Histogram from pre-computed bins (federated) |
| `boxplot(ax, data, labels, orient, ...)` | Box-and-whisker, outliers as dots |
| `violin(ax, data, labels, ...)` | Violin with median line |
| `bar_chart(ax, categories, values, ...)` | Vertical or horizontal, auto-truncates at 20 |
| `stacked_bar(ax, df, ...)` | Stacked bars from a DataFrame |
| `scatter(ax, x, y, hue, reg_line, ...)` | Scatter with optional regression overlay |
| `line_chart(ax, x, y, fill, ...)` | Line with optional area fill |
| `heatmap(ax, matrix, cmap, annotate, ...)` | Seaborn heatmap wrapper |
| `pie_chart(ax, sizes, labels, ...)` | Pie with small-slice merging |
| `count_plot(ax, series, top_n, ...)` | Bar chart of value counts |

**Auto-binning** (`_auto_bins`): Freedman-Diaconis (`2 * IQR * n^(-1/3)`),
Sturges fallback for zero-IQR, capped `[5, 50]`.

---

### Step 6 — Create `generate_figures/data_quality_plots.py`

**What:** Created three data-quality plot functions and wired them into `analyze.py`,
replacing inline `missingno` blocks.

**Why:** The inline missingno blocks in `analysis_method` and `_save_local_node_results`
contained a bug: the bar chart was being computed on a 5-row truncated DataFrame,
making the completeness bars statistically meaningless. Extracting to a dedicated module
also allows the functions to be tested independently.

**Functions:**

| Function | Input | Output |
|---|---|---|
| `save_missing_bar(df, path)` | Raw df | Column completeness bar chart (missingno) |
| `save_missing_heatmap(df, path)` | Raw df | Nullity correlation heatmap (missingno) |
| `save_missing_by_column(missing_counts, n_rows, path)` | Stats dict | Stacked horizontal bar: present % vs missing % |

**Fix:** Changed truncation from 5 rows to all rows, with a 50-column cap for
readability. This makes the completeness bars reflect the actual dataset.

---

### Step 7 — Move `pca.py` → `generate_figures/pca_plots.py`

**What:** Moved `statistical_analysis/local/pca.py` to `generate_figures/pca_plots.py`
unchanged. Deleted the original. Updated the import in `analyze.py`.

**Why:** PCA here is 80–90% visualization code — it exists entirely to produce plots.
Keeping it in `statistical_analysis/local/` was misleading. Moving it to `generate_figures`
reflects what the module actually does and avoids any risk of the analysis code
trying to re-use PCA outputs downstream.

**Docstring cross-references** updated from `:mod:pca` to `:mod:pca_plots`.

---

### Step 8 — Move `mca.py` → `generate_figures/mca_plots.py`

**What:** Same operation as Step 7, for MCA.

**Why:** Same rationale — MCA is purely descriptive/exploratory visualization,
not an input to any downstream analysis.

---

### Step 9–11 — Create `generate_figures/local_descriptive_plots.py`

**What:** Created a 634-line module covering all three local descriptive plot sections:
numeric, categorical, and temporal. Also contains the shared `_periods_to_timestamps`
helper.

#### `_periods_to_timestamps(obs: dict) → (timestamps, counts)`

Centralises the Period→Timestamp conversion that was previously duplicated inline
three times in `analyze.py`. Handles both `pd.Period` objects and strings (forward
compatible), drops unparseable keys silently, returns lists sorted chronologically.

#### Numeric section

| Function | Requires | Notes |
|---|---|---|
| `save_numeric_histograms` | raw df | One histogram per column, grid layout, Freedman-Diaconis bins |
| `save_numeric_boxplots` | raw df | All columns side-by-side in one figure |
| `save_correlation_heatmap` | raw df | Pearson correlation, constant columns dropped |
| `save_scatter_matrix` | raw df | Seaborn pairplot, capped at 10 columns |
| `save_age_distribution` | edges + hist | Histogram from pre-computed bins (no raw df needed) |
| `save_data_type_distribution` | stats dicts | Pie chart: numeric / categorical / temporal counts |

#### Categorical section

| Function | Notes |
|---|---|
| `save_sex_distribution` | Bar chart from pre-computed sex_counts dict |
| `save_categorical_bar_charts` | Top-20 bar per column, grid layout |
| `save_stacked_bar_charts` | Pairwise stacked bars for 2–10 level columns |
| `save_column_availability_chart` | Pie: common-all / common-partial / unique-local |

#### Temporal section

| Function | Notes |
|---|---|
| `save_temporal_line_charts` | Line chart per column, most-active period annotated |
| `save_temporal_area_charts` | Filled area chart (cumulative view) |
| `save_temporal_bar_charts` | Horizontal bar per period (sparse series) |
| `save_temporal_heatmap` | Month × year heatmap; only when ≥ 2 distinct years |

---

### Step 12 — Refactor `generate_figures/clustering_plots.py`

**What:** Updated all cluster plot functions to use primitives. Added
`save_cluster_sizes_bar`.

**Changes per function:**

| Function | Old | New |
|---|---|---|
| `save_cluster_histograms` | `bins=30`, manual grid, `.plot.bar()` for categorical | Freedman-Diaconis via `histogram`, `make_subplots`, `count_plot` |
| `save_cluster_boxplots` | `df.boxplot(rot=45)`, hardcoded figsize | `boxplot` primitive, dynamic width |
| `save_cluster_violinplots` | `sns.violinplot` on melted df | `violin` primitive on list-of-arrays |
| `save_cluster_scatterplots` | `plt.scatter` + manual labels | `scatter` primitive |
| TOM/heatmap functions | `plt.figure()` + `sns.heatmap` inline | `fig, ax = plt.subplots` + `heatmap` primitive + `save_fig` |
| `save_tom_dendrogram` | `plt.figure()` + bare `dendrogram()` | `fig, ax = plt.subplots` + `dendrogram(ax=ax)` + `save_fig` |
| DPI | Hardcoded `dpi=200` everywhere | `style.DPI` via `save_fig` |

**New:** `save_cluster_sizes_bar` — horizontal bar chart of variables-per-cluster.
Useful for spotting unbalanced clusters (e.g. one cluster absorbing half the variables).

---

### Step 13 — Create `generate_figures/federated_descriptive_plots.py`

**What:** Created a dedicated module for federated-level plots. Replaced the
entire inline `_make_plots` method body in `DataReportAggregator` with a
two-line delegation.

**Why:** `_make_plots` was 60 lines of inline plt calls with no tests and no
reuse path. The aggregator class should orchestrate, not render.

**New functions vs. old `_make_plots`:**

| | Old | New |
|---|---|---|
| Data-type pie | Inline `plt.pie` | `save_federated_data_type_distribution` |
| Age histogram | Inline `plt.bar` with manual centers | `save_federated_age_distribution` via `histogram_from_bins` |
| Sex bar chart | Inline `plt.bar` | `save_federated_sex_distribution` |
| Numeric summary | Not present | `save_federated_numeric_summary_bars` — mean ± std bars |
| Categorical distributions | Not present | `save_federated_categorical_distributions` — top-N bars |
| Temporal charts | Not present | `save_federated_temporal_charts` — line chart per column |

**`_make_plots` after refactoring:**
```python
def _make_plots(self, federated_results):
    from data_report.generate_figures.federated_descriptive_plots import save_all_federated_plots
    save_all_federated_plots(federated_results, FEDERATED_RESULTS_DIR)
```

---

### Steps 14–17 — Create `generate_figures/inferential_plots.py`

**What:** Created a 798-line module covering all four inferential plot sections.

#### Group comparison (Step 14)

| Function | Purpose |
|---|---|
| `save_two_group_comparison` | Boxplot + significance bracket (*** / ** / * / n.s.), effect size in title |
| `save_two_group_violins` | Violin complement for distribution shape |
| `save_one_way_comparison` | N-group boxplot; bracket added automatically for N=2 |
| `save_group_comparisons_summary` | Effect-size bar chart, significant vs. n.s. highlighted |
| `save_association_screening` | Two-panel: effect-size bars (left) + volcano scatter (right) |

The significance bracket is drawn by `_annotate_significance` — a private helper
that places a bracket + `***`/`**`/`*`/`n.s.` label between two box positions.

#### Correlation (Step 15)

| Function | Purpose |
|---|---|
| `save_correlation_scatter` | Scatter + regression overlay, r and p-value in title |
| `save_correlation_matrix` | Heatmap rebuilt from screening DataFrame (num-num pairs) |
| `save_cramers_v_bars` | Horizontal bars of Cramer's V, sorted by strength |

#### Regression (Step 16)

| Function | Purpose |
|---|---|
| `save_regression_coefficients` | Forest-plot style: one row per predictor, 95% CI bars |
| `save_regression_residuals` | Predicted vs. actual + residuals histogram (linear only) |
| `save_logistic_predicted_proba` | Predicted probability histogram split by true class |

#### Time series (Step 17)

| Function | Purpose |
|---|---|
| `peak_annotation` | Moved from `inferential_analysis.py`; signature unchanged |
| `save_power_spectrum` | FFT spectrum with `peak_annotation` overlay |

**`peak_annotation` relocation rationale:** It was plotting code living in the
statistics module. Moving it to `inferential_plots.py` keeps `inferential_analysis.py`
free of matplotlib imports while maintaining backward compatibility (same signature).

---

### Step 18 — Wire inferential plots into `analyze.py` + federated trend

#### Local wiring

The inferential section in `analysis_method` now:
1. Calls `save_association_screening(screening, inferential_dir)` immediately after
   saving the screening CSVs — generates the two-panel volcano + effect-size chart.
2. Stores each `cmp` result dict in `cmp_results` alongside the summary row.
3. After outcome comparison CSVs: calls `save_group_comparisons_summary` for the
   effect-size overview bar chart.
4. Calls `save_two_group_comparison` or `save_one_way_comparison` per numeric column
   into `inferential/comparisons/`.

#### Federated inferential — `save_federated_trend_summary`

Only one federated inferential operation is feasible from the existing aggregated
data: **simple linear trend analysis on temporal counts**.

> Pearson correlation requires E[XY] (cross-product moments not transmitted).
> Chi-square requires joint contingency tables (only marginals are transmitted).
> Federated t-test is feasible but complex — deferred per the design decision.

`save_federated_trend_summary` fits OLS on each temporal column's
`counts_per_period`, plots a slope bar chart coloured ↑ (increasing) / ↓
(decreasing), and annotates R² when ≥ 0.1.

---

### Step 19 — Delete `generate_plots.py`, move `compute_age_histogram`

**What:** Moved `compute_age_histogram` from `generate_figures/generate_plots.py`
to `statistical_analysis/local/compute_statistics.py`. Deleted `generate_plots.py`.

**Why:** `compute_age_histogram` performs a statistical computation (numpy histogram),
not a visualization. It belongs in `compute_statistics.py` alongside the other
`compute_*` functions. `generate_plots.py` existed only to hold this one function —
once moved, the file had no reason to exist.

**Improvement:** Added a guard for an entirely non-numeric or empty age column
(previously would return empty lists silently; now returns `(None, None)`).

**Import in `analyze.py` updated:**
```python
# before
from data_report.generate_figures.generate_plots import compute_age_histogram
# after
from data_report.statistical_analysis.local.compute_statistics import compute_age_histogram
```

---

### Step 20 — `should_apply_reductions(df, column_types) → dict[str, bool]`

**What:** Added a module-level gating function in `analyze.py` (before the class
definitions), called once after `detect_column_types`.

**Why:** PCA and MCA had ad-hoc `>= 2` guards that were far too permissive —
running PCA on 2 numeric columns produces a single component which is meaningless.
The real thresholds were scattered in comments and documentation but not enforced.
Centralising them in one function makes them easy to find, adjust, and test.

**Thresholds:**

| Flag | Threshold | Rationale |
|---|---|---|
| `pca` | n_numeric ≥ 10 | Fewer components give trivial decompositions |
| `numeric_clustering` | n_numeric ≥ 10 | TOM/correlation matrix needs enough variables |
| `mca` | n_cat_usable ≥ 8 | Usable = ≤ 30 unique values |
| `categorical_clustering` | n_cat_usable ≥ 12 | Larger requirement for stable clusters |
| `temporal_clustering` | n_temporal ≥ 6 | Needs a meaningful number of time columns |

**Usage in `analysis_method`:**
```python
column_types = detect_column_types(df)
reductions = should_apply_reductions(df, column_types)

# PCA — was: if len(numeric_feature_cols) >= 2
if reductions["pca"]:
    save_pca_outputs(...)

# MCA — was: if len(categorical_feature_cols) >= 2
if reductions["mca"]:
    save_mca_outputs(...)
```

---

## Final Architecture

### Layer 1 — Style and Primitives

```
generate_figures/
├── __init__.py           (empty package marker)
├── style.py              PALETTE, DPI, THEMES, set_palette(), set_theme()
└── primitives.py         12 axis-level functions; reads style.* at call time
```

All palette access goes through `style.PALETTE[n]` at call time, never at import
time. `style.set_theme("colorblind")` before report generation propagates to
every plot automatically.

### Layer 2 — Domain Modules

```
generate_figures/
├── data_quality_plots.py         save_missing_bar, save_missing_heatmap,
│                                 save_missing_by_column
├── local_descriptive_plots.py    Numeric, categorical, temporal sections;
│                                 _periods_to_timestamps helper
├── federated_descriptive_plots.py  save_all_federated_plots + 7 sub-functions
│                                   + save_federated_trend_summary
├── clustering_plots.py           TOM/dendrogram/heatmap views + df-dependent plots
│                                 + save_cluster_sizes_bar (new)
├── pca_plots.py                  run_pca, PCAResult, save_pca_outputs + helpers
├── mca_plots.py                  run_mca, MCAResult, save_mca_outputs + helpers
└── inferential_plots.py          Group comparison, correlation, chi-square,
                                  regression, time series; peak_annotation (moved)
```

### Layer 3 — Orchestration

```
analyze.py                        DataReportAnalyzer.analysis_method()
│                                 DataReportAggregator.aggregation_method()
│                                 should_apply_reductions() ← new gating function
│
└── calls domain modules directly, never re-implements rendering logic
```

### Moved out of `generate_figures`

```
statistical_analysis/local/
└── compute_statistics.py         + compute_age_histogram (moved from generate_plots.py)
```

### Deleted

```
generate_figures/generate_plots.py          (only function moved to compute_statistics.py)
statistical_analysis/local/pca.py           (moved to generate_figures/pca_plots.py)
statistical_analysis/local/mca.py           (moved to generate_figures/mca_plots.py)
```

---

## Design Decisions Recorded

| Decision | Rationale |
|---|---|
| Axis-level primitives (no figure creation) | Domain modules own layout; primitives own rendering. Separation makes both independently testable. |
| `style.PALETTE` accessed at call time | Ensures `set_theme()` propagates globally without re-importing |
| PCA/MCA kept as full-file moves, no split | Both are 80–90% visualization; splitting would create a circular import risk with no benefit |
| Column availability: pie only, no bar | Bar chart of 3 categories is redundant; pie communicates composition directly |
| Federated inferential: trend only | Pearson requires E[XY]; chi-square requires contingency table; both need joint data not transmitted by nodes |
| `peak_annotation` moved to `inferential_plots.py` | Statistics modules should not import matplotlib |
| `compute_age_histogram` moved to `compute_statistics.py` | It computes statistics (numpy histogram), not a visualization |
| `should_apply_reductions` as module-level function | Makes thresholds independently testable; called once, result reused |

---

## Files Changed

| File | Status | Change |
|---|---|---|
| `analyze.py` | Modified | −138 lines net: removed duplicate function, inline plots migrated, imports updated, gating function added |
| `generate_figures/__init__.py` | Created | Empty package marker |
| `generate_figures/style.py` | Created | 65 lines |
| `generate_figures/primitives.py` | Created | 443 lines |
| `generate_figures/data_quality_plots.py` | Created | 110 lines |
| `generate_figures/local_descriptive_plots.py` | Created | 634 lines |
| `generate_figures/federated_descriptive_plots.py` | Created | 388 lines |
| `generate_figures/clustering_plots.py` | Rewritten | 209 lines (was 176) |
| `generate_figures/pca_plots.py` | Created (moved) | 329 lines |
| `generate_figures/mca_plots.py` | Created (moved) | 340 lines |
| `generate_figures/inferential_plots.py` | Created | 798 lines |
| `generate_figures/generate_plots.py` | **Deleted** | Only function moved |
| `statistical_analysis/local/pca.py` | **Deleted** | Moved to `generate_figures/` |
| `statistical_analysis/local/mca.py` | **Deleted** | Moved to `generate_figures/` |
| `statistical_analysis/local/compute_statistics.py` | Modified | +23 lines (`compute_age_histogram`) |
