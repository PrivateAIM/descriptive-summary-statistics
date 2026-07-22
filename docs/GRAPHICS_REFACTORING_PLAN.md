# Graphics Module Refactoring Plan

---

## 1. Requirements Analysis

### What is currently implemented vs. required

**Local descriptive — Numeric:**

| Plot | Required | Status |
|------|----------|--------|
| Bar chart | ✓ | Missing |
| Histogram | ✓ | Only age (`compute_age_histogram` in `generate_plots.py`) |
| Box plot | ✓ | Missing |
| Scatter plot | ✓ | Missing (only in clustering context) |
| Correlation heatmap | ✓ | Missing (TOM/similarity heatmaps exist for clustering, not descriptive) |

**Local descriptive — Categorical:**

| Plot | Required | Status |
|------|----------|--------|
| Bar chart | ✓ | Only sex (`analyze.py:1254`) |
| Stacked bar | ✓ | Missing |
| Pie chart | ✓ | Only data-type distribution pie (`analyze.py:1077`) |
| Count plot | ✓ | Missing |

**Local descriptive — Temporal:**

| Plot | Required | Status |
|------|----------|--------|
| Time-series | ✓ | Done (`analyze.py:1148–1225`) |
| Area chart | ✓ | Missing |
| Bar chart over time | ✓ | Missing |
| Heatmap | ✓ | Missing |

**Inferential plots:**

| Analysis | Required plots | Status |
|----------|----------------|--------|
| Two groups | Boxplot + Violin + Effect size | Missing entirely |
| ANOVA | Boxplot + Mean CI + Post-hoc significance | Missing |
| Correlation | Scatter + Regression line | Missing |
| Chi-square | Contingency heatmap + Mosaic | Missing |
| Linear regression | Actual vs. Predicted + Residual + Coefficient plot | Missing |
| Logistic regression | ROC + Confusion Matrix + Odds Ratio | Missing |
| Time series | Line + Trend + FFT Spectrum | FFT computation exists in `inferential_analysis.py`; no plots |
| Longitudinal | Spaghetti + Mean trajectory + Slope distribution | Missing |
| Panel | Multi-line + Heatmap | Missing |
| Event data | Event timeline + Inter-event histogram | Missing |
| FFT | Power spectrum + Peak annotations | `peak_annotation()` exists in `inferential_analysis.py` but no save function |

**Federated descriptive:**

| Section | Status |
|---------|--------|
| Bar chart by hospital | Missing |
| Stacked bar across hospitals | Missing |
| Boxplots across hospitals | Missing |
| Numeric histogram | Only age |
| Categorical bar | Only sex |
| Temporal time-series | Missing |
| Data quality overview | Missingno bar and heatmap only (truncated to 5 rows, 30 cols — wrong) |

**Clustering:**

| Plot | Status |
|------|--------|
| TOM heatmap / clustered TOM heatmap | Done (`clustering_plots.py`) |
| Dendrogram | Done (`clustering_plots.py`) |
| Cluster heatmaps per cluster | Done |
| Cluster histograms | Done |
| Cluster boxplots | Done (numeric + temporal only) |
| Violin plots | Done |
| Scatter plots (strongest pair) | Done |
| Cluster bar charts | Missing |
| Correlation clustermap | Done |
| Cluster summary table | Done |

**PCA:** Fully implemented in `pca.py` (explained variance, 2D scatter, loadings biplot, scatter matrix, 3D HTML, overview panel).

**MCA:** Fully implemented in `mca.py` (inertia plot, row scatter, column map, scatter matrix, 3D HTML, overview panel).

---

## 2. Gap Analysis

### Missing modules entirely

- No `inferential_plots.py` anywhere — the most glaring gap. `inferential_analysis.py` has rich, well-designed analysis functions with zero corresponding visualizations.
- No `local_descriptive_plots.py` (or equivalent) for non-hardcoded numeric/categorical columns.
- No `federated_descriptive_plots.py` covering hospital comparison and full variable distributions.
- No `data_quality_plots.py` beyond the truncated missingno calls embedded in `analyze.py`.

### Misplaced modules

- `pca.py` and `mca.py` live in `statistical_analysis/local/` but are 80–90% visualization code. They import and call matplotlib, seaborn, and plotly. They belong in `generate_figures/`.
- `peak_annotation()` in `inferential_analysis.py` is a plotting helper function mixed into a statistics file.

### Duplicated code

1. `save_cluster_outputs` exists in both `analyze.py:65–130` and `clustering_plots.py:61–72`. They do almost the same thing with slightly different file naming conventions. The one in `analyze.py` handles per-dtype subdirectories; the one in `clustering_plots.py` handles a single result. Neither is a subset of the other — they evolved in parallel.
2. Pie chart for data-type distribution: identical code block at `analyze.py:828–839` (federated) and `analyze.py:1077–1088` (local).
3. `save_cluster_histograms` in `clustering_plots.py` uses `col.hist(bins=30, ax=ax)` — hardcoded 30 bins, same problem as the spec describes.

### Dead / vestigial code

- `generate_plots.py` contains only `compute_age_histogram()`, which computes data, not a plot. The function name misleads — it should be in `compute_statistics.py`.
- `generate_tables.py` is empty.
- Missing values bar chart at `analyze.py:1070` is commented out — the code exists but the `plt.savefig` call is disabled.

### Silent bug

At `analyze.py:970–980`, the bar chart plot overwrites the pie chart that was just saved to `output_path`. Both save to the same path. The bar chart wins; the pie chart is generated and immediately discarded without being saved.

---

## 3. Reusable Visualization Primitives

Every domain module needs histograms, bar charts, boxplots, scatter plots, heatmaps, and line charts. Without a shared rendering layer, each module will hand-write `plt.figure(); plt.bar(); plt.tight_layout(); plt.savefig()` — exactly the pattern already causing duplication in `analyze.py`.

The primitives layer should be **axis-level** (takes an `ax`, not a figure), so domain modules control layout (single plot, subplot grid, composite panel) while the primitive handles styling, smart defaults, and consistency.

### Required primitives

```python
# primitives.py — axis-level rendering, returns nothing (mutates ax)
def histogram(ax, values, *, bins=None, color=None, title=None, xlabel=None, ylabel="Count")
def histogram_from_bins(ax, edges, counts, *, color=None, title=None, xlabel=None)
def boxplot(ax, data, *, labels=None, orient="v", title=None, ylabel=None)
def violin(ax, data, *, labels=None, orient="v", title=None, ylabel=None)
def bar_chart(ax, categories, values, *, horizontal=False, colors=None, title=None, xlabel=None, ylabel=None)
def stacked_bar(ax, df, *, colors=None, title=None, xlabel=None, ylabel=None)
def scatter(ax, x, y, *, hue=None, alpha=0.7, reg_line=False, title=None, xlabel=None, ylabel=None)
def line_chart(ax, x, y, *, markers=True, fill=False, color=None, title=None, xlabel=None, ylabel=None)
def heatmap(ax, matrix, *, cmap="viridis", vmin=None, vmax=None, annotate=False, title=None)
def pie_chart(ax, sizes, labels, *, colors=None, min_slice_pct=3.0, title=None)
def count_plot(ax, series, *, top_n=20, horizontal=True, title=None)

# Figure-level utilities (not axis-level)
def make_subplots(n, ncols=3, *, width=5, height=4) -> tuple[Figure, ndarray]
def save_fig(fig, path, *, dpi=200) -> None
```

### Smart defaults to implement

| Default | Rule |
|---------|------|
| Auto-bin histogram | Freedman-Diaconis rule (`2 * IQR * n^(-1/3)`); cap at 50 bins; fall back to Sturges if IQR is zero |
| Auto figure width | `make_subplots` scales width by `n_cols` so cluster grids don't produce squashed subplots |
| Consistent palette | Define a project-wide `PALETTE` list in `style.py`; primitives use it by index so colors are stable across related plots |
| Pie `min_slice_pct` | Merge slices smaller than 3% into "Other" to prevent label overlap |
| Bar chart truncation | When `categories` > 20 items, auto-truncate to top-20 by value and append an "Other" bar |

### Full primitives.py design

```python
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

# -- Style constants (override via style.py) ---------------------------
PALETTE = ["#4C9BE8", "#F28B30", "#3BB37E", "#E85C4C",
           "#A77FD3", "#F7C948", "#72C9CF", "#D98CB0"]
DPI = 200

# -- Figure utilities --------------------------------------------------
def make_subplots(n: int, ncols: int = 3, *, width=5, height=4):
    nrows = math.ceil(n / ncols)
    ncols = min(ncols, n)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows), squeeze=False)
    for ax in axes.flatten()[n:]:
        ax.set_visible(False)
    return fig, axes.flatten()[:n]

def save_fig(fig: Figure, path, *, dpi: int = DPI) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# -- Axis-level primitives ---------------------------------------------
def histogram(ax, values, *, bins=None, color=None, title=None, xlabel=None, ylabel="Count"):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if bins is None:
        bins = _auto_bins(values)
    ax.hist(values, bins=bins, color=color or PALETTE[0], edgecolor="white")
    _decorate(ax, title, xlabel, ylabel)

def _auto_bins(values: np.ndarray) -> int:
    n = len(values)
    if n < 4:
        return max(n, 1)
    iqr = np.percentile(values, 75) - np.percentile(values, 25)
    if iqr == 0:
        return min(int(np.ceil(1 + np.log2(n))), 50)  # Sturges
    width = 2 * iqr * n ** (-1/3)                      # Freedman-Diaconis
    data_range = values.max() - values.min()
    return min(max(int(np.ceil(data_range / width)), 5), 50)

def histogram_from_bins(ax, edges, counts, *, color=None, title=None, xlabel=None, ylabel="Count"):
    edges = np.asarray(edges, dtype=float)
    counts = np.asarray(counts, dtype=float)
    centers = (edges[:-1] + edges[1:]) / 2
    widths  = (edges[1:] - edges[:-1]) * 0.9
    ax.bar(centers, counts, width=widths, color=color or PALETTE[0], edgecolor="white")
    _decorate(ax, title, xlabel, ylabel)

def boxplot(ax, data: list, *, labels=None, orient="v", title=None, ylabel=None):
    ax.boxplot(data, labels=labels, vert=(orient == "v"), patch_artist=True,
               boxprops=dict(facecolor=PALETTE[0], alpha=0.6))
    _decorate(ax, title, ylabel=ylabel)
    if labels and orient == "v":
        ax.set_xticklabels(labels, rotation=45, ha="right")

def violin(ax, data: list, *, labels=None, title=None, ylabel=None):
    parts = ax.violinplot(data, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(PALETTE[0])
        pc.set_alpha(0.6)
    if labels:
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right")
    _decorate(ax, title, ylabel=ylabel)

def bar_chart(ax, categories, values, *, horizontal=False, colors=None, title=None, xlabel=None, ylabel=None):
    categories, values = _maybe_truncate(categories, values)
    c = colors or PALETTE[:len(categories)]
    if horizontal:
        ax.barh(categories, values, color=c)
    else:
        ax.bar(categories, values, color=c)
        ax.set_xticklabels(categories, rotation=45, ha="right")
    _decorate(ax, title, xlabel, ylabel)

def stacked_bar(ax, df, *, colors=None, title=None, xlabel=None, ylabel=None):
    c = colors or PALETTE[:len(df.columns)]
    bottom = np.zeros(len(df))
    for i, col in enumerate(df.columns):
        ax.bar(df.index, df[col], bottom=bottom, label=col, color=c[i % len(c)])
        bottom += df[col].values
    ax.legend()
    ax.set_xticklabels(df.index, rotation=45, ha="right")
    _decorate(ax, title, xlabel, ylabel)

def scatter(ax, x, y, *, hue=None, alpha=0.6, reg_line=False, title=None, xlabel=None, ylabel=None):
    if hue is not None:
        sns.scatterplot(x=x, y=y, hue=hue, alpha=alpha, ax=ax)
    else:
        ax.scatter(x, y, alpha=alpha, color=PALETTE[0])
    if reg_line:
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() > 2:
            m, b = np.polyfit(np.asarray(x)[valid], np.asarray(y)[valid], 1)
            xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
            ax.plot(xs, m * xs + b, color=PALETTE[1], linewidth=1.5, linestyle="--")
    _decorate(ax, title, xlabel, ylabel)

def line_chart(ax, x, y, *, markers=True, fill=False, color=None, title=None, xlabel=None, ylabel=None):
    c = color or PALETTE[0]
    ax.plot(x, y, marker="o" if markers else None, color=c, linewidth=2)
    if fill:
        ax.fill_between(x, y, alpha=0.3, color=c)
    _decorate(ax, title, xlabel, ylabel)
    ax.tick_params(axis="x", rotation=45)

def heatmap(ax, matrix, *, cmap="viridis", vmin=None, vmax=None, annotate=False, title=None):
    sns.heatmap(matrix, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                annot=annotate, fmt=".2f" if annotate else "")
    if title:
        ax.set_title(title)

def pie_chart(ax, sizes, labels, *, colors=None, min_slice_pct=3.0, title=None):
    sizes, labels = _merge_small_slices(np.asarray(sizes, float), list(labels), min_slice_pct)
    c = colors or PALETTE[:len(sizes)]
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, colors=c)
    ax.axis("equal")
    if title:
        ax.set_title(title)

def count_plot(ax, series, *, top_n=20, horizontal=True, title=None):
    counts = series.value_counts(dropna=True).head(top_n)
    bar_chart(ax, list(counts.index.astype(str)), list(counts.values),
              horizontal=horizontal, title=title, ylabel="Count")

# -- Private helpers ---------------------------------------------------
def _decorate(ax, title=None, xlabel=None, ylabel=None):
    if title:  ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)

def _maybe_truncate(categories, values, max_n=20):
    if len(categories) <= max_n:
        return categories, values
    paired = sorted(zip(values, categories), reverse=True)[:max_n]
    vals, cats = zip(*paired)
    return list(cats), list(vals)

def _merge_small_slices(sizes, labels, threshold_pct):
    total = sizes.sum()
    mask = (sizes / total * 100) >= threshold_pct
    other = sizes[~mask].sum()
    sizes_f  = list(sizes[mask]) + ([other] if other > 0 else [])
    labels_f = [l for l, m in zip(labels, mask) if m] + (["Other"] if other > 0 else [])
    return sizes_f, labels_f
```

---

## 4. Clustering / PCA / MCA Applicability Rules

The threshold must be **per type**, not on total column count.

The core reason: these techniques answer the question "do I have too many variables of this type to visualize directly?" Dimensionality reduction is only meaningful when the dimensionality problem is real for that specific type.

### Decision variables

```python
n_numeric     = len(column_types["numeric"])
n_cat_usable  = len([c for c in column_types["categorical"] if df[c].nunique() <= 30])
n_temporal    = len(column_types["temporal"])
```

### Thresholds

| Technique | Apply when | Skip: do this instead |
|-----------|-----------|----------------------|
| Numeric clustering | `n_numeric >= 10` | Show correlation heatmap directly |
| PCA | `n_numeric >= 10` | Same threshold |
| Categorical clustering | `n_cat_usable >= 12` | Show Cramér's V heatmap directly |
| MCA | `n_cat_usable >= 8` | Show direct bar/stacked bar charts |
| Temporal clustering | `n_temporal >= 6` | Show each time-series individually |

### Rationale

- With 9 numeric variables, a 9×9 correlation heatmap has 36 unique cells — a person can scan it. With 15+, the heatmap becomes a wall and clustering adds real navigation value.
- MCA needs enough category-level geometry to produce a meaningful plot. Below 8 usable categorical variables, a category map is trivially sparse.
- The `n_numeric >= 10` threshold applies to both numeric clustering AND PCA simultaneously — they share the same cutoff and in practice will always fire together.

### Cross-type independence

Total column count is irrelevant for per-type thresholds.

**Your example** (100 cols: 5 numeric, 80 categorical, 15 temporal):
- `n_numeric = 5` < 10 → **no PCA, no numeric clustering** → show direct 5×5 correlation heatmap
- `n_cat_usable ≈ 70` ≥ 12 → **apply categorical clustering AND MCA**
- `n_temporal = 15` ≥ 6 → **apply temporal clustering**

This produces exactly the right behavior automatically.

### Additional minimum sample size rules

- **PCA**: skip if `n_rows < 2 * n_numeric` (degenerate fit — already enforced in `pca.py:87`)
- **MCA**: skip if `n_rows < 5 * n_cat_usable` (sparse one-hot encoding destabilizes geometry)

### Implementation

Encode as a single utility function, not scattered `if len(...) >= X` checks throughout `analyze.py`:

```python
def should_apply_reductions(df, column_types) -> dict[str, bool]:
    n_num = len(column_types["numeric"])
    n_cat = len([c for c in column_types["categorical"] if df[c].nunique(dropna=True) <= 30])
    n_tmp = len(column_types["temporal"])
    n_rows = len(df)
    return {
        "numeric_clustering": n_num >= 10,
        "pca":                n_num >= 10 and n_rows >= 2 * n_num,
        "categorical_clustering": n_cat >= 12,
        "mca":                n_cat >= 8 and n_rows >= 5 * n_cat,
        "temporal_clustering": n_tmp >= 6,
    }
```

---

## 5. Evaluation of Plan 1

```
basic_plots.py
descriptive_plots.py
  ├── local_plots.py
  ├── clustering_plots.py
  ├── pca_mca_plots.py
  └── federated_plots.py
inferential_plots.py
```

**Problems:**

1. `descriptive_plots.py` as a parent with children is confusing unless it is a namespace package. If it is a single file containing submodule imports, it is just indirection with no value. If it is a package, the submodule list groups by context which is inconsistent — clustering is not "descriptive" in the same sense as histograms.

2. `clustering_plots.py` already exists and works well as a top-level module. Nesting it under `descriptive_plots` would require a rename and break existing imports in `analyze.py`.

3. `pca_mca_plots.py` as a single merged file is fine given their parallel structure, but `pca.py` and `mca.py` are already well-separated with clear docstrings distinguishing them. Merging them removes clarity for marginal line-count savings.

4. The plan does not address where the *primitives* live. Without a primitives layer, `basic_plots.py` and all the submodules will end up duplicating `plt.figure(); plt.tight_layout()` boilerplate.

**Verdict:** Reasonable direction but imprecise grouping and missing the primitives layer entirely.

---

## 6. Evaluation of Plan 2

```
basic_plots.py
local_plots.py   (descriptive + clustering + inferential)
federated_plots.py  (descriptive + inferential)
```

**Problems:**

1. `local_plots.py` will be 1500–2000 lines covering at least 25 distinct plot functions across three very different analysis types. Finding "the correlation scatter function" requires knowing which section it is in.

2. "Descriptive + clustering + inferential" do not form a natural cohesion. They share chart types (all use boxplots), not data structures — clustering plots take `ClusterResult`, inferential plots take raw test result dicts, descriptive plots take column statistics dicts. Putting them together just because they are "local" is grouping by accident.

3. Local and federated share more code than the plan implies. The boxplot primitive for a local two-group comparison and the boxplot for a federated hospital comparison are literally the same rendering call with different data labels. A hard boundary between `local_plots.py` and `federated_plots.py` encourages duplication of the rendering logic.

4. Does nothing to address where PCA/MCA plots go — they do not fit "local descriptive" cleanly since they are currently in `statistical_analysis/local/`.

**Verdict:** Too coarse. Turns into a maintenance problem as soon as the inferential plots section is fully implemented.

---

## 7. Proposed Architecture

Neither Plan 1 nor Plan 2 is right. The correct design is a **three-layer architecture**:

```
Layer 1 — Rendering primitives (no domain knowledge)
  primitives.py      axis-level drawing functions, smart defaults, consistent style
  style.py           color palettes, theme constants, figure sizing helpers

Layer 2 — Analysis-type modules (use primitives, know about data structures)
  local_descriptive_plots.py      uses raw DataFrame
  federated_descriptive_plots.py  uses aggregated statistics dicts
  inferential_plots.py            uses test result dicts; works for local AND federated
  clustering_plots.py             uses ClusterResult; works for local AND federated
  pca_plots.py                    uses PCAResult
  mca_plots.py                    uses MCAResult
  data_quality_plots.py           uses missing data dicts + raw df

Layer 3 — Orchestrators (in analyze.py)
  _save_local_node_results()      calls layer-2 modules; no plt.* calls
  _make_plots()                   calls layer-2 modules; no plt.* calls
```

### Why this beats Plan 1 and Plan 2

- `inferential_plots.py` and `clustering_plots.py` are **reusable across both local and federated** — there is no "local inferential" vs "federated inferential" split because the plot functions take test result dicts, not raw DataFrames. The local/federated distinction lives in the orchestrator, not in the plot module.
- The primitives layer is the **only** place that calls `plt.figure()`, `ax.set_title()`, `plt.tight_layout()`. Every other module calls primitives. Styling changes and rename-fixes touch exactly one file.
- `analyze.py` becomes an orchestrator only — no inline `plt.bar()` calls remaining in it.

---

## 8. Proposed Module Structure

```
data_report/
  generate_figures/
    __init__.py
    primitives.py                     NEW — axis-level rendering functions
    style.py                          NEW — palette, theme, figure sizing helpers

    local_descriptive_plots.py        NEW — histograms, boxplots, bars, scatter, corr heatmap
    federated_descriptive_plots.py    NEW — federated counterparts + hospital comparison plots

    inferential_plots.py              NEW — all inferential plot types (local + federated)
    clustering_plots.py               EXISTING — update to use primitives
    pca_plots.py                      MOVED from statistical_analysis/local/pca.py (save_* only)
    mca_plots.py                      MOVED from statistical_analysis/local/mca.py (save_* only)
    data_quality_plots.py             NEW — missing data, quality overview plots

    generate_plots.py                 DELETE — its one function moves to compute_statistics.py
    generate_tables.py                KEEP but implement, or delete if out of scope

  statistical_analysis/
    local/
      pca.py                          KEEP — PCAResult dataclass + run_pca() stay here
      mca.py                          KEEP — MCAResult dataclass + run_mca() stay here
      inferential_analysis.py         UPDATE — remove peak_annotation() (moves to inferential_plots.py)
      ...
```

### PCA/MCA split detail

The current `pca.py` mixes computation and visualization. After refactoring:

| Stays in `statistical_analysis/local/pca.py` | Moves to `generate_figures/pca_plots.py` |
|----------------------------------------------|------------------------------------------|
| `PCAResult` dataclass | `save_explained_variance_plot` |
| `run_pca()` | `save_pca_scatter` |
| | `save_pca_scatter_matrix` |
| | `save_pca_loadings_biplot` |
| | `save_pca_3d_html` |
| | `save_pca_overview` |
| | `save_pca_outputs` (entry point, imports `run_pca`) |

Same split applies to MCA.

The import in `analyze.py` changes from:
```python
from data_report.statistical_analysis.local.pca import save_pca_outputs
```
to:
```python
from data_report.generate_figures.pca_plots import save_pca_outputs
```

---

## 9. Plot Abstraction Design

### How domain modules use primitives

```python
# local_descriptive_plots.py example

from pathlib import Path
import matplotlib.pyplot as plt
from data_report.generate_figures.primitives import histogram, boxplot, make_subplots, save_fig

def save_numeric_histograms(df, numeric_cols: list[str], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = make_subplots(len(numeric_cols), ncols=3)
    for ax, col in zip(axes, numeric_cols):
        histogram(ax, df[col].dropna(), title=col, xlabel=col)
    save_fig(fig, output_dir / "numeric_histograms.png")

def save_numeric_boxplots(df, numeric_cols: list[str], output_path: Path):
    fig, ax = plt.subplots(figsize=(max(8, len(numeric_cols) * 0.6), 5))
    data = [df[col].dropna().values for col in numeric_cols]
    boxplot(ax, data, labels=numeric_cols, title="Numeric Variables — Boxplots")
    save_fig(fig, output_path)
```

### Inferential plots dispatcher

```python
# inferential_plots.py

def save_inferential_plots(test_result: dict, output_dir: Path):
    """Dispatcher: reads test_result['method'] and calls the right plot function."""
    method = test_result.get("method", "")
    dispatch = {
        "student_ttest":  _save_two_group_plots,
        "welch_ttest":    _save_two_group_plots,
        "mannwhitney":    _save_two_group_plots,
        "welch":          _save_anova_plots,
        "kruskal":        _save_anova_plots,
        "pearson":        _save_correlation_plots,
        "spearman":       _save_correlation_plots,
        "chi2":           _save_chi_square_plots,
        "linear":         _save_linear_regression_plots,
        "logistic":       _save_logistic_regression_plots,
    }
    fn = dispatch.get(method)
    if fn:
        fn(test_result, output_dir)
```

### Clustering plots with primitives

```python
# clustering_plots.py (updated)

from data_report.generate_figures.primitives import histogram, boxplot, make_subplots, save_fig, heatmap

def save_cluster_histograms(df, result, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for cluster_id, variables in result.clusters.items():
        n = len(variables)
        fig, axes = make_subplots(n, ncols=3)
        for ax, variable in zip(axes, variables):
            col = df[variable]
            if pd.api.types.is_numeric_dtype(col):
                histogram(ax, col.dropna(), title=variable)  # auto-bins, no hardcoding
            else:
                count_plot(ax, col, title=variable)
        save_fig(fig, output_dir / f"cluster_{cluster_id}_histograms.png")
```

---

## 10. Refactoring Strategy

In order of surgical precision — each step is independently testable.

**Step 1 — Delete duplicate `save_cluster_outputs` in `analyze.py:65–130`**
It is dead code: `_save_local_node_results` uses `cp_save_cluster_outputs` from `clustering_plots.py`, not the local version. Delete it.

**Step 2 — Fix the overwritten pie/bar chart at `analyze.py:970–980`**
Both save to the same `output_path`. Give them different filenames.

**Step 3 — Uncomment missing-values-by-column plot (`analyze.py:1070`)**
The code exists but the `plt.savefig` call is commented out. Uncomment it.

**Step 4 — Create `primitives.py` and `style.py`**
No existing code changes yet — just new files. Run existing tests to confirm no regressions.

**Step 5 — Create `data_quality_plots.py` and move missingno calls**
Implement `save_missing_bar(df, path)`, `save_missing_heatmap(df, path)`, `save_missing_by_column(missing_dict, n_rows, path)`.
Replace the inline code in `analyze.py:415–428` with calls to these functions.

**Step 6 — Move PCA `save_*` functions → `generate_figures/pca_plots.py`**
Keep `PCAResult` and `run_pca` in `statistical_analysis/local/pca.py`.
Update one import in `analyze.py`. Low risk, no behavior change.

**Step 7 — Move MCA `save_*` functions → `generate_figures/mca_plots.py`**
Same as step 6 for MCA.

**Step 8 — Create `local_descriptive_plots.py` — numeric section**
Implement: `save_numeric_histograms`, `save_numeric_boxplots`, `save_scatter_matrix`, `save_correlation_heatmap`.
Move age histogram and sex bar chart from `analyze.py:1240–1263` here.

**Step 9 — Create `local_descriptive_plots.py` — categorical section**
Implement: `save_categorical_bar_charts`, `save_stacked_bar`, `save_count_plots`.

**Step 10 — Create `local_descriptive_plots.py` — temporal section**
Move temporal time-series from `analyze.py:1148–1225` here.
Add `save_temporal_area_chart` and `save_temporal_heatmap`.

**Step 11 — Update `clustering_plots.py` to use primitives**
Replace `col.hist(bins=30, ax=ax)` with `primitives.histogram(ax, df[col])`.
Replace hardcoded `figsize=` values with `primitives.make_subplots(n)`.

**Step 12 — Create `federated_descriptive_plots.py` and migrate `_make_plots`**
Move inline plt.* code from `_make_plots` into this module.
Add hospital comparison plots (bar by hospital, stacked bars, boxplots across hospitals).

**Step 13 — Create `inferential_plots.py` — group comparison section**
Implement two-group boxplot+violin+effect size, ANOVA boxplot+CI.

**Step 14 — Create `inferential_plots.py` — correlation + chi-square sections**
Scatter+regression line, contingency heatmap.

**Step 15 — Create `inferential_plots.py` — regression section**
Linear: actual vs. predicted + residual + coefficient plot.
Logistic: ROC + confusion matrix + odds ratio plot.

**Step 16 — Create `inferential_plots.py` — time series + longitudinal + panel + event sections**
Move `peak_annotation()` from `inferential_analysis.py` here.
Implement spaghetti plots, mean trajectory, slope distribution, event timeline.

**Step 17 — Wire inferential plots into `analyze.py`**
Call `save_inferential_plots(screening, inferential_dir)` from the inferential statistics section.

**Step 18 — Delete `generate_figures/generate_plots.py`**
Move `compute_age_histogram` to `compute_statistics.py`.
Update the one import in `analyze.py`.

---

## 11. Migration Strategy from `analyze.py`

`analyze.py` currently contains inline plot generation in three locations. Each has a clear target.

### Location 1: `analysis_method()` lines 232–241 (clustering df-dependent plots)

These calls already use `clustering_plots.py` correctly. No migration needed — just ensure `clustering_plots.py` uses primitives internally when you update it in Step 11.

### Location 2: `_save_local_node_results()` lines 956–1263 (~300 lines)

**Before (current state):** Inline `plt.figure()`, `plt.pie()`, `plt.bar()`, temporal loop, age histogram, sex bar chart, all mixed in.

**After (target state):**
```python
def _save_local_node_results(self, analysis_results, comparison_results):
    from data_report.generate_figures import (
        local_descriptive_plots as ldp,
        data_quality_plots as dqp,
    )
    ...
    # Data quality
    dqp.save_missing_bar(truncated_df, overview_dir / "missingno_bar.png")
    dqp.save_missing_heatmap(truncated_df, overview_dir / "missingno_heatmap.png")
    dqp.save_missing_by_column(r["missing_by_col"], r["n_rows"],
                                overview_dir / "missing_by_column.png")

    # Overview
    ldp.save_data_type_distribution(
        numeric_statistics, categorical_statistics, temporal_statistics,
        overview_dir / "data_type_distribution.png"
    )

    # Comparison
    ldp.save_column_distribution_charts(column_counts, comparison_dir)

    # Temporal
    ldp.save_temporal_plots(temporal_statistics, temporal_dir, node_number)

    # Numeric
    ldp.save_age_distribution(r["age_hist"], r["age_edges"], numeric_dir, node_number)

    # Categorical
    ldp.save_sex_distribution(r["sex_counts"], categorical_dir, node_number)
```

### Location 3: `_make_plots()` lines 815–875

**Before:** ~60 lines of inline plt.* calls.

**After:**
```python
def _make_plots(self, federated_results):
    from data_report.generate_figures import federated_descriptive_plots as fdp
    fdp.save_data_type_distribution(federated_results, overview_dir)
    fdp.save_age_distribution(federated_results, numeric_dir)
    fdp.save_sex_distribution(federated_results, categorical_dir)
    fdp.save_hospital_comparison(federated_results, overview_dir)
    fdp.save_numeric_distributions(federated_results, numeric_dir)
    fdp.save_categorical_distributions(federated_results, categorical_dir)
    fdp.save_temporal_distributions(federated_results, temporal_dir)
```

---

## 12. Implementation Roadmap

Ordered by: (1) fix existing bugs first, (2) infrastructure, (3) fill gaps in already-implemented features, (4) new modules.

| # | Task | Type | Notes |
|---|------|------|-------|
| 1 | Delete duplicate `save_cluster_outputs` in `analyze.py:65–130` | Bug | Dead code, never called |
| 2 | Fix overwritten pie/bar chart at `analyze.py:970–980` | Bug | Two plots saving to same path |
| 3 | Uncomment `plt.savefig` for missing-values-by-column (`analyze.py:1070`) | Bug | Existing code, disabled |
| 4 | Create `primitives.py` | Infrastructure | Foundation for all below |
| 5 | Create `style.py` | Infrastructure | Color, font, sizing constants |
| 6 | Create `data_quality_plots.py` + move missingno calls | Migration | Easiest domain module |
| 7 | Move PCA `save_*` → `generate_figures/pca_plots.py` | Migration | One import change in `analyze.py` |
| 8 | Move MCA `save_*` → `generate_figures/mca_plots.py` | Migration | Same |
| 9 | `local_descriptive_plots.py` — numeric section | New feature | Histogram, boxplot, scatter, corr heatmap for all numeric cols |
| 10 | `local_descriptive_plots.py` — categorical section | New feature | Bar, stacked bar, pie, count plot |
| 11 | `local_descriptive_plots.py` — temporal section | Migration + new | Move time-series, add area chart and heatmap |
| 12 | Update `clustering_plots.py` to use primitives | Refactor | Remove hardcoded bins, auto-sizing |
| 13 | `federated_descriptive_plots.py` + migrate `_make_plots` | Migration + new | Add hospital comparison plots |
| 14 | `inferential_plots.py` — group comparison | New feature | Two-group + ANOVA plots |
| 15 | `inferential_plots.py` — correlation + chi-square | New feature | Scatter+regression, contingency heatmap |
| 16 | `inferential_plots.py` — regression | New feature | Linear and logistic plots |
| 17 | `inferential_plots.py` — time series + longitudinal + panel + event | New feature | Most complex section |
| 18 | Wire inferential plots into `analyze.py` | Integration | Call dispatcher from inferential section |
| 19 | Delete `generate_figures/generate_plots.py` | Cleanup | Move `compute_age_histogram` to `compute_statistics.py` |
| 20 | Implement `should_apply_reductions()` utility | Cleanup | Centralize applicability thresholds |

---

## 13. Risks and Future Extensions

### Risks

**1. PCA/MCA split circular import risk**
When you split `pca.py` into computation and visualization, `pca_plots.py` imports from `statistical_analysis/local/pca.py` but that module should never import from `generate_figures/`. There is no cycle here — analysis only calls visualization, never the reverse. However: verify this before doing the split.

**2. Primitives style lock-in**
Once you use `PALETTE[0]` everywhere, changing the first color changes 30 plots simultaneously. This is the intended behavior, but validate styling on a real dataset before wiring primitives everywhere. A wrong color choice touches everything at once.

**3. Complex Period-to-Timestamp conversion in temporal plots**
The `analyze.py:1166–1176` code that converts `pd.Period` objects to `pd.Timestamp` handles a subtle type mismatch. When you extract this into `local_descriptive_plots.py`, do not simplify the conversion — it handles a real edge case specific to how `compute_temporal_statistics` returns its data.

**4. Inferential plots need raw df, not just the stats dict**
The inferential visualization functions will need the raw DataFrame (for boxplots, scatter, etc.). `_save_local_node_results` only has serialized statistics dicts. This is already solved by the pattern used for df-dependent clustering plots: `save_cluster_histograms` is called from `analysis_method()` where `df` is available, not from `_save_local_node_results`. Inferential plots must follow the same pattern.

**5. The silent pie/bar overwrite bug (item 2 in roadmap)**
This bug means the column distribution pie chart has never appeared in any output — ever. When you fix it, it will appear for the first time. Check that the pie chart with `column_counts` actually contains meaningful data before treating the fix as done.

**6. `generate_tables.py` is empty**
It is imported nowhere and contains nothing. Either implement it or delete it. Leaving it creates confusion for future contributors who may assume it is working code.

### Future extensions

**Kaplan-Meier survival curve**
Requires `lifelines` or `scikit-survival`. Add as a separate section in `inferential_plots.py` with an explicit optional-dependency guard — same pattern as `statsmodels` and `scikit_posthocs` already use in `inferential_analysis.py`.

**Forest plot**
Needs confidence intervals from regression or meta-analysis. Wire to the `regression()` output which already returns `coef_table` with `ci_lower`/`ci_upper` — all the data is there.

**Federated inferential plots**
The federated aggregation currently produces no inferential statistics. When it does (Pearson from aggregates, chi-square, federated t-test), `inferential_plots.py` will work for federated outputs unchanged — because the dispatcher takes test result dicts, not raw DataFrames. This is the main benefit of a single shared `inferential_plots.py` over a local/federated split.

**Report PDF generation**
`generate_tables.py` is empty. The most natural extension is to pair each visualization module with a table module so plots and tables are generated together. A future PDF report assembler could import both and lay them out side by side.

**Interactive plots beyond 3D HTML**
`pca.py` and `mca.py` already export interactive 3D HTML via Plotly. The same approach could be extended to the inferential scatter plots and temporal line charts for the final report, giving readers pan/zoom on dense distributions.
