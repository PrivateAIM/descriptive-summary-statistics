# Data Summary Report

Federated exploratory data analysis and descriptive summary statistics for
privacy-preserving medical research, built on
[FLAME](https://github.com/PrivateAIM/python-sdk-patterns).

Each participating node (hospital) computes descriptive, comparative, and
inferential statistics on its own data. A central aggregator combines the
per-node results into federated (cross-node) statistics, without any node
sharing raw patient data.

## Installation

```bash
pip install -e .
```

Requires Python >= 3.10.

## Usage

```bash
dr-analyze
```

This runs the full pipeline (via `StarModelTester`) against the dataset
configured in `data_report/cli.py::analyze_main` (`data/<dataset_name>/`,
currently `dataset5`), and writes all outputs to `results/`.

## Outputs

For each node, under `results/local_results/<node>/`:

- `overview/`, `numeric/`, `categorical/`, `temporal/`, `comparison/`,
  `inferential/`, `pca/`, `mca/` -- CSVs and figures for each analysis
  section. Numeric histograms/boxplots, temporal activity charts, and
  categorical distributions are batched into several images
  (`numeric_histograms_01.png`, `numeric_boxplots_01.png`,
  `temporal_activity_batch_01.png`, `categorical_distributions_01.png`, etc.)
  rather than one figure per column, so reports stay readable regardless of
  how many columns a dataset has.
- `local_report_<node>_short.pdf` / `local_report_<node>_full.pdf` -- PDF
  reports.
- `summary.json` -- a single machine-readable JSON bundling the node's
  summary-table CSVs (overview, numeric/categorical/temporal summaries,
  comparisons, and inferential associations), with stringified dict/list
  cells parsed into real JSON objects/arrays.

For the federated results, under `results/federated_results/`:

- `overview/`, `numeric/`, `categorical/`, `temporal/` -- pooled CSVs and
  figures.
- `global_report_short.pdf` / `global_report_full.pdf` -- PDF reports.
- `summary.json` -- the same kind of consolidated JSON, for the federated
  statistics.

`results/data_report.pkl` holds the pickled aggregated result from the
StarModel run.

## Testing

```bash
pytest
```

## Project layout

- `data_report/` -- the installable package: `analyze.py`
  (`DataReportAnalyzer` / `DataReportAggregator`), `cli.py` (`dr-analyze`
  entry point), `statistical_analysis/`, `generate_figures/`, `get_data/`.
- `generate_reports/` -- PDF report builders (`generate_local_report.py`,
  `generate_global_report.py`), the JSON summary builder
  (`generate_json_summary.py`), and shared `report_utils.py`.
- `data/` -- input datasets, one subdirectory per dataset, each with one
  subdirectory per node.
- `notes/` -- session notes documenting non-obvious fixes and dataset design
  decisions.

## Recent changes

### Identifier-keyword and pie-chart fixes (2026-07)

Two small, targeted fixes from testing against a new dataset (`dataset6`,
the GBSG/Rotterdam breast-cancer survival files).

#### `detect_id_column` missed `rownames` and `pid`

`detect_id_column` (`compute_statistics.py`, mirrored in
`hub_entrypoint_10.py`) matches column names by splitting on non-alphanumeric
characters and checking the resulting tokens against a fixed keyword set. A
column literally named `rownames` (R's default row-index column, exported by
`write.csv`) or `pid` (a fused "patient ID" abbreviation) never decomposes
into a recognized token — `"rownames"` doesn't contain `"name"` as a separate
token, and `"pid"` doesn't contain `"id"` as one — so both columns fell
through to being treated as ordinary numeric features despite being 100%
unique row/patient identifiers in both `dataset6` node files. Fixed by adding
`"pid"` and `"rownames"` to `detect_id_column`'s `strong_keywords` set in
both files.

#### Pie charts merged small-but-real categories into "Other"

`pie_chart()` (`primitives.py`, mirrored in `hub_entrypoint_10.py`) merges
any slice under `min_slice_pct` (default 3.0%) into an "Other" bucket, to
avoid label overlap on charts with many small categories. "Data Type
Distribution" (Numerical/Categorical/Temporal) and "Column Availability"
(Common/Partial/Unique) only ever have 3 possible slices, so there's no
overlap risk to guard against — but a small-but-real category (e.g. 1 out of
46 columns being temporal) still got relabeled "Other" instead of shown by
its actual name. Fixed by passing `min_slice_pct=0` at all 5 live call sites
across both files (one call site, `save_column_availability_chart` in
`local_descriptive_plots.py`, was confirmed dead code with zero callers and
left untouched).

### Deferred design-decision fixes, hub/local parity tests, graphics-overlap audit (2026-07)

Follow-up to the full code review documented in `FULL_CODE_REVIEW_2026-07.md`:
the three items that review deliberately deferred (they change semantics,
not just fix bugs) are now implemented, a hub/local parity test suite was
added specifically to catch drift bugs like `is_binary`'s, and every plot
type in both the modular project and `hub_entrypoint_10.py` was audited for
title/axis-label/legend overlap.

#### Censored-numeric-string categorical notice

A lab-result column recorded as e.g. `["12.5", "8.1", "<5", "300.2"]` is
classified as categorical because the censored value `"<5"` keeps the whole
column at object dtype, with no indication anywhere why. A new
`detect_quasi_numeric_categorical_columns()` (in `compute_statistics.py`,
mirrored in `hub_entrypoint_10.py`) flags columns where most (`>=50%`) but
not all values parse as numeric; a new `quasi_numeric_categorical_notice()`
surfaces the affected column names in the categorical section of both local
and hub reports.

#### Out-of-range age notice

`compute_age_histogram`'s fixed 0-100 (step-5) bins silently drop ages
outside that range (a negative age, or a typo like 999) -- they simply don't
appear in any bin, with the plotted total not matching the patient count and
no explanation why. A new `count_out_of_range_ages()` counts them (summed
across nodes for the federated report too); a new `out_of_range_age_notice()`
surfaces the count in the numeric section of local, federated, and hub
reports.

#### Minimum group-size guard in `screen_associations`

`compare_two_groups`' effect-size math (`ddof=1` variance) is undefined for a
group of size 1 -- a num-cat pair where one category has a single observation
produced a `NaN` effect size instead of being skipped. `screen_associations`'
num-cat loop now checks each group's size against `_MIN_GROUP_SIZE_FOR_COMPARISON`
(2) before calling `compare_two_groups`, in both `inferential_analysis.py`
and `hub_entrypoint_10.py`.

#### Hub/local parity test suite

`tests/test_hub_local_parity.py` loads `hub_entrypoint_10.py` as a module
(via `importlib`, with `_load_analysis_dependencies()` called explicitly
since hub lazy-loads scipy/sklearn/etc.) and runs the modular project's
functions and hub's copies on identical synthetic inputs, asserting matching
output -- `is_binary`, `detect_column_types`, `detect_id_column`,
`detect_quasi_numeric_categorical_columns`, `compute_age_histogram`,
`count_out_of_range_ages`, `compute_numeric_statistics`,
`compute_categorical_statistics`, the `_cohens_d`/`_hedges_g` effect-size
helpers, and a full `screen_associations` run including the new group-size
guard. This is exactly the test class that would have caught the `is_binary`
drift bug from the prior review before it shipped.

Building the suite immediately found a **second, previously-unknown drift
bug**: hub's `is_binary` returned `False` immediately when a column had zero
numeric-parseable values (e.g. a pure `"yes"`/`"no"` string column), skipping
the bool/semantic-binary check entirely -- so hub misclassified purely
string-valued binary columns (`yes`/`no`, `y`/`n`, `true`/`false`, `ja`/`nein`)
as non-binary, while the modular project correctly classified them as binary.
Fixed by removing the early return so the code falls through to the semantic
check, matching the modular project's logic.

#### Graphics overlap audit

Every plot-producing function in `data_report/generate_figures/*.py` and
`hub_entrypoint_10.py`'s aggregator-side plotting code was rendered (via a
full dataset1 and dataset5 run of both the modular pipeline and a direct
`MockFlameCoreSDK` hub run) and visually inspected. Three real overlap bugs
were found and fixed in both files:

- **Missing-values-by-column legend**: `ax.legend()` with no explicit
  location defaults to `loc="best"`, which has nowhere free to go on a
  stacked-bar chart spanning the full 0-100% width and settles on top of the
  first bar's data. Moved outside the axes (`bbox_to_anchor=(1.01, 1)`).
  Hub's version used `loc="lower right"`, which has the same problem for a
  bottom row with 0% missing (fully-present, spanning to 100%) -- given the
  same fix.
- **PCA loadings biplot / "Top Feature Contributions" panel**: labels for
  arrows pointing in nearly the same direction (common for correlated
  features, e.g. several lab values that move together) were placed at a
  fixed offset from each arrow tip, stacking into unreadable overlapping
  text. A new `declutter_radial_labels()` (in `primitives.py`, mirrored in
  `hub_entrypoint_10.py`) groups arrows by angle and fans each group's
  labels out perpendicular to the group's shared direction (alternating
  sides, growing distance), connecting offset labels back to their arrow
  with a thin leader line. Wired into `save_pca_loadings_biplot`,
  `save_pca_overview`, and hub's equivalent aggregator-side PCA loadings
  plot (the standalone `save_pca_outputs`/`save_pca_overview` functions in
  hub turned out to be dead code -- never called; the actual live path is
  inline in the aggregator, and that's what needed the fix).
- **MCA category map / "Category Map" panel**: category-level points that
  land close together in MCA space (a real occurrence, not a plotting
  artifact) had their text labels drawn directly on top of each other. A
  new `declutter_point_labels()` groups points by proximity and stacks each
  group's labels vertically with leader lines. The compact 3-panel MCA/PCA
  overview images (`mca_overview.png`) also had no cap on how many
  categories get labeled (unlike the standalone category map's existing
  top-20 cap) -- capped at 15 there too, since no amount of label
  decluttering fits 30+ labels legibly into a panel a third the width.

Two design bugs in the decluttering algorithm itself were found and fixed
during testing on real (not synthetic) data, both via direct hub execution
on dataset1/dataset5 rather than unit tests:

- `declutter_radial_labels`'s angle-grouping compared each new arrow only
  to the *first* member of a candidate group, so a long fan of near-
  collinear arrows (e.g. 7-8 lab-value loadings) where cumulative drift
  across the fan exceeds the grouping threshold -- even though each
  neighboring pair is close -- never grouped at all. Fixed with proper
  consecutive-chain grouping (compare each point to its immediate neighbor
  in angle-sorted order).
- The first fix attempt staggered labels by multiplying each arrow's own
  (x, y) by a growing radius factor -- which breaks down when a group's
  members already have very different native magnitudes: it applies the
  *smallest* push to the most important (largest-magnitude, already
  furthest-out) loading and the *largest* push to a nearby smaller one,
  shoving the small loading's label into the big one's space instead of
  separating them. Replaced with the perpendicular-fan approach described
  above, which fans labels in the direction the group's own labels don't
  already extend into, regardless of how the native magnitudes differ.

Full prior review this follows up on: `FULL_CODE_REVIEW_2026-07.md`.

### Clustering removed; association screening reworked to full-pairwise (2026-07)

The WGCNA/TOM-style variable-clustering feature (`data_report/statistical_analysis/local/clustering.py`,
`data_report/generate_figures/clustering_plots.py`, and every mirrored section
in `hub_entrypoint_10.py`) was removed entirely. TOM is a real, published
method (Zhang & Horvath, 2005), but it was designed for gene co-expression
networks; applying it to generic clinical variables, and using cluster
co-membership to decide which pairs get a significance test, has no
literature backing and isn't defensible for the thesis. Full rationale and
file-by-file changes: `CLUSTERING_REMOVAL_README.md`.

The one place clustering was load-bearing rather than cosmetic --
`screen_associations`'s pair selection for numeric-numeric and
categorical-categorical association screening -- now tests every pair
directly (`itertools.combinations`), corrected once via the existing
Benjamini-Hochberg FDR, matching the design already used for
numeric-categorical pairs. This avoids the circularity of selecting pairs by
the same statistic being tested, and drops several previously-unvalidated
hyperparameters (TOM power, distance_cut, linkage method) with no loss of
statistical rigor.

Local/global mean and variance already used sample statistics (`ddof=1` /
`N-1`) throughout, including the federated pooling formula in
`combine_node_variances` (Chan, Golub & LeVeque, 1979) -- confirmed correct
and left unchanged; no code changes were needed there.

Clustering-related report sections, plots, JSON summary keys, and glossary
entries were removed from both the modular project and `hub_entrypoint_10.py`;
`hub_entrypoint_9.py` was intentionally left untouched as a frozen prior
snapshot.

### Report readability fixes: exclusion notices, batched distributions, an MCA bug (2026-07)

A readability checkup of the local report sections turned up several gaps,
fixed in `hub_entrypoint_10.py` first, then mirrored into the modular
project. Full detail: `REPORT_READABILITY_FIXES_2.md`.

- **Categorical single-value notice**: a categorical column with only one
  observed value has no distribution plot to show and used to just vanish
  from "3.2 Distributions" with no explanation. A new
  `categorical_excluded_from_distributions_notice` names any such columns.
- **Numeric histograms/boxplots were dead code in the modular project**:
  `save_numeric_histograms`/`save_numeric_boxplots` existed but were never
  called from `analyze.py` -- every report's "2.2 Distributions" section
  showed only the age histogram, no other numeric column. Wired up and given
  `batch_size`-based splitting (matching `save_categorical_distributions`'s
  existing pattern) so reports stay readable regardless of column count.
  `hub_entrypoint_10.py`'s equivalent was already working, just unbatched --
  batching was added there without needing to wire anything new up.
- **Temporal activity charts batched for full-mode reports**: a new
  `save_temporal_activity_batched` groups several columns' line charts per
  image for full mode. The existing one-file-per-column generation is kept
  (not replaced) because short-mode picks a top-N-by-activity subset of
  named files, a ranking only known at report-generation time.
- **PCA excluded-columns notice + a real MCA bug fix**: `run_pca` already
  silently dropped entirely-missing numeric columns (by design); this is now
  surfaced as a report notice. `run_mca` had **no equivalent guard at all** --
  an entirely-missing categorical column reaching it flowed straight into
  `prince.MCA.fit()` unguarded. Fixed by adding the same drop-empty-columns
  step `run_pca` already had.
- **Hub-only latent bug found via live execution, not just syntax-checking**:
  `hub_entrypoint_10.py`'s numeric boxplots have never rendered, for a reason
  unrelated to this round's work -- the aggregator reconstructs boxplots from
  `q25`/`q75` fields it expects on each column's `numeric_statistics` entry,
  but `compute_numeric_statistics` computed `q1`/`q3` and then never stored
  them (left commented out). Fixed by exposing them as `q25`/`q75` in the hub
  file (scoped there only -- the modular project's boxplot function builds
  directly from the raw DataFrame and never needed these fields).
- A stale docstring in `save_categorical_distributions` claiming binary
  columns are excluded (they aren't -- only single-valued columns are) was
  corrected to match the actual filter.

### Hub report: cluster similarity, PCA/MCA plots from summary statistics (2026-06)

A second pass over `hub_entrypoint_10.py` implements all remaining plots that
are feasible without raw patient data. Everything new is in `hub_entrypoint_10.py`.

#### Analyzer: three new compact serialized statistics

All three are column-level summaries — no per-patient rows leave the node.

**`similarity_matrix_values`** added inside every cluster dict entry (alongside the
existing `tom_matrix_cols`, `tom_matrix_values`, `linkage_matrix`). It is the
N×N Pearson / Cramér's V similarity matrix that clustering is computed from,
sent as a list-of-lists. The existing `tom_matrix_cols` is reused as the shared
column index.

**`pca_loadings`, `pca_feature_names`, `pca_explained_variance`,
`pca_recommended_n_components`** — computed by `run_pca()` on all numeric
columns (requires ≥2). `loadings` is an n_features × n_components matrix of
scaled eigenvectors (no patient rows). `explained_variance` is a list of
per-component variance ratios. Serialized with `.tolist()`; safe through the
existing `_make_serializable` + `json.dumps` check. Skipped with a printed
warning when PCA cannot run (e.g. fewer than 2 numeric columns or all-missing).

#### `_collect_cluster_plots`: five new plot types

The function signature gains `node_stats: dict | None = None` (defaulting to
`None` for backward compatibility). The caller in `_collect_local_node_files`
passes `node_stats=r` so the full node result dict is available. Five new
blocks are appended after the existing TOM plots:

**Correlation clustermap** — similarity matrix reordered by the hierarchical
linkage (same linkage already sent for TOM). Guards `_members_in_sim` to only
include variables actually present in the matrix index, preventing KeyError
when a cluster has members not in the matrix.
Output: `{dtype}/correlation_clustermap.png`.

**Per-cluster similarity heatmaps** — one per cluster with ≥2 members present
in the similarity matrix. Cell annotations suppressed above 15 variables.
Output: `{dtype}/cluster_{id}_similarity_heatmap.png`.

**Cluster histograms** — one panel per variable, batched at 6 per figure.
Numeric variables use pre-sent `numeric_histograms` bins; categorical variables
use `category_counts` as a horizontal bar chart (top 20 categories). Variables
with no available data leave the panel empty (handled gracefully by tight_layout).
Output: `{dtype}/cluster_{id}_histograms.png` or `_01.png`, `_02.png` … when batched.

**Cluster boxplots** — numeric variables only, ≥2 required. Uses Q25/median/Q75/
min/max from `numeric_statistics` with `ax.bxp()`; whiskers clamped to
`[Q1−1.5×IQR, Q3+1.5×IQR]` ∩ `[min, max]`.
Output: `{dtype}/cluster_{id}_boxplot.png`.

**Cluster violin plots** (approximate) — ≥2 numeric variables with histogram
data required. A synthetic sample is constructed via `np.repeat(bin_midpoints,
bin_counts)` and passed to `ax.violinplot()`. Reproduces the distribution shape
without transmitting per-patient values.
Output: `{dtype}/cluster_{id}_violin.png`.

#### Aggregator: three new plot types in `_collect_local_node_files`

**MCA explained inertia plot** — bar chart of per-dimension inertia plus
cumulative inertia line. X-axis auto-spaced (≤12 ticks). `mca_explained_inertia`
was already serialized in the prior session; this just generates the plot.
Output: `clustering/categorical/mca_explained_inertia.png`.

**PCA explained variance (scree) plot** — bar chart + cumulative variance line
with recommended component count marked. X-axis auto-spaced (≤12 ticks).
Output: `clustering/numeric/pca_explained_variance.png`.

**PCA loadings biplots** — arrow biplot of feature contributions to PC1 vs PC2
(and PC1 vs PC3 when ≥3 components exist). Capped at 20 features by loading
magnitude (matching local `save_pca_loadings_biplot`). Axis limit floored at 1.0
to handle degenerate all-zero loadings without producing an unusable `(-0, 0)`
plot range.
Output: `clustering/numeric/pca_loadings_pc1_pc2.png` (and `pc1_pc3` when applicable).

#### What still requires raw patient data (3 items)

| Plot | Why |
|---|---|
| Cluster scatterplots | Needs per-patient `(x, y)` values for the most-similar variable pair |
| PCA scatter / scatter matrix / 3D HTML | `result.components` are per-patient PC projections |
| MCA row scatter / scatter matrix / 3D HTML | `result.row_coordinates` are per-patient MCA projections |

The PCA and MCA overview panels (composite figures that include a row-scatter
subplot) are also excluded for the same reason.

---

### Hub report quality: plots and layout alignment (2026-06)

A targeted pass over `hub_entrypoint_10.py` to bring the hub-generated reports
as close as possible to local reports, without transmitting raw patient data.
All changes are in `hub_entrypoint_10.py`.

#### Fix 1 — Privacy notice: federation node-count bullet always shown

The privacy notice in local node reports included a bullet warning that
"above/below average" comparisons can indirectly reveal information about other
nodes. This bullet was guarded by `n_nodes < 5`, so it was silently absent for
hub runs involving five or more nodes — the most common real-world case.

The guard was removed. The bullet is now shown whenever `n_nodes is not None`
(i.e. for every hub-generated local report), and the wording was updated from
"With only N node(s)" to "With N node(s)".

#### Fix 2 — Data type pie chart: crowded autopct labels

The per-node data type distribution pie chart used default matplotlib layout,
causing the percentage labels to overlap the slice labels on small slices.

Three layout parameters were added: `figsize=(6, 6)`, `pctdistance=0.75`
(percentage labels drawn at 75% of radius, inside the slice), and
`labeldistance=1.15` (outer labels pushed further out). The title now includes
`— node{N}` to distinguish charts when viewing multiple nodes.

#### Fix 3a — Missing values: batched stacked bar charts

The aggregator called `_make_missing_by_column_fig`, which produced a single
vertical bar chart of raw missing counts. With many columns this chart was
unreadable (bars too thin, x-axis labels overlapping).

Replaced with a batching loop:
- Columns are sorted by missing count descending (worst first).
- **Full mode**: groups of 20 columns per figure → `missing_values_by_column_01.png`,
  `_02.png`, … Each bar is a horizontal stacked bar (green = present %, red = missing %).
  The title shows the column range: "Missing Values — node{N} (columns X–Y of Z)".
- **Short mode**: top 10 most-missing columns → `missing_values_by_column_short.png`.

#### Fix 3b — Missing values: false "all complete" narrative

The "All columns are fully complete for this node" narrative was shown whenever
the nullity heatmap was absent — which is always the case in hub mode, because
missingno needs the raw dataframe. This caused the note to appear even when
missing-values bar charts were present and visible.

Split into two independent conditions: the "all complete" note fires only when
`not has_quality_plots`; a separate note "Nullity correlation heatmap is not
available in federated mode" fires only in full mode when the heatmap file does
not exist.

#### Fix 4 — Age distribution: wrong output directory

The age distribution PNG was being saved to `overview/age_distribution.png` in
the `output_files` dict, but the report builder reads it from
`numeric/age_distribution.png`. The chart was therefore always absent from
section 2.2. Fixed: the key in `output_files` now uses the `numeric/` prefix.

#### Fix 5 — Categorical summary table: missing top cat % columns

`add_categorical_comparison()` reads federated categorical CSVs from `fed_dir`
to add the "top cat % (local)", "top cat % (global)", and "vs_global" columns
to the categorical summary table. In hub mode, `generate_local_report_bytes`
never wrote anything to `fed_dir`, so `global_categorical` was always `None`
and the three columns were always absent.

Two changes:
1. `generate_local_report_bytes` gains a `federated_results` parameter. When
   provided, `_write_federated_csvs_to_dir(federated_results, fed_dir)` is
   called so the PDF builder can find the global statistics.
2. `_collect_pdf_reports` passes `federated_results` to every
   `generate_local_report_bytes` call.

#### Fix 6 — Local categorical bars: horizontal for multi-category variables

Bar charts for categorical variables with more than 2 categories used
`ax.bar()` with 45° rotated x-axis labels. With 10–20 category levels the
labels overlapped heavily.

For variables with > 2 categories `ax.barh()` (horizontal bars) is now used
instead. Binary variables (2 categories) keep vertical bars. Figure height
increased from 4 to 5 inches; `fig.tight_layout(h_pad=3.0)` added.

#### Fix 7 — Federated categorical distributions: axis label spacing

The federated multi-category bar chart loop used `height=4` with no explicit
tight-layout padding. When batches contained many categories per variable the
subplots ran into each other.

Height increased to 5 and `fig.tight_layout(h_pad=3.0)` added before the byte
conversion.

---

#### New: numeric histograms for all numeric columns

The local pipeline defines `save_numeric_histograms()` and
`save_numeric_boxplots()` in `local_descriptive_plots.py` but never calls them
from `analyze.py` (pre-existing gap). The hub now generates equivalent plots
from compact summary data:

- **Analyzer** computes histogram bin counts and edges for every numeric column
  via `np.histogram(values, bins="auto")` and includes them in the return dict
  as `"numeric_histograms": {col: {"hist": [...], "edges": [...]}}`.
- **Aggregator** builds a grid figure (3 columns, rows as needed) and saves it
  as `numeric/numeric_histograms.png`.

#### New: numeric boxplots for all numeric columns

- **Analyzer** already returns Q25, median, Q75, min, max in
  `numeric_statistics` for every column.
- **Aggregator** builds a grid of synthetic box plots using `ax.bxp()` with
  whiskers computed as `Q1 − 1.5×IQR` / `Q3 + 1.5×IQR`, clamped to [min, max].
  Saved as `numeric/numeric_boxplots.png`.

#### New: nullity correlation heatmap

- **Analyzer** computes Pearson correlation on the binary missingness indicator
  matrix (`df.isnull().corr()`), restricted to columns that have at least one
  missing value. Sent as `"nullity_correlation": {"columns": [...], "values": [[...]]}`.
  Skipped when fewer than 2 columns have missing values.
- **Aggregator** renders a seaborn heatmap (coolwarm, −1 to 1 scale; cell
  annotations shown when ≤15 columns) and saves it as
  `overview/missingno_heatmap.png` — the same path the report builder reads for
  section 1.2.

#### New: TOM heatmap, clustered heatmap, and dendrogram

- **Analyzer** expands the `clusters` dict to include `tom_matrix_cols`,
  `tom_matrix_values`, and `linkage_matrix` for each data type that produced a
  clustering result. These are compact (N×N floats for N variables, typically
  < 100 values).
- **Aggregator** (`_collect_cluster_plots`) reconstructs the matrix as a
  DataFrame and calls the same matplotlib logic as `save_tom_heatmap`,
  `save_clustered_tom_heatmap`, and `save_tom_dendrogram`. Outputs:
  `{dtype}/tom_heatmap.png`, `{dtype}/tom_heatmap_clustered.png`,
  `{dtype}/tom_dendrogram.png`.

#### New: MCA column map batches

- **Analyzer** runs `run_mca()` on non-binary categorical columns (≥2 required)
  and sends `mca_column_coordinates` (dict form of the column-coordinates
  DataFrame) and `mca_explained_inertia` (list of per-dimension inertia ratios).
  Skipped with a warning when MCA fails (e.g. insufficient columns or
  high-cardinality variables).
- **Aggregator** reconstructs the coordinates DataFrame, groups labels by source
  variable using the `__` separator, and produces batch PNGs of 5 variables each
  (`mca_column_map_batch_01.png`, `_02.png`, …) with colored scatter + category
  labels — matching the output of `save_mca_outputs()` for the column-map plots.
  Saved under `clustering/categorical/`.

---

#### New: correlation clustermap and per-cluster similarity heatmaps

- **Analyzer** now includes `similarity_matrix_values` in every cluster dict
  entry (a list-of-lists, same N×N shape as the TOM matrix already sent).
  The `tom_matrix_cols` key is reused as the shared column index.
- **Aggregator** (`_collect_cluster_plots`) reconstructs the similarity
  DataFrame and generates:
  - `correlation_clustermap.png` — the full N×N similarity matrix reordered
    by the hierarchical linkage (variables in the same cluster grouped visually).
  - `cluster_{id}_similarity_heatmap.png` — one per cluster with ≥2 members,
    showing only the sub-matrix for that cluster's variables.

#### New: cluster histograms, boxplots, and violin plots from summary statistics

`_collect_cluster_plots` now accepts a `node_stats` dict (passed as `r` from
the caller), giving it access to `numeric_histograms`, `numeric_statistics`,
and `categorical_statistics`. For every cluster:

- **Histograms** — one panel per variable, batched at 6 per figure. Numeric
  variables are reconstructed from bin counts and edges. Categorical variables
  use `category_counts` as a horizontal bar chart (top 20 categories).
  Output: `cluster_{id}_histograms.png` (or `_01.png`, `_02.png` … when batched).
- **Boxplots** — for numeric variables in the cluster with ≥2 members. Uses
  Q25/median/Q75/min/max from `numeric_statistics`; whiskers are clamped to
  `[min, Q1−1.5×IQR]` / `[Q3+1.5×IQR, max]` using `ax.bxp()`.
  Output: `cluster_{id}_boxplot.png`.
- **Violin plots** — approximate. A synthetic sample is constructed from
  histogram bin midpoints repeated by count (`np.repeat(midpoints, counts)`),
  then passed to `ax.violinplot()`. Reproduces the main distribution shape
  without transmitting per-patient values. Requires ≥2 numeric variables with
  histogram data in the cluster.
  Output: `cluster_{id}_violin.png`.

#### New: PCA loadings biplot and explained variance (scree) plot

- **Analyzer** runs `run_pca()` on all numeric columns (if ≥2) and serializes
  three column-level statistics: `pca_loadings` (n_features × n_components as
  a list-of-lists), `pca_feature_names`, `pca_explained_variance`
  (list of per-component variance ratios), and `pca_recommended_n_components`.
  These contain no per-patient information.
- **Aggregator** generates:
  - `clustering/numeric/pca_explained_variance.png` — bar chart of per-component
    variance + cumulative line, with the recommended component count marked.
    X-axis ticks are auto-spaced to show at most ~12 labels (same fix as MCA).
  - `clustering/numeric/pca_loadings_pc1_pc2.png` — arrow biplot showing each
    variable's contribution to PC1 and PC2. Capped at 20 features by magnitude
    (matching local `save_pca_loadings_biplot`).
  - `clustering/numeric/pca_loadings_pc1_pc3.png` — same for PC1/PC3 when
    ≥3 components exist.

#### New: MCA explained inertia plot

`mca_explained_inertia` was already serialized by the analyzer in the previous
session. The aggregator now also generates a scree-style bar+line chart from it:

- `clustering/categorical/mca_explained_inertia.png` — per-dimension inertia
  bars + cumulative inertia line. X-axis auto-spaced (max ~12 ticks).

#### What still differs from local reports (raw-df-dependent plots)

A small number of visualizations genuinely require per-patient row data and
cannot be reconstructed from summary statistics:

| Plot | Why it needs raw data |
|---|---|
| `missingno_bar.png` | missingno library visualizes row-level sparsity patterns |
| `missingno_heatmap.png` (local version) | missingno co-occurrence heatmap uses row-level nullity |
| Cluster scatterplots | Needs per-patient `(x, y)` values for the most-similar variable pair |
| PCA scatter / scatter matrix / 3D HTML | `result.components` = per-patient projections onto PC axes |
| MCA row scatter / scatter matrix / 3D HTML | `result.row_coordinates` = per-patient projections onto MCA axes |
| PCA / MCA overview panel | Composite panels that include row-scatter subplots |

The hub nullity correlation heatmap is computed via `df.isnull().corr()` (Pearson
correlation of binary missingness indicators), not the missingno algorithm, but
it conveys the same column-level co-missingness information without raw data.

### Hub second audit: additional fixes (2026-06)

A follow-up audit of `hub_entrypoint_10.py` after the first round of fixes
identified seven further issues, all now resolved.

#### `_posthoc_to_pvalue_matrix` — wrong column name for Games-Howell output (High)

`pg.pairwise_gameshowell` returns a `"pval"` column, not `"p-adj"`.  The
branch `{"A", "B", "p-adj"}.issubset(posthoc_df.columns)` never matched,
so the welch post-hoc p-value matrix was always empty — post-hoc heatmaps for
Welch-path outcomes were silently blank even when tests ran.  Fixed by changing
the column name check and accessor to `"pval"`.

#### Fourth `stats` shadow loop in aggregator not renamed (Medium)

Three of the four per-column loops inside `_aggregation_method_impl` were
already renamed from `for col, stats in …` to `for col, col_stats in …` (first
audit), but one loop in the per-node numeric comparison block was missed.  The
loop variable `stats` shadowed the module-level `scipy.stats` alias for the
remainder of the function.  Renamed to `col_stats`; inner references updated.

#### `one_way_group_comparison` and `posthoc_test` not fully synced to local (Medium)

The hub version of these two functions had diverged from the local pipeline:

- **`one_way_group_comparison`** still had `method="auto"` as an explicit
  parameter and a dead `if method == "anova": stats.f_oneway(…)` branch.
  Both removed; method selection (`"welch"` or `"kruskal"`) is now
  unconditional and internal, matching the local `inferential_analysis.py`.

- **`posthoc_test`** still had a dead `if method == "anova"` Tukey HSD branch
  (the anova path is never taken by the pipeline), and the kruskal branch used
  `sp.posthoc_dunn` (scikit-posthocs) instead of the local implementation.
  Both removed.  The kruskal path is now identical to the local pipeline:
  pairwise Mann-Whitney U + Holm-Bonferroni correction using only
  `scipy.stats.mannwhitneyu` and `statsmodels.stats.multitest.multipletests`,
  with no new Docker image dependencies.

#### Three more dead functions removed (Low)

The following functions were defined in the hub but never called anywhere:

| Function | Notes |
|---|---|
| `detect_event_columns` | Longitudinal event-column scorer; no callers |
| `event_dataset_score` | Row-variability scorer for event datasets; no callers |
| `detect_seasonality_fft` | FFT-based seasonality detector; no callers |

All three remain in their original local modules where they may be used in
future analyses.

#### Stale `main()` comment updated (Low)

The docstring comment in `main()` still described the old output contract
(`output_type=["bytes"]`, `filename=["results.tar.gz"]`) instead of the
current one (`output_type="str"`, `filename="results.tar.gz.b64.txt"`).
Updated to match the actual `StarModel` call directly below it.

#### Redundant `import base64 as _b64` removed (Low)

Inside `aggregation_method`, the aggregator imported `import base64 as _b64`
at function scope and used `_b64.b64encode(…)`.  `base64` is already
imported at module level.  The local import and its alias were removed; the
call now uses the module-level `base64.b64encode(…)` consistently.

#### Missing space in assignment fixed (Low)

`categorical_statistics= compute_categorical_statistics(categorical_df)`
(missing space before `=`) was corrected to
`categorical_statistics = compute_categorical_statistics(categorical_df)`.

#### Unused optional-dependency stubs fully removed (Cleanup)

After removing all callers of Tukey HSD (`pairwise_tukeyhsd`) and
scikit-posthocs (`sp`), the optional-import block that set
`sm = ols = pairwise_tukeyhsd = sp = None` and the corresponding
`try/except` blocks that imported them were also removed.  statsmodels is
still used (via a local `from statsmodels.stats.multitest import multipletests`
inside the kruskal post-hoc path), but the module-level stubs for the
four unused names are gone.

---

### Hub/local pipeline alignment and code audit fixes (2026-06)

#### Hub deployment fix: `import re` missing (Critical)

`detect_id_column()` in `hub_entrypoint_10.py` used `re.split()` and
`re.compile()` for token-level keyword matching, but `import re` was absent
from the module-level imports. Every analyzer invocation crashed with
`NameError: name 're' is not defined` before returning any result, causing the
aggregator to raise `ValueError("no analysis results from any node")`. Added
`import re` to the imports block.

#### `_json_sanitize`: `inf` values caused silent `summary.json` loss (High)

`_json_sanitize()` checked `math.isnan()` but not `math.isinf()`. Any numeric
statistic that evaluates to `inf` or `-inf` (e.g. from a near-zero-variance
division in clustering) passed through as a plain Python `float`, causing
`json.dumps()` to raise `ValueError: Out of range float values are not JSON
compliant`. The `summary.json` file was silently never written. Fixed:

```python
# Before
return None if math.isnan(value) else float(value)
# After
return None if (math.isnan(value) or math.isinf(value)) else float(value)
```

#### `generate_local_report_bytes`: inferential tables missing from PDF (High)

Only `.png` files were copied from `output_files` to the temp directory used
by the PDF builder. Inferential CSV files (`significant_associations.csv` etc.)
were in `output_files` but never written to disk, so `_build_cross_variable_section`
always received `None` from `safe_read_csv` and the pairwise associations table
was always empty in local PDFs. Fixed by also copying `.csv` files.

#### `n_total_values` denominator wrong when nodes have different column counts (Medium)

The aggregator computed `n_total_values = total_rows × max(n_cols)` where
`max(n_cols)` is the largest column count across nodes. When nodes differ in
column count this over-estimates the true cell count, making
`total_missing_percentage`, `completeness`, and `usable_data_contribution`
inaccurate. Fixed: `n_total_values = sum(r["n_rows"] * r["n_cols"] for r in analysis_results)`.

#### `global_missing_rate` computed inside per-node loop (Low)

`global_missing_rate` was assigned inside the per-node for-loop even though
both operands (`total_missing`, `n_total_values`) are loop-invariant constants.
The value was correct (identical on every iteration) but the placement was
misleading and would become a real bug if the formula ever depended on a
per-node variable. Moved to before the loop.

#### Dead `self.cluster_results` instance write removed (Low)

`self.cluster_results = cluster_results` in `DataReportAnalyzer.analysis_method`
set an instance attribute that was never read by any method or by the FLAME
framework. Removed.

#### Duplicate PDF sizing constants removed (Low)

`_MAX_W_IN`, `_MAX_H_IN`, and `_MARGIN_IN` were defined identically twice in
the hub file (once for the local report builder, once for the global report
builder). The second definition silently overwrote the first. Consolidated to
a single definition.

#### `stats` loop variable shadowed `scipy.stats` module name (Low)

In `_aggregation_method_impl`, the inner loop variable `stats` shadowed the
module-level `stats` alias for `scipy.stats`. Renamed to `col_stats` across
all three aggregation loops (numeric, categorical, temporal).

#### `posthoc_test` wired up in hub analyzer (previously missing)

The hub's `_analysis_method_impl` called `one_way_group_comparison` for 3+
group outcomes but never followed up with `posthoc_test`, leaving post-hoc
heatmaps and CSVs absent from hub reports even when the omnibus test was
significant. Now matches the local pipeline: when omnibus p < 0.05, posthoc
tests run; results are saved as `posthoc_<var>_vs_<outcome>.png/.csv` and
encoded for transport. `save_posthoc_heatmap()` and `_posthoc_to_pvalue_matrix()`
were also ported into the hub file.

#### 7 dead functions removed from `hub_entrypoint_10.py`

The following functions were defined but never called anywhere in the hub file:

| Function | Reason removed |
|---|---|
| `two_way_anova` | Unused regression-adjacent helper; no callers |
| `_validate_regression_inputs` | Part of unused regression pipeline |
| `_prepare_regression_data` | Part of unused regression pipeline |
| `_select_regression_model` | Part of unused regression pipeline |
| `_fit_regression_model` | Part of unused regression pipeline |
| `_build_regression_result` | Part of unused regression pipeline |
| `check_duplicates` | Utility with no callers |

`posthoc_test` was retained and wired up (see above). All removed functions
remain in the local `inferential_analysis.py` where some are used.

#### Identifier detection: value-based check for `patient_NNN` pattern

`detect_id_column()` (both `compute_statistics.py` and `hub_entrypoint_10.py`)
gains a value-based check: if the first 50 non-null values in an object column
all match the pattern `^(patient|pat|subject|sub|person|record|case|participant)[_-]?\d+$`
(case-insensitive), the column is classified as an identifier. This correctly
detects the `Unnamed: 0` column in dataset1 (values: `patient_001`,
`patient_002`, …) without relying on the column name or the bare uniqueness
heuristic.

#### Categorical distributions: threshold changed to `>= 2` distinct values

`save_categorical_distributions()` previously filtered to columns with `> 2`
distinct values (multi-category only), excluding binary clinical flags like
ICU admission (0/1), comorbidities, and symptom indicators. The threshold is
now `>= 2`: columns with at least 2 distinct non-null values are shown, while
single-value constant columns (which have no distribution to show) are excluded.
Applied to both local (`local_descriptive_plots.py`) and hub (local and
federated sections).

#### `patient_series = None` when no identifier column found

When no identifier column is detected, `patient_series` was set to
`pd.Series(dtype="object")` — an empty Series with a mismatched index. This
caused `compute_temporal_statistics()` to raise `ValueError: patient_series
must have the same index as temporal_df`. Fixed to `None` in both `analyze.py`
and `hub_entrypoint_10.py`; the temporal statistics function already handles
`None` correctly.

---

### MCA x-axis and `is_binary` safeguard (2026-06)

#### MCA explained inertia plot — x-axis tick spacing

When a dataset has many MCA dimensions (e.g. 109 on dataset1), the x-axis
ticks `1, 2, 3, …, 109` in `save_explained_inertia_plot()` and the
`save_mca_overview()` subplot were overlapping and unreadable.

A step is now computed as `step = max(1, n // 12)` so at most ~12 tick marks
appear regardless of the number of dimensions.  Both the standalone
`mca_explained_inertia.png` and the "Explained Inertia" panel inside
`mca_overview.png` apply the same fix.

#### `is_binary` — sparse-column safeguard

`is_binary()` in `compute_statistics.py` previously classified any numeric
column whose non-null values were a subset of `{0, 1}` as binary.  A
continuous column with only 2 non-null observations (e.g. a rarely-filled
vaccination date stored as 0/1) would be misclassified.

Two complementary guards were added:

1. **Minimum count** (`_BINARY_MIN_COUNT = 3`): a column must have at least
   3 non-null observations to be classified as binary on value evidence alone.

2. **Medical keyword list** (`_MEDICAL_BINARY_KEYWORDS`): ~70 clinical/medical
   terms (comorbidities, symptoms, treatments, hospital events, …). If the
   column name contains any of these tokens the minimum-count guard is waived,
   because a sparse `icu` or `cardiovasc` column is almost certainly a
   binary clinical flag even if only 1–2 patients have a recorded value.

The function signature gains a `column_name=""` parameter; `detect_column_types()`
passes `column_name=col` at every call site.  All 144 binary columns in
dataset1 (EUCARE Covid cohort) are preserved after the change.

---

### P-value display and summary.json fixes (2026-06)

#### P-value formatting in PDF tables

Previously, `create_table()` in `report_utils.py` applied `round(3)` to all
float columns uniformly.  Any p-value below 0.001 (e.g. `1.74e-137`) was
collapsed to `0.0` and displayed as `0.000` in the table — factually wrong
and potentially alarming (a reader might think the test produced a literal
zero probability).

P-value columns (`p_value`, `p_adj`, `p-val`, `p-unc`, `pval`) are now
formatted **before** the general rounding pass, using the standard medical
journal convention:

| Raw value | Displayed as |
|---|---|
| Exactly `0.0` | `0` |
| `0 < p < 0.001` | `< 0.001` |
| `p ≥ 0.001` | 3 decimal places, e.g. `0.049` |

All other float columns (means, standard deviations, effect sizes, etc.)
continue to be rounded to 3 decimal places as before.

A plain-language note was added to the section 5.1 narrative in the local
report explaining that `< 0.001` is **not zero** — it means the probability
of the result being due to chance is less than 1 in 1000.

#### `summary.json` — dynamic outcome column key

`generate_json_summary.py` previously looked for
`inferential/comparisons_by_mortality.csv` by name.  The actual filename is
dynamic: `comparisons_by_{outcome_col}.csv` (e.g.
`comparisons_by_mortality_status.csv` on dataset5).  The builder now finds
the file by glob at runtime.  The key in `summary.json` is renamed from
`comparisons_by_mortality` → **`comparisons_by_outcome`** to be dataset-agnostic.

---

### Codebase audit fixes (2026-06)

A targeted audit of the codebase produced a prioritised fix list; the
following issues were addressed.

#### C-1 — Duplicate federated categorical CSV write

`DataReportAggregator.aggregate()` in `data_report/analyze.py` wrote the
federated categorical statistics CSV **twice** in a row.  The first write
(`pd.DataFrame.from_dict(…, orient="index")`) produced a misaligned file (row
index stored as an unnamed column, feature names lost as the column header).
The correct write (building rows from the dict and calling
`result.to_csv(…, index=False)`) immediately followed and overwrote the bad
file.  The duplicate first write was removed.

#### C-2 — Unguarded `int()` on `node_id = "unknown"`

`analysis_method()` derives `node_number` from the FLAME node ID via
`int(node_id.split("_")[-1]) + 1`.  When the StarModel runner cannot resolve
a node ID, it falls back to the sentinel string `"unknown"`, which causes
`int("unknown")` to raise a `ValueError` at runtime.  A guard identical to
the one already present earlier in the same method was added:

```python
if node_id != "unknown":
    node_index = int(node_id.split("_")[-1])
    node_number = node_index + 1
else:
    node_number = 0
```

#### H-1 — Dead `anova` branch

`one_way_group_comparison()` selects its method as either `"welch"` or
`"kruskal"` — never `"anova"`.  The `if method == "anova": stat, p =
stats.f_oneway(…)` branch was therefore unreachable.  The same dead branch
existed in `posthoc_test()` (Tukey HSD path).  Both were removed.

#### H-3 — Dead functions removed from `inferential_analysis.py`

Eight functions with zero call-site references were removed:

| Function | Reason |
|---|---|
| `two_way_anova` | Depended on optional statsmodels `ols`; never called |
| `compare_multiple_groups` | Wrapper around `one_way_group_comparison`; replaced by direct calls |
| `correlation_between_two_variables` | Replaced by private `_auto_correlation` used only within `screen_associations` |
| `categorical_association` | Replaced by private `_chi2_association` used only within `screen_associations` |
| `_validate_regression_inputs`, `_prepare_regression_data`, `_select_regression_model`, `_fit_regression_model`, `_build_regression_result`, `regression` | Full regression pipeline; unused in the running pipeline |
| `check_duplicates` | Utility with no callers |
| `detect_dataset_type` | High-level dataset classifier; unused after pipeline refactor |
| `analyze_cross_sectional` | Dataset-type-specific analysis; unused after pipeline refactor |

The logic from `correlation_between_two_variables` and
`categorical_association` that is still needed by `screen_associations` was
extracted into two private helpers (`_auto_correlation`,
`_chi2_association`) defined in the same module.

The unused optional-dependency block (`import statsmodels.api as sm`, `from
statsmodels.formula.api import ols`, `from statsmodels.stats.multicomp import
pairwise_tukeyhsd`) was removed from the module header; only
`statsmodels.stats.multitest.multipletests` (needed by `posthoc_test`) is
imported.

#### L-2 — Missing space around `=` in `analyze.py`

`categorical_statistics= compute_categorical_statistics(…)` → corrected to
`categorical_statistics = compute_categorical_statistics(…)`.

#### L-3 — New tests

`tests/test_inferential_analysis.py` was updated:

- Removed five test classes that tested the deleted functions
  (`TestRegressionValidation`, `TestPrepareRegressionData`,
  `TestSelectRegressionModel`, `TestRegressionEndToEnd`,
  `TestDetectDatasetType`).
- Updated the NaN-FDR regression test to monkeypatch `_auto_correlation`
  instead of the deleted `correlation_between_two_variables`.
- Added `TestPosthocTest` — six tests covering the welch and kruskal paths
  of `posthoc_test()`: return shape, matrix symmetry, NaN diagonal,
  well-separated p-values, and unsupported method error.
- Added `TestDataset5Structure` — five tests verifying the structure of
  the generated dataset5 CSVs (four groups, correct group names, ≥ 20 rows
  per group, expected columns, three nodes present).
- Added `TestReportFallbackMessages` — two tests checking that
  `_build_cross_variable_section()` emits the correct fallback Note blocks
  when no 3+-group plots exist and when no post-hoc plots exist.

Total test count: **300 tests, 0 failures**.

#### Unused import removed

`import seaborn as sns` in `data_report/analyze.py` was unused and not
declared in `requirements.txt`; it was removed.

---

## Recent changes

### Table display polish (2026-06)

Two small readability fixes applied to `create_table()` in `report_utils.py`,
so both local and global reports benefit automatically.

#### Float rounding

All `float` columns are rounded to **3 decimal places** before rendering.
Previously only the pairwise-associations table was rounded; columns such as
`top cat % (local)`, `top cat % (global)`, and `class imbalance ratio` in the
categorical summary could display values like `26.400000000000002`.
Integer columns are unaffected.

#### System-value humanization

Cell values in system-generated descriptor columns now have underscores
replaced with spaces for readability.  Dataset-originated values (feature
names, category labels, group names) are left unchanged.

| Column | Example before | Example after |
|---|---|---|
| `vs_global` | `above_average` | `above average` |
| `vs_global` | `below_average` | `below average` |
| `test` | `student_ttest` | `student ttest` |
| `test` | `welch_ttest` | `welch ttest` |
| `effect_size_metric` | `hedges_g` | `hedges g` |
| `effect_size_metric` | `rank_biserial` | `rank biserial` |
| `comparison` | `above_global` | `above global` |

The `availability` column already had a hand-written lookup dict
(`common_all` → `"common in all"`, etc.); that is unchanged.

---

### Multi-group inferential statistics and post-hoc tests (2026-06)

#### Overview

When an outcome column with **3 or more groups** is detected (e.g. a severity
score with Mild / Standard / Complex / Deceased outcomes), the pipeline now
runs a full one-way omnibus test followed by pairwise post-hoc tests for all
significant results.  These results appear in a new, clearly labelled
**Section 5.2 / 5.3** of the local PDF report.

#### Outcome detection

`detect_outcome_column()` uses `OUTCOME_KEYWORD_GROUPS` with priority-ordered
keyword matching.  The "mortality" keyword group has the highest priority, so
a four-group column named `mortality_status` is detected automatically and the
multi-group path is taken.  Requirements: at least 3 distinct groups, each
with ≥ 20 observations; at most 5 groups total.

#### One-way omnibus test (`one_way_group_comparison`)

Already implemented in `inferential_analysis.py`; now wired into `analyze.py`.
The test is selected automatically:
- **Welch's ANOVA** when each group passes the Shapiro-Wilk normality check.
- **Kruskal-Wallis** when any group is skewed or has unequal variance.

#### Post-hoc pairwise tests (`posthoc_test`)

When the omnibus p-value is < 0.05, `posthoc_test()` runs pairwise follow-up
tests to identify *which* group pairs differ:

| Omnibus test | Post-hoc test |
|---|---|
| Welch's ANOVA | Games-Howell |
| Kruskal-Wallis | Pairwise Mann-Whitney U + Holm-Bonferroni correction |
| One-way ANOVA (equal variance) | Tukey HSD |

The Kruskal post-hoc uses only `scipy.stats.mannwhitneyu` and
`statsmodels.stats.multitest.multipletests` — **no new dependencies**
(drop-in replacement for `scikit-posthocs` Dunn test, avoiding Docker
image changes).

#### Post-hoc heatmap (`save_posthoc_heatmap`)

`inferential_plots.py` gains two new functions:
- `_posthoc_to_pvalue_matrix()` — normalizes the three possible output shapes
  from `posthoc_test` (welch long-form A/B/pval, anova long-form
  group1/group2/p-adj, kruskal square matrix) into a single symmetric
  group × group p-value matrix.
- `save_posthoc_heatmap()` — renders the matrix as a Blues_r heatmap (0–1
  scale, dark = low p-value = significant difference).  Saved as
  `inferential/comparisons/posthoc_<var>_vs_<outcome>.png`; the companion
  CSV is `posthoc_<var>_vs_<outcome>.csv`.

Heatmaps are picked up by `add_plots_from_dir` automatically — no changes to
`section_definitions.py` were needed.

#### Dataset 5 (`data/dataset5/`)

A new 3-node synthetic hospital dataset designed to exercise the full
multi-group pipeline.  Identical column structure to dataset4 except:
- `mortality` (binary 0/1) is replaced by `mortality_status` (4 ordered
  categories: Mild_Recovery / Standard_Recovery / Complex_Recovery / Deceased).
- Group membership is derived from a continuous `severity_score` using fixed
  thresholds (−0.5, 0.5, 1.5), guaranteeing ≥ 20 patients per group on every
  node even with ±0.4 severity shifts between nodes.
- Severity-driven variables (heart_rate, temperature_c, wbc_count, systolic_bp,
  glucose_mg_dl, crp_level, age, HbA1c, LDL, diastolic_bp, length_of_stay)
  show monotonically increasing means across the four groups.
- Non-severity variables (height_cm, weight_kg, bmi) show no pattern — a
  sanity-check that the generator is correct.

Verified output: node1 groups = {Standard_Recovery: 155, Mild_Recovery: 144,
Complex_Recovery: 110, Deceased: 41}; 11 post-hoc heatmaps on node1 (10 on
node2, node3); all omnibus p-values < 1e-5 for severity-driven variables.

#### Report section 5 restructuring

`generate_local_report.py` — `_build_cross_variable_section()` now has three
explicit subsections, each preceded by a plain-language **Note** box:

| Subsection | Content |
|---|---|
| **5.1 Pairwise Associations** | Two-group tests (t-test / Mann-Whitney U). Explains association screening heatmap, significance brackets (*** / ** / * / n.s.), and effect sizes (Hedges' g, rank-biserial r). |
| **5.2 Multi-Group Outcome Comparisons (3+ Groups)** | Omnibus boxplots (Welch ANOVA or Kruskal-Wallis). Explains test selection and what non-significant omnibus results mean. Shows a **"No outcome column with 3 or more groups detected"** Note if the dataset has only a binary outcome. |
| **5.3 Post-Hoc Pairwise Tests** | Post-hoc p-value heatmaps. Explains how to read the heatmap: darker blue = lower p-value = more significant difference; blank diagonal; near 0.000 = very strong evidence; near 1.000 = no difference. Shows a **"No significant multi-group associations found"** Note if all omnibus tests were non-significant. Subsection only appears when 3+ group comparisons exist. |

---

### Report readability and usability overhaul (2026-06)

A large batch of fixes across the pipeline was applied after running on
datasets 1 and 4 and a German-language dataset. Changes are organized by
category below.

#### A — Section headers
- Local and global PDF reports: "Numeric Variables" → **"Numeric Section"**,
  "Categorical Variables" → **"Categorical Section"**,
  "Temporal Variables" → **"Temporal Section"**.  
  "Cross-Variable Associations" is unchanged.

#### B — Inferential statistics
- `compare_two_groups()` and `one_way_group_comparison()` in
  `inferential_analysis.py`: removed the `method="auto"` parameter; method
  selection (Shapiro-Wilk → Mann-Whitney U or t-test) is now unconditional and
  not exposed as a user option.

#### C — Data quality / missing value charts
- `save_missing_bar()`: adds a subtitle showing column count ("Showing N of M
  columns"); when truncating to 50, sorts by ascending completeness so the
  worst-quality columns appear first.
- `save_missing_heatmap()`: returns `False` without writing a file when the
  dataframe has no missing values; the report builder shows a narrative message
  instead of a blank chart.
- `_build_overview_section()` in local report:
  - **Short mode**: shows only `missingno_bar.png`.
  - **Full mode**: shows split `missing_values_by_column_*.png` charts +
    `missingno_heatmap.png`; no `missingno_bar.png`.

#### D — Summary tables
- `create_table()` in `report_utils.py`: column headers have underscores
  replaced with spaces for display; the `availability` column cell values
  (`common_all`, `unique_local`, …) are rendered as human-readable text
  ("common in all", "unique to this node", …) while the CSV values are
  unchanged.
- `add_categorical_comparison()`: renamed `top_category_share` →
  **"top cat % (local)"** and `top_category_share_global` →
  **"top cat % (global)"**; an explanatory narrative was added above the table.
- Pairwise association table: all numeric values are now rounded to 3 decimal
  places.

#### E — Visualizations
- **PCA loadings biplot**: when there are more than 20 features, only the top
  20 by loading magnitude are shown; title says "— top 20 features".
- **MCA column map** (`save_mca_column_map()`): when there are more than 20
  category labels, only the top 20 by distance from the origin are annotated;
  title notes "— top 20 of N categories shown".
- **MCA column maps are now batched**: `save_mca_outputs()` generates
  `mca_column_map_batch_01.png`, `_02.png`, … (5 source variables per chart)
  instead of one combined image.
- **Correlation / TOM heatmaps**: cell annotations are suppressed when the
  matrix has more than 15 rows/columns (prevents unreadable overlapping numbers).
- **Cluster histograms**: split into batches of 6 variables per image when a
  cluster exceeds 6 variables; files named
  `cluster_N_histograms_01.png`, `_02.png`, …
- **Categorical distributions** (new, local): `save_categorical_distributions()`
  in `local_descriptive_plots.py` produces bar charts for non-binary
  multi-category variables only (> 2 distinct values), batched 6 per image,
  named `categorical_distributions_01.png`, etc. Called from `analysis_method`
  while the raw dataframe is available.
- **Categorical distributions** (federated): `save_federated_categorical_distributions()`
  now produces the same batched output instead of a single combined grid.
- **PCA/MCA short vs. full mode**: short mode shows only the overview panel;
  full mode shows the individual plots and excludes the overview panel.

#### F — Section definitions
- `LOCAL_PCA.short_plot` → `"pca_overview.png"`;
  `LOCAL_MCA.short_plot` → `"mca_overview.png"`.

#### I — Identifier column detection
- `detect_id_column()` in `compute_statistics.py`: expanded `strong_keywords`
  to include name columns (`name`, `surname`, `first_name`, `last_name`,
  `family_name`, `firstname`, `lastname`, `familyname`, `patient_name`,
  `patientname`) and German equivalents (`vorname`, `nachname`, `familienname`,
  `patientenname`), plus contact fields (`telefon`, `phone`, `email`,
  `adresse`, `address`, `postcode`, `zip`, `ssn`, `dob`, `birthdate`,
  `geburtsdatum`).
- `analysis_method()` in `analyze.py`: **all** identifier columns are now
  detected (not just the first). All are excluded from column types, statistics,
  PCA, MCA, clustering, and inferential analysis.
- `total_values` is now computed from analytical columns only (excludes
  identifier columns).
- The "Features" row in the local overview now reads:
  **"X analytical columns + Y identifier columns detected (col1, col2)"**.

#### J — Explanatory narratives and glossary
- Plain-language explanation boxes (`NarrativeMessage`) were added before each
  major section: TOM clustering, PCA, MCA, inferential statistics, and the
  categorical summary table (explaining "top cat %").
- A **28-term medical/statistical glossary appendix** was added to both local
  and global PDF reports. Terms include p-value, FDR, effect sizes (Cohen's d,
  Hedges' g, Cramér's V), PCA, MCA, TOM, IQR, SD, skewness, kurtosis, and
  common hospital/clinical abbreviations.

#### Bug fixes and code cleanup
- Sex distribution bar chart: replaced inline `plt.bar` with
  `save_sex_distribution()` (uses palette per category).
- Age distribution: replaced inline `plt.bar` with `save_age_distribution()`
  (uses the `histogram_from_bins` primitive).
- Data type pie chart: replaced inline `plt.pie` with
  `save_data_type_distribution()` (uses the `pie_chart` primitive with correct
  title padding).
- Fixed pre-existing double-percent bug in `patient_contribution` display
  (`33.333%%` → `33.333%`).

#### Fix: `detect_id_column` false positives on drug names and sparse columns

Caught by running on dataset1 (EUCARE Covid cohort, 213 columns). Two root
causes:

1. **Substring matching** — `"id" in name` fired on `paxlovid`,
   `glucocorticoids`, and `other_covid_treatment` (via the substring `"covid"`).
   Replaced with token-level matching: the column name is split on
   non-alphanumeric characters and the resulting tokens are intersected against
   the keyword set as whole words.

2. **No minimum fill requirement for the uniqueness heuristic** — ultra-sparse
   columns such as `vac4_date` (2 non-null out of 2798 rows) trivially reach
   `uniqueness_ratio = 1.0` and were incorrectly flagged. A minimum of
   `max(20, 10 % of rows)` non-null values is now required before the
   uniqueness check applies.

3. **Datetime columns now unconditionally skipped** — any column already
   converted to `datetime64` is temporal data, never an identifier.

After the fix, dataset1 correctly identifies only `unnamed:_0` (the bare row
index) as an identifier; all 212 treatment flags, symptom indicators, and
visit-date columns are preserved for analysis.

Running on dataset2 exposed a further ordering bug: ID detection originally
ran before the date-conversion loop, so `discharge_date` and `therapy_session1`
were still `object` dtype when `detect_id_column` fired and were flagged via
the uniqueness heuristic. Fixed by moving ID detection to after the
date-conversion loop, so the `datetime64` guard correctly skips all temporal
columns before uniqueness is checked.

---

### Previous changes

- Added `generate_reports/generate_json_summary.py`, producing a
  `summary.json` per node and one for the federated results -- a
  machine-readable counterpart to the PDF reports, sourced from the same
  summary-table CSVs. Wired into `dr-analyze` (`data_report/cli.py`)
  alongside the PDF report generation.
- Fixed `is_binary` (`data_report/statistical_analysis/local/compute_statistics.py`):
  semantic checks for `yes`/`no`, `y`/`n`, `true`/`false`, etc. were
  unreachable whenever a column's values weren't numeric-convertible; and a
  prior "any column with <=2 distinct numeric values is binary" rule
  misclassified sparse numeric columns (e.g. a continuous column with only 2
  non-null values). Restored the `{0, 1}`-subset check for numeric binary
  columns.
- Fixed two broken imports that were failing test collection:
  `tests/test_report_utils.py` imported a removed `summarise_overview`
  helper, and `test_clustering_integration.py` imported a non-existent
  `save_cluster_outputs` from `data_report.analyze` (it's defined in
  `data_report.generate_figures.clustering_plots`).
