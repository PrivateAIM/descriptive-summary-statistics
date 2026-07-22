# Report Generation Refactoring Plan (v2)

This supersedes the v1 plan. It folds in: section restructuring (PCA/MCA as
subsections, hybrid inferential placement), report-time comparison columns,
a ranking-based short-report strategy, a generalized narrative/warning
system, and a privacy & data governance notice — all without requiring
changes to the existing analysis CSVs or `dr-analyze` pipeline.

---

## 1. Current State & Problems

Both `generate_local_report.py` and `generate_global_report.py` are
essentially identical flat scripts.

| Problem | Detail |
|---|---|
| **Flat scripts, not functions** | Global state (`styles` at module level), hardcoded paths — not importable, not testable |
| **Missing sections** | Only cover overview / numeric / categorical / temporal. Miss clustering, PCA, MCA, inferential, comparison entirely |
| **Blind glob** | `add_plots` globs `*.png` with no ordering, no context, no narrative text |
| **Fixed image size** | Every plot forced to 5 × 3.5 inch regardless of its actual aspect ratio |
| **Duplicate code** | Both files copy the same helpers (`create_table`, `add_plots`, `make_header`, etc.) |
| **No hierarchy** | Clustering output lives in nested subdirectories the current code never visits |

---

## 2. Two-Version Strategy

For every dataset (local node + global federated), generate **two PDFs**:

- **Short version** (~4–8 pages): stakeholders, first-pass review. One
  narrative sentence per subsection, key charts only, ranked top-N tables.
- **Long/full version**: analysts, archiving, audit. Everything in short
  plus all per-variable plots, full cluster breakdowns, full inferential
  suite.

Both share the same builder — a `mode: Literal["short", "full"]` parameter
controls inclusion. No code duplication between the two.

---

## 3. Report Structure

### 3.1 Local Report — Section Registry

Top-level sections, each with subsections where applicable. Order = order
of appearance in the PDF.

**0. Title Page + Data & Privacy Notice** (short + full)
- Title, node name, generation date
- Privacy notice block (see §5)

**1. Overview** (short + full)
- 1.1 Overview table (`overview.csv`) + data type distribution pie
- 1.2 Data quality: missing values by column (short: top 10, full: all),
  missingno bar + heatmap (full only)
- 1.3 Column availability across nodes — **chart only**
  (`column_availability.png`). No separate table here; per-column
  availability is shown instead as a column in each variable's own
  descriptive table (2.1, 3.1, 4.1 below).

**2. Numeric Variables** (short + full)
- 2.1 Descriptive statistics — `numeric_summary.csv`, which already
  includes an `availability` column (`common_all` / `common_partial` /
  `unique_local`, computed during analysis), plus a `vs_global` comparison
  column added at report time (short: ranked top ≤10 by `|local − global|`
  deviation; full: complete table, both columns shown)
- 2.2 Distributions — age distribution (if present), histograms (full only),
  boxplots (full only)
- 2.3 Clustering *(subsection)* — short: one narrative sentence from
  `*_clusters.csv`; full: dendrograms, TOM heatmaps, per-cluster
  histograms/boxplots/violins
- 2.4 PCA *(subsection)* — short: explained variance chart; full: all 7
  PCA outputs (variance, scatter 2D/3D, loadings, scatter matrix, overview)
- 2.5 Correlations *(subsection, numeric↔numeric inferential)* — short:
  correlation heatmap + top ≤10 significant pairs; full: full pairs table

**3. Categorical Variables** (short + full)
- 3.1 Descriptive statistics — `categorical_summary.csv` (already includes
  `availability`) + `vs_global` comparison column added at report time
  (short: ranked top ≤10 by distribution skew / involvement in significant
  associations; full: complete table, both columns shown)
- 3.2 Distributions — sex distribution (if present), bar charts (short: top
  N from ranking above; full: all), stacked bar charts (full only)
- 3.3 Clustering *(subsection)* — short: one narrative sentence; full: all
  cluster plots
- 3.4 MCA *(subsection)* — short: explained inertia chart; full: all 7 MCA
  outputs
- 3.5 Associations *(subsection, categorical↔categorical, Cramér's V)* —
  short: bar chart + top ≤10 pairs; full: full pairs table

**4. Temporal Variables** (short + full)
- 4.1 Descriptive statistics — temporal summary (already includes
  `availability`) + `vs_global` comparison (activity level vs. global
  activity level)
- 4.2 Line charts — short: top N by `|trend slope|`; full: all columns
- 4.3 Clustering *(subsection, if temporal clusters exist)* — short:
  narrative; full: all plots

**5. Cross-Variable Associations & Outcome Comparisons** (short + full)
*(numeric↔categorical inferential — group comparisons against the detected
outcome/target, plus the overall screening view)*
- Association screening / volcano plot (short + full)
- Significant associations table (short: top ≤10 by effect size; full: all)
- Per-comparison group plots (full only)

> Temporal gets no inferential subsection — there are no
> temporal↔temporal or temporal↔outcome tests in the current pipeline.

### 3.2 Global Report — Section Registry

**0. Title Page + Data & Privacy Notice** (short + full)

**1. Overview** (short + full)
- Federated overview table, data type distribution pie, data quality summary

**2. Numeric Variables** (short + full)
- Federated means ± std bar chart
- Federated numeric stats table (short: condensed; full: complete)
- Age distribution (federated, if present)

**3. Categorical Variables** (short + full)
- Federated categorical distributions (short: top 3 columns; full: all)
- Sex distribution (federated, if present)

**4. Temporal Variables** (short + full)
- Federated trend summary chart
- Narrative note: *"Federated inferential analysis is currently limited to
  trend-slope estimation; per-pair correlation/association tests are
  computed locally only and are not aggregated."*
- Per-column line charts (full only)

**No standalone Inferential section** (federated inferential capability is
limited to the trend summary above, which stays in Temporal). **No
Clustering section** (`federated_results/` has no clustering output and no
raw data is available federally to compute one — flagged as a possible
future enhancement using cross-node co-occurrence of local `*_clusters.csv`
groupings, not part of this refactor).

---

## 4. Comparison Columns (report-time, not baked into CSVs)

For each local descriptive table (numeric, categorical, temporal), a
`vs_global` column is computed **when the report is generated**, by joining
the local CSV against the corresponding federated CSV:

```python
def add_comparison_column(local_df, global_df, key_col, value_col, threshold=0.1):
    merged = local_df.merge(
        global_df[[key_col, value_col]], on=key_col, suffixes=("", "_global")
    )
    rel_diff = (merged[value_col] - merged[f"{value_col}_global"]) / merged[f"{value_col}_global"]
    merged["vs_global"] = np.select(
        [rel_diff > threshold, rel_diff < -threshold],
        ["above_average", "below_average"],
        default="similar",
    )
    return merged
```

Reused for:
- **Numeric**: local mean vs. global mean
- **Categorical**: top-category share vs. global share
- **Temporal**: activity level (e.g. records per period) vs. global activity level

**Why report-time, not in the analysis CSVs:**
1. Local CSVs are written during `analysis_method`, *before* federated
   aggregation completes — global stats don't exist yet at that point.
2. Keeps the CSV exports stable, type-consistent data — not mixed with
   report-layer interpretation/thresholds.
3. Faster iteration — threshold/logic tweaks don't require re-running the
   federated pipeline.

**Note on `availability`**: local descriptive CSVs already contain a
per-row `availability` column (`common_all` / `common_partial` /
`unique_local`, computed during analysis from `compute_column_distribution`
/ `classify_local_columns`). This is displayed as-is alongside `vs_global`
in the descriptive tables — no report-time computation needed for it.

**Optional, non-default export**: if a user wants the comparison persisted
as data (not just shown in the PDF), report generation can optionally also
write `numeric_summary_with_comparison.csv` (and categorical/temporal
equivalents) alongside the PDF. Off by default; enabled via a flag, e.g.
`generate_local_report(..., export_comparison_csv=False)`.

---

## 5. Privacy & Data Governance Notice

A short block on the title page (both report types, both modes) — purely
descriptive, computed from data already available, **no suppression or
masking of values**.

**Local report notice includes:**
- Provenance: *"Generated from this node's local data only. No row-level
  data was transmitted to other nodes or to a central server."*
- Identifier exclusion: *"Identifier columns (e.g. Patient_ID) are excluded
  from all tables and plots in this report."* (uses existing
  `detect_id_column`)
- **Minimum group size indicator**: smallest category/group count across
  all categorical breakdowns and outcome-comparison groups in this report,
  e.g. *"Smallest displayed group size: 3 (in `smoker` distribution)."*
- **Small-group count**: *"X of Y categorical breakdowns contain at least
  one group below the reporting threshold (k=5)."*

**Global report notice includes:**
- Provenance: *"Contains only aggregated statistics computed across N
  nodes. No row-level/patient data is shared or displayed."*
- Aggregation methodology: *"Means, standard deviations, and category
  counts are computed via federated aggregation; trend slopes via federated
  OLS on per-period counts."*
- Identifier exclusion (same as local)

**Section-level small-group warnings** (local report only): wherever a
categorical breakdown, outcome-comparison group, or cluster has n below the
threshold (k=5, configurable), attach a `warning`-level narrative, e.g.:
*"`smoker = unknown` has only 3 records — interpret with caution; small
groups combined with other displayed variables may carry re-identification
risk."* This reuses the narrative/warning mechanism in §7 — no new
infrastructure.

**Comparison-column caveat**: if the federated `n_nodes` is small (< 5), a
single conditional warning is attached the first time a `vs_global` column
appears: *"With only N participating nodes, 'above/below average'
comparisons may indirectly reveal information about other individual
nodes."*

**Explicitly out of scope for this refactor** (documented as future work):
- Small-cell **suppression** (masking counts below k as `<k`) — would
  change displayed values, requires its own design/decision
- Differential privacy noise disclosure — not applicable, no DP mechanism
  exists in the pipeline today

---

## 6. Short-Report Summarization Strategy

Short mode does **not** truncate by column position. Each truncatable
table/section is ranked, and only the top N are shown, with an `info`
narrative pointing to the full CSV:

| Section | Ranking criterion |
|---|---|
| Numeric descriptive | `\|local − global\|` relative deviation (reuses `vs_global`) |
| Categorical descriptive | distribution skew/entropy, or involvement in significant associations |
| Clustering | most coherent clusters (highest avg within-cluster correlation/TOM) |
| Inferential (cross-variable) | already ranked by effect size / significance |
| Temporal | `\|trend slope\|` or R² |

Every truncated table/section gets:
> *"Showing the N most [criterion] of M total [items] (ranked by
> [method]). Full results: `<filename>.csv`."*

---

## 7. Narrative & Warning System

```python
@dataclass
class NarrativeMessage:
    level: Literal["info", "warning", "insight"]
    text: str
```

- **info** — truncation/navigation notes (§6), section-level summaries in
  short mode (e.g. *"2 numeric clusters: cluster 1 groups weight + waist."*)
- **warning** — small sample sizes (§5), degenerate tests, high
  missingness
- **insight** — headline findings (e.g. *"Strongest association: weight ×
  waist (r = 0.81, p < 0.001)."*)

Each level gets distinct visual styling (e.g. colored left-border box).
Rendered via a single helper, `render_narrative(elements, msg)`.

---

## 8. File Structure

```
generate_reports/
  report_utils.py            shared helpers (see §9)
  section_definitions.py      section registries for local + global (see §10)
  generate_local_report.py    generate_local_report(node_dir, output_dir, mode) -> Path
  generate_global_report.py   generate_global_report(results_dir, output_dir, mode) -> Path
```

No new dependencies — stays on ReportLab + pandas + Pillow + numpy
(`add_comparison_column`).

---

## 9. `report_utils.py` — Shared Helpers

- **`auto_fit_image(path, max_width, max_height)`** — Pillow-based, computes
  largest size preserving aspect ratio. Replaces fixed `5*inch x 3.5*inch`.
- **`create_table(df, available_width, max_rows=None)`** — column widths
  proportional to content; `max_rows` for short-mode truncation.
- **`make_header(section_title)`** — unchanged from current.
- **`render_narrative(elements, msg: NarrativeMessage)`** — styled box per
  level (§7).
- **`add_section_heading(elements, title, level=1)`**
- **`add_figure(elements, path, max_width, max_height, caption=None)`**
- **`add_plots_from_dir(elements, directory, max_width, max_height, only=None, exclude=None)`**
- **`add_comparison_column(local_df, global_df, key_col, value_col, threshold=0.1)`** (§4)
- **`build_privacy_notice(node_dir_or_results_dir, report_type)`** (§5) —
  computes min group size, small-group count, returns list of `Paragraph`
  elements for the title page
- **Narrative summarizers**: `summarise_overview`, `summarise_numeric`,
  `summarise_categorical`, `summarise_clustering`, `summarise_inferential`,
  `summarise_temporal` — each reads a CSV and returns a `NarrativeMessage`

---

## 10. `section_definitions.py` — Section Registry

```python
@dataclass
class SectionSpec:
    key: str
    title: str
    subdir: str | list[str]
    include_in: set[str]            # {"short"} | {"full"} | {"short", "full"}
    short_plots: list[str]
    all_plots_in_full: bool
    short_table: str | None         # CSV to load for short mode (ranked/truncated)
    full_table: str | None          # CSV to load for full mode
    comparison: dict | None         # {"key_col":..., "value_col":..., "global_csv":...} or None
    narrative_fn: Callable | None
    nested_subdirs: bool
    subsections: list["SectionSpec"] | None = None
```

Registries (`LOCAL_SECTIONS`, `GLOBAL_SECTIONS`) implement the structures in
§3.1 / §3.2 as nested `SectionSpec` trees — top-level sections (Overview,
Numeric, Categorical, Temporal, Cross-Variable Associations) with
`subsections` for Clustering / PCA / MCA / Correlations / Associations
where applicable.

---

## 11. Output File Naming

```
results/
  local_results/
    node1/
      local_report_node1_short.pdf
      local_report_node1_full.pdf
    node2/
      ...
  federated_results/
    global_report_short.pdf
    global_report_full.pdf
```

(Optional, if `export_comparison_csv=True`):
```
results/local_results/node1/
  numeric_summary_with_comparison.csv
  categorical_summary_with_comparison.csv
  temporal_summary_with_comparison.csv
```

---

## 12. Execution Plan

| Step | File(s) | What |
|---|---|---|
| **1** | `report_utils.py` (new) | `auto_fit_image`, `create_table`, `make_header`, `render_narrative`, `add_section_heading`, `add_figure`, `add_plots_from_dir`, `add_comparison_column`, `NarrativeMessage` |
| **2** | `report_utils.py` | `build_privacy_notice` + min-group-size / small-group detection |
| **3** | `report_utils.py` | Narrative summarizer functions (`summarise_*`) |
| **4** | `section_definitions.py` (new) | `SectionSpec` + `LOCAL_SECTIONS` / `GLOBAL_SECTIONS` registries per §3 |
| **5** | `generate_local_report.py` (rewrite) | `generate_local_report(node_dir, output_dir, mode="full") -> Path`; walks `LOCAL_SECTIONS`, applies comparison columns, ranking, narratives, privacy notice |
| **6** | `generate_local_report.py` | `__main__`: generate short + full for every node in `results/local_results/` |
| **7** | `generate_global_report.py` (rewrite) | `generate_global_report(results_dir, output_dir, mode="full") -> Path`; walks `GLOBAL_SECTIONS` |
| **8** | `generate_global_report.py` | `__main__`: generate short + full |
| **9** | tests | Unit tests for `report_utils` helpers (especially `add_comparison_column`, ranking, privacy-notice computations) and a smoke test that both report functions produce non-empty PDFs and return valid `Path`s |

CLI (`cli.py` / `dr-analyze`) stays unchanged for now, per earlier
agreement — these functions will be wired into a pipeline entry point
later.
