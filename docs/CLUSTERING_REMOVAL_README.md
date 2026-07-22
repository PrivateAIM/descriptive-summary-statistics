# Clustering Removal

## Rationale

The pipeline previously ran a WGCNA-style TOM (Topological Overlap Measure)
clustering step over numeric, binary, categorical, and temporal variables.
TOM itself is published (Zhang & Horvath, 2005) but was designed for gene
co-expression networks; applying it to generic clinical/tabular variables —
and using cluster co-membership as the gatekeeper for which pairs get a
formal significance test — is not an established, validated method in the
literature. It was removed for scientific-rigor reasons ahead of thesis
write-up.

The one place clustering was load-bearing (not just a report section) was
`screen_associations`: it used cluster membership as the *only* pair-selection
mechanism for numeric–numeric and categorical–categorical association
screening, with no all-pairs fallback. Removing clustering without a
replacement would have silently dropped that screening entirely.

**Replacement:** `screen_associations` now tests every numeric–numeric pair
and every (cardinality-limited) categorical–categorical pair directly,
correcting once across all tests with the existing Benjamini-Hochberg FDR
procedure — the same design already used for numeric–categorical pairs. This
was chosen over a raw-correlation/Cramér's-V threshold pre-filter because
thresholding a pair by the same statistic you're about to test it with is
circular (a selection-bias / "winner's curse" pattern); BH-FDR is specifically
designed to stay valid regardless of how many tests are run, so no pre-filter
is statistically necessary at this scale (~200 columns → tens of thousands of
cheap tests, seconds of compute).

## What changed

**Deleted:**
- `data_report/statistical_analysis/local/clustering.py` (TOM/WGCNA clusterer classes)
- `data_report/generate_figures/clustering_plots.py` (TOM heatmaps, dendrograms, per-cluster plots)
- `tests/test_clustering.py`, `test_clustering_integration.py`

**Reworked:**
- `data_report/statistical_analysis/local/inferential_analysis.py` —
  `screen_associations` no longer takes a `cluster_results` argument; pair
  generation for numeric–numeric and categorical–categorical uses
  `itertools.combinations` over all typed columns instead of within-cluster
  pairs. `_cluster_pairs` helper removed.
- `data_report/analyze.py` — `ClusteringManager` invocation, cluster-plot
  saving (both the df-dependent block in the analyzer and the
  `ClusterResult`-only block in the aggregator), the `cluster_results`/
  `clusters` keys in the analyzer's return dict, and the three
  `*_clustering` flags in `should_apply_reductions` are all removed. `pca`
  and `mca` flags are untouched (they were never gated by clustering, despite
  living in the same threshold dict).
- Report layer (`generate_reports/section_definitions.py`,
  `generate_local_report.py`, `report_utils.py`, `generate_json_summary.py`)
  — clustering subsections, `ClusteringSubsection` spec, `summarise_clustering`,
  the `clustering` key in `summary.json`, and the WGCNA/TOM/Dendrogram/
  Clustermap glossary entries are removed. Numeric/Categorical section
  subsections renumbered (PCA 2.3, Correlations 2.4, MCA 3.3, Associations
  3.4 — previously 2.4/2.5/3.4/3.5).
- `hub_entrypoint_10.py` (canonical single-file Hub deployment artifact) —
  every change above mirrored inline: the inlined clusterer classes, the
  inlined clustering-plot functions, the aggregator-side
  `_collect_cluster_plots` reconstruction function (this file serializes
  `ClusterResult` into a plain dict with expanded TOM/linkage arrays since
  raw DataFrames can't cross the analyzer→aggregator boundary — that whole
  path is now gone), the `scipy.cluster.hierarchy` lazy-import scaffolding
  (`linkage`/`fcluster`/`dendrogram`/`leaves_list`/`squareform` — confirmed
  unused elsewhere in the file before removing), and the report-layer
  duplicates of everything above.
- `hub_entrypoint_9.py` — **intentionally left untouched** (frozen historical
  snapshot, not a deploy target; user confirmed).
- `data_report/generate_figures/pca_plots.py`, `mca_plots.py`, `primitives.py`
  — stale docstring references to the now-deleted `ClusteringManager` cleaned
  up (in both the local multi-file project and the mirrored copies inside
  `hub_entrypoint_10.py`).
- `data_report/get_data/generate_synthetic_data.py` — module docstring no
  longer justifies its correlation structure via `ClusteringManager` (the
  correlated variable groups are unchanged and still exercise PCA/MCA and the
  new full-pairwise screening).
- Tests updated to match: `test_generate_json_summary.py`,
  `test_inferential_analysis.py` (full-pairwise behavior, `_cluster_pairs`
  tests removed), `test_new_modules.py` (`should_apply_reductions` now
  returns only `{"pca", "mca"}`), `test_report_utils.py`.

## Final review

An 8-angle review pass (correctness line-by-line scan, removed-behavior audit,
cross-file caller tracing, reuse, simplification, efficiency, altitude,
CLAUDE.md conventions) was run against the full diff. Five of eight angles
came back clean; three surfaced minor, low-risk cleanup that was applied:

- `tests/test_new_modules.py` — `_make_column_types` had an unused
  `n_temporal` parameter and dead temporal-column-generation loop left over
  after the temporal-clustering tests were deleted. Removed.
- `hub_entrypoint_10.py` — a `# Source: figures_clustering_plots.py` section
  header remained over `save_categorical_distributions_local`, an unrelated
  categorical-distributions helper that happened to be co-located in that
  source block. Retitled to reflect its actual origin
  (`figures_local_descriptive_plots.py`).
- `data_report/statistical_analysis/local/compute_statistics.py` (and its
  mirror in `hub_entrypoint_10.py`) — the `detect_column_types` comment
  explaining why empty columns are dropped referenced a "numeric TOM heatmap"
  and "clustering similarity matrices" that no longer exist. Reworded to
  describe the actual (non-clustering) reason.

One out-of-scope finding was surfaced and intentionally left alone: the
numeric–categorical loop in `screen_associations` recomputes
`df[group_col].nunique()` once per (numeric, categorical) pair rather than
once per categorical column. This is pre-existing code untouched by the
clustering removal (confirmed via `git diff` — zero changes to that section)
and unrelated to clustering, so per the instruction to only change
clustering-related code, it was left as-is rather than fixed here.

All 194 tests pass after the cleanup; `hub_entrypoint_10.py` and
`compute_statistics.py` re-verified importable.

## Verification performed

- `python -m pytest tests/` — 194 passed, 0 failed.
- Full pipeline run on `data/dataset1` via `run_local_sim.py` (2 nodes): completed
  with no errors, no `clustering/` directories produced, PDF report text
  scanned for "Clustering"/"TOM"/"WGCNA"/"Dendrogram"/"Clustermap" — 0
  occurrences, section numbering confirmed correct (2.3 PCA, 2.4
  Correlations, 3.3 MCA, 3.4 Associations).
- Full pipeline run on `data/dataset5` via `data_report.cli.analyze_main` (3
  nodes): completed with no errors, `summary.json` confirmed to have no
  `clustering` key, association screening confirmed full-pairwise (e.g. 21
  numeric columns → 210 = C(21,2) numeric–numeric pairs tested), same PDF
  text scan came back clean.
- Both runs log (and gracefully skip, via the pre-existing try/except) a
  handful of degenerate categorical pairs newly exposed by full-pairwise
  screening (e.g. a zero-variance column tested against another column,
  raising `chi2_contingency`'s "no data" error) that cluster-based selection
  happened not to pair before. This is expected, non-fatal, and was already
  handled by the existing per-pair error handling in `screen_associations`.
