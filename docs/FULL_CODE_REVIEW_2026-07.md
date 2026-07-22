# Full Code Review — hub_entrypoint_10.py and Local Implementation

A thorough correctness audit of the entire pipeline (not a diff review): the
full 7,367-line `hub_entrypoint_10.py` and all 38 files of the local
implementation (`data_report/`, `generate_reports/`), split across 10
parallel deep-review passes by subsystem (column detection & statistics,
inferential analysis, PCA/MCA, plotting/figures, federated/inferential
plots, report builders, JSON/comparison/data-loading, the analyzer, the
aggregator, and a dead-code sweep). Every finding below was independently
verified by direct code execution, not just static reading, before being
accepted.

## Bugs found and fixed

**1. Duplicate column names crash the entire pipeline.** `analyze.py`'s own
header normalization (`.str.strip().str.lower().str.replace(" ", "_")`) can
*create* a collision that didn't exist in the raw CSV — e.g. `"Blood
Pressure"` and `"blood_pressure"` both normalize to `blood_pressure`. Once
duplicated, `df[col]` returns a DataFrame instead of a Series and
`detect_column_types` crashes with `ValueError: truth value of a Series is
ambiguous`. This is not a contrived edge case — it's a byproduct of the
pipeline's own preprocessing step. Fixed in both files by deduplicating
column names (appending `_1`, `_2`, …) immediately after normalization.

**2. `is_binary` classified data differently in the Hub deployment than
locally.** The local `compute_statistics.py` requires either ≥3 valid
observations or a medical/binary keyword in the column name before
classifying a sparse numeric `{0,1}` column as binary — a deliberate
safeguard against a sparse continuous measurement coincidentally showing
only 0s and 1s. `hub_entrypoint_10.py`'s copy never received this safeguard
in an earlier round of fixes and classified any `{0,1}`-only column as
binary regardless of sample size or name. The same input dataset could
produce a different numeric/categorical split — and therefore different
descriptive statistics, PCA feature sets, and report sections — depending on
whether it ran locally or through the Hub. Ported the safeguard to hub.

**3. `n_total_values` used two different (and one wrong) formulas.** Local
`analyze.py` computed `total_rows × max(n_cols across nodes)`; the hub file
computed `Σ(n_rows × n_cols)` per node. These only agree when every node has
an identical schema. Verified with dataset5's actual heterogeneous
node schemas (46/43/40 columns): the old local formula gave 62,100; the
correct value is 57,840. This value feeds `total_missing_percentage`,
`completeness`, `usable_data_contribution`, and every per-node comparison
metric — a federation with differing node schemas (the norm for this
project, not the exception) was getting systematically wrong completeness
numbers. Made local match hub's already-correct formula.

**4. Hub-only doubled `%` in the Patients overview line.** `patient_contribution`
was stored as an already-`%`-suffixed string, then the display template
appended `%` again — e.g. `"45.678%% of all patients"`. Every field around
it (`completeness`, `total_value_contribution`, etc.) was stored as a plain
float; this one was the anomaly. Fixed to match the plain-float convention
used everywhere else (and used by the local project).

**5. Hub's small-federation privacy caveat had no floor.** `build_privacy_notice`
is documented to show a caveat only for federations under 5 nodes (where
above/below-average comparisons could indirectly reveal information about a
specific other node) — the local project correctly gates this
(`n_nodes < 5`); the hub copy dropped the `< 5` condition somewhere along the
way, so it fires for a federation of any size, contradicting its own
docstring.

**6. Federation-wide age histogram silently disappears if the first node
happens to lack a usable age column.** Both files took `age_edges` from
`analysis_results[0]` unconditionally. If that specific node has no valid
age data (but other nodes do), the whole federation's age chart vanishes —
even though any node's edges would work (the binning grid is fixed). Fixed
in both files to take the first *non-None* value across all nodes.

**7. Hub's nullity-correlation matrix could include identifier columns.**
Every other analysis step explicitly excludes detected identifier columns;
this one iterated `df.columns` directly. Only the *first* identifier gets
renamed/excluded from `column_types` — any additional (secondary) ID column
in a dataset would leak into the missing-value correlation matrix sent to
the aggregator. Fixed to exclude `id_columns` explicitly.

**8. `run_mca`'s new empty-column guard (added in the previous round of
fixes) didn't actually work for the general case.** The dtype-based
categorical filter ran *before* the entirely-missing-column check. An
all-NaN column inferred as `float64` (rather than `object`) by pandas — the
common case for a genuinely empty CSV column — got excluded by the dtype
filter first and raised `"non-categorical feature"` instead of being
silently dropped, contradicting the function's own docstring claim to mirror
`run_pca`. Verified directly: `pd.Series([nan]*30)` has dtype `float64`, not
`object`. Fixed in both files by checking for emptiness on the raw requested
columns *before* the dtype filter, and added a regression test that
specifically avoids the `dtype="object"` override my earlier test used
(which is why this had gone unnoticed).

**9. MCA category labels could be mis-attributed when one feature name is a
prefix of another.** `_variable_for_label` used first-match `startswith`
search; for feature names `["site", "site__region"]`, every category level of
`site__region` (e.g. `"site__region__x"`) resolved to `"site"` instead.
Fixed to pick the *longest* matching feature name in both `mca_plots.py` and
`hub_entrypoint_10.py`. The hub file also had a second, independent
reconstruction of this same logic in the aggregator (`lbl.split("__")[0]`,
used because the aggregator never had access to the real feature list) with
the identical failure mode — fixed by serializing the analyzer's actual MCA
feature-name list (`mca_feature_names`, new field) and having the aggregator
reuse the same corrected `_variable_for_label` logic instead of a naive
split, with a documented fallback for older cached results that predate this
field.

**10. Hub's short-mode PCA/MCA sections always showed a false message.**
`LOCAL_PCA`/`LOCAL_MCA` (both files' copies) pointed `short_plot` at
`pca_overview.png`/`mca_overview.png` — files the *local* project's
`save_pca_outputs`/`save_mca_outputs` genuinely produce, but which the hub's
live aggregator path never generates (it builds individual plots inline
instead, with no combined overview panel). Every hub-generated short-mode
PDF therefore showed *"PCA/MCA plots are available in the full version of
this report"* — which was also false, since full mode doesn't have an
overview file to show either. Repointed hub's copies to files it actually
always produces when PCA/MCA succeed (`pca_explained_variance.png`,
`mca_explained_inertia.png`), with a comment explaining why this
intentionally differs from the local project's config.

**11. Hub's live categorical-distribution plot had no cardinality cap.**
The local project's `count_plot(top_n=20)` truncates to the top 20 categories
by frequency; the hub's inline reconstruction (built from serialized
`category_counts`, not raw data) drew every single category with no cap —
a diagnosis-code-style column with 100+ distinct values would produce a
subplot with 100+ overlapping bars. Fixed to sort by count and cap at 20,
matching the local behavior.

**12. Federated categorical-distributions filter drifted between the two
files.** Local's `save_federated_categorical_distributions` (correctly,
per its own docstring) excludes binary columns (`> 2` non-null categories)
and filters out NaN keys before counting. Hub's inline federated version
used `>= 2` (including binary columns, duplicating what the summary table
already shows) and never filtered NaN keys (so a column with 1 real category
plus a missing-value bucket could incorrectly qualify as "multi-category").
Fixed to match.

**13. Local project's PCA/MCA short-mode fallback message was silently
dropped.** `_render_reduction_subsection` didn't capture
`add_heading_and_plots`'s return value, so it could never render *"plots are
available in the full version"* when a short-mode overview file was missing
(e.g. because `run_pca`/`run_mca` raised on too few usable columns after
dropping empty ones) — hub's version already had this. Ported to local.

**14. Local `generate_json_summary.py` didn't guard against `Inf` values.**
`_sanitize()` checked `math.isnan()` but not `math.isinf()` — an exact
recurrence of a bug the project's own changelog documents having fixed in
the hub file previously (`_json_sanitize`), which apparently never got
ported to (or regressed in) the local project's independent implementation.
An `Inf`/`-Inf` value (e.g. from a near-zero-variance ratio) would serialize
as the bare `Infinity` token, which is invalid per RFC 8259 and breaks
non-Python JSON consumers. Fixed to match hub's check, and removed a
now-fully-redundant trailing NaN check in the same function.

**15. Local `_csv_to_records` had no error handling.** Hub's equivalent
degrades gracefully (returns `[]`) if a mapped CSV is corrupt/genuinely
empty; local's raised uncaught, aborting the entire `summary.json` build for
one bad file. Added the same try/except.

## Verified, deliberately not auto-fixed (design decisions, not bugs)

These are real, concrete findings the review surfaced, but fixing them
changes behavior/semantics rather than correcting an unambiguous defect —
flagging for your decision rather than guessing:

- **Numeric-looking censored values** (e.g. a lab column containing `"<5"`
  alongside real numbers) get classified as categorical, not numeric, since
  the column becomes `object` dtype and nothing coerces it back. Forcing
  coercion would silently convert `"<5"` to `NaN`, losing the
  below-detection-limit signal — a real design tradeoff, not obviously wrong
  as-is.
- **`compute_age_histogram` silently drops out-of-range ages** (negative, or
  above the fixed 0–100 grid) from the total count with no indication in the
  returned histogram. A data-entry error (age in months, a sentinel like
  999) would silently understate the age distribution.
- **`_cohens_d`/`_hedges_g` return `NaN`** when a group has exactly 1
  observation (0/0 division), and `screen_associations`'s numeric-vs-binary-
  categorical loop has no minimum group-size floor (unlike
  `detect_outcome_column`'s `min_class_size=20`) — so a real hospital dataset
  with a rare binary category (1–2 patients) produces a row with a valid
  p-value but `effect_size: NaN`, silently marked `significant: False`
  rather than excluded or flagged. Adding a floor is a threshold choice, not
  a pure bug fix.
- **A stray whitespace-only string** (`"  "`) is treated as its own distinct
  category rather than coalesced with missing values — no stated intent
  either way in the code.

## Dead code found (not deleted — flagging for your decision)

The sweep confirmed (via whole-repo grep, zero importers) these are safe to
delete if you want the cleanup, or safe to leave if you'd rather keep them
as documented future-work stubs:

- `data_report/statistical_analysis/federated/federated_categorical_analysis.py`,
  `federated_temporal_analysis.py`, `federated_numeric_analysis.py` — all
  three are empty (0 bytes) placeholder files. The real federated-pooling
  logic lives in `analyze.py`'s `DataReportAggregator`, not here (the
  `LOCAL_PIPELINE_README.md` module-map table incorrectly implied otherwise
  — corrected as part of the documentation update).
- `data_report/generate_figures/generate_tables.py` — empty, documented in
  `GRAPHICS_REFACTORING_PLAN.md` as unimplemented future work.
- `data_report/statistical_analysis/local/categorical_analysis.py` — 69
  lines with real implementations, never imported anywhere.
- `data_report/label.py` — 182 lines (PASC label generation), never wired
  into either pipeline; consistent with the project's earlier decision to
  exclude PASC labeling.
- In `hub_entrypoint_10.py`: six full analyzer-side plotting orchestrator
  functions (`save_categorical_distributions_local`, `save_pca_outputs`,
  `save_mca_outputs`, `save_two_group_comparison`, `save_one_way_comparison`,
  `save_correlation_matrix`) plus four smaller helpers (`_fig_to_base64`,
  `violin`, `stacked_bar`, `peak_annotation_inferential`) and unused
  lazy-loaded FFT globals — leftovers from `hub_entrypoint_9.py`'s
  architecture (base64-encoded PNGs crossing the analyzer→aggregator
  boundary) that were never removed when the design changed to build all
  charts inline at the aggregator from serialized statistics. ~550+ lines
  total. These being unreachable is *why* bug #11 above went unnoticed for
  as long as it did (a reviewer diffing "the mirrored section" would find
  this dead code and think there was nothing to compare against the real,
  separately-located live logic).
- `import os` (unused) and a duplicate local `import sys` in
  `hub_entrypoint_10.py`; `List`/`Optional`/`Tuple`/`PASC_SYMPTOM_KEYWORDS`/
  `detect_file_type` imported-but-unused in `data_report/analyze.py`.
- A duplicate `peak_annotation()` in `inferential_analysis.py` — superseded
  by (and never removed after) its relocation to `inferential_plots.py`.

I did not delete any of this. It's inert (confirmed zero live impact) and
removing ~600+ lines across many files is a bigger, separate decision than
a bug fix — let me know if you'd like it cleaned up as a follow-up.

## Documentation updated

- **`README.md`** — removed stale clustering references from the
  "Outputs"/architecture description, fixed the stale "currently dataset1"
  (actual default is `dataset5`), and added two new changelog entries
  (clustering removal + full-pairwise rework; report-readability fixes)
  following the file's existing dated-entry convention, rather than
  rewriting the historical entries that document clustering's original
  development.
- **`LOCAL_PIPELINE_README.md`** — this is a living architecture reference
  (not a changelog), so it needed a real rewrite: removed the entire
  "Clustering" section (renumbering every section after it), updated the
  reduction-gating table, the association-screening description (full-pairwise
  instead of within-cluster), the serialized analyzer-output dict, the output
  directory tree (removed `clustering/`, added the batched file naming and
  `excluded_columns.csv`), the report-generation mode description, and the
  module map (also corrected the pre-existing, unrelated inaccuracy that
  `data_report/statistical_analysis/federated/` — confirmed dead code —
  was documented as the real federated-pooling implementation).
- **`CHANGES_README.md`** — left untouched; it's a historical fix-log for a
  specific past debugging session, not a living document.

## Verification

- Full repo test suite (`pytest`, 260 tests including the standalone
  genomic/imaging/column-detection tests that live outside `tests/`):
  **all passing**, plus 2 new regression tests added for fixes #8 and the
  temporal-batching error-isolation fix from the previous round.
- Full pipeline run on `data/dataset1` (2 nodes, `run_local_sim.py`) and
  `data/dataset5` (3 nodes, `dr-analyze`): both clean, no errors.
- **`hub_entrypoint_10.py` verified by actually executing it** (its
  `DataReportAnalyzer`/`DataReportAggregator` instantiated directly via
  `MockFlameCoreSDK`, bypassing only the outer `StarModel` wrapper that
  needs real FLAME infrastructure) against `dataset5`'s 3 heterogeneous-schema
  nodes — decoded the resulting `results.tar.gz` and confirmed: the
  corrected `n_total_values` (57,840, matching the modular project's run on
  the same data exactly), no doubled `%`, no false "available in full
  version" messages for PCA/MCA short mode, the privacy notice still
  correctly firing for this 3-node federation, and the `_variable_for_label`
  fix resolving a manufactured prefix-collision case correctly.
