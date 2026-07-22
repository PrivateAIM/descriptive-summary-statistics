# Inferential statistics module — changes, fixes, and refactorings

This documents everything changed in and around
`data_report/statistical_analysis/local/inferential_analysis.py` and its
wiring into `data_report/analyze.py`, in the order it was done.

## 1. Refactoring: decomposed `regression()`

`regression()` (~160 lines) was split into five single-purpose helpers, with
`regression()` left as a thin orchestrator:

- `_validate_regression_inputs`
- `_prepare_regression_data`
- `_select_regression_model`
- `_fit_regression_model`
- `_build_regression_result`

**Why:** the function mixed input validation, data prep, model selection,
fitting, and result assembly in one block, making it hard to test or modify
any one concern in isolation. Verified behaviorally identical via smoke tests
(linear + logistic, synthetic data) and the full suite (99 passed both
before and after).

## 2. Bug fix: `detect_dataset_type` called `detect_id_column` with the wrong signature

**Before:**
```python
patient_col = detect_id_column(df)
```

**After:**
```python
# detect_id_column inspects one column (series + its name) at a time, so
# scan the columns and take the first match -- mirrors the detection loop
# in DataReportAnalyzer.analysis_method.
patient_col = next((col for col in df.columns if detect_id_column(df[col], col)), None)
if patient_col is None:
    raise ValueError("detect_dataset_type could not find a patient/ID column in the dataframe.")
```

**Why:** `detect_id_column` takes `(series, column_name)`, not a whole
DataFrame — calling it with `df` either raised or silently misbehaved. This
was a genuine pre-existing bug, found by running every public function in
the module against real hospital data (2798 patients × 214 columns) and
checking outputs by hand.

## 3. New: generic outcome-column detection

Added `OUTCOME_KEYWORD_GROUPS` to `data_report/config.py` (additive — the
existing `LABEL_COL = "pasc"` is untouched, still used for PCA/MCA coloring):

```python
OUTCOME_KEYWORD_GROUPS: List[List[str]] = [
    ["death", "died", "deceased", "mortality", "survival"],
    ["icu", "intensive_care", "ventilation", "intubation"],
    ["readmission", "re-admission", "admission"],
    ["outcome", "status", "diagnosis", "condition", "label", "pasc"],
    ["complication", "adverse"],
]
```

Added `detect_outcome_column(df, column_types, keyword_groups, min_class_size=20, max_levels=5)`
to `inferential_analysis.py`: walks the keyword groups in priority order and
returns the first matching categorical column that is *usable* — not
constant, ≤ `max_levels` categories, every level has ≥ `min_class_size` rows.
Returns `None` if nothing usable is found.

**Why:** you decided `pasc`/`LABEL_COL` shouldn't be hardcoded as *the*
pivot variable for inferential analyses, since the program must run
unattended on arbitrary hospital datasets across the hub (no user
interaction possible). This generalizes the existing inline `sex_col`
detection pattern already in `analyze.py` into something reusable, dataset-
agnostic, and validated for actual usability before being trusted.

## 4. New: bounded association screening

Added `screen_associations(df, column_types, cluster_results=None, alpha=0.05, effect_size_thresholds=None, max_group_levels=6)`
to `inferential_analysis.py` — the "screen → filter → detail" approach:

- **num-num**: only pairs *within the same numeric cluster* (from
  `cluster_results["numeric"]`) → `correlation_between_two_variables`
- **cat-cat**: only pairs within the same categorical cluster, *and* both
  members low-cardinality (≤ `max_group_levels`) → `categorical_association`
- **num-cat**: every numeric column × every *binary* categorical column →
  `compare_two_groups` (the only comparison function that already returns a
  standardized effect size regardless of which test it auto-selects)
- Applies **Benjamini-Hochberg FDR correction** (`statsmodels.stats.multitest.multipletests`)
  to get `p_adj`, and flags `significant` only when `p_adj < alpha` **and**
  the effect size clears a Cohen's-convention "small effect" threshold
  (`|r|`/rank-biserial ≥ 0.2, Cramér's V ≥ 0.1, Hedges' g ≥ 0.2)

**Why bounded, not brute-force:** the dataset has ~150 categorical columns;
all-pairs chi-square would be ~10k tests. Reusing the clustering already
computed earlier in `analysis_method` gives a principled, data-driven way to
decide *which* pairs are worth testing, with no duplicated clustering work.

**Why FDR + effect size, not p-value alone:** running hundreds of tests at
α = 0.05 flags ~5% as "significant" by chance; and at n ≈ 2800, trivially
small differences become p < 0.05. FDR correction plus a minimum effect size
filters out both failure modes.

### Iteration on cat-cat pair selection (worth knowing about)

1. **v1** — used raw cluster pairs directly → surfaced meaningless
   high-cardinality pairs like `('unnamed:_0', 'sex')` (an ID column paired
   with `sex`), because `ClusteringManager` clusters by association/TOM, not
   cardinality.
2. **v2** — added a brute-force fallback "if no cluster pairs survive
   filtering, test all pairs among low-cardinality categorical columns" →
   recreated exactly the combinatorial explosion the clustering bound was
   meant to avoid: **9,840 pairs, 17.2 seconds**.
3. **v3 (final)** — restrict cluster-derived cat-cat pairs to members where
   *both* columns are low-cardinality, **with no brute-force fallback at
   all**. If clustering yields nothing usable, the section honestly
   contributes zero cat-cat rows — there's no principled way to bound
   brute-force testing without it. Final: **79 bounded pairs in ~5.4s**.

This is a deliberate deviation from the plan text (which mentioned "else a
capped shortlist of low-cardinality columns" as a fallback) — the fallback
was tried, measured, and removed because it reproduced the exact problem it
was meant to solve.

### Validation

- **Real data** (2798 × 214): 79 sensible bounded pairs in ~5s; correctly
  found nothing "significant" (this particular dataset has weak signal).
- **Synthetic data with known structure**: correctly flagged all 6 genuine
  relationships (`num_a~num_b` r = 0.65; `grp2~cat1` Cramér's V = 0.62; four
  group-difference pairs, Hedges' g / rank-biserial up to 1.17) and correctly
  rejected all 4 noise pairs.

## 5. Bug fix: FDR correction broken by a single `NaN` p-value

Found while end-to-end testing the wired-up pipeline: one `welch_ttest` /
`welch` ANOVA call returned a `NaN` p-value (degenerate group variance), and
`multipletests` propagated that single `NaN` to **every** `p_adj` value in
the table — silently making `significant` always `False` for the whole
screen.

**Fix** (`inferential_analysis.py`, in `screen_associations`):
```python
screening["p_adj"] = np.nan
valid = screening["p_value"].notna()
if valid.any():
    _, p_adj, _, _ = multipletests(screening.loc[valid, "p_value"], alpha=alpha, method="fdr_bh")
    screening.loc[valid, "p_adj"] = p_adj
```
Correct only over the valid p-values; rows with `NaN` p-values stay
`NaN`/`not significant` instead of corrupting the whole correction.

## 6. Wiring into `analyze.py`

Added imports:
```python
from data_report.config import LABEL_COL, PASC_SYMPTOM_KEYWORDS, OUTCOME_KEYWORD_GROUPS
from data_report.statistical_analysis.local.inferential_analysis import (
    screen_associations,
    detect_outcome_column,
    compare_two_groups,
    one_way_group_comparison,
)
```

Added a new section inside `analysis_method`, after the MCA block, following
the same `try/except` + `print(f"... section error: {e}")` pattern as
PCA/MCA (one section's failure can't take down the pipeline):

- Runs `screen_associations`, saves `node{N}/inferential/association_screening.csv`
  and `significant_associations.csv`
- Runs `detect_outcome_column`; if a usable outcome is found, compares every
  numeric column across its groups (`compare_two_groups` for 2 groups,
  `one_way_group_comparison` for 3+), flattens each result dict (`method`,
  `statistic`, `p_value`, one `effect_size_<metric>` column per entry) into
  a row, and saves `comparisons_by_<outcome_col>.csv` — only if at least one
  comparison succeeded
- No `regression(...)` call anywhere in this automatic pipeline — per your
  explicit decision that auto-selecting a regression target on unattended
  hub runs risks data dredging / circular analysis (e.g. if a demographic
  like age or sex got picked as the "outcome")

### Deviations from the originally-approved plan text

1. **Dropped the planned post-hoc branch** (`posthoc_test` for "significant
   num-cat pairs with > 2 groups"). The actual `screen_associations` only
   ever produces `num-cat` rows for *binary* categoricals (the only path
   with a standardized effect size), so that condition could never be true —
   removed rather than kept as dead code.
2. **Resolved the plan's `flatten()` placeholder** with explicit row
   construction (shown above) — `flatten` doesn't exist anywhere in the
   codebase.
3. **Outcome-comparison failures are skipped silently** (`continue`), not
   printed per-column. Many numeric columns are sparse enough that one
   outcome group ends up empty (e.g. visit-specific fields not recorded for
   every patient) — that's expected for ~150 columns × 1 outcome per node,
   and per-failure prints would flood hub logs.

## 7. Verification performed

- Smoke-tested every public function in `inferential_analysis.py` against
  real hospital data (2798 × 214) — found and fixed the `detect_dataset_type`
  bug (§2)
- Unit/synthetic tests for `detect_outcome_column` (keyword match + usable;
  keyword match but unusable, falls through; no match → `None`) and
  `screen_associations` (real + synthetic data, see §4)
- Ran the wired-up `analysis_method` block end-to-end on real node data;
  confirmed `node{N}/inferential/*.csv` files are created, non-empty, and
  well-formed
- Confirmed graceful degradation: dropped every outcome-keyword-matching
  column from the dataset, re-ran — `detect_outcome_column` returns `None`,
  no `comparisons_by_*` file is produced, no exception bubbles up
- Full suite: `.venv/bin/python -m pytest tests/ -q` → **99 passed** (no
  regressions, before and after every change above)
