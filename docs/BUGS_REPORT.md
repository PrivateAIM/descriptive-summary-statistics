# Bug & Issue Report — DataSummaryReport

This report catalogs issues found across the codebase, ordered by priority.
Nothing in this report has been fixed except the item marked **[FIXED]**.

---

## [FIXED] `dr-analyze` did not regenerate PDF reports / left stale results when switching datasets

- **File**: `data_report/cli.py`
- **Description**: `analyze_main` only ran the `StarModelTester` pipeline (CSV/PNG/pkl
  outputs). The PDF reports (`generate_reports/generate_global_report.py`,
  `generate_reports/generate_local_report.py`) were never invoked, and
  `results/local_results/` & `results/federated_results/` were never cleared
  between runs.
- **Impact**: After switching `dataset_name` from `dataset2` to `dataset1` and
  re-running, the PDF reports in `results/federated_results/` still showed
  dataset2's data (3 nodes / 17 features), and no per-node local PDF reports
  existed at all.
- **Status**: Fixed — `cli.py` now clears `local_results`/`federated_results`
  at the start of each run and generates global + per-node local PDF reports
  (short & full) at the end.

---

## Critical

### 1. `analyze.py:106-113` — Non-deterministic file selection per node
When a node folder contains multiple files (e.g. `dataset1`'s nodes each have
a raw CSV *and* a `_labeled` CSV), `analysis_method` does:
```python
file_bytes = next(iter(data[0].values()))
```
This picks **one arbitrary file** based on dict/iteration order from
`os.iterdir()`/`Path.iterdir()`, which is not guaranteed to be sorted.
- **Impact**: Which file gets analyzed (raw vs. labeled) can vary across
  runs/machines, silently changing which columns/labels are present in the
  report (e.g. `pasc`/label column may or may not be included).
- **Where**: `data_report/analyze.py:106-113`, `get_data/load_data.py:7-19`.

### 2. `analyze.py:470` — `max()` on empty sequence crashes aggregation
```python
n_cols = max(r["n_cols"] for r in analysis_results)
```
If `analysis_results` is empty (e.g. all nodes failed to return data, or a
dataset folder has zero node subdirectories), this raises
`ValueError: max() arg is an empty sequence` and the whole aggregation crashes
with no useful message.

### 3. `analyze.py:644` — `analysis_results[0]` IndexError on empty results
```python
age_edges = analysis_results[0].get("age_edges")
```
Same root cause as #2 — no guard for an empty `analysis_results` list.

---

## High

### 4. `analyze.py:631` — Invalid temporal periods aggregated under the literal string `"NaT"`
```python
k = str(pd.to_datetime(k, errors="coerce"))
global_counts[k] = global_counts.get(k, 0) + v
```
If a period key can't be parsed, `pd.to_datetime` returns `NaT`, and
`str(NaT)` is `"NaT"` — which becomes a real bucket in
`global_temporal[col]["counts_per_period"]` rather than being dropped.
- **Impact**: Federated temporal statistics report a bogus "NaT" period with
  a non-zero count, and it can even be picked as `most_active_period` if it
  accumulates enough observations.

### 5. `analyze.py:403` and `analyze.py:473` — Possible division by zero
```python
missing_values_percentage = (total_missing / total_values) * 100   # line 403
total_missing_percentage  = (total_missing / n_total_values) * 100  # line 473
```
If a node's dataframe is empty (`df.size == 0`) or `n_total_values == 0`
(possible if `n_cols` ends up `0`, see #2), these raise
`ZeroDivisionError`/produce `inf`/`NaN`.

### 6. `analyze.py:1131` and `analyze.py:1158` — Direct `r["age_hist"]` / `r["sex_counts"]` access
```python
if r["age_hist"] is not None:      # line 1131
...
if r["sex_counts"]:                # line 1158
```
These keys are always returned by the current `analysis_method`, so this is
not exploitable *today*, but it's inconsistent with the `.get(...)` pattern
used everywhere else in `_save_local_node_results` and will raise `KeyError`
the moment either key is renamed/omitted (e.g. by a future refactor of the
analyzer's return dict).

### 7. `compute_statistics.py` — `.iloc[0]` / `.iloc[-1]` on possibly-empty `value_counts()`
Around lines 112-156, several statistics (mode, first/last category, etc.) are
read via `.iloc[0]`/`.iloc[-1]` from a `value_counts()` result. If a column is
entirely `NaN` (or empty after filtering), `value_counts()` returns an empty
Series and these raise `IndexError`. Only one of the relevant call sites has
an `.empty` guard.

---

## Medium

### 8. `analyze.py:819` — Dead, hardcoded absolute path
```python
DATA_DIR = '/Users/nouryassine/Desktop/DataSummaryReport/data'
```
Defined inside `_save_local_node_results` but never referenced afterwards.
Aside from being dead code, a hardcoded path like this would break for any
other user/checkout if it were ever wired up.

### 9. Inconsistent/silent exception handling around plot generation
- `generate_figures/local_descriptive_plots.py` (e.g. ~379, 460, 518, 391-392)
  and `generate_figures/federated_descriptive_plots.py` (~260, 308): some
  failures are caught and printed, some are caught and silently `continue`d,
  and figures opened via `plt.subplots()` may not be closed (`plt.close()`)
  if an exception occurs mid-function — these are matplotlib figure leaks
  that accumulate over a long run with many columns.
- The inconsistency means it's hard to tell, just from the report output,
  whether a missing plot reflects "not applicable" or "crashed silently".

### 10. `inferential_analysis.py:1210` — Assumes `.iloc` is available
```python
x = (time_values - time_values.iloc[0]).dt.total_seconds().values / 86400.0
```
Assumes `time_values` is always a `pandas.Series`. If it's ever passed as a
plain numpy array or list (e.g. from a future caller), this raises
`AttributeError`.

### 11. `inferential_analysis.py` (~1419, 1444, 1466) — Bare `except Exception:` without logging
Multiple association-screening code paths swallow all exceptions silently.
Combined with #9, this makes it very hard to distinguish "this pair genuinely
has no association" from "the test crashed".

---

## Low (code quality / cleanup)

### 12. Empty placeholder modules
- `data_report/statistical_analysis/local/imaging_data_analysis.py` — 0 bytes.
- `data_report/statistical_analysis/federated/federated_categorical_analysis.py`,
  `federated_numeric_analysis.py`, `federated_temporal_analysis.py` — all 0
  bytes; the federated logic actually lives in `analyze.py`'s
  `DataReportAggregator`.

These are either unfinished scaffolding or leftover stubs from an earlier
module layout, and currently just add confusion about where federated logic
lives.

### 13. `_save_local_node_results` overview text — double `%%`
```python
"Patients": f"{r['n_rows']} ({patient_contribution}% of all patients in the federation)",
```
`patient_contribution` is already formatted as a string ending in `%`
(`overview_comp["patient_contribution"] = str(round(...,3)) + "%"`), so the
f-string above produces `"... (50.0%% of all patients...)"`. Cosmetic, but
shows up in every node's `overview.csv` and PDF report.

### 14. `analyze.py` ~302, ~342 — bare `except Exception:` without logging in outcome comparisons
Skips a numeric column's outcome comparison silently; acceptable per the
existing code comment for *expected* sparse-data cases, but the same blanket
`except Exception` would also hide a genuine bug (e.g. a typo causing
`KeyError` in `cmp["effect_size"]`).

### 15. `git status` shows deleted files with no remaining references
`data_report/generate_figures/generate_plots.py`,
`data_report/statistical_analysis/local/mca.py`, and
`data_report/statistical_analysis/local/pca.py` are marked deleted. A repo-wide
search found **no remaining imports** of these modules, so the deletions
appear safe/already complete — just needs the deletion to be committed.

---

## Suggested order of attack

1. Fix #1 (deterministic file selection) — affects correctness of *every*
   dataset with multi-file node folders (currently `dataset1`).
2. Guard against empty `analysis_results` (#2, #3) and zero-division (#5) —
   cheap defensive checks that turn hard crashes into clear messages.
3. Fix the `"NaT"` bucket bug (#4) — silently corrupts federated temporal
   stats.
4. Harden `compute_statistics.py` empty-`value_counts()` paths (#7).
5. Clean up dead code / cosmetic issues (#8, #12, #13, #15) and tighten
   exception handling (#6, #9, #10, #11, #14) as time allows.
