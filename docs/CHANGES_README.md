# hub_entrypoint.py — Changes and Fixes Applied

This document describes every change applied to the original `hub_entrypoint.py`
in comparison to the version before debugging. Only fixes that are present in the
final working version are listed here.

---

## 1. Lazy Loading of Heavy Dependencies

**Problem:** All heavy libraries (`PIL`, `reportlab`, `seaborn`, `sklearn`,
`scipy`, `plotly`, `prince`, `missingno`, `pingouin`) were imported at module
level. Python executes all top-level imports when the file is loaded, which
caused the container to exceed the FLAME platform's startup timeout
(`ANALYSISSTARTUPERROR`) before `StarModel()` was ever called.

**Fix:** All heavy imports were moved into two lazy-loader functions that are
called only after the FLAME SDK handshake completes:

`_load_analysis_dependencies()` — called at the start of `_analysis_method_impl`.
Loads seaborn, plotly, prince, missingno, pingouin, sklearn, scipy.

`_load_reporting_dependencies()` — called just before PDF generation in the
aggregator. Loads PIL and reportlab, which are the heaviest libraries and are
only needed at the very end of the aggregation phase.

Only `matplotlib`, `numpy`, and `pandas` remain at module level since they
are fast to import and are used throughout.

Module-level placeholder variables are set to `None` for all lazy names
(`PILImage = None`, `colors = None`, etc.) so the file parses without errors.

---

## 2. Deferred Module-Level Constants That Used Lazy Names

**Problem:** Several constants and expressions at module level called into
lazy-loaded names before those names were loaded, causing `TypeError` or
`AttributeError` crashes at import time:

- `MAX_W = 6.8 * inch` — `inch` is `None` at module level
- `MAX_H = 4 * inch`
- `PAGE_MARGIN = 0.4 * inch`
- `STYLES = getSampleStyleSheet()` — `getSampleStyleSheet` is `None` at module level
- `PILImage.MAX_IMAGE_PIXELS = None` — `PILImage` is `None` at module level
- `_NARRATIVE_STYLE = {"info": {"bg": colors.HexColor(...)}}` — `colors` is `None`
- `def add_figure(..., max_width=6.5 * inch, ...)` — default argument evaluated at definition time
- `def add_plots_from_dir(..., max_width=6.5 * inch, ...)` — same issue
- `def add_heading_and_plots(..., max_width=6.5 * inch, ...)` — same issue

**Fix:**

`MAX_W`, `MAX_H`, `PAGE_MARGIN` are set to `None` at module level and computed
inside `_load_reporting_dependencies()` after `inch` is available.

`STYLES` is set to `None` at module level and assigned inside
`_load_reporting_dependencies()`.

`PILImage.MAX_IMAGE_PIXELS = None` is called inside `_load_reporting_dependencies()`
after `PILImage` is loaded.

`_NARRATIVE_STYLE` is set to `None` at module level. A new function
`_get_narrative_style()` builds and caches it on first call (after lazy load).
All callers use `_get_narrative_style()[msg.level]` instead of `_NARRATIVE_STYLE[msg.level]`.

The three function default arguments (`add_figure`, `add_plots_from_dir`,
`add_heading_and_plots`) use `None` as the default and compute the real value
inside the function body:
```python
def add_figure(elements, path, max_width=None, max_height=None, caption=None):
    if max_width is None: max_width = 6.5 * inch
    if max_height is None: max_height = 4 * inch
```

---

## 3. Serialization Helpers Added

**Problem:** Values returned by `analysis_method` travel over the network as
JSON. Non-JSON-native types (`numpy` arrays, `np.int64`, `np.float64`,
`pd.Period`, `pd.Timestamp`, `float NaN/Inf`, raw objects) silently break
serialization, causing the aggregator to hang waiting for results that never
arrive.

**Fix:** Four helper functions added at module level:

`_make_serializable(obj)` — recursively converts any non-JSON-native type
to a plain Python type. Handles `np.ndarray` → `list`, `np.integer` → `int`,
`np.floating` → `float` (with `NaN`/`Inf` → `None`), `pd.Period` → `str`,
`pd.Timestamp` → `isoformat()` string.

`_sanitize_temporal_statistics(ts)` — converts `pd.Period` keys and values
inside temporal statistics dicts to strings. `compute_temporal_statistics`
returns `pd.Period` objects as dict keys, which cannot be JSON-serialized.

`_fig_to_base64(fig)` — renders a matplotlib Figure to PNG and returns it as
a base64 string. Used by the analyzer to encode df-dependent plots so they
can travel through JSON serialization.

`_fig_to_bytes(fig)` — renders a matplotlib Figure to PNG bytes. Used by the
aggregator to build plots directly into `output_files`.

`_df_to_csv_bytes(df)` — serializes a DataFrame to CSV bytes via `BytesIO`.

`_build_tar(file_dict)` — packs a `{relative_path: bytes}` dict into a
gzip-compressed tar archive and returns the archive as bytes.

---

## 4. Serialization Validation in analysis_method

**Problem:** If any value in the return dict was not JSON-serializable, the
failure was silent — the aggregator simply never received the result and hung.

**Fix:** A JSON validation step added at the end of `_analysis_method_impl`
before returning:
```python
try:
    json.dumps(result)
    print("Serialization check: OK")
except (TypeError, ValueError) as exc:
    raise RuntimeError(f"analysis_method return value is not JSON-serializable: {exc}") from exc
```
This surfaces the exact non-serializable field in the node logs instead of
causing a silent hang.

---

## 5. Fixed age_hist and age_edges in Analyzer Return Dict

**Problem:** The original code called `.tolist()` on `age_hist` and `age_edges`:
```python
"age_edges": age_edges.tolist() if age_edges is not None else None,
"age_hist":  age_hist.tolist()  if age_hist  is not None else None,
```
But `compute_age_histogram` already returns plain Python lists. Calling
`.tolist()` on a list raises `AttributeError: 'list' object has no attribute 'tolist'`.

**Fix:** Changed to `list()` which works safely on both lists and numpy arrays:
```python
"age_edges": list(age_edges) if age_edges is not None else None,
"age_hist":  list(age_hist)  if age_hist  is not None else None,
```

---

## 6. Robust CSV File Selection in analysis_method

**Problem:** The original code used `next(iter(data[0].values()))` to get the
CSV file — grabbing the first value in the dict regardless of filename. Dict
ordering is not guaranteed when multiple files are present, so this could pick
the wrong file.

**Fix:** A priority-based selection chain that searches by filename suffix:

1. Any file ending in `unlabeled.csv` (datasets that ship both labeled/unlabeled versions)
2. Any `.csv` file whose name does not contain `"labeled"` (robust to multi-file datasets)
3. Any `.csv` file at all (fallback — also handles datasets with only a labeled file)

```python
csv_key = (
    next((k for k in files if k.lower().endswith("unlabeled.csv")), None)
    or next((k for k in files if k.lower().endswith(".csv")
             and "labeled" not in k.lower()), None)
    or next((k for k in files if k.lower().endswith(".csv")), None)
)
```
A `print` statement logs the selected filename and byte size for debugging.

---

## 7. Node Index Derived from Sorted UUID List

**Problem:** The original code used `int(node_id.split("_")[-1])` to extract
a node number, assuming node IDs look like `"node_0"`, `"node_1"`. Real FLAME
node IDs are UUIDs (e.g. `"bcb3ac02-d73e-43ca-9418-2a7c01b9f58f"`). Splitting
on `"_"` and converting to int raises `ValueError`.

**Fix:** The approach recommended by the privateAIM support team — sort all
participant UUIDs alphabetically (including the own node's ID) and use the
position in that sorted list as the node index:

In `_analysis_method_impl` (uses `self.id` and `self.partner_node_ids`):
```python
node_id = self.id
all_n_ids = self.partner_node_ids.copy()
all_n_ids.append(self.id)
node_index = sorted(all_n_ids).index(self.id)
node_number = node_index + 1
```

In `_collect_local_node_files` and `_collect_pdf_reports` (uses the full list
from `analysis_results`):
```python
all_n_ids = sorted(r["node_id"] for r in analysis_results)
node_index = all_n_ids.index(raw_node_id)
node_number = node_index + 1
```

The function `generate_local_report_bytes` now accepts `node_number` as a
parameter instead of computing it internally.

---

## 8. Df-Dependent Plots Encoded as Base64 in the Analyzer

**Problem:** Several plots (missing value bar/heatmap, age distribution, sex
distribution, data type pie chart, temporal activity charts) require the raw
DataFrame. The raw DataFrame is only available in `analysis_method` on the
node — it is never sent to the aggregator. These plots were previously either
missing or being generated incorrectly.

**Fix:** All df-dependent plots are now generated inside `_analysis_method_impl`
while the DataFrame is available. Each figure is encoded as a base64 PNG string
using `_fig_to_base64()` and included in the return dict under two keys:

`"node_plots"` — dict of `{plot_name: base64_string}` for overview plots
(missing bar, missing heatmap, age distribution, sex distribution, data type
distribution).

`"temporal_activity_plots"` — dict of `{feature_name: base64_string}` for
per-column temporal activity line charts.

Two helper functions added just before the analyzer class:
`_make_missing_bar_fig(df)` and `_make_missing_heatmap_fig(df)` — return
matplotlib Figure objects using missingno, with graceful error handling.

The aggregator decodes these base64 strings back to PNG bytes and stores them
in `output_files`.

---

## 9. LABEL_COL Defined

**Problem:** `LABEL_COL` was used in PCA and MCA plot code to optionally colour
plots by a target column, but it was never defined, causing `NameError`.

**Fix:** Added in the configuration section:
```python
LABEL_COL = None
```
Set to a column name string to enable label-based colouring in PCA/MCA plots.
Defaults to `None` (colouring disabled).

---

## 10. Stale _save_local_node_results Call Removed

**Problem:** A call to `self._save_local_node_results(analysis_results, comparison_results_per_node)`
remained in `_aggregation_method_impl` from the original disk-writing approach.
The method no longer exists in the class, causing `AttributeError`.

**Fix:** The call was removed. The equivalent functionality is now handled by
`_collect_local_node_files()` which is called later in the same method.

---

## 11. Aggregator Output: In-Memory Files, Tar Archive, Base64 String

**Problem:** The original aggregator wrote files to disk paths and tried to
read them back as a list. The FLAME platform's `output_type="bytes"` failed
with `TypeError: 'bytes' object cannot be interpreted as an integer` regardless
of how the bytes were produced — this is a known limitation of the FLAME SDK's
bytes translation layer.

**Fix:** A three-step output pipeline:

Step 1 — All output files (CSVs, PNGs, PDFs) are collected into an
`output_files: dict[str, bytes]` mapping archive-internal relative paths to
raw bytes. Four collector methods populate this dict:
`_collect_federated_files`, `_collect_federated_plots`,
`_collect_local_node_files`, `_collect_pdf_reports`.

Step 2 — `_build_tar(output_files)` packs everything into a single
`.tar.gz` archive. The archive is written to `RESULTS_DIR/results.tar.gz`
on disk (mirroring the HALTA example's pattern of writing to disk then
reading back).

Step 3 — The tar bytes are read back from disk, base64-encoded to a UTF-8
string, and returned as `[result_str]`. `output_type="str"` is used in
`StarModel` since it is the most reliably handled output type in the FLAME SDK.

```python
with open(str(tar_path), "rb") as f:
    result_bytes = f.read()
import base64 as _b64
result_str = _b64.b64encode(result_bytes).decode("utf-8")
return [result_str]
```

`StarModel` configuration:
```python
StarModel(
    ...
    multiple_results=True,
    simple_analysis=False,
    output_type="str",
    filename="results.tar.gz.b64.txt",
)
```

---

## 12. PDF Report Images Fixed

**Problem:** The PDF reports contained no images. The PDF builders
(`generate_local_report`, `generate_global_report`) look for PNG files at
disk paths inside a directory tree. These paths only existed in the
`output_files` dict in memory — never on disk. `add_figure` checks
`path.exists()` and silently skips any file not found.

**Fix:** `generate_local_report_bytes` and `generate_global_report_bytes`
now accept an `output_files` parameter. Before calling the PDF builder,
they write all relevant PNG files from `output_files` into the temp
directory alongside the CSVs:

```python
if output_files:
    prefix = f"local/node{node_number}/"
    for rel_path, content in output_files.items():
        if rel_path.startswith(prefix) and rel_path.endswith(".png"):
            dest = tmp / "local" / rel_path[len("local/"):]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)
```

The callers in `_collect_pdf_reports` pass `output_files=output_files` to
both functions.

---

## 13. Error Surfacing in analysis_method and aggregation_method

**Problem:** Exceptions inside `analysis_method` were swallowed by the FLAME
framework and never appeared in the node logs, making silent failures impossible
to diagnose.

**Fix:** Both methods are wrapped in a try/except that prints the full traceback
with `flush=True` before re-raising, ensuring it appears in the platform logs:

```python
def analysis_method(self, data, aggregator_results):
    import traceback as _tb, sys
    print("analysis_method called", flush=True)
    try:
        return self._analysis_method_impl(data, aggregator_results)
    except Exception as _exc:
        msg = _tb.format_exc()
        print("ANALYSIS_METHOD FAILED:\n" + msg, flush=True)
        raise
```

The actual logic lives in `_analysis_method_impl` and `_aggregation_method_impl`.

---

## 14. Federated Plot Functions Rewritten for In-Memory Output

**Problem:** `save_all_federated_plots` and related functions
(`save_federated_data_type_distribution`, `save_federated_age_distribution`,
etc.) all wrote PNG files to disk paths. This was incompatible with the
in-memory output pipeline.

**Fix:** `save_all_federated_plots` was rewritten to accept
`output_files: dict` and `archive_base: str` instead of `output_dir`.
It generates all federated plots using `_fig_to_bytes()` and stores them
directly into `output_files` under keys like
`"federated/overview/data_type_distribution.png"`.

Similarly, `_collect_federated_files` writes all federated CSV tables
as bytes into `output_files` using `_df_to_csv_bytes()`.

---

## 15. Inferential Outputs Routed Through the Analyzer→Aggregator Boundary

**Problem:** The inferential analysis section in `_analysis_method_impl` saved
`association_screening.csv`, `significant_associations.csv`,
`comparisons_by_<outcome>.csv`, and all inferential plots to the analyzer node's
local disk. Because the analyzer and aggregator run on different machines, those
files were never available to the aggregator. They were absent from the output
`tar.gz` and appeared as `null` in `summary.json`.

**Fix:** After the existing disk-write calls (which are kept for local
debugging), each output is also captured in memory and added to the analyzer
return dict under `"inferential_data"`:

- `"association_screening_records"` / `"significant_associations_records"` —
  DataFrames serialized as `list[dict]` via `_make_serializable`.
- `"outcome_col"` — the detected outcome column name (or `None`).
- `"comparisons_by_outcome_records"` — outcome comparison DataFrame as records.
- `"association_screening_png"` / `"group_comparisons_summary_png"` — the
  already-saved PNG files read back from disk and base64-encoded.
- `"comparison_plots"` — dict of `{col_vs_outcome: base64_png}` for per-column
  comparison plots, read back from disk after `save_two_group_comparison` /
  `save_one_way_comparison` write them.

In `_collect_local_node_files` the aggregator reconstructs everything:
records are converted back to CSV bytes with `_df_to_csv_bytes` and written to
`output_files["{base}/inferential/..."]`; base64 strings are decoded and stored
as PNG bytes. All paths match what the PDF builder and JSON summary generator
expect.

`_collect_json_summaries` was updated to dynamically find
`comparisons_by_<outcome>.csv` in `output_files` (the filename depends on the
detected outcome column name) so `summary.json["comparisons_by_outcome"]` is
populated instead of always being `null`.

`_csv_bytes_to_records` was hardened with a try/except around `pd.read_csv` to
handle the `EmptyDataError` raised when `significant_associations.csv` is an
empty DataFrame (no significant pairs found).

---

## 16. Cluster Size Bar Chart Fixed in _collect_cluster_plots

**Problem:** `_collect_cluster_plots` iterated `clusters.items()` treating the
dict as `{variable_name: cluster_id}`, but the actual structure stored in the
analyzer return dict is `{cluster_id: [member_variable_names]}` (the inverse).
The loop tried to use the list of member names as a dict key, which raises
`TypeError: unhashable type: 'list'`.

**Fix:** Corrected the iteration to match the actual structure:
```python
sizes = {
    cluster_id: len(members) if isinstance(members, list) else 1
    for cluster_id, members in clusters.items()
}
```
Bar labels are now `"Cluster {id}"` and the sort key handles string-form
integer cluster IDs produced by JSON round-trip.

---

## 17. MCA Fixed for pandas 2.1+ / NumPy 2.0

**Problem:** `run_mca` called `select_dtypes(include=["object", "str",
"category", "bool"])`. In pandas 2.1+, `"str"` in the include list is
interpreted as NumPy's fixed-width string dtype (`np.str_`), which is
explicitly rejected with `TypeError: numpy string dtypes are not allowed,
use 'str' or 'object' instead`. This crashed the entire MCA section on every
node.

**Fix:** Removed `"str"` from the `select_dtypes` include list:
```python
categorical_df = df[features].select_dtypes(include=["object", "category", "bool"])
```
All text columns in practice have `object` dtype, so no columns are dropped by
this change.

Additionally, the dtype-normalisation loop in `_analysis_method_impl` that
prepares `mca_df` before calling `save_mca_outputs` was extended to coerce any
remaining extension string types (e.g. `pd.StringDtype`, Arrow-backed strings)
to plain `object` dtype, ensuring compatibility regardless of how a hospital
dataset was loaded:
```python
elif not (
    pd.api.types.is_object_dtype(dtype)
    or isinstance(dtype, pd.CategoricalDtype)
    or pd.api.types.is_bool_dtype(dtype)
):
    mca_df[col] = mca_df[col].astype(object)
```

---

## 18. Dead Code Removal (W-1, W-2, W-3)

**Problem:** 49 functions totalling 1,796 lines were reachable by Python's parser
but never called from any live code path. They added maintenance weight and
created confusion about which functions were actually in use.

Key dead functions identified:

- `_load_heavy_dependencies` (W-1) — a single-call convenience wrapper around
  `_load_analysis_dependencies()` and `_load_reporting_dependencies()`. All
  callers already used the two real loaders directly; this wrapper was never
  called.
- `peak_annotation` (W-2) — FFT/signal peak annotation helper originally brought
  in for spectral analysis. Never called from any analysis path.
- 47 additional functions (W-3) — including `set_theme`, `detect_file_type`,
  `regression`, `detect_dataset_type`, `compare_multiple_groups`, and large
  blocks of standalone plotting helpers that were superseded by the in-memory
  `_make_*_fig` helpers when the output pipeline was rewritten.

**Fix:** All 49 functions removed. Stale comments and docstring references to
removed functions updated throughout:

- Module-level comment on line 18 updated from `_load_heavy_dependencies()` to
  the two real loader names.
- `set_palette` docstring updated to remove the `set_theme` reference.
- `save_pca_outputs` and `save_mca_outputs` docstrings updated to remove the
  reference to the removed `save_cluster_outputs` function.
- The `figures_data_quality_plots` section header updated to name the current
  `_make_missing_*_fig` helpers instead of the removed `save_missing_*` functions.

**Result:** 9,050 → 7,304 lines. No functional behaviour changed — cross-reference
audit confirmed all 49 removed functions had zero call sites in live code.

---

## 19. Cluster Detail Plots Routed Through the Analyzer→Aggregator Boundary (W-4)

**Problem:** `save_cluster_histograms`, `save_cluster_boxplots`,
`save_cluster_violinplots`, and `save_cluster_scatterplots` all write PNG files
to the analyzer node's local disk (`clustering_dir`). Because the analyzer and
aggregator run on different machines, those plots were never available to the
aggregator and were absent from the output `tar.gz`.

**Fix:** After the cluster detail plot loop in `_analysis_method_impl`, all PNG
files under `clustering_dir` are read back from disk and encoded as base64
strings:

```python
cluster_detail_plots: dict[str, str] = {}
for _png in sorted(clustering_dir.rglob("*.png")):
    _rel = str(_png.relative_to(clustering_dir))
    cluster_detail_plots[_rel] = base64.b64encode(_png.read_bytes()).decode("utf-8")
```

`cluster_detail_plots` is added to the analyzer return dict (alongside the
existing `node_plots` and `temporal_activity_plots` keys). In
`_collect_local_node_files` the aggregator decodes each entry and stores it
under `output_files[f"{base}/clustering/{rel_path}"]`, which places it
correctly inside the tar archive.

---

## 20. PCA and MCA Plots Routed Through the Analyzer→Aggregator Boundary (W-5)

**Problem:** `save_pca_outputs` and `save_mca_outputs` both write all their PNG
files to the analyzer node's local `pca/` and `mca/` directories. Like the
cluster detail plots, these were never available to the aggregator.

**Fix:** After each `save_*_outputs` call, all PNG files in the output directory
are read back and encoded as base64, then returned in the analyzer result dict
under `"pca_plots"` and `"mca_plots"`:

```python
pca_plots: dict[str, str] = {}
for _png in sorted(_pca_dir.rglob("*.png")):
    pca_plots[str(_png.relative_to(_pca_dir))] = base64.b64encode(
        _png.read_bytes()
    ).decode("utf-8")
```

Both dicts initialized before their respective `try` blocks so they remain empty
(rather than raising `NameError`) if the section is skipped due to insufficient
columns or an exception.

In `_collect_local_node_files` the aggregator decodes these and stores them at
`output_files[f"{base}/pca/{rel}"]` and `output_files[f"{base}/mca/{rel}"]`.

---

## How to Decode the Output File

The downloaded file (`results.tar.gz.b64.txt`) contains a base64-encoded
`.tar.gz` archive. Use the provided `decode_results.py` script:

```bash
python decode_results.py results.tar.gz.b64.txt --output my_results/
```

Or manually in Python:
```python
import base64, tarfile, io
with open("results.tar.gz.b64.txt") as f:
    tar_bytes = base64.b64decode(f.read())
with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
    tar.extractall("output/")
```

The extracted archive contains the following structure:
```
federated/
    overview/          — federated overview CSV and plots
    numeric/           — federated numeric statistics CSV and plots
    categorical/       — federated categorical statistics CSV and plots
    temporal/          — federated temporal statistics CSV and plots
    summary.json
    report_short.pdf
    report_full.pdf
local/
    node1/
        overview/
        numeric/
        categorical/
        temporal/
        comparison/
        clustering/
            numeric/
                histograms/        — per-cluster variable histograms (fix 19)
                boxplots/          — per-cluster variable boxplots   (fix 19)
                violinplots/       — per-cluster violin plots        (fix 19)
                scatterplots/      — per-cluster scatter plots       (fix 19)
            categorical/
                histograms/
            binary/
                histograms/
            temporal/
                histograms/
                boxplots/
                violinplots/
                scatterplots/
        pca/                       — PCA plots (if ≥10 numeric cols) (fix 20)
        mca/                       — MCA plots (if categorical cols)  (fix 20)
        inferential/
            association_screening.csv
            association_screening.png
            significant_associations.csv
            comparisons_by_<outcome>.csv  (if an outcome column was detected)
            group_comparisons_summary.png
            comparisons/                  (per-column comparison plots)
        summary.json
        report_short.pdf
        report_full.pdf
    node2/
        ...
```
