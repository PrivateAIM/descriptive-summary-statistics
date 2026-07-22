# Fixes and Improvements Plan — Final (Approved)

All decisions have been confirmed. Implementation can begin.

---

## Confirmed Decisions

| Point | Decision |
|-------|----------|
| Section headers | Change all except "Cross-Variable Associations" which stays |
| Long version missing plots | Keep split `missing_values_by_column_*.png`, remove `missingno_bar.png`, keep `missingno_heatmap.png` |
| Short version missing plots | Show only `missingno_bar.png` (with column-count annotation) |
| D2 top_category_share | Rename columns + add explanatory narrative in the report |
| Categorical distributions | Only multi-category (non-binary) variables; split by batch |
| Local categ. distributions | Currently **only sex_distribution.png** exists locally. Add standalone charts for all non-binary multi-category variables; split into batches of 6 per image |
| Federated categ. distributions | `categorical_distributions.png` is one big combined grid — split into batches of 6 columns per image |
| Cluster histograms | Also split into batches of max 6 per image when a cluster is large |
| Availability labels | Keep CSV values unchanged; only fix the display in summary tables (replace underscores with spaces in the `availability` column cells) |
| MCA column map | Split by batch of 5 variables per chart |
| PCA/MCA redundancy | Short mode: overview panel only. Full mode: individual plots only (no overview) |
| Identifier columns | Show both counts: "X analytical columns + Y identifier columns detected" |
| Glossary | In both local and global reports |
| Multi-variable associations | Deferred |

---

## Implementation Plan (File by File)

### 1. `generate_reports/section_definitions.py`
- F1: Change `LOCAL_PCA.short_plot` → `"pca_overview.png"`
- F1: Change `LOCAL_MCA.short_plot` → `"mca_overview.png"`

### 2. `generate_reports/report_utils.py`
- D1: In `create_table()`, replace underscores with spaces in column header display
- D1/H1: In `create_table()`, also replace underscores with spaces in cell values of the `availability` column
- D2: Rename `top_category_share` → `"top cat % (local)"` and `top_category_share_global` → `"top cat % (global)"` in `add_categorical_comparison()`

### 3. `generate_reports/generate_local_report.py`
- A1: Rename page template headers: "Numeric Variables" → "Numeric Section", "Categorical Variables" → "Categorical Section", "Temporal Variables" → "Temporal Section"
- A1: Rename section headings in `_build_numeric_section`, `_build_categorical_section`, `_build_temporal_section`
- C2/F2: `_build_overview_section()`: in short mode show only `missingno_bar.png`; in long mode show split charts + heatmap but NOT `missingno_bar.png`
- D2 explanation: add `NarrativeMessage` explaining `top cat % (local)` and `top cat % (global)` after categorical table
- D3: Ensure overview table renders all rows (investigate why it may appear cut)
- D4/B3: Round numeric values to 3 decimal places in `_render_pairwise_table()` before calling `create_table()`
- E4: In `_build_categorical_section()`, read all `*_distributions_*.png` files from `categorical_dir` instead of only `sex_distribution.png`
- F2: In `_render_reduction_subsection()` full mode, exclude `*_overview.png` from the plot list
- J1: Add explanatory `NarrativeMessage` boxes after section headings for TOM clustering, PCA, MCA, and inferential statistics
- J2: Add medical/hospital glossary appendix section at the end

### 4. `generate_reports/generate_global_report.py`
- A1: Same header renames as local report (check which ones apply)
- J2: Add same glossary appendix section

### 5. `data_report/analyze.py`
- E1: Sex distribution bar — use `PALETTE[i % len(PALETTE)]` per category instead of default single color
- E2: Pie chart title — add `pad=20, y=1.04` to push title above category labels
- I1: Expand identifier detection (see compute_statistics.py change)
- I2: Detect ALL identifier columns into a list, not just the first
- I3/I4: Compute `n_analytical_cols` and `total_values` excluding identifier columns; show both counts in overview
- K1: Span-based temporal frequency selection (D, M, or Y) before calling `compute_temporal_statistics()`

### 6. `data_report/statistical_analysis/local/compute_statistics.py`
- I1: Expand `detect_id_column()` keywords: add `"name"`, `"surname"`, `"first_name"`, `"last_name"`, `"family_name"`, `"firstname"`, `"lastname"`, German: `"vorname"`, `"nachname"`, `"familienname"` to strong_keywords

### 7. `data_report/statistical_analysis/local/inferential_analysis.py`
- B1: Remove `method="auto"` parameter from `compare_two_groups()` and `one_way_group_comparison()`; inline auto-detection unconditionally

### 8. `data_report/generate_figures/data_quality_plots.py`
- C1: `save_missing_heatmap()`: if df has zero missing values, return a sentinel (`None`) so the report builder can show a narrative instead
- C3: `save_missing_bar()`: add subtitle showing column count (e.g., "Showing 47 of 47 columns" or "Showing top 50 of 120 most incomplete columns"); if truncating, sort by ascending completeness (most incomplete first)

### 9. `data_report/generate_figures/clustering_plots.py`
- E3: `save_cluster_heatmaps()`: pass `annotate=False` when matrix size > 15
- E3: `save_tom_heatmap()`, `save_clustered_tom_heatmap()`, `save_correlation_clustermap()`: pass `annotate=False` (these never annotated, confirm)
- E4: `save_cluster_histograms()`: split into chunks of max 6 variables per image when a cluster exceeds 6 variables

### 10. `data_report/generate_figures/pca_plots.py`
- E5: `save_pca_loadings_biplot()`: if number of features > 20, show only top 20 by loading magnitude; add note in title "top 20 features"
- F2: `save_pca_outputs()`: do NOT save `pca_overview.png` as a separate step — it is still saved, but the report builder will exclude it in full mode

### 11. `data_report/generate_figures/mca_plots.py`
- E5: `save_mca_column_map()`: if total labels > 20, show only top 20 by distance from origin; add note "top 20 categories"
- E6: `save_mca_outputs()`: instead of one `mca_column_map_dim1_dim2.png`, generate multiple column map images (batches of 5 source variables), named `mca_column_map_batch_01.png`, etc.

### 12. `data_report/generate_figures/local_descriptive_plots.py`
- E4 (new function): `save_categorical_distributions(df, cols, output_dir, batch_size=6)` — bar chart for each non-binary multi-category column, split into batches; saves `categorical_distributions_01.png`, etc.

### 13. `data_report/generate_figures/federated_descriptive_plots.py`
- E4: `save_federated_categorical_distributions()` — instead of one combined grid, produce batches of 6 columns per image; saves `categorical_distributions_01.png`, etc.

### 14. `data_report/analyze.py` (E4 wiring)
- After computing `categorical_statistics`, call `save_categorical_distributions()` for non-binary, multi-category columns

---

## Items NOT changing (confirmed)
- "Cross-Variable Associations" section name: keep as-is
- Multi-variable (N>2) associations: deferred
- CSV availability values (`common_all` etc.): unchanged
- TOM vs clustered TOM: both kept, just with clearer titles
