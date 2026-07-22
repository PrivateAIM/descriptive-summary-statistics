# Synthetic multi-hospital test dataset (`dataset4`) and related fixes

## Why

The existing test datasets (`dataset1`, `dataset2`, `dataset3`) don't have
enough correlated numeric/categorical columns, or enough hospital "nodes", to
exercise the inferential-statistics screening (`screen_associations`,
`compare_two_groups`, `one_way_group_comparison`,
`correlation_between_two_variables`) and the TOM-based clustering
(`ClusteringManager`) in a meaningful way. This adds a purpose-built
synthetic dataset with deliberate correlation structure across numeric,
categorical, binary, and temporal columns, plus standalone datasets for the
longitudinal / event-based / panel analysis functions.

## What was added

### `data_report/get_data/generate_synthetic_data.py`

Generates four datasets under `data/`:

- **`dataset4/node{1,2,3}/hospital_node{N}.csv`** -- cross-sectional,
  3-hospital, 33-column dataset (380-520 patients/node). Built from shared
  latent severity factors so that:
  - `weight_kg`, `bmi`, `waist_circumference_cm` form a numeric cluster
    (anthropometric).
  - `systolic_bp`, `diastolic_bp`, `glucose_mg_dl`, `hba1c_percent`,
    `ldl_cholesterol` form a numeric cluster (cardiometabolic).
  - `heart_rate`, `temperature_c`, `wbc_count` form a numeric cluster
    (inflammation).
  - `admission_date`, `discharge_date`, `followup_date` form a temporal
    cluster.
  - `smoking_status` and `respiratory_diagnosis` form a categorical cluster.
  - `current_smoker` and `copd` form a binary cluster.
  - `mortality` / `icu_admission` are usable outcome columns (matches
    `OUTCOME_KEYWORD_GROUPS`), driving the automatic outcome comparisons.
  - Each node has different demographics/severity baselines to simulate
    inter-hospital heterogeneity.

- **`dataset4_longitudinal/longitudinal_patients.csv`** -- 150 patients with
  3-6 visits each and trending weight/glucose/BP. `detect_dataset_type`
  classifies this as `"longitudinal"`.

- **`dataset4_event_based/event_based_patients.csv`** -- irregular event log
  (689 events) with a per-patient-constant severity score (no trend).
  `detect_dataset_type` classifies this as `"event_based"`.

- **`dataset4_panel/hospital_panel.csv`** -- 5 hospitals x 36 months of
  admissions/LOS/occupancy/mortality, for `analyze_panel`.

## Tuning the correlation/association strength

### Numeric clusters

Initial pairwise correlations (~0.6-0.76) within the cardiometabolic and
inflammation groups were too weak: with `power=3` and `distance_cut=0.25`,
`ClusteringManager` requires roughly Pearson |r| > ~0.85 for a cluster to
form (similarity is cubed before the TOM transform, and TOM must exceed
0.75). The latent-factor weight/noise ratios were tuned to push pairwise
correlations to ~0.85, which produced 3-5 member numeric clusters as
intended.

### Categorical clusters: `smoking_status` <-> `respiratory_diagnosis`

For a **2-member** cluster, the threshold is higher: Cramer's V must exceed
`0.5**(1/3) ~= 0.794` (derived from the TOM formula for a 2-node clique,
where `TOM = 1.5 * similarity**3`).

- First attempt: `respiratory_diagnosis` sampled conditional on
  `smoking_status` with probabilities `[0.95/0.04/0.01]`,
  `[0.15/0.65/0.20]`, `[0.03/0.07/0.90]` -> Cramer's V ~0.65. Below
  threshold, stayed as separate singleton clusters.
- Final: sharpened to `[0.99/0.01/0.00]`, `[0.05/0.85/0.10]`,
  `[0.01/0.03/0.96]` -> Cramer's V ~0.82-0.86 across all 3 nodes. The pair
  now clusters together and `screen_associations` reports it as a
  significant `cat-cat` association.
- `copd` is now derived as `respiratory_diagnosis == "COPD"` (previously
  computed independently from `respiratory_risk`/`current_smoker`), which
  also pulled `current_smoker` and `copd` into a 2-member binary cluster as a
  side effect.

### Risk check: spurious third-variable clustering

Strengthening one pairwise association risks TOM's shared-neighbour term
pulling in a third variable with only a moderate *direct* association into
the same cluster. `diagnosis_category` was the candidate (it shares the same
underlying severity/respiratory-risk factors). After the change:

- `diagnosis_category` remained a singleton cluster in all 3 nodes.
- Its direct Cramer's V with `smoking_status` / `respiratory_diagnosis` is
  only ~0.16-0.24, far below the clustering threshold.
- `region` / `insurance_type` (Cramer's V ~0.3) also remained unclustered.

No spurious clustering was observed.

## Bug fix: `data_report/generate_figures/mca_plots.py`

`run_mca` called `df[features].select_dtypes(include=["object", "str",
"category", "bool"])`. Under pandas 2.3, the literal `"str"` in `include`
raises `TypeError: numpy string dtypes are not allowed, use 'str' or
'object' instead` via `select_dtypes`'s `invalidate_string_dtypes` check.
This silently broke the MCA section (caught by a try/except in
`analyze.py`) for **every** dataset, not just `dataset4`. Fixed by dropping
`"str"` from the `include` list -- `"object"` already covers plain Python
string columns.

## Verification

- `ClusteringManager` produces the intended multi-member numeric, temporal,
  categorical, and binary clusters on `dataset4/node{1,2,3}`.
- `screen_associations` returns 16 significant `num-num` correlations, ~39
  significant `num-cat` group comparisons, and a significant `cat-cat`
  (`smoking_status` vs `respiratory_diagnosis`) row on node1.
- `detect_dataset_type` correctly labels `dataset4_longitudinal` as
  `"longitudinal"` and `dataset4_event_based` as `"event_based"`.
- `analyze_longitudinal`, `analyze_event_based`, and `analyze_panel` all run
  without errors on the corresponding datasets.
- The full StarModel pipeline (clustering, PCA, MCA, inferential screening,
  local + global PDF reports) runs end-to-end on `dataset4` across all 3
  nodes with no errors (verified by temporarily pointing `dr-analyze` at
  `dataset4`).

## Column-availability heterogeneity in `dataset4`

`make_hospital_dataset` now also generates a `referral_source` categorical
column. `generate_synthetic_data.main()` applies per-node column drops on
top of the shared 34-column schema:

- `node1`: full 34 columns (includes `referral_source`).
- `node2`: drops `referral_source` and `ldl_cholesterol` (32 columns).
- `node3`: drops `referral_source`, `crp_level`, and `followup_date`
  (31 columns).

This gives `DataReportAggregator` a realistic mix across all three column
types for `compute_column_distribution` / `classify_local_columns` to
classify:

- `common_all` (29 columns, present in all 3 nodes).
- `common_partial`: `ldl_cholesterol` (node1+node3), `crp_level`
  (node1+node2), `followup_date` (node1+node2) -- one numeric/numeric/
  temporal example.
- `unique`: `referral_source` (node1 only).

Verified via `results/data_report.pkl`'s `column_distribution_summary` after
a `dr-analyze` run on `dataset4` -- the split matches exactly (29/3/1), and
the federated numeric/categorical/temporal statistics CSVs tag
`ldl_cholesterol`, `crp_level`, `referral_source`, and `followup_date` with
`availability="not_common_all"` while still computing correct pooled
statistics from whichever nodes report them. Each local report's
`column_availability.png` pie chart reflects the per-node common/partial/
unique split (e.g. node1: 30 common_all / 3 common_partial / 1 unique;
node2: 30/2/0; node3: 30/1/0).

Note: `DataReportAggregator`'s per-column `availability` field is currently
binary (`common_all` / `not_common_all`) -- it doesn't distinguish
`common_partial` from `unique` the way `column_distribution_summary` and the
local `column_availability.png` pie charts do. Not changed here since it
wasn't part of this task, but worth knowing if finer-grained federated
labeling is wanted later.

## 2-node restructuring of longitudinal/event-based/panel datasets

`dataset4_longitudinal`, `dataset4_event_based`, and `dataset4_panel` were
each split into `node1`/`node2` subdirectories (previously a single flat
file per dataset), matching the multi-node directory layout `load_dataset`
expects:

- `dataset4_longitudinal/node{1,2}/longitudinal_node{N}.csv` -- 100/80
  patients (seeds 123/124), patient IDs prefixed `L1_`/`L2_`.
- `dataset4_event_based/node{1,2}/event_based_node{N}.csv` -- 100/80
  patients (seeds 456/457), patient IDs prefixed `E1_`/`E2_`.
- `dataset4_panel/node{1,2}/hospital_panel_node{N}.csv` -- node1 has
  `Hospital_1..3`, node2 has `Hospital_4..5` (disjoint hospital sets, seeds
  789/790).

`make_longitudinal_dataset`, `make_event_based_dataset`, and
`make_panel_dataset` were refactored to take a node label / patient-ID
prefix / hospital-ID list, respectively.

### Standalone inspection script

`detect_dataset_type`, `analyze_longitudinal`, `analyze_event_based`, and
`analyze_panel` (in `inferential_analysis.py`) are **not called anywhere in
`analyze.py`** -- the `DataReportAnalyzer`/StarModel pipeline only runs the
cross-sectional path, so `dr-analyze` doesn't exercise these functions
regardless of dataset layout. Added
`data_report/get_data/inspect_special_dataset_types.py`, a standalone script
(run via `python -m data_report.get_data.inspect_special_dataset_types`)
that loads each node's CSV directly and calls the relevant functions so
node1/node2 results can be compared.

Results:

- `dataset4_longitudinal/node{1,2}`: `detect_dataset_type` -> `"longitudinal"`
  for both nodes; `analyze_longitudinal` on `weight_kg` returns per-subject
  slopes for all patients with >=2 visits (100/80 subjects).
- `dataset4_event_based/node{1,2}`: `detect_dataset_type` -> `"event_based"`
  for both nodes; `analyze_event_based` returns event counts, daily density,
  peak day, and inter-event timing for both nodes.
- `dataset4_panel/node{1,2}`: `analyze_panel` on `admissions` returns
  per-hospital trend stats (mean/growth) for all 5 hospitals across both
  nodes. **Note**: `detect_dataset_type` classifies this data as
  `"longitudinal"`, not `"panel"` -- its `"panel"` branch requires at least
  one detected event column (`detect_event_columns`), which hospital-panel
  data doesn't have. This is a pre-existing limitation of
  `detect_dataset_type`'s heuristic, not something introduced here.

## Re-verification

Re-ran `dr-analyze` against the updated `dataset4` (3 nodes, with the
column-availability heterogeneity above) a second time after the changes
above were committed: the pipeline still completes with exit code 0,
generates all 8 local PDFs + 2 global PDFs with no errors, and
`column_distribution_summary` still reports the expected 29 `common_all` /
3 `common_partial` (`ldl_cholesterol`, `crp_level`, `followup_date`) / 1
`unique` (`referral_source`) split.
