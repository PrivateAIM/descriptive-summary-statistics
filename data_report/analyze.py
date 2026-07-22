"""Federated data-report analysis: per-node analyzer and central aggregator.

This module provides the two FLAME ``Star`` components that implement the
federated data-report pipeline:

* ``DataReportAnalyzer`` — runs on every participating node; parses the node's
  raw CSV bytes, computes local descriptive statistics, runs dimensionality
  reduction (PCA / MCA), inferential association screening, and returns a
  serialisable result dictionary.

* ``DataReportAggregator`` — runs on the central coordinator; aggregates the
  per-node result dictionaries into federation-wide statistics, writes all CSV
  tables and PNG figures to ``results/federated_results/``, and triggers
  per-node output writing to ``results/local_results/``.

Module-level helpers ``should_apply_reductions`` and ``combine_node_variances``
support the analyzer and aggregator respectively.
"""
import logging
import os
from io import BytesIO
# from multiprocessing.reduction import duplicate
from pathlib import Path
from typing import Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flame.star import StarAnalyzer, StarAggregator
import json
from collections import Counter

from data_report.config import LABEL_COL, OUTCOME_KEYWORD_GROUPS
from data_report.statistical_analysis.local.compute_statistics import (
    detect_column_types,
    detect_id_column,
    detect_quasi_numeric_categorical_columns,
    compute_numeric_statistics,
    compute_categorical_statistics,
    compute_temporal_statistics
)
from data_report.statistical_analysis.local.compute_statistics import (
    compute_age_histogram,
    count_out_of_range_ages,
)
from data_report.statistical_analysis.local.data_quality import (
    compute_missing_by_column,
    compute_total_missing,
)
from data_report.comparison.utils import (
    compute_column_distribution,
    classify_local_columns)

from data_report.generate_figures.pca_plots import save_pca_outputs
from data_report.generate_figures.mca_plots import save_mca_outputs
from data_report.statistical_analysis.local.inferential_analysis import (
    screen_associations,
    detect_outcome_column,
    compare_two_groups,
    one_way_group_comparison,
    posthoc_test,
)
from data_report.generate_figures.primitives import pie_chart, save_fig

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
# folder for local results
# each node has its own folder inside with the csv files for the tables and the png images for the plots
LOCAL_RESULTS_DIR = Path("results/local_results")
# folder for federated results
FEDERATED_RESULTS_DIR = Path("results/federated_results")


# ---------------------------------------------------------------------------
# Reduction gating
# ---------------------------------------------------------------------------

def should_apply_reductions(df: pd.DataFrame, column_types: dict) -> dict:
    """Return flags indicating which dimensionality-reduction methods have enough data.

    Centralises all threshold decisions so they are easy to find, adjust, and
    reason about rather than being scattered as ad-hoc guards throughout
    ``analysis_method``.

    Thresholds are chosen conservatively so that results remain interpretable:

    * ``pca``: requires at least 10 numeric columns so the PCA decomposition
      is non-trivial.
    * ``mca``: counts only low-cardinality categorical columns (≤ 30 unique
      levels). Requires 8 for MCA.

    Args:

        df (pd.DataFrame): The node's data frame, used to count unique levels
            of categorical columns.
        column_types (dict): Mapping produced by ``detect_column_types``; keys
            are ``"numeric"``, ``"categorical"``, and ``"temporal"``, values
            are lists of column names.

    Returns:

        dict: Boolean flags with keys ``"pca"`` and ``"mca"``.
    """
    n_numeric = len(column_types.get("numeric", []))
    n_cat_usable = sum(
        1 for col in column_types.get("categorical", [])
        if df[col].nunique(dropna=True) <= 30
    )
    return {
        "pca": n_numeric >= 10,
        "mca": n_cat_usable >= 8,
    }


def combine_node_variances(node_stats, global_mean: float) -> float:
    """Pool per-node sample variances into the combined-dataset sample variance.

    Uses the exact decomposition from Chan, Golub & LeVeque (1979): the total
    sum of squared deviations from the global mean equals the sum of each
    node's within-node sum of squares ``(n - 1) * var`` plus the between-node
    term ``n * (mean - global_mean) ** 2``.  Dividing by ``N - 1`` (not
    ``N - K``) yields the exact sample variance (``ddof=1``) of the pooled
    dataset.

    Args:

        node_stats (iterable): Iterable of ``(n, mean, var)`` tuples, one per
            node.  ``var`` must be the local sample variance (``ddof=1``,
            i.e. divided by ``n - 1``).
        global_mean (float): The already-computed weighted global mean across
            all nodes.

    Returns:

        float: Pooled sample variance of the combined dataset.
    """
    node_stats = list(node_stats)
    total_n = sum(n for n, _, _ in node_stats)
    if total_n <= 1:
        return 0.0

    sum_squares = sum(
        (n - 1) * var + n * (mean - global_mean) ** 2
        for n, mean, var in node_stats
    )
    return sum_squares / (total_n - 1)


# ---------------------------------------------------------------------------
# Analyzer  (runs on each node)
# ---------------------------------------------------------------------------
class DataReportAnalyzer(StarAnalyzer):
    """FLAME StarAnalyzer that computes local statistics for one federated node.

    On each invocation the analyzer receives the node's raw file bytes, parses
    them into a DataFrame, detects column types and identifier columns, runs
    PCA / MCA / inferential screening, and returns a result dictionary that
    the central ``DataReportAggregator`` will aggregate.

    All figures that require the raw DataFrame (histograms, boxplots, scatter
    plots, missing-value heatmaps, etc.) are generated here before returning
    because the DataFrame is not forwarded to the aggregator.
    """

    def __init__(self, flame):
        """Initialise the analyzer with the FLAME runtime handle."""
        super().__init__(flame)

    def analysis_method(self, data, aggregator_results):
        """Run all local analyses on the node's data and return a result dict.

        Steps performed in order:

        1. Parse the raw CSV bytes (auto-detect delimiter).
        2. Normalise column names and replace sentinel missing values.
        3. Convert date-like object columns to ``datetime64`` and detect
           identifier columns (which are excluded from all analyses).
        4. Determine which dimensionality-reduction methods are applicable.
        5. Run PCA and MCA if thresholds are met and save their figures.
        6. Save per-category distribution plots.
        7. Run inferential association screening and optional outcome-driven
           group comparisons.
        8. Compute descriptive statistics (numeric, categorical, temporal),
           detect sex / age distributions, and measure data quality.
        9. Save missing-value bar and heatmap figures.

        Args:

            data (list | bytes): Raw CSV file bytes, either as plain bytes or
                wrapped in a ``[{filename: bytes}]`` list as produced by the
                FLAME S3 data source.
            aggregator_results: Results returned by the aggregator in previous
                iterations; unused in the current single-iteration design.

        Returns:

            dict: Serialisable result dictionary containing node metadata,
                column-type mappings, descriptive statistics, data-quality
                metrics, and age / sex distributions.
        """
        # --- Parse CSV bytes ------------------------------------------------
        if isinstance(data, list) and data and isinstance(data[0], dict):
            files = data[0]
            # Priority-based file selection (mirrors hub_entrypoint_10.py) --
            # a node's payload can carry more than one file (e.g. a README
            # alongside the CSV), and picking "whichever key comes first in
            # dict order" is filesystem-iteration-order-dependent, not a
            # real file-type check. First match wins:
            #   1. Explicitly unlabeled file (datasets shipping both labeled/unlabeled)
            #   2. Any CSV that is NOT the labeled version
            #   3. Any CSV at all (last-resort fallback)
            csv_key = (
                next((k for k in files if k.lower().endswith("unlabeled.csv")), None)
                or next((k for k in files if k.lower().endswith(".csv")
                         and "labeled" not in k.lower()), None)
                or next((k for k in files if k.lower().endswith(".csv")), None)
            )
            if csv_key is None:
                raise ValueError(
                    f"No CSV file found in data payload. Available keys: {list(files.keys())}"
                )
            file_bytes = files[csv_key]
        else:
            file_bytes = data
        if isinstance(file_bytes, str):
            file_bytes = file_bytes.encode("utf-8")
        # sep=None tells pandas to figure out the delimiter automatically
        df = pd.read_csv(BytesIO(file_bytes), sep=None, engine="python")
        # clean
        print("Columns:", df.columns.tolist())
        print("Shape:", df.shape)
        df = df.replace(["", "NULL", "null", "NaN"], pd.NA)
        # normalize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        # Normalization can collapse two originally-distinct headers into the
        # same name (e.g. "Blood Pressure" and "blood_pressure" both become
        # "blood_pressure"). df[col] on a duplicated name returns a DataFrame
        # instead of a Series and crashes detect_column_types downstream, so
        # any collision must be resolved here before anything else sees it.
        if df.columns.duplicated().any():
            seen: dict = {}
            deduped = []
            for col in df.columns:
                if col in seen:
                    seen[col] += 1
                    deduped.append(f"{col}_{seen[col]}")
                else:
                    seen[col] = 0
                    deduped.append(col)
            df.columns = deduped

        node_id = getattr(self, "id", "unknown")

        # Convert date-like object columns to datetime64 BEFORE identifier
        # detection, so that detect_id_column's is_datetime64 guard correctly
        # skips temporal columns (e.g. discharge_date, therapy_session1 in
        # small datasets where high uniqueness ratios would otherwise fire).
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                continue

            numeric_try = pd.to_numeric(df[col], errors="coerce")
            if numeric_try.notna().mean() > 0.9:
                continue

            converted = pd.to_datetime(df[col], errors="coerce", format="mixed")

            original_non_null = df[col].notna().sum()
            converted_non_null = converted.notna().sum()

            if original_non_null > 0 and (converted_non_null / original_non_null) > 0.6:
                df[col] = converted

        # Detect ALL identifier columns (not just the first one).
        # Must run after date conversion so datetime64 columns are skipped.
        id_columns = [col for col in df.columns if detect_id_column(df[col], col)]
        if id_columns:
            print(f"Identifier columns detected: {id_columns}")
        else:
            print("No identifier columns detected")

        # Rename the first identifier column to patient_id for temporal analysis
        if id_columns:
            primary_id = id_columns[0]
            df = df.rename(columns={primary_id: "patient_id"})
            id_columns[0] = "patient_id"
            patient_series = df["patient_id"]
        else:
            patient_series = None

        n_rows, n_cols = df.shape
        n_analytical_cols = n_cols - len(id_columns)

        column_types = detect_column_types(df)

        # All identifier columns are kept in df for record counts/joins but
        # must not enter PCA/MCA, descriptive statistics, or inferential
        # screening.
        for dtype in column_types:
            column_types[dtype] = [c for c in column_types[dtype] if c not in id_columns]

        reductions = should_apply_reductions(df, column_types)

        if node_id != "unknown":
            node_index = int(node_id.split("_")[-1])
            node_number = node_index + 1
        else:
            node_number = 0

        # ----------------------------------
        # PCA (standalone descriptive visualization)
        #
        # PCA here projects *samples* into a reduced numeric space purely to
        # visualize and describe structure -- see pca.py docstring.
        try:
            if reductions["pca"]:
                pca_target = LABEL_COL if LABEL_COL in df.columns else None
                save_pca_outputs(
                    df,
                    column_types["numeric"],
                    LOCAL_RESULTS_DIR / f"node{node_number}" / "pca",
                    target=pca_target,
                )
        except Exception as e:
            print(f"PCA section error: {e}")

        # ----------------------------------
        # MCA (standalone descriptive visualization)
        #
        # Categorical analog of the PCA section above: projects categorical
        # *levels* and samples into a low-dimensional association space for
        # description/visualization only -- independent of the PCA section.
        # See mca.py docstring.
        try:
            if reductions["mca"]:
                categorical_feature_cols = [
                    col for col in column_types["categorical"]
                    if df[col].nunique(dropna=True) <= 30
                ]
                mca_target = LABEL_COL if LABEL_COL in df.columns else None

                # Binary indicator columns (e.g. symptom flags coded 0/1) are
                # classified as "categorical" by detect_column_types but are
                # often stored as numeric (float64/int64) dtype. run_mca
                # only accepts object/category/bool columns, so recode any
                # numeric-dtype features here -- a 2-level numeric column is
                # just as valid a categorical feature for MCA as "yes"/"no".
                mca_df = df.copy()
                for col in categorical_feature_cols:
                    if pd.api.types.is_numeric_dtype(mca_df[col]):
                        mca_df[col] = mca_df[col].astype("category")

                save_mca_outputs(
                    mca_df,
                    categorical_feature_cols,
                    LOCAL_RESULTS_DIR / f"node{node_number}" / "mca",
                    target=mca_target,
                )
        except Exception as e:
            print(f"MCA section error: {e}")

        # ----------------------------------
        # Categorical distributions (non-binary multi-category variables only)
        try:
            from data_report.generate_figures.local_descriptive_plots import (
                save_categorical_distributions,
            )
            save_categorical_distributions(
                df,
                column_types["categorical"],
                LOCAL_RESULTS_DIR / f"node{node_number}" / "categorical",
                node_label=f"node{node_number}",
            )
        except Exception as e:
            print(f"Categorical distributions error: {e}")

        # Note columns that ended up categorical only because a few values
        # (e.g. a censored lab result like "<5") couldn't be parsed as
        # numbers -- surfaced as a report notice rather than silently
        # coercing those values to NaN.
        quasi_numeric_cols = []
        try:
            quasi_numeric_cols = detect_quasi_numeric_categorical_columns(
                df, column_types["categorical"]
            )
            if quasi_numeric_cols:
                categorical_dir = LOCAL_RESULTS_DIR / f"node{node_number}" / "categorical"
                categorical_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"feature": quasi_numeric_cols}).to_csv(
                    categorical_dir / "quasi_numeric_columns.csv", index=False
                )
        except Exception as e:
            print(f"Quasi-numeric column detection error: {e}")

        # ----------------------------------
        # Numeric distributions (histograms and boxplots, batched)
        try:
            from data_report.generate_figures.local_descriptive_plots import (
                save_numeric_histograms,
                save_numeric_boxplots,
            )
            save_numeric_histograms(
                df,
                column_types["numeric"],
                LOCAL_RESULTS_DIR / f"node{node_number}" / "numeric",
                node_label=f"node{node_number}",
            )
            save_numeric_boxplots(
                df,
                column_types["numeric"],
                LOCAL_RESULTS_DIR / f"node{node_number}" / "numeric",
                node_label=f"node{node_number}",
            )
        except Exception as e:
            print(f"Numeric distributions error: {e}")

        # ----------------------------------
        # Inferential statistics (automatic association screening + optional
        # outcome-driven comparisons). Runs unattended -- no fixed "the"
        # variable is assumed; every numeric-numeric and categorical-categorical
        # pair (cardinality permitting) is tested and flagged by FDR-corrected
        # significance + effect size, not p-value alone. No automatic
        # regression-target selection: using an auto-detected outcome as a
        # regression target risks data dredging / circular analysis.
        try:
            inferential_dir = LOCAL_RESULTS_DIR / f"node{node_number}" / "inferential"
            screening = screen_associations(df, column_types)
            inferential_dir.mkdir(parents=True, exist_ok=True)
            screening.to_csv(inferential_dir / "association_screening.csv", index=False)
            screening[screening["significant"]].to_csv(
                inferential_dir / "significant_associations.csv", index=False
            )

            from data_report.generate_figures.inferential_plots import (
                save_association_screening,
                save_two_group_comparison,
                save_one_way_comparison,
                save_group_comparisons_summary,
                save_posthoc_heatmap,
            )
            save_association_screening(
                screening, inferential_dir, node_label=f"node{node_number}"
            )

            # Outcome-driven comparisons (only if a usable outcome column is
            # detected). Every numeric column is compared across the outcome's
            # groups -- two groups -> compare_two_groups, 3+ -> one_way_group_comparison
            # -- and flattened into one row per comparison.
            outcome_col = detect_outcome_column(df, column_types, OUTCOME_KEYWORD_GROUPS)
            if outcome_col is not None:
                outcome_rows = []
                n_outcome_groups = df[outcome_col].nunique(dropna=True)
                cmp_results = {}
                for num_col in column_types["numeric"]:
                    if num_col == outcome_col:
                        continue
                    try:
                        if n_outcome_groups == 2:
                            cmp = compare_two_groups(df, num_col, outcome_col)
                        else:
                            cmp = one_way_group_comparison(df, num_col, outcome_col)
                            if (
                                cmp.get("p_value") is not None
                                and not np.isnan(cmp["p_value"])
                                and cmp["p_value"] < 0.05
                            ):
                                try:
                                    cmp["posthoc"] = posthoc_test(
                                        df, num_col, outcome_col, cmp["method"]
                                    )
                                except Exception:
                                    logger.debug(
                                        "Post-hoc test skipped for %s vs %s",
                                        num_col, outcome_col, exc_info=True,
                                    )
                    except Exception:
                        # Expected for sparsely-populated columns where one or
                        # more outcome groups end up with no usable values
                        # (e.g. visit-specific fields not recorded for every
                        # patient) -- not every numeric column is testable
                        # against every outcome, so this is logged at debug
                        # level rather than as a warning.
                        logger.debug(
                            "Outcome comparison skipped for %s vs %s",
                            num_col, outcome_col, exc_info=True,
                        )
                        continue
                    row = {
                        "value_col": num_col,
                        "outcome_col": outcome_col,
                        "method": cmp["method"],
                        "statistic": cmp["statistic"],
                        "p_value": cmp["p_value"],
                    }
                    for metric, value in cmp.get("effect_size", {}).items():
                        row[f"effect_size_{metric}"] = value
                    outcome_rows.append(row)
                    cmp_results[num_col] = cmp

                if outcome_rows:
                    comparison_df = pd.DataFrame(outcome_rows)
                    comparison_df.to_csv(
                        inferential_dir / f"comparisons_by_{outcome_col}.csv", index=False
                    )
                    save_group_comparisons_summary(
                        comparison_df, inferential_dir, node_label=f"node{node_number}"
                    )
                    comparisons_dir = inferential_dir / "comparisons"
                    for num_col, cmp in cmp_results.items():
                        try:
                            if n_outcome_groups == 2:
                                save_two_group_comparison(
                                    df, num_col, outcome_col, cmp, comparisons_dir,
                                    node_label=f"node{node_number}",
                                )
                            else:
                                save_one_way_comparison(
                                    df, num_col, outcome_col, cmp, comparisons_dir,
                                    node_label=f"node{node_number}",
                                )
                        except Exception:
                            logger.warning(
                                "Comparison plot failed for %s vs %s",
                                num_col, outcome_col, exc_info=True,
                            )
                            continue
                        posthoc_df = cmp.get("posthoc")
                        if posthoc_df is not None:
                            try:
                                save_posthoc_heatmap(
                                    posthoc_df, cmp["method"], num_col, outcome_col,
                                    comparisons_dir, node_label=f"node{node_number}",
                                )
                                posthoc_df.to_csv(
                                    comparisons_dir / f"posthoc_{num_col}_vs_{outcome_col}.csv"
                                )
                            except Exception:
                                logger.warning(
                                    "Post-hoc results failed for %s vs %s",
                                    num_col, outcome_col, exc_info=True,
                                )
        except Exception as e:
            print(f"Inferential statistics section error: {e}")

        temporal_cols = column_types["temporal"]
        numeric_cols = column_types["numeric"]
        categorical_cols = column_types["categorical"]
        all_columns = list(df.columns)

        # split data
        numeric_df = df[column_types["numeric"]]
        categorical_df = df[column_types["categorical"]]
        temporal_df = df[column_types["temporal"]]

        numeric_statistics = compute_numeric_statistics(numeric_df)
        categorical_statistics = compute_categorical_statistics(categorical_df)
        temporal_statistics = compute_temporal_statistics(temporal_df, patient_series, freq="M")

        means = {
            col: stats["mean"]
            for col, stats in numeric_statistics.items()
        }

        # detect sex/gender column
        sex_col = next(
            (col for col in df.columns if any(k in col for k in ["sex", "gender"])),
            None
        )

        # normalize values
        sex_counts = {}
        if sex_col:
            df[sex_col] = (
                df[sex_col]
                .astype(str)
                .str.strip()
                .str.lower()
                )
            # map known values
            df[sex_col] = df[sex_col].replace({
                "m": "male",
                "f": "female",
                "nb": "non-binary",
                "nonbinary": "non-binary"
            })
            valid_categories = {"male", "female", "non-binary"}
            df[sex_col] = df[sex_col].where(df[sex_col].isin(valid_categories))
            # nans are automatically excluded when counting
            sex_counts = df[sex_col].value_counts(dropna=True).to_dict()

        # data quality
        missing_by_col = compute_missing_by_column(df)
        total_missing = compute_total_missing(df)

        # Identifier columns carry no missing-value signal (they're always
        # filled), so completeness is measured on analytical columns only.
        total_values = n_rows * n_analytical_cols
        missing_values_percentage = (total_missing / total_values * 100) if total_values else 0.0
        # check how many duplicates there are
        n_duplicates = int(df.duplicated().sum())

        # age histogram
        age_hist, age_edges = compute_age_histogram(df)
        age_out_of_range_count = count_out_of_range_ages(df)

        # data quality overview plots (generated here while raw df is available)
        from data_report.generate_figures.data_quality_plots import (
            save_missing_bar,
            save_missing_heatmap,
        )
        if node_id != "unknown":
            node_index = int(node_id.split("_")[-1])  # → 0
            node_number = node_index + 1  # → 1
        else:
            node_number = 0
        node_dir = LOCAL_RESULTS_DIR / f"node{node_number}"
        node_dir.mkdir(parents=True, exist_ok=True)
        overview_dir = node_dir / "overview"
        overview_dir.mkdir(parents=True, exist_ok=True)

        save_missing_bar(df, overview_dir / "missingno_bar.png")
        save_missing_heatmap(df, overview_dir / "missingno_heatmap.png")

        return {
            "node_id": node_id,
            "all_columns": all_columns,
            "n_rows": int(n_rows),
            "n_cols": int(n_cols),
            "n_analytical_cols": int(n_analytical_cols),
            "id_columns": id_columns,
            "total_values": total_values,
            "numeric_statistics": numeric_statistics,
            "categorical_statistics": categorical_statistics,
            "temporal_statistics": temporal_statistics,
            "means": means,
            "missing_by_col": missing_by_col,
            "total_missing": total_missing,
            "missing_values_percentage": missing_values_percentage,
            "n_duplicates": n_duplicates,
            "age_edges": age_edges,
            "age_hist": age_hist,
            "age_out_of_range_count": age_out_of_range_count,
            "quasi_numeric_categorical_columns": quasi_numeric_cols,
            "sex_counts": sex_counts,
            "column_types": column_types,
        }

# ---------------------------------------------------------------------------
# Aggregator  (central coordinator)
# ---------------------------------------------------------------------------

class DataReportAggregator(StarAggregator):
    """FLAME StarAggregator that federates per-node analysis results.

    Receives one result dictionary per node from ``DataReportAnalyzer``,
    computes federation-wide descriptive statistics (pooled mean / variance,
    merged category counts, merged temporal observations), generates
    per-node comparison metrics, writes all CSV tables and PNG figures, and
    returns the federated result dictionary.

    Outputs are written to:

    * ``results/local_results/node<N>/`` — per-node CSVs, plots, and
      comparison metrics.
    * ``results/federated_results/`` — federation-wide CSVs and plots.
    """

    def __init__(self, flame):
        """Initialise the aggregator with the FLAME runtime handle."""
        super().__init__(flame)
    # --- main aggregation ---------------------------------------------------
    def aggregation_method(self, analysis_results: list):
        """Aggregate per-node analysis results into federation-wide statistics.

        Computes:

        * Federated overview metrics (total rows, total missing, etc.).
        * Column availability across nodes (common to all / partial / unique).
        * Global numeric statistics (weighted mean, pooled variance, global
          min / max) using ``combine_node_variances``.
        * Global categorical statistics (merged counts, relative frequencies,
          mode).
        * Global temporal statistics (merged observations per period).
        * Age histogram (sum of per-node histograms) and sex counts.
        * Per-node comparison metrics (patient contribution, completeness
          contribution, local vs. global missingness ratio, numeric column
          alignment with global means).

        After aggregation, calls ``_save_local_node_results``,
        ``_save_federated_tables``, and ``_make_plots`` to persist all outputs.

        Args:

            analysis_results (list): List of result dictionaries, one per
                node, as returned by ``DataReportAnalyzer.analysis_method``.

        Returns:

            dict: Federated result dictionary with keys for global statistics,
                column coverage, age / sex distributions, and iteration count.

        Raises:

            ValueError: If ``analysis_results`` is empty.
        """
        if not analysis_results:
            raise ValueError(
                "aggregation_method received no analysis results from any node"
            )

        n_nodes = len(analysis_results)

        # Aggregate descriptive statistics
        total_rows = sum(r["n_rows"] for r in analysis_results)
        n_cols = max(r["n_cols"] for r in analysis_results)
        total_missing = sum(r["total_missing"] for r in analysis_results)
        # Sum of each node's own row*col count, not total_rows * max(n_cols) --
        # the latter overstates the true cell count whenever nodes differ in
        # column count (a federation with heterogeneous schemas is the norm,
        # not the exception here). Uses n_analytical_cols (excludes identifier
        # columns), matching every other quantity computed from this total
        # (total_values, total_missing) -- identifier columns are never missing
        # and carry no completeness signal, so counting their cells here would
        # inflate the denominator relative to every numerator derived from it.
        n_total_values = sum(r["n_rows"] * r["n_analytical_cols"] for r in analysis_results)
        total_missing_percentage = (total_missing / n_total_values * 100) if n_total_values else 0.0

        #-----
        # count how many nodes contain each column (no raw data, only presence)
        # total number of nodes
        total_sites = n_nodes
        column_node_counts, column_distribution_summary = compute_column_distribution(
            analysis_results,
            total_sites=n_nodes
        )
        global_availability_map = {}
        for col, count in column_node_counts.items():
            if count == total_sites:
                global_availability_map[col] = "common_all"
            else:
                global_availability_map[col] = "not_common_all"
        coverage_df = pd.DataFrame.from_dict(
            column_node_counts,
            orient="index",
            columns=["count"]
        )

        missing_by_col: Dict[str, int] = {}
        for r in analysis_results:
            for k, v in r.get("missing_by_col", {}).items():
                missing_by_col[k] = missing_by_col.get(k, 0) + v

        # federated global numeric statistics
        global_numeric = {}
        # collect all numeric columns
        all_numeric_cols = []
        for r in analysis_results:
            for col in r.get("numeric_statistics", {}).keys():
                # avoid duplicates
                if col not in all_numeric_cols:
                    # append to keep the right order
                    all_numeric_cols.append(col)
        for col in all_numeric_cols:
            # total number of samples across all nodes
            total_n = 0
            # sum of(mean * count)
            weighted_mean_sum = 0
            # per-node (n, mean, var) used by combine_node_variances below
            node_stats = []
            global_min = None
            global_max = None
            # loop over each node to aggregate statistics for this column
            for r in analysis_results:
                stats = r.get("numeric_statistics", {}).get(col)
                if not stats:
                    continue
                # extract local statistics
                # number of samples in the node
                n = stats.get("count", 0)
                # number of missing values
                # local statistics
                mean = stats.get("mean", 0)
                var = stats.get("variance", 0)
                col_min = stats.get("min")
                col_max = stats.get("max")

                # skip empty nodes
                if n == 0:
                    continue
                # update totals for federated mean/variance
                total_n += n
                # accumulate weighted mean
                weighted_mean_sum += n * mean
                node_stats.append((n, mean, var))
                # min / max
                if global_min is None or col_min < global_min:
                    global_min = col_min
                if global_max is None or col_max > global_max:
                    global_max = col_max
            # if we have data we compute the statistics
            if total_n > 0:
                # weighted mean
                global_mean = weighted_mean_sum / total_n
                # pooled sample variance across nodes (Chan et al., 1979) --
                # see combine_node_variances for the exact formula and why
                # this replaces the population-variance identity used before
                global_var = combine_node_variances(node_stats, global_mean)
                # std is the square root of the variance
                global_std = np.sqrt(global_var)

                # store results for the column
                global_numeric[col] = {
                    "mean": round(global_mean, 3),
                    "variance": round(global_var, 3),
                    "std": round(global_std, 3),
                    "min": global_min,
                    "max": global_max,
                    "count": total_n,
                    # if column exists, return total missing
                    # if column doesn’t exist, return 0 (safe fallback)
                    "missing_count": missing_by_col.get(col, 0),
                    "availability": global_availability_map.get(col, "unknown")
                }

        # federated global categorical statistics
        global_categorical = {}
        # get all categorical columns
        all_categorical_cols = []
        for r in analysis_results:
            for col in r.get("categorical_statistics", {}).keys():
                # avoid duplicates
                if col not in all_categorical_cols and col != "patient_id":
                    # append to keep the right order
                    all_categorical_cols.append(col)

        for col in all_categorical_cols:
            total_counts = Counter()
            total_n = 0
            for r in analysis_results:
                stats = r.get("categorical_statistics", {}).get(col)
                if not stats:
                    continue
                counts = stats.get("category_counts", {})
                total_counts.update(counts)
                total_n += sum(counts.values())

            if total_n > 0:
                rel_freq = {k: v / total_n for k, v in total_counts.items()}
                mode = max(total_counts, key=total_counts.get)

                global_categorical[col] = {
                    "counts": dict(total_counts),
                    "relative_freq": {k: round(v, 3) for k, v in rel_freq.items()},
                    "mode": mode,
                    "num_categories": len(total_counts),
                    "missing_count": missing_by_col.get(col, 0),
                    "availability": global_availability_map.get(col, "unknown")
                }

        # federated global temporal statistics
        global_temporal = {}
        # get all categorical columns
        all_temporal_cols = []
        for r in analysis_results:
            for col in r.get("temporal_statistics", {}).keys():
                # avoid duplicates
                if col not in all_temporal_cols :
                    # append to keep the right order
                    all_temporal_cols.append(col)

        for col in all_temporal_cols:
            global_counts = {}
            for r in analysis_results:
                stats = r.get("temporal_statistics", {}).get(col)
                if not stats:
                    continue
                # merge counts
                obs = stats.get("observations_per_period", {})
                for k, v in obs.items():
                    if isinstance(k, pd.Period):
                        k = str(k)  # e.g. "2021-01"
                    else:
                        parsed = pd.to_datetime(k, errors="coerce")
                        if pd.isna(parsed):
                            # unparseable period -- drop rather than bucket
                            # observations under the literal string "NaT"
                            continue
                        k = str(parsed)

                    global_counts[k] = global_counts.get(k, 0) + v
            if global_counts:
                most_active = max(global_counts, key=global_counts.get)

                global_temporal[col] = {
                    "counts_per_period": global_counts,
                    "most_active_period": most_active,
                    "missing_count": missing_by_col.get(col, 0),
                    "availability": global_availability_map.get(col, "unknown")
                }

        # Any node's edges work equally well (compute_age_histogram uses a fixed
        # 0..100 step-5 grid) -- take the first non-None one rather than
        # assuming analysis_results[0] has usable age data, so the federation
        # doesn't silently lose its age histogram just because the first node
        # happens to lack a valid age column.
        age_edges = next(
            (r["age_edges"] for r in analysis_results if r.get("age_edges") is not None),
            None,
        )
        age_hist = None
        if age_edges is not None:
            age_hist = np.zeros(len(age_edges) - 1)
            for r in analysis_results:
                if r.get("age_hist") is not None:
                    age_hist += np.array(r["age_hist"])
            age_hist = age_hist.tolist()

        sex_counts: Dict[str, int] = {}
        for r in analysis_results:
            for k, v in r.get("sex_counts", {}).items():
                sex_counts[k] = sex_counts.get(k, 0) + v

        age_out_of_range_count = sum(
            r.get("age_out_of_range_count", 0) or 0 for r in analysis_results
        )

        # comparison
        comparison_results_per_node = []

        for r in analysis_results:
            node_comparison = {
                "node_id": r["node_id"],
                "column_comparison": {},
                "overview_comparison": {},
                "numeric_comparison": {},
                "categorical_comparison": {}
                # "temporal_comparison": {},
            }

            local_columns = set()

            local_columns.update(r.get("numeric_statistics", {}).keys())
            local_columns.update(r.get("categorical_statistics", {}).keys())
            local_columns.update(r.get("temporal_statistics", {}).keys())

            column_labels = classify_local_columns(local_columns, total_sites, column_node_counts)
            node_comparison["column_comparison"] = column_labels

            #-----general comparison
            n_rows = r.get("n_rows", 0)
            local_total_values = r.get("total_values", 0)
            local_missing = r.get("total_missing", 0)

            # 1.
            patient_contribution = (n_rows / total_rows) * 100 if n_rows else 0
            # 2.
            completeness = (
                (local_total_values - local_missing) / n_total_values
                if n_total_values else 0
            )
            # 3.
            usable_data_contribution = (
                (local_total_values - local_missing) / (n_total_values - total_missing)
                if (n_total_values - total_missing) else 0
            )
            # 4.
            local_missing_rate = (
                local_missing / local_total_values
                if local_total_values else 0
            )
            # 5.
            global_missing_rate = (
                total_missing / n_total_values
                if n_total_values else 0
            )

            relative_missing = (
                local_missing_rate / global_missing_rate
                if global_missing_rate else 0
            )

            total_value_contribution = (
                local_total_values / n_total_values
                if n_total_values else 0
            )
            overview_comp = {
                "patient_contribution": round(patient_contribution, 3),
                "completeness": round(completeness, 3),
                "usable_data_contribution": round(usable_data_contribution, 3),
                "local_missing_rate": round(local_missing_rate, 3),
                "relative_missing": round(relative_missing, 3),
                "total_value_contribution": round(total_value_contribution, 3),
            }
            node_comparison["overview_comparison"] = overview_comp

            # ---------- NUMERIC COMPARISON (UNCHANGED LOGIC) ----------
            local_numeric = r.get("numeric_statistics", {})
            numeric_comp = {}

            for col, stats in local_numeric.items():
                if col not in global_numeric:
                    continue

                local_mean = stats.get("mean")
                global_mean = global_numeric[col].get("mean")

                if local_mean is None or global_mean is None:
                    continue

                diff = local_mean - global_mean

                if abs(diff) < 1e-6:
                    category = "aligned"
                elif diff > 0:
                    category = "above_global"
                else:
                    category = "below_global"

                numeric_comp[col] = {
                    "comparison_category": category
                }

            node_comparison["numeric_comparison"] = numeric_comp

            # only once at  the end append all results
            comparison_results_per_node.append(node_comparison)


        self._save_local_node_results(analysis_results, comparison_results_per_node)

        federated_results = {
            "n_nodes": n_nodes,
            "total_rows": total_rows,
            "n_cols": n_cols,
            "n_total_values": n_total_values,
            "column_node_counts": column_node_counts,
            "column_coverage": coverage_df.to_dict(orient="index"),
            "column_distribution_summary": column_distribution_summary,
            "global_numeric": global_numeric,
            "global_categorical": global_categorical,
            "global_temporal": global_temporal,
            "total_missing": total_missing,
            "total_missing_percentage": total_missing_percentage,
            "global_missing_rate": global_missing_rate,
            "missing_by_col": missing_by_col,
            "age_edges": age_edges,
            "age_hist": age_hist,
            "age_out_of_range_count": age_out_of_range_count,
            "sex_counts": sex_counts,
            "iteration": self.num_iterations,
        }
        # federated_results["column_coverage"] = coverage_df.to_dict(orient="index")


        # save federated results
        self._save_federated_tables(
          federated_results,
        )

        # save federated plots
        self._make_plots(federated_results)

        return federated_results

    # --- convergence check --------------------------------------------------

    def has_converged(self, result, last_result) -> bool:
        """Signal that aggregation has converged after the first iteration.

        The data-report pipeline is a single-pass, non-iterative analysis, so
        this method always returns ``True``.

        Args:

            result: The current aggregated result (unused).
            last_result: The aggregated result from the previous iteration
                (unused).

        Returns:

            bool: Always ``True``.
        """
        # TODO (optional): if the parameter 'simple_analysis' in 'StarModel' is set to False,
        #  this function defines the exit criteria in a multi-iterative analysis (otherwise ignored)
        # return self.num_iterations >= 1
        return True  # Return True to indicate convergence in this simple analysis
#-----------------------------

    def _make_plots(self, federated_results):
        """Generate and save all federated descriptive figures."""
        from data_report.generate_figures.federated_descriptive_plots import save_all_federated_plots
        save_all_federated_plots(federated_results, FEDERATED_RESULTS_DIR)

#----------------------------------------------------------------------------------------------
    # create csv and png files for each node
    def _save_local_node_results(self, analysis_results, comparison_results):
        """Write per-node CSV tables, figures, and comparison outputs to disk."""
        LOCAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        # DATA_DIR = '/Users/nouryassine/Desktop/DataSummaryReport/data'  # unused
        # process results for each node
        for r in analysis_results:

            raw_node_id = r["node_id"]  # "node_0"
            # extract number after underscore
            node_index = int(raw_node_id.split("_")[-1])  # → 0
            # human-readable index
            node_number = node_index + 1  # → 1

            # clean folder name
            node_dir = LOCAL_RESULTS_DIR / f"node{str(node_number)}"
            node_dir.mkdir(parents=True, exist_ok=True)

            # make a folder for overview results inside the node folder
            overview_dir = node_dir / "overview"
            overview_dir.mkdir(parents=True, exist_ok=True)
            # make a folder for numeric results inside the node folder
            numeric_dir = node_dir / "numeric"
            numeric_dir.mkdir(parents=True, exist_ok=True)
            # categorical
            categorical_dir = node_dir / "categorical"
            categorical_dir.mkdir(parents=True, exist_ok=True)
            # temporal
            temporal_dir = node_dir / "temporal"
            temporal_dir.mkdir(parents=True, exist_ok=True)
            # comparison
            comparison_dir = node_dir / "comparison"
            comparison_dir.mkdir(parents=True, exist_ok=True)

            # extract the statistical information
            numeric_statistics = r.get("numeric_statistics", {})
            categorical_statistics = r.get("categorical_statistics", {})
            temporal_statistics = r.get("temporal_statistics", {})

            node_comp = next(
                (c for c in comparison_results if c["node_id"] == r["node_id"]),
                {}
            )
            column_comparison = node_comp.get("column_comparison", {})
            column_status_map = column_comparison
            overview_comparison = node_comp.get("overview_comparison", {})
            numeric_comparison = node_comp.get("numeric_comparison", {})


            # count column categories
            all_common_counts = 0
            partially_common_counts = 0
            unique_counts = 0
            for i in column_comparison.values():
                if i == "common_all":
                    all_common_counts += 1
                elif  i == "common_partial":
                    partially_common_counts += 1
                elif i == "unique_local":
                    unique_counts += 1
                else:
                    print(f"Unexpected column status: {i}")

            column_counts = {
                "common_all": all_common_counts,
                "common_partial": partially_common_counts,
                "unique_local": unique_counts
            }

            # pie chart: proportion of columns that are common/partial/unique across nodes
            n_nodes = len(analysis_results)
            fig, ax = plt.subplots()
            pie_chart(
                ax,
                [all_common_counts, partially_common_counts, unique_counts],
                ["Common", "Partial", "Unique"],
                colors=["#A6D8A8", "#86C5F9", "#FFCC80"],
                title=f'Column Availability ({n_nodes} node{"s" if n_nodes > 1 else ""})',
                # Only 3 possible slices here -- no label-overlap risk to
                # guard against, so a small category shouldn't be renamed
                # "Other" instead of shown by its real name.
                min_slice_pct=0,
            )
            save_fig(fig, os.path.join(comparison_dir, "column_availability.png"))

            # Column comparison!!!
            # overview comparison metrics computed in the aggregator
            patient_contribution = overview_comparison.get("patient_contribution", 0)
            completeness = overview_comparison.get("completeness", 0)
            usable_data_contribution = overview_comparison.get("usable_data_contribution", 0)
            relative_missing = overview_comparison.get("relative_missing", 0)
            local_missing_rate = overview_comparison.get("local_missing_rate", 0)
            total_value_contribution = overview_comparison.get("total_value_contribution", 0)

            # interpret missingness compared to federation
            if relative_missing > 1:
                missing_label = "above federation average"
            elif relative_missing < 1:
                missing_label = "below federation average"
            else:
                missing_label = "equal to federation average"

            # extract information for the local overview section
            n_analytical = r.get("n_analytical_cols", r["n_cols"])
            id_cols = r.get("id_columns", [])
            n_id = len(id_cols)
            if n_id:
                id_suffix = (f" + {n_id} identifier column{'s' if n_id != 1 else ''} "
                             f"detected ({', '.join(id_cols)})")
            else:
                id_suffix = ""
            overview = {
                "Patients":
                    f"{r['n_rows']} "
                    f"({patient_contribution}% of all patients in the federation)",
                "Features":
                    f"{n_analytical} analytical columns{id_suffix}",
                "Total Values":
                    f"{r['total_values']:,} "
                    f"({round(total_value_contribution * 100, 2)}% of all values in the federation)",

                "Missing Values":
                    f"{r['total_missing']:,} "
                    f"({round(r['missing_values_percentage'], 2)}%)",

                "Missingness Compared to Federation":
                    f"{round(relative_missing, 2)}× "
                    f"({missing_label})",

                "Completeness Contribution":
                    f"{round(completeness * 100, 2)}% "
                    f"of all values in the federation",

                "Usable Data Contribution":
                    f"{round(usable_data_contribution * 100, 2)}% "
                    f"of all non-missing values in the federation",

                "Duplicates":
                    r["n_duplicates"]
            }

            # changed from json to csv because table would be better
            df_overview = pd.DataFrame([
                {"metric": col, "value": val}
                for col, val in overview.items()
            ])
            df_overview.to_csv(overview_dir / "overview.csv", index=False)

            # numeric comparison
            df_numeric_comp = pd.DataFrame([
                {"column": col, "comparison": info["comparison_category"]}
                for col, info in numeric_comparison.items()
            ])
            df_numeric_comp.to_csv(comparison_dir / "numeric_comparison.csv", index=False)
            #-----------

            # missing values per column plot
            try:
                from data_report.generate_figures.data_quality_plots import (
                    save_missing_by_column,
                )
                save_missing_by_column(
                    r["missing_by_col"],
                    r["n_rows"],
                    overview_dir / "missing_values_by_column.png",
                    node_label=f"node{node_number}",
                )
            except Exception as e:
                print(f"Missing plot error: {e}")

            from data_report.generate_figures.local_descriptive_plots import (
                save_data_type_distribution,
            )
            save_data_type_distribution(
                numeric_statistics, categorical_statistics, temporal_statistics,
                overview_dir, node_label=f"node{node_number}",
            )

            df_missing = pd.DataFrame(
                list(r["missing_by_col"].items()),
                columns=["column", "missing_count"]
            )
            # df_missing.to_csv(overview_dir / "missing_values.csv", index=False)

            # summary table for numeric statistics
            if numeric_statistics:
                rows = []
                for feature, metrics in numeric_statistics.items():
                    row = {"feature": feature,
                           "availability": column_status_map.get(feature, "unknown") }
                    for metric, value in metrics.items():
                        row[metric] = value
                    rows.append(row)
                result = pd.DataFrame(rows)
                result.to_csv(numeric_dir / "numeric_summary.csv", index=False)

            # summary table for categorical statistics
            if categorical_statistics:
                rows = []

                for feature, metrics in categorical_statistics.items():
                    row = {"feature": feature,
                           "availability": column_status_map.get(feature, "unknown") }
                    for metric, value in metrics.items():
                        # remove category_counts
                        if metric == "category_counts":
                            continue
                        row[metric] = value

                    rows.append(row)
                result= pd.DataFrame(rows)
                # result.to_csv(node_dir / "categorical_summary.csv", index=False)
                result.to_csv(categorical_dir / "categorical_summary.csv", index=False)

            # summary table for temporal statistics
            if temporal_statistics:
                rows = []
                for feature, metrics in temporal_statistics.items():
                    row = {"feature": feature,
                           "availability": column_status_map.get(feature, "unknown") }
                    for metric, value in metrics.items():
                        if (metric == "observations_per_period"
                                or metric == "time_range" or metric == "range_days"):
                            continue

                        row[metric] = value
                    rows.append(row)
                result = pd.DataFrame(rows)
                # result.to_csv(node_dir / "temporal_summary.csv", index=False)
                result.to_csv(temporal_dir / "temporal_summary.csv", index=False)

            # local plots
            # example plot for temporal analysis
            if temporal_statistics:
                for feature, metrics in temporal_statistics.items():
                    obs = metrics.get("observations_per_period")
                    # skip if no data
                    if not obs or len(obs) == 0:
                        continue
                    try:
                        # convert periods to timestamps!
                        # the obs from observations_per_period likely has keys like:
                        # Period('2021-01', 'M')
                        # Period('2021-02', 'M')
                        # these are pandas.Period objects, not strings or timestamps.
                        # and pd.to_datetime() does not accept Period objects directly
                        # so it needs to be converted to Timestamp
                        # Period vs. Timestamp
                        # Timestamp = exact point in time (2021-01-01 00:00:00)
                        # Period = span of time (January 2021)
                        # Matplotlib needs timestamps, not periods!
                        periods = []
                        for p in obs.keys():
                            if isinstance(p, pd.Period):
                                periods.append(p.to_timestamp())
                            else:
                                periods.append(pd.to_datetime(p, errors="coerce"))
                        # build a dataframe
                        df_temp = pd.DataFrame({
                            "period": periods,
                            "count": list(obs.values())
                        })
                        # drop invalid dates and sort
                        df_temp = (
                            df_temp
                            .dropna(subset=["period"])
                            .sort_values("period")
                        )
                        # skip if empty after cleaning
                        if df_temp.empty:
                            continue

                        plt.figure(figsize=(10, 5))
                        plt.plot(
                            df_temp["period"],
                            df_temp["count"],
                            marker="o",
                            linewidth=2
                        )
                        # highlight most active period
                        most_active = metrics.get("most_active_period")

                        if most_active:
                            if isinstance(most_active, pd.Period):
                                most_active = most_active.to_timestamp()
                            else:
                                most_active = pd.to_datetime(most_active, errors="coerce")

                            match = df_temp[df_temp["period"] == most_active]
                            if not match.empty:
                                plt.scatter(
                                    match["period"],
                                    match["count"],
                                    s=100,
                                    label="Most Active Period"
                                )
                                plt.legend()
                        plt.xlabel("Time")
                        plt.ylabel("Number of Observations")
                        plt.title(f"{feature} - Temporal Activity (Node {node_number})")
                        plt.xticks(rotation=45)
                        plt.grid(alpha=0.3)
                        plt.tight_layout()

                        output_path = temporal_dir / f"{feature}_activity.png"
                        plt.savefig(output_path, dpi=200)
                        plt.close()

                    except Exception as e:
                        print(f"Temporal plot error ({feature}): {e}")

                # Batched grid version for full-mode reports (readability at scale);
                # the per-feature files above remain for short mode's top-N selection.
                try:
                    from data_report.generate_figures.local_descriptive_plots import (
                        save_temporal_activity_batched,
                    )
                    save_temporal_activity_batched(
                        temporal_statistics, temporal_dir, node_label=f"node{node_number}",
                    )
                except Exception as e:
                    print(f"Temporal batched activity plot error: {e}")

            if r.get("age_hist") is not None:
                edges = r["age_edges"]
                hist = r["age_hist"]
                bins = [
                    f"{int(edges[i])}-{int(edges[i + 1])}"
                    for i in range(len(edges) - 1)
                ]
                df_age = pd.DataFrame({
                    "age_bin": bins,
                    "count": hist
                })
                df_age.to_csv(numeric_dir / "age_distribution.csv", index=False)

            if r.get("age_out_of_range_count"):
                pd.DataFrame({"count": [r["age_out_of_range_count"]]}).to_csv(
                    numeric_dir / "age_out_of_range.csv", index=False
                )

            try:
                from data_report.generate_figures.local_descriptive_plots import (
                    save_age_distribution,
                    save_sex_distribution,
                )
                save_age_distribution(
                    r.get("age_edges"), r.get("age_hist"),
                    numeric_dir, node_label=f"node{node_number}",
                )
                if r.get("sex_counts"):
                    save_sex_distribution(
                        r["sex_counts"], categorical_dir,
                        node_label=f"node{node_number}",
                    )
            except Exception as e:
                print(f"Local plot error: {e}")
    #----------------------------------------------------------------------------------
    def _save_federated_tables(self, federated_results):
        """Write federation-wide CSV summary tables to disk."""
        FEDERATED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        numeric_dir = FEDERATED_RESULTS_DIR / "numeric"
        numeric_dir.mkdir(parents=True, exist_ok=True)
        categorical_dir = FEDERATED_RESULTS_DIR / "categorical"
        categorical_dir.mkdir(parents=True, exist_ok=True)
        temporal_dir = FEDERATED_RESULTS_DIR / "temporal"
        temporal_dir.mkdir(parents=True, exist_ok=True)
        overview_dir = FEDERATED_RESULTS_DIR / "overview"
        overview_dir.mkdir(parents=True, exist_ok=True)

        # extract information for the overview section
        overview = {
            "number of hospitals": federated_results.get("n_nodes"),
            "total number of patients": federated_results.get("total_rows"),
            "total number of features": federated_results.get("n_cols"),
            "total number of values": federated_results.get("n_total_values"),
            "total missing values": federated_results.get("total_missing"),
            "total missing values percentage": f"{round(federated_results.get('total_missing_percentage'), 3)}%",
        }
        # save the overview information in a table
        # changed from json to csv because table would be better
        df_overview = pd.DataFrame([
            {"metric": col, "value": val}
            for col, val in overview.items()
        ])
        df_overview.to_csv(overview_dir / "overview.csv", index=False)

        # with open(overview_dir / "overview.json", "w") as f:
        #     json.dump(overview, f, indent=4)


        global_numeric = federated_results.get("global_numeric", {})
        num_rows = []
        for feature, metrics in global_numeric.items():
            row = {"feature": feature}
            row.update(metrics)
            num_rows.append(row)
        result = pd.DataFrame(num_rows)
        result.to_csv(numeric_dir / "federated_numeric_statistics.csv", index=False)

        global_categorical = federated_results.get("global_categorical", {})
        cat_rows = []
        for feature, metrics in global_categorical.items():
            row = {"feature": feature}
            row.update(metrics)
            cat_rows.append(row)
        result = pd.DataFrame(cat_rows)
        result.to_csv(categorical_dir / "federated_categorical_statistics.csv", index=False)

        temp_rows = []
        # added fix here because if global_temporal is None then there is a crash
        global_temporal = federated_results.get("global_temporal", {})
        for feature, stats in global_temporal.items():
            row = {"feature": feature}
            row.update(stats)

            # fix serialization after update
            row["counts_per_period"] = json.dumps(row.get("counts_per_period", {}))
            row["most_active_period"] = str(row.get("most_active_period"))

            temp_rows.append(row)

        df_temporal = pd.DataFrame(temp_rows)
        df_temporal.to_csv(temporal_dir/ "federated_temporal_statistics.csv", index=False)

        # missing values
        df_missing = pd.DataFrame(
            list(federated_results.get("missing_by_col").items()),
            columns=["column", "missing_count"]
        )
        # df_missing.to_csv(overview_dir/ "missing_values_federated.csv", index=False)

        # sex distribution
        df_sex = pd.DataFrame(
            list(federated_results.get("sex_counts").items()),
            columns=["sex", "count"]
        )
        df_sex.to_csv(categorical_dir / "sex_distribution_federated.csv", index=False)

        age_hist = federated_results.get("age_hist")
        age_edges = federated_results.get("age_edges")
        # age distribution
        if age_hist is not None:
            bins = [
                f"{int(age_edges[i])}-{int(age_edges[i + 1])}"
                for i in range(len(age_edges) - 1)
            ]

            df_age = pd.DataFrame({
                "age_bin": bins,
                "count": age_hist
            })
            df_age.to_csv(numeric_dir / "age_distribution_federated.csv", index=False)

        if federated_results.get("age_out_of_range_count"):
            pd.DataFrame({"count": [federated_results["age_out_of_range_count"]]}).to_csv(
                numeric_dir / "age_out_of_range.csv", index=False
            )