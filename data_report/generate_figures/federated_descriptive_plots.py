"""
Federated (aggregator-level) descriptive visualizations.

All functions receive pre-aggregated statistics dicts — no raw DataFrames
are available at the aggregator. Each function is a standalone entry point
that can be called independently; `save_all_federated_plots` is the
single orchestrator that replaces the old inline `_make_plots` method.

Sections:
  Overview   — data-type distribution pie, age histogram, sex bar chart
  Numeric    — mean ± std bar chart (only summary stats are available)
  Categorical — top-category bar charts per column
  Temporal   — line chart per column using federated counts_per_period
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_report.generate_figures import style

logger = logging.getLogger(__name__)
from data_report.generate_figures.primitives import (
    bar_chart,
    histogram_from_bins,
    line_chart,
    make_subplots,
    pie_chart,
    save_fig,
)
from data_report.generate_figures.local_descriptive_plots import _periods_to_timestamps


# ---------------------------------------------------------------------------
# Overview section
# ---------------------------------------------------------------------------

def save_federated_data_type_distribution(
    global_numeric: dict,
    global_categorical: dict,
    global_temporal: dict,
    output_dir,
) -> None:
    """Save a pie chart of federated column counts broken down by data type.

    Args:
        global_numeric (dict): Federated statistics for numeric columns.
        global_categorical (dict): Federated statistics for categorical
            columns.
        global_temporal (dict): Federated statistics for temporal columns.
        output_dir (str or Path): Directory where the image is written.
    """
    counts = [len(global_numeric), len(global_categorical), len(global_temporal)]
    if sum(counts) == 0:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    pie_chart(
        ax,
        counts,
        ["Numerical", "Categorical", "Temporal"],
        colors=["#A6D8A8", "#86C5F9", "#FFCC80"],
        title="Federated Data Type Distribution",
        # Only 3 possible slices here -- no label-overlap risk to guard
        # against, so a small category shouldn't be renamed "Other" instead
        # of shown by its real name.
        min_slice_pct=0,
    )
    save_fig(fig, output_dir / "data_type_distribution.png")


def save_federated_age_distribution(
    age_edges,
    age_hist,
    output_dir,
) -> None:
    """Save a federated age histogram from aggregated bin edges and counts.

    The bin edges and per-bin counts are aggregated across all nodes before
    being passed here; no raw subject-level data is required.

    Args:
        age_edges (array-like): Bin boundary values; length must be
            len(age_hist) + 1.
        age_hist (array-like): Per-bin observation counts summed across nodes.
        output_dir (str or Path): Directory where the image is written.
    """
    if age_edges is None or age_hist is None:
        return
    edges = np.asarray(age_edges, dtype=float)
    hist = np.asarray(age_hist, dtype=float)
    if len(edges) < 2 or hist.sum() == 0:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    histogram_from_bins(
        ax, edges, hist,
        title="Federated Age Distribution",
        xlabel="Age",
        ylabel="Count (federated)",
    )
    save_fig(fig, output_dir / "age_distribution_federated.png")


def save_federated_sex_distribution(
    sex_counts: dict,
    output_dir,
) -> None:
    """Save a federated sex/gender bar chart from aggregated counts.

    Args:
        sex_counts (dict): Mapping of sex/gender label to aggregated
            observation count across all nodes.
        output_dir (str or Path): Directory where the image is written.
    """
    if not sex_counts:
        return
    keys = [k for k in sex_counts if pd.notna(k)]
    vals = [sex_counts[k] for k in keys]
    if not keys:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    bar_chart(
        ax, keys, vals,
        colors=[style.PALETTE[i % len(style.PALETTE)] for i in range(len(keys))],
        title="Federated Sex Distribution",
        xlabel="Sex",
        ylabel="Count (federated)",
    )
    save_fig(fig, output_dir / "sex_distribution_federated.png")


# ---------------------------------------------------------------------------
# Numeric section
# ---------------------------------------------------------------------------

def save_federated_numeric_summary_bars(
    global_numeric: dict,
    output_dir,
) -> None:
    """Save a horizontal bar chart of federated means with standard-deviation error bars.

    This is the only numeric distribution visualisation available at the
    aggregator level: histogram bins are not preserved across nodes for
    general numeric columns. Age has its own dedicated histogram.

    Args:
        global_numeric (dict): Federated statistics for numeric columns;
            each entry must contain "mean" and "std" keys.
        output_dir (str or Path): Directory where the image is written.
    """
    if not global_numeric:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cols = list(global_numeric.keys())
    means = [global_numeric[c].get("mean", 0) for c in cols]
    stds = [global_numeric[c].get("std", 0) for c in cols]

    fig, ax = plt.subplots(figsize=(10, max(5, len(cols) * 0.4)))
    y_pos = range(len(cols))
    ax.barh(
        list(y_pos), means,
        xerr=stds,
        color=style.PALETTE[0],
        ecolor=style.PALETTE[3],
        capsize=4,
        alpha=0.8,
    )
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(cols)
    ax.set_xlabel("Mean ± Std (federated)")
    ax.set_title("Federated Numeric Feature Summary")
    ax.grid(axis="x", alpha=0.3)
    save_fig(fig, output_dir / "numeric_summary_bars.png")


# ---------------------------------------------------------------------------
# Categorical section
# ---------------------------------------------------------------------------

def save_federated_categorical_distributions(
    global_categorical: dict,
    output_dir,
    *,
    top_n: int = 20,
    batch_size: int = 6,
) -> list[Path]:
    """Save batched bar charts for non-binary multi-category federated variables.

    Binary columns (2 or fewer distinct categories in the federated counts)
    are excluded because they are already visible in the summary table and
    sex-distribution chart. Columns with more than 2 categories are packed
    in batches of batch_size per image file.

    Args:
        global_categorical (dict): Federated statistics for categorical
            columns; each entry must contain a "counts" sub-dict.
        output_dir (str or Path): Directory where images are written.
        top_n (int): Maximum number of categories shown per column.
            Defaults to 20.
        batch_size (int): Number of column charts packed into each image
            file. Defaults to 6.

    Returns:
        list[Path]: Paths of the written image files.
    """
    if not global_categorical:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Only multi-category columns with > 2 distinct non-null values
    multi_cat = [
        c for c in global_categorical
        if len([k for k in global_categorical[c].get("counts", {}) if pd.notna(k)]) > 2
    ]
    if not multi_cat:
        return []

    n_batches = max(1, math.ceil(len(multi_cat) / batch_size))
    written: list[Path] = []
    for b in range(n_batches):
        batch = multi_cat[b * batch_size:(b + 1) * batch_size]
        fig, axes = make_subplots(len(batch), ncols=2, width=7, height=4)
        for ax, col in zip(axes, batch):
            counts = global_categorical[col]["counts"]
            sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
            if not sorted_items:
                continue
            cats, vals = zip(*sorted_items)
            bar_chart(
                ax,
                list(cats), list(vals),
                horizontal=True,
                title=col,
                xlabel="Count (federated)",
            )

        title = "Federated Categorical Distributions (multi-category)"
        if n_batches > 1:
            title += f" ({b + 1}/{n_batches})"
        fig.suptitle(title, fontsize=13, y=1.01)

        out_path = output_dir / f"categorical_distributions_{b + 1:02d}.png"
        save_fig(fig, out_path)
        written.append(out_path)

    return written


# ---------------------------------------------------------------------------
# Temporal section
# ---------------------------------------------------------------------------

def save_federated_temporal_charts(
    global_temporal: dict,
    output_dir,
) -> None:
    """Save one line chart per temporal column using federated period counts.

    Period keys in global_temporal are typically strings because the
    aggregation layer serialises them. _periods_to_timestamps handles both
    string and Period keys, so this is forward-compatible.

    Args:
        global_temporal (dict): Federated statistics for temporal columns;
            each entry must contain a "counts_per_period" sub-dict.
        output_dir (str or Path): Directory where images are written.
    """
    if not global_temporal:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for feature, stats in global_temporal.items():
        counts_per_period = stats.get("counts_per_period")
        if not counts_per_period:
            continue
        fig = None
        try:
            timestamps, counts = _periods_to_timestamps(counts_per_period)
            if not timestamps:
                continue

            fig, ax = plt.subplots(figsize=(10, 5))
            line_chart(
                ax, timestamps, counts,
                title=f"{feature} — Federated Temporal Activity",
                xlabel="Time",
                ylabel="Observations (federated)",
            )

            most_active = stats.get("most_active_period")
            if most_active is not None:
                most_active_ts = pd.to_datetime(most_active, errors="coerce")
                if not pd.isna(most_active_ts) and most_active_ts in timestamps:
                    idx = timestamps.index(most_active_ts)
                    ax.scatter(
                        [most_active_ts], [counts[idx]],
                        s=100, zorder=5, color=style.PALETTE[1],
                        label="Most Active Period",
                    )
                    ax.legend()

            save_fig(fig, output_dir / f"{feature}_activity_federated.png")
        except Exception:
            logger.warning("Federated temporal chart error (%s)", feature, exc_info=True)
            if fig is not None:
                plt.close(fig)


# ---------------------------------------------------------------------------
# Federated inferential: simple trend analysis
# ---------------------------------------------------------------------------

def save_federated_trend_summary(
    global_temporal: dict,
    output_dir,
) -> None:
    """Save a bar chart of linear trend slopes for each federated temporal column.

    Computes a simple OLS slope (observations ~ time index) from the
    federated counts_per_period. Bars are coloured by direction (increasing
    or decreasing). The R² value is printed inside each bar when it is 0.1
    or higher to indicate goodness of fit.

    Args:
        global_temporal (dict): Federated statistics for temporal columns;
            each entry must contain a "counts_per_period" sub-dict.
        output_dir (str or Path): Directory where the image is written.

    Note:
        Columns with fewer than 3 time periods are excluded because OLS
        slope estimation is unreliable with fewer data points.
    """
    if not global_temporal:
        return

    from scipy.stats import linregress
    from data_report.generate_figures.local_descriptive_plots import _periods_to_timestamps

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slopes, r2s, labels = [], [], []
    for feature, stats in global_temporal.items():
        counts_per_period = stats.get("counts_per_period")
        if not counts_per_period:
            continue
        try:
            _, counts = _periods_to_timestamps(counts_per_period)
            if len(counts) < 3:
                continue
            x = np.arange(len(counts), dtype=float)
            y = np.asarray(counts, dtype=float)
            slope, _, r, _, _ = linregress(x, y)
            slopes.append(slope)
            r2s.append(r ** 2)
            labels.append(feature)
        except Exception:
            logger.warning("Federated trend slope error (%s)", feature, exc_info=True)
            continue

    if not slopes:
        return

    order = np.argsort(np.abs(slopes))
    slopes = [slopes[i] for i in order]
    r2s = [r2s[i] for i in order]
    labels = [labels[i] for i in order]

    colors = [
        style.PALETTE[0] if s >= 0 else style.PALETTE[3]
        for s in slopes
    ]

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.5)))
    bars = ax.barh(labels, slopes, color=colors, alpha=0.85)

    # Annotate R² when it's meaningful
    for bar, r2 in zip(bars, r2s):
        if r2 >= 0.1:
            x_pos = bar.get_width()
            ax.text(
                x_pos / 2, bar.get_y() + bar.get_height() / 2,
                f"R²={r2:.2f}", ha="center", va="center", fontsize=8, color="white",
            )

    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Linear Trend Slope (observations / period)")
    ax.set_title("Federated Temporal Trend Summary")

    import matplotlib.patches as mpatches
    up = mpatches.Patch(color=style.PALETTE[0], label="Increasing ↑")
    down = mpatches.Patch(color=style.PALETTE[3], label="Decreasing ↓")
    ax.legend(handles=[up, down], loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    save_fig(fig, output_dir / "temporal_trend_summary.png")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def save_all_federated_plots(federated_results: dict, output_dir) -> None:
    """Generate all federated plots from a federated results dictionary.

    Dispatches to each individual save_federated_* function and organises
    outputs into sub-directories:

    - overview/    — data-type distribution, age distribution, sex distribution
    - numeric/     — mean ± std summary bars
    - categorical/ — batched multi-category bar charts
    - temporal/    — per-column line charts and trend summary

    Args:
        federated_results (dict): Aggregated results dict produced by the
            aggregation layer; expected keys are "global_numeric",
            "global_categorical", "global_temporal", "age_edges",
            "age_hist", and "sex_counts".
        output_dir (str or Path): Root directory; sub-directories are
            created automatically.
    """
    output_dir = Path(output_dir)
    overview_dir = output_dir / "overview"
    numeric_dir = output_dir / "numeric"
    categorical_dir = output_dir / "categorical"
    temporal_dir = output_dir / "temporal"

    global_numeric = federated_results.get("global_numeric", {})
    global_categorical = federated_results.get("global_categorical", {})
    global_temporal = federated_results.get("global_temporal", {})

    save_federated_data_type_distribution(
        global_numeric, global_categorical, global_temporal, overview_dir
    )
    save_federated_age_distribution(
        federated_results.get("age_edges"),
        federated_results.get("age_hist"),
        numeric_dir,
    )
    save_federated_sex_distribution(
        federated_results.get("sex_counts", {}), categorical_dir
    )
    save_federated_numeric_summary_bars(global_numeric, numeric_dir)
    save_federated_categorical_distributions(global_categorical, categorical_dir)
    save_federated_temporal_charts(global_temporal, temporal_dir)
    save_federated_trend_summary(global_temporal, temporal_dir)
