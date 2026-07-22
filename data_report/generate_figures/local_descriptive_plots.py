"""
Local (per-node) descriptive visualizations.

All public functions accept either a raw DataFrame (when df is available
inside analysis_method) or pre-computed statistics dicts (when called from
_save_local_node_results, which only sees serialized stats). The docstring of
each function states which form it expects.

Sections in this file:
  Numeric     — histograms, boxplots, correlation heatmap, scatter matrix,
                age distribution, data-type distribution
  Categorical — sex distribution, per-column bar charts, stacked bars,
                column-availability pie
  Temporal    — time-series line chart, area chart, bar-over-time, heatmap
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns

from data_report.generate_figures import style

logger = logging.getLogger(__name__)
from data_report.generate_figures.primitives import (
    bar_chart,
    boxplot,
    heatmap,
    histogram,
    histogram_from_bins,
    line_chart,
    make_subplots,
    pie_chart,
    save_fig,
    scatter,
    stacked_bar,
)


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _periods_to_timestamps(obs: dict) -> tuple[list, list]:
    """Convert a period-keyed observations dict to sorted (timestamps, counts) lists."""
    pairs = []
    for k, v in obs.items():
        if isinstance(k, pd.Period):
            ts = k.to_timestamp()
        else:
            ts = pd.to_datetime(k, errors="coerce")
        if pd.isna(ts):
            continue
        pairs.append((ts, v))
    pairs.sort(key=lambda x: x[0])
    if not pairs:
        return [], []
    timestamps, counts = zip(*pairs)
    return list(timestamps), list(counts)


# ===========================================================================
# Numeric section
# ===========================================================================

def save_numeric_histograms(
    df: pd.DataFrame,
    numeric_cols: list[str],
    output_dir,
    *,
    batch_size: int = 6,
    node_label: Optional[str] = None,
) -> list[Path]:
    """Save batched grids of histograms, one subplot per numeric column.

    Columns with fewer than 2 non-null values are skipped silently. Charts
    are packed in batches of ``batch_size`` per image file so each image
    stays readable regardless of how many numeric variables are present.

    Args:
        df (pd.DataFrame): DataFrame containing the numeric columns.
        numeric_cols (list[str]): Column names to plot.
        output_dir (str or Path): Directory where images are written.
        batch_size (int): Number of column histograms packed into each
            image file. Defaults to 6.
        node_label (str, optional): Node identifier appended to plot
            titles.

    Returns:
        list[Path]: Paths of the written image files.
    """
    if not numeric_cols:
        return []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_cols = [c for c in numeric_cols if df[c].notna().sum() >= 2]
    if not valid_cols:
        return []

    n_batches = max(1, math.ceil(len(valid_cols) / batch_size))
    written: list[Path] = []
    for b in range(n_batches):
        batch = valid_cols[b * batch_size:(b + 1) * batch_size]
        fig, axes = make_subplots(len(batch), ncols=3, width=5, height=4)
        for ax, col in zip(axes, batch):
            histogram(ax, df[col].dropna().values, title=col, xlabel=col)

        title = "Numeric Feature Distributions"
        if node_label:
            title += f" — {node_label}"
        if n_batches > 1:
            title += f" ({b + 1}/{n_batches})"
        fig.suptitle(title, fontsize=13, y=1.01)

        out_path = output_dir / f"numeric_histograms_{b + 1:02d}.png"
        save_fig(fig, out_path)
        written.append(out_path)

    return written


def save_numeric_boxplots(
    df: pd.DataFrame,
    numeric_cols: list[str],
    output_dir,
    *,
    batch_size: int = 6,
    node_label: Optional[str] = None,
) -> list[Path]:
    """Save batched side-by-side boxplots for all numeric columns.

    Columns with fewer than 2 non-null values are skipped. Boxplots are
    packed in batches of ``batch_size`` per image file so each image stays
    readable regardless of how many numeric variables are present.

    Args:
        df (pd.DataFrame): DataFrame containing the numeric columns.
        numeric_cols (list[str]): Column names to plot.
        output_dir (str or Path): Directory where images are written.
        batch_size (int): Number of columns packed into each image file.
            Defaults to 6.
        node_label (str, optional): Node identifier appended to plot
            titles.

    Returns:
        list[Path]: Paths of the written image files.
    """
    if not numeric_cols:
        return []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_cols = [c for c in numeric_cols if df[c].notna().sum() >= 2]
    if not valid_cols:
        return []

    import matplotlib.pyplot as plt
    n_batches = max(1, math.ceil(len(valid_cols) / batch_size))
    written: list[Path] = []
    for b in range(n_batches):
        batch = valid_cols[b * batch_size:(b + 1) * batch_size]
        data = [df[c].dropna().values for c in batch]

        fig, ax = plt.subplots(figsize=(max(8, len(batch) * 0.7), 6))
        boxplot(ax, data, labels=batch)

        title = "Numeric Feature Boxplots"
        if node_label:
            title += f" — {node_label}"
        if n_batches > 1:
            title += f" ({b + 1}/{n_batches})"
        ax.set_title(title)

        out_path = output_dir / f"numeric_boxplots_{b + 1:02d}.png"
        save_fig(fig, out_path)
        written.append(out_path)

    return written


def save_correlation_heatmap(
    df: pd.DataFrame,
    numeric_cols: list[str],
    output_dir,
    *,
    node_label: Optional[str] = None,
) -> None:
    """Save a Pearson correlation matrix as an annotated heatmap.

    Requires at least 2 numeric columns with non-constant values; returns
    early otherwise.

    Args:
        df (pd.DataFrame): DataFrame containing the numeric columns.
        numeric_cols (list[str]): Column names to include in the matrix.
        output_dir (str or Path): Directory where the image is written.
        node_label (str, optional): Node identifier appended to the plot
            title.
    """
    if len(numeric_cols) < 2:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sub = df[numeric_cols].select_dtypes(include="number").dropna(how="all")
    # Drop constant columns (zero variance → undefined correlation)
    sub = sub.loc[:, sub.std() > 0]
    if sub.shape[1] < 2:
        return

    corr = sub.corr(method="pearson")
    size = max(6, corr.shape[0] * 0.6)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    heatmap(
        ax, corr,
        cmap="coolwarm", vmin=-1, vmax=1,
        annotate=True, fmt=".2f",
        title="Pearson Correlation" + (f" — {node_label}" if node_label else ""),
    )
    save_fig(fig, output_dir / "correlation_heatmap.png")


def save_scatter_matrix(
    df: pd.DataFrame,
    numeric_cols: list[str],
    output_dir,
    *,
    max_cols: int = 10,
    node_label: Optional[str] = None,
) -> None:
    """Save a pairwise scatter matrix for the first max_cols numeric columns.

    Uses seaborn pairplot. The column count is capped at max_cols because
    the number of panels grows as O(n²).

    Args:
        df (pd.DataFrame): DataFrame containing the numeric columns.
        numeric_cols (list[str]): Column names to include.
        output_dir (str or Path): Directory where the image is written.
        max_cols (int): Maximum number of columns to include. Defaults to 10.
        node_label (str, optional): Node identifier appended to the plot
            title.
    """
    valid_cols = [c for c in numeric_cols if df[c].notna().sum() >= 2]
    if len(valid_cols) < 2:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cols = valid_cols[:max_cols]
    plot_df = df[cols].dropna(how="all")

    import matplotlib.pyplot as plt
    grid = sns.pairplot(plot_df, corner=True, plot_kws={"alpha": 0.5, "s": 15})
    title = "Numeric Feature Scatter Matrix"
    if node_label:
        title += f" — {node_label}"
    grid.figure.suptitle(title, y=1.01)
    grid.savefig(output_dir / "scatter_matrix.png", dpi=style.DPI, bbox_inches="tight")
    plt.close(grid.figure)


def save_age_distribution(
    age_edges,
    age_hist,
    output_dir,
    *,
    node_label: Optional[str] = None,
) -> None:
    """Save an age histogram from pre-computed bin edges and counts.

    Called with serialized statistics rather than raw data. Returns silently
    when either argument is None or the histogram is empty.

    Args:
        age_edges (array-like): Bin boundary values; length must be
            len(age_hist) + 1.
        age_hist (array-like): Per-bin observation counts.
        output_dir (str or Path): Directory where the image is written.
        node_label (str, optional): Node identifier appended to the plot
            title.
    """
    if age_edges is None or age_hist is None:
        return
    edges = np.asarray(age_edges, dtype=float)
    hist = np.asarray(age_hist, dtype=float)
    if len(edges) < 2 or hist.sum() == 0:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    histogram_from_bins(
        ax, edges, hist,
        title="Age Distribution" + (f" — {node_label}" if node_label else ""),
        xlabel="Age",
    )
    save_fig(fig, output_dir / "age_distribution.png")


def save_data_type_distribution(
    numeric_stats: dict,
    categorical_stats: dict,
    temporal_stats: dict,
    output_dir,
    *,
    node_label: Optional[str] = None,
) -> None:
    """Save a pie chart of column counts broken down by data type.

    Slice sizes are derived from the number of keys in each stats dict.

    Args:
        numeric_stats (dict): Statistics dict for numeric columns.
        categorical_stats (dict): Statistics dict for categorical columns.
        temporal_stats (dict): Statistics dict for temporal columns.
        output_dir (str or Path): Directory where the image is written.
        node_label (str, optional): Node identifier appended to the plot
            title.
    """
    counts = [len(numeric_stats), len(categorical_stats), len(temporal_stats)]
    if sum(counts) == 0:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    pie_chart(
        ax,
        counts,
        ["Numerical", "Categorical", "Temporal"],
        colors=["#A6D8A8", "#86C5F9", "#FFCC80"],
        title="Data Type Distribution" + (f" — {node_label}" if node_label else ""),
        # Only 3 possible slices here -- no label-overlap risk to guard
        # against, so a small category shouldn't be renamed "Other" instead
        # of shown by its real name.
        min_slice_pct=0,
    )
    save_fig(fig, output_dir / "data_type_distribution.png")


# ===========================================================================
# Categorical section
# ===========================================================================

def save_sex_distribution(
    sex_counts: dict,
    output_dir,
    *,
    node_label: Optional[str] = None,
) -> None:
    """Save a bar chart of sex/gender category counts.

    Args:
        sex_counts (dict): Mapping of sex/gender label to observation count.
        output_dir (str or Path): Directory where the image is written.
        node_label (str, optional): Node identifier appended to the plot
            title.
    """
    if not sex_counts:
        return
    keys = [k for k in sex_counts if pd.notna(k)]
    vals = [sex_counts[k] for k in keys]
    if not keys:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    bar_chart(
        ax, keys, vals,
        colors=[style.PALETTE[i % len(style.PALETTE)] for i in range(len(keys))],
        title="Sex Distribution" + (f" — {node_label}" if node_label else ""),
        xlabel="Sex",
        ylabel="Count",
    )
    save_fig(fig, output_dir / "sex_distribution.png")


def save_categorical_bar_charts(
    df: pd.DataFrame,
    categorical_cols: list[str],
    output_dir,
    *,
    top_n: int = 20,
    node_label: Optional[str] = None,
) -> None:
    """Save one horizontal bar chart per categorical column showing top-N categories.

    Args:
        df (pd.DataFrame): DataFrame containing the categorical columns.
        categorical_cols (list[str]): Column names to plot.
        output_dir (str or Path): Directory where the image is written.
        top_n (int): Maximum number of categories to show per column.
            Defaults to 20.
        node_label (str, optional): Node identifier appended to the plot
            title.
    """
    if not categorical_cols:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_cols = [c for c in categorical_cols if df[c].notna().sum() > 0]
    if not valid_cols:
        return

    import matplotlib.pyplot as plt
    from data_report.generate_figures.primitives import count_plot
    fig, axes = make_subplots(len(valid_cols), ncols=2, width=7, height=4)
    for ax, col in zip(axes, valid_cols):
        count_plot(ax, df[col], top_n=top_n, title=col)

    title = "Categorical Feature Distributions"
    if node_label:
        title += f" — {node_label}"
    fig.suptitle(title, fontsize=13, y=1.01)
    save_fig(fig, output_dir / "categorical_bar_charts.png")


def save_categorical_distributions(
    df: pd.DataFrame,
    categorical_cols: list[str],
    output_dir,
    *,
    batch_size: int = 6,
    top_n: int = 20,
    node_label: Optional[str] = None,
) -> list[Path]:
    """Save batched bar charts for all columns with at least 2 distinct values.

    Columns with only a single observed value are excluded (there is no
    distribution to show; see ``categorical_excluded_from_distributions_notice``
    for the report-facing note about this). Binary and multi-category columns
    both get one bar chart each (up to top_n categories); charts are packed
    in batches of batch_size per image file so each image stays readable
    regardless of how many categorical variables are present.

    Args:
        df (pd.DataFrame): DataFrame containing the categorical columns.
        categorical_cols (list[str]): Column names to consider.
        output_dir (str or Path): Directory where images are written.
        batch_size (int): Number of column charts packed into each image
            file. Defaults to 6.
        top_n (int): Maximum number of categories shown per column.
            Defaults to 20.
        node_label (str, optional): Node identifier appended to plot titles.

    Returns:
        list[Path]: Paths of the written image files.
    """
    import matplotlib.pyplot as plt
    from data_report.generate_figures.primitives import count_plot

    # All categorical columns with at least 2 distinct non-null values
    # (includes binary; excludes columns with only 1 value, which have no distribution to show)
    multi_cat = [
        c for c in categorical_cols
        if c in df.columns and df[c].nunique(dropna=True) >= 2
    ]
    if not multi_cat:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_batches = max(1, math.ceil(len(multi_cat) / batch_size))
    written: list[Path] = []
    for b in range(n_batches):
        batch = multi_cat[b * batch_size:(b + 1) * batch_size]
        fig, axes = make_subplots(len(batch), ncols=2, width=7, height=4)
        for ax, col in zip(axes, batch):
            count_plot(ax, df[col], top_n=top_n, title=col)

        title = "Categorical Distributions (multi-category)"
        if node_label:
            title += f" — {node_label}"
        if n_batches > 1:
            title += f" ({b + 1}/{n_batches})"
        fig.suptitle(title, fontsize=13, y=1.01)

        out_path = output_dir / f"categorical_distributions_{b + 1:02d}.png"
        save_fig(fig, out_path)
        written.append(out_path)

    return written


def save_stacked_bar_charts(
    df: pd.DataFrame,
    categorical_cols: list[str],
    output_dir,
    *,
    node_label: Optional[str] = None,
) -> None:
    """Save stacked bar charts for pairs of low-cardinality categorical columns.

    Only columns with 2 to 10 unique values are used; higher cardinality
    produces unreadable stacked bars. The first eligible column is paired
    against each subsequent eligible column.

    Args:
        df (pd.DataFrame): DataFrame containing the categorical columns.
        categorical_cols (list[str]): Column names to consider.
        output_dir (str or Path): Directory where images are written.
        node_label (str, optional): Node identifier appended to plot titles.
    """
    usable = [
        c for c in categorical_cols
        if 2 <= df[c].nunique(dropna=True) <= 10
    ]
    if len(usable) < 2:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    pivot_col = usable[0]
    for other_col in usable[1:]:
        fig = None
        try:
            ct = pd.crosstab(df[pivot_col], df[other_col])
            fig, ax = plt.subplots(figsize=(10, 5))
            stacked_bar(
                ax, ct,
                title=(
                    f"{pivot_col} × {other_col}"
                    + (f" — {node_label}" if node_label else "")
                ),
                xlabel=str(pivot_col),
                ylabel="Count",
            )
            fname = f"stacked_{pivot_col}_vs_{other_col}.png"
            save_fig(fig, output_dir / fname)
        except Exception:
            logger.warning(
                "Stacked bar chart error (%s vs %s)", pivot_col, other_col, exc_info=True
            )
            if fig is not None:
                plt.close(fig)


def save_column_availability_chart(
    column_comparison: dict,
    n_nodes: int,
    output_dir,
) -> None:
    """Save a pie chart of column availability categories across nodes.

    Slices represent the proportion of columns classified as common to all
    nodes, common to a partial set of nodes, or unique to this node.

    Args:
        column_comparison (dict): Mapping of column name to availability
            label ("common_all", "common_partial", or "unique_local").
        n_nodes (int): Total number of nodes in the federation, used for
            the chart title.
        output_dir (str or Path): Directory where the image is written.
    """
    common = sum(1 for v in column_comparison.values() if v == "common_all")
    partial = sum(1 for v in column_comparison.values() if v == "common_partial")
    unique = sum(1 for v in column_comparison.values() if v == "unique_local")
    total = common + partial + unique
    if total == 0:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    pie_chart(
        ax,
        [common, partial, unique],
        ["Common", "Partial", "Unique"],
        colors=["#A6D8A8", "#86C5F9", "#FFCC80"],
        title=f'Column Availability ({n_nodes} node{"s" if n_nodes != 1 else ""})',
    )
    save_fig(fig, output_dir / "column_availability.png")


# ===========================================================================
# Temporal section
# ===========================================================================

def save_temporal_line_charts(
    temporal_statistics: dict,
    output_dir,
    *,
    node_label: Optional[str] = None,
) -> None:
    """Save one line chart per temporal column showing observations over time.

    The most active period is annotated with a highlighted scatter point.

    Args:
        temporal_statistics (dict): Mapping of column name to temporal
            statistics, each entry containing an "observations_per_period"
            sub-dict.
        output_dir (str or Path): Directory where images are written.
        node_label (str, optional): Node identifier appended to plot titles.
    """
    if not temporal_statistics:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    for feature, metrics in temporal_statistics.items():
        obs = metrics.get("observations_per_period")
        if not obs:
            continue
        fig = None
        try:
            timestamps, counts = _periods_to_timestamps(obs)
            if not timestamps:
                continue

            fig, ax = plt.subplots(figsize=(10, 5))
            line_chart(
                ax, timestamps, counts,
                title=(
                    f"{feature} — Temporal Activity"
                    + (f" ({node_label})" if node_label else "")
                ),
                xlabel="Time",
                ylabel="Observations",
            )

            most_active = metrics.get("most_active_period")
            if most_active is not None:
                if isinstance(most_active, pd.Period):
                    most_active_ts = most_active.to_timestamp()
                else:
                    most_active_ts = pd.to_datetime(most_active, errors="coerce")
                if not pd.isna(most_active_ts) and most_active_ts in timestamps:
                    idx = timestamps.index(most_active_ts)
                    ax.scatter(
                        [most_active_ts], [counts[idx]],
                        s=100, zorder=5, color=style.PALETTE[1],
                        label="Most Active Period",
                    )
                    ax.legend()

            save_fig(fig, output_dir / f"{feature}_activity.png")
        except Exception:
            logger.warning("Temporal line chart error (%s)", feature, exc_info=True)
            if fig is not None:
                plt.close(fig)


def save_temporal_activity_batched(
    temporal_statistics: dict,
    output_dir,
    *,
    batch_size: int = 6,
    node_label: Optional[str] = None,
) -> list[Path]:
    """Save batched grids of temporal activity line charts, one subplot per column.

    Full-mode reports use these combined grid images instead of one
    separate file per temporal column, for the same readability reason
    numeric/categorical distributions are batched. Short mode still
    selects specific top-N-by-activity features from the individual
    per-feature files this module also writes (see
    ``save_temporal_line_charts``), since that ranking is only known at
    report-generation time.

    Args:
        temporal_statistics (dict): Mapping of column name to temporal
            statistics, each entry containing an "observations_per_period"
            sub-dict.
        output_dir (str or Path): Directory where images are written.
        batch_size (int): Number of column line charts packed into each
            image file. Defaults to 6.
        node_label (str, optional): Node identifier appended to plot
            titles.

    Returns:
        list[Path]: Paths of the written image files.
    """
    if not temporal_statistics:
        return []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    usable = [
        (feature, metrics) for feature, metrics in temporal_statistics.items()
        if metrics.get("observations_per_period")
    ]
    if not usable:
        return []

    n_batches = max(1, math.ceil(len(usable) / batch_size))
    written: list[Path] = []
    for b in range(n_batches):
        batch = usable[b * batch_size:(b + 1) * batch_size]
        fig, axes = make_subplots(len(batch), ncols=2, width=6, height=4)
        for ax, (feature, metrics) in zip(axes, batch):
            try:
                timestamps, counts = _periods_to_timestamps(metrics["observations_per_period"])
                if not timestamps:
                    continue
                ax.plot(timestamps, counts, marker="o", linewidth=2)
                most_active = metrics.get("most_active_period")
                if most_active is not None:
                    if isinstance(most_active, pd.Period):
                        most_active_ts = most_active.to_timestamp()
                    else:
                        most_active_ts = pd.to_datetime(most_active, errors="coerce")
                    if not pd.isna(most_active_ts) and most_active_ts in timestamps:
                        idx = timestamps.index(most_active_ts)
                        ax.scatter([most_active_ts], [counts[idx]], s=60, zorder=5)
                ax.set_title(feature, fontsize=9)
                ax.tick_params(axis="x", rotation=45, labelsize=7)
                ax.grid(alpha=0.3)
            except Exception:
                # One malformed feature must not blank out the whole batch image --
                # the per-feature save_temporal_line_charts loop has the same guard.
                logger.warning("Temporal batched activity subplot error (%s)", feature, exc_info=True)

        title = "Temporal Activity"
        if node_label:
            title += f" — {node_label}"
        if n_batches > 1:
            title += f" ({b + 1}/{n_batches})"
        fig.suptitle(title, fontsize=13, y=1.01)

        out_path = output_dir / f"temporal_activity_batch_{b + 1:02d}.png"
        save_fig(fig, out_path)
        written.append(out_path)

    return written


def save_temporal_area_charts(
    temporal_statistics: dict,
    output_dir,
    *,
    node_label: Optional[str] = None,
) -> None:
    """Save one filled area chart per temporal column.

    Area charts provide a cumulative-volume view of temporal activity and
    complement the line charts produced by save_temporal_line_charts.

    Args:
        temporal_statistics (dict): Mapping of column name to temporal
            statistics, each entry containing an "observations_per_period"
            sub-dict.
        output_dir (str or Path): Directory where images are written.
        node_label (str, optional): Node identifier appended to plot titles.
    """
    if not temporal_statistics:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    for feature, metrics in temporal_statistics.items():
        obs = metrics.get("observations_per_period")
        if not obs:
            continue
        fig = None
        try:
            timestamps, counts = _periods_to_timestamps(obs)
            if not timestamps:
                continue

            fig, ax = plt.subplots(figsize=(10, 5))
            line_chart(
                ax, timestamps, counts,
                fill=True,
                title=(
                    f"{feature} — Temporal Volume"
                    + (f" ({node_label})" if node_label else "")
                ),
                xlabel="Time",
                ylabel="Observations",
            )
            save_fig(fig, output_dir / f"{feature}_area.png")
        except Exception:
            logger.warning("Temporal area chart error (%s)", feature, exc_info=True)
            if fig is not None:
                plt.close(fig)


def save_temporal_bar_charts(
    temporal_statistics: dict,
    output_dir,
    *,
    node_label: Optional[str] = None,
) -> None:
    """Save one horizontal bar chart of observation counts per period per temporal column.

    Useful for sparse time series where a line chart would be misleading
    due to the implied continuity between widely-spaced data points.

    Args:
        temporal_statistics (dict): Mapping of column name to temporal
            statistics, each entry containing an "observations_per_period"
            sub-dict.
        output_dir (str or Path): Directory where images are written.
        node_label (str, optional): Node identifier appended to plot titles.
    """
    if not temporal_statistics:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    for feature, metrics in temporal_statistics.items():
        obs = metrics.get("observations_per_period")
        if not obs:
            continue
        fig = None
        try:
            timestamps, counts = _periods_to_timestamps(obs)
            if not timestamps:
                continue
            labels = [str(t)[:7] for t in timestamps]  # "YYYY-MM"

            fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.3)))
            bar_chart(
                ax, labels, counts,
                horizontal=True,
                title=(
                    f"{feature} — Observations per Period"
                    + (f" ({node_label})" if node_label else "")
                ),
                xlabel="Count",
                ylabel="Period",
            )
            save_fig(fig, output_dir / f"{feature}_bar.png")
        except Exception:
            logger.warning("Temporal bar chart error (%s)", feature, exc_info=True)
            if fig is not None:
                plt.close(fig)


def save_temporal_heatmap(
    temporal_statistics: dict,
    output_dir,
    *,
    node_label: Optional[str] = None,
) -> None:
    """Save a month-by-year heatmap of observation counts per temporal column.

    Only produced when at least 2 distinct years are present in the data.
    Useful for spotting seasonal patterns or data collection gaps.

    Args:
        temporal_statistics (dict): Mapping of column name to temporal
            statistics, each entry containing an "observations_per_period"
            sub-dict.
        output_dir (str or Path): Directory where images are written.
        node_label (str, optional): Node identifier appended to plot titles.
    """
    if not temporal_statistics:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    for feature, metrics in temporal_statistics.items():
        obs = metrics.get("observations_per_period")
        if not obs:
            continue
        fig = None
        try:
            timestamps, counts = _periods_to_timestamps(obs)
            if not timestamps:
                continue

            ts_series = pd.Series(counts, index=pd.DatetimeIndex(timestamps))
            pivot = ts_series.groupby(
                [ts_series.index.year, ts_series.index.month]
            ).sum().unstack(fill_value=0)
            pivot.columns = [
                pd.Timestamp(2000, m, 1).strftime("%b")
                for m in pivot.columns
            ]

            if pivot.shape[0] < 2:
                continue

            fig, ax = plt.subplots(figsize=(12, max(4, pivot.shape[0] * 0.5)))
            heatmap(
                ax, pivot,
                cmap="YlOrRd",
                annotate=True, fmt=".0f",
                title=(
                    f"{feature} — Month × Year Heatmap"
                    + (f" ({node_label})" if node_label else "")
                ),
            )
            ax.set_xlabel("Month")
            ax.set_ylabel("Year")
            save_fig(fig, output_dir / f"{feature}_heatmap.png")
        except Exception:
            logger.warning("Temporal heatmap error (%s)", feature, exc_info=True)
            if fig is not None:
                plt.close(fig)
