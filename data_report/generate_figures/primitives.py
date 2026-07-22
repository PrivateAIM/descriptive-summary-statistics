"""
Axis-level rendering primitives.

Every function takes a matplotlib Axes object as its first argument and
mutates it in place — it never creates a figure or calls plt.savefig.
Domain modules (local_descriptive_plots, etc.) are responsible for figure
layout; primitives only handle rendering and styling.

Palette and DPI are read from style.py at call time (not at import time),
so style.set_theme() / style.set_palette() called before report generation
propagates to all plots automatically.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from data_report.generate_figures import style

# ---------------------------------------------------------------------------
# Figure-level utilities
# ---------------------------------------------------------------------------

def make_subplots(
    n: int,
    ncols: int = 3,
    *,
    width: float = 5,
    height: float = 4,
) -> tuple[Figure, np.ndarray]:
    """Create a grid of n axes laid out in up to ncols columns.

    Unused axes at the end are hidden so callers can zip(axes, items)
    without worrying about extra panels.

    Args:
        n (int): Total number of subplots required.
        ncols (int): Maximum number of columns in the grid.
        width (float): Width in inches of each individual subplot.
        height (float): Height in inches of each individual subplot.

    Returns:
        tuple[Figure, np.ndarray]: The matplotlib Figure and a flat array
            of exactly n visible Axes objects.

    Raises:
        ValueError: If n is less than 1.
    """
    if n == 0:
        raise ValueError("make_subplots requires n >= 1")
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(width * ncols, height * nrows),
        squeeze=False,
    )
    flat = axes.flatten()
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


def save_fig(fig: Figure, path, *, dpi: Optional[int] = None) -> None:
    """Apply tight_layout, save the figure to disk, and close it.

    Args:
        fig (Figure): The matplotlib Figure to save.
        path (str or Path): Destination file path.
        dpi (int, optional): Output resolution in dots per inch. Defaults
            to style.DPI when not provided.
    """
    fig.tight_layout()
    fig.savefig(path, dpi=dpi or style.DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Axis-level primitives
# ---------------------------------------------------------------------------

def histogram(
    ax,
    values,
    *,
    bins: Optional[int] = None,
    color: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: str = "Count",
) -> None:
    """Draw a histogram on ax with automatic bin count selection.

    Bin count is computed via Freedman-Diaconis when IQR > 0, with a
    Sturges fallback for constant or near-constant data. Capped at 50 bins
    so wide-range variables stay readable. NaN values are silently dropped.

    Args:
        ax (Axes): Matplotlib axes to draw on.
        values (array-like): Raw numeric values to bin.
        bins (int, optional): Explicit bin count; computed automatically
            when omitted.
        color (str, optional): Bar fill colour; defaults to style.PALETTE[0].
        title (str, optional): Axes title.
        xlabel (str, optional): X-axis label.
        ylabel (str): Y-axis label. Defaults to "Count".
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        _decorate(ax, title, xlabel, ylabel)
        return
    n_bins = bins if bins is not None else _auto_bins(values)
    ax.hist(values, bins=n_bins, color=color or style.PALETTE[0], edgecolor="white")
    _decorate(ax, title, xlabel, ylabel)


def histogram_from_bins(
    ax,
    edges,
    counts,
    *,
    color: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: str = "Count",
) -> None:
    """Draw a histogram from pre-computed bin edges and counts.

    Used for federated data where raw values are unavailable; the caller
    supplies aggregated edges and per-bin counts directly.

    Args:
        ax (Axes): Matplotlib axes to draw on.
        edges (array-like): Bin boundary values; length must be
            len(counts) + 1.
        counts (array-like): Per-bin observation counts.
        color (str, optional): Bar fill colour; defaults to style.PALETTE[0].
        title (str, optional): Axes title.
        xlabel (str, optional): X-axis label.
        ylabel (str): Y-axis label. Defaults to "Count".
    """
    edges = np.asarray(edges, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if len(edges) < 2:
        _decorate(ax, title, xlabel, ylabel)
        return
    centers = (edges[:-1] + edges[1:]) / 2
    widths = (edges[1:] - edges[:-1]) * 0.9
    ax.bar(centers, counts, width=widths, color=color or style.PALETTE[0], edgecolor="white")
    _decorate(ax, title, xlabel, ylabel)


def boxplot(
    ax,
    data: list,
    *,
    labels: Optional[list] = None,
    orient: str = "v",
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    """Draw a box-and-whisker plot on ax.

    Outliers are rendered as small semi-transparent dots to reduce visual
    clutter.

    Args:
        ax (Axes): Matplotlib axes to draw on.
        data (list): List of 1-D arrays, one array per group.
        labels (list, optional): Group labels aligned with data.
        orient (str): Orientation — "v" for vertical, "h" for horizontal.
            Defaults to "v".
        title (str, optional): Axes title.
        ylabel (str, optional): Y-axis label.
    """
    if not data:
        return
    bp = ax.boxplot(
        data,
        labels=labels,
        vert=(orient == "v"),
        patch_artist=True,
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(style.PALETTE[0])
        patch.set_alpha(0.6)
    _decorate(ax, title, ylabel=ylabel)
    if labels and orient == "v":
        ax.set_xticklabels(labels, rotation=45, ha="right")


def violin(
    ax,
    data: list,
    *,
    labels: Optional[list] = None,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    """Draw a violin plot with a median line on ax.

    Args:
        ax (Axes): Matplotlib axes to draw on.
        data (list): List of 1-D arrays, one array per group.
        labels (list, optional): Group labels aligned with data.
        title (str, optional): Axes title.
        ylabel (str, optional): Y-axis label.
    """
    if not data or all(len(d) == 0 for d in data):
        return
    parts = ax.violinplot(data, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(style.PALETTE[0])
        pc.set_alpha(0.6)
    for part_name in ("cbars", "cmins", "cmaxes", "cmedians"):
        if part_name in parts:
            parts[part_name].set_color(style.PALETTE[3])
    if labels:
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right")
    _decorate(ax, title, ylabel=ylabel)


def bar_chart(
    ax,
    categories: Sequence,
    values: Sequence,
    *,
    horizontal: bool = False,
    colors: Optional[list] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    max_n: int = 20,
) -> None:
    """Draw a bar chart on ax, truncating to the top max_n categories if needed.

    When more than max_n categories are supplied they are ranked by value and
    only the highest max_n are plotted.

    Args:
        ax (Axes): Matplotlib axes to draw on.
        categories (Sequence): Category labels for each bar.
        values (Sequence): Numeric height of each bar.
        horizontal (bool): When True, draws horizontal bars. Defaults to
            False.
        colors (list, optional): Per-bar colours; cycles from style.PALETTE
            when omitted.
        title (str, optional): Axes title.
        xlabel (str, optional): X-axis label.
        ylabel (str, optional): Y-axis label.
        max_n (int): Maximum number of bars to draw. Defaults to 20. Callers
            that already truncated ``categories``/``values`` themselves (e.g.
            to build a title like "top N") should pass ``max_n=len(categories)``
            so this doesn't silently re-truncate below what the title claims
            is shown.
    """
    categories, values = _maybe_truncate(list(categories), list(values), max_n=max_n)
    c = colors if colors is not None else style.PALETTE[:len(categories)]
    if horizontal:
        ax.barh(categories, values, color=c)
    else:
        ax.bar(range(len(categories)), values, color=c)
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=45, ha="right")
    _decorate(ax, title, xlabel, ylabel)


def stacked_bar(
    ax,
    df: pd.DataFrame,
    *,
    colors: Optional[list] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    """Draw a stacked bar chart on ax from a tidy DataFrame.

    DataFrame rows correspond to x-axis positions and columns to stack
    layers. Index values are used as x-axis tick labels.

    Args:
        ax (Axes): Matplotlib axes to draw on.
        df (pd.DataFrame): Pivot-style table of counts or proportions.
        colors (list, optional): One colour per column; cycles from
            style.PALETTE when omitted.
        title (str, optional): Axes title.
        xlabel (str, optional): X-axis label.
        ylabel (str, optional): Y-axis label.
    """
    if df.empty:
        return
    c = colors if colors is not None else style.PALETTE[:len(df.columns)]
    bottom = np.zeros(len(df))
    for i, col in enumerate(df.columns):
        ax.bar(
            range(len(df)),
            df[col].values,
            bottom=bottom,
            label=str(col),
            color=c[i % len(c)],
        )
        bottom += df[col].values
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([str(v) for v in df.index], rotation=45, ha="right")
    ax.legend()
    _decorate(ax, title, xlabel, ylabel)


def scatter(
    ax,
    x,
    y,
    *,
    hue=None,
    alpha: float = 0.6,
    reg_line: bool = False,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    """Draw a scatter plot on ax with an optional regression line overlay.

    Args:
        ax (Axes): Matplotlib axes to draw on.
        x (array-like): Numeric values for the x-axis.
        y (array-like): Numeric values for the y-axis.
        hue (array-like, optional): Category labels for colour-coding points.
        alpha (float): Point transparency. Defaults to 0.6.
        reg_line (bool): When True, overlays an OLS regression line.
            Defaults to False.
        title (str, optional): Axes title.
        xlabel (str, optional): X-axis label.
        ylabel (str, optional): Y-axis label.

    Note:
        The regression line is omitted when fewer than three valid
        (non-NaN) point pairs are available.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if hue is not None:
        sns.scatterplot(x=x, y=y, hue=hue, alpha=alpha, ax=ax)
    else:
        ax.scatter(x, y, alpha=alpha, color=style.PALETTE[0])
    if reg_line:
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() > 2:
            m, b = np.polyfit(x[valid], y[valid], 1)
            xs = np.linspace(np.nanmin(x), np.nanmax(x), 200)
            ax.plot(xs, m * xs + b, color=style.PALETTE[1], linewidth=1.5, linestyle="--")
    _decorate(ax, title, xlabel, ylabel)


def line_chart(
    ax,
    x,
    y,
    *,
    markers: bool = True,
    fill: bool = False,
    color: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    """Draw a line chart on ax, optionally shaded as an area chart.

    Args:
        ax (Axes): Matplotlib axes to draw on.
        x (array-like): X-axis values (typically timestamps or indices).
        y (array-like): Y-axis values.
        markers (bool): When True, adds circular markers at each data point.
            Defaults to True.
        fill (bool): When True, shades the region between the line and zero
            to produce an area chart. Defaults to False.
        color (str, optional): Line and fill colour; defaults to
            style.PALETTE[0].
        title (str, optional): Axes title.
        xlabel (str, optional): X-axis label.
        ylabel (str, optional): Y-axis label.
    """
    c = color or style.PALETTE[0]
    ax.plot(x, y, marker="o" if markers else None, color=c, linewidth=2)
    if fill:
        ax.fill_between(x, y, alpha=0.25, color=c)
    ax.tick_params(axis="x", rotation=45)
    _decorate(ax, title, xlabel, ylabel)


def heatmap(
    ax,
    matrix,
    *,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    annotate: bool = False,
    fmt: str = ".2f",
    title: Optional[str] = None,
    tick_fontsize: float = 10,
    annot_fontsize: float = 10,
) -> None:
    """Draw a heatmap on ax using seaborn.

    Args:
        ax (Axes): Matplotlib axes to draw on.
        matrix (DataFrame or array-like): 2-D numeric matrix to visualise.
        cmap (str): Matplotlib colour-map name. Defaults to "viridis".
        vmin (float, optional): Lower bound of the colour scale.
        vmax (float, optional): Upper bound of the colour scale.
        annotate (bool): When True, prints cell values inside each cell.
            Defaults to False.
        fmt (str): Format string for cell annotations. Defaults to ".2f".
        title (str, optional): Axes title.
        tick_fontsize (float): Font size for axis tick labels. Defaults to 10.
        annot_fontsize (float): Font size for cell-value annotations.
            Defaults to 10.

    Note:
        tick_fontsize and annot_fontsize are exposed explicitly because the
        seaborn defaults become illegible once figures are scaled down to
        fit a report page.
    """
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=annotate,
        fmt=fmt if annotate else "",
        annot_kws={"fontsize": annot_fontsize} if annotate else None,
    )
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    if title:
        ax.set_title(title, fontsize=tick_fontsize + 2)


def pie_chart(
    ax,
    sizes: Sequence,
    labels: Sequence,
    *,
    colors: Optional[list] = None,
    min_slice_pct: float = 3.0,
    title: Optional[str] = None,
) -> None:
    """Draw a pie chart on ax, merging small slices into an "Other" category.

    Slices that represent less than min_slice_pct percent of the total are
    collapsed into a single "Other" slice to prevent label overlap.

    Args:
        ax (Axes): Matplotlib axes to draw on.
        sizes (Sequence): Numeric size of each slice.
        labels (Sequence): Label for each slice, aligned with sizes.
        colors (list, optional): Per-slice colours; cycles from style.PALETTE
            when omitted.
        min_slice_pct (float): Minimum slice size as a percentage of the
            total below which slices are merged. Defaults to 3.0.
        title (str, optional): Axes title.
    """
    sizes_arr = np.asarray(sizes, dtype=float)
    labels_list = list(labels)
    if sizes_arr.sum() == 0:
        _decorate(ax, title)
        return
    sizes_arr, labels_list = _merge_small_slices(sizes_arr, labels_list, min_slice_pct)
    c = colors if colors is not None else style.PALETTE[:len(sizes_arr)]
    ax.pie(sizes_arr, labels=labels_list, autopct="%1.1f%%", startangle=90, colors=c)
    ax.axis("equal")
    if title:
        ax.set_title(title, pad=20)


def count_plot(
    ax,
    series: pd.Series,
    *,
    top_n: int = 20,
    horizontal: bool = True,
    title: Optional[str] = None,
) -> None:
    """Draw a bar chart of value counts for a pandas Series on ax.

    Args:
        ax (Axes): Matplotlib axes to draw on.
        series (pd.Series): Categorical series whose value counts are plotted.
        top_n (int): Maximum number of categories to show, ranked by
            frequency. Defaults to 20.
        horizontal (bool): When True, draws horizontal bars. Defaults to True.
        title (str, optional): Axes title.
    """
    counts = series.value_counts(dropna=True).head(top_n)
    if counts.empty:
        _decorate(ax, title)
        return
    # The count axis is x for horizontal bars, y for vertical ones -- always
    # labeling ylabel="Count" put the label on the category axis instead of
    # the value axis whenever horizontal=True (the default), leaving the
    # real count axis unlabeled.
    bar_chart(
        ax,
        list(counts.index.astype(str)),
        list(counts.values),
        horizontal=horizontal,
        title=title,
        xlabel="Count" if horizontal else None,
        ylabel=None if horizontal else "Count",
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _auto_bins(values: np.ndarray) -> int:
    """Return an auto-computed histogram bin count clamped to [5, 50]."""
    n = len(values)
    if n < 4:
        return max(n, 1)
    iqr = float(np.percentile(values, 75) - np.percentile(values, 25))
    if iqr == 0:
        # Sturges' formula
        return min(int(math.ceil(1 + math.log2(n))), 50)
    data_range = float(values.max() - values.min())
    if data_range == 0:
        return 1
    bin_width = 2.0 * iqr * (n ** (-1.0 / 3.0))
    return int(min(max(math.ceil(data_range / bin_width), 5), 50))


def _decorate(
    ax,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    """Apply title, axis labels, and a faint y-axis grid."""
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)


def _maybe_truncate(
    categories: list,
    values: list,
    max_n: int = 20,
) -> tuple[list, list]:
    """Return the top max_n (categories, values) pairs ranked by value descending."""
    if len(categories) <= max_n:
        return categories, values
    paired = sorted(zip(values, categories), reverse=True)[:max_n]
    vals, cats = zip(*paired)
    return list(cats), list(vals)


def declutter_radial_labels(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    names: Sequence[str],
    *,
    base_offset: float = 1.08,
    perp_step_frac: float = 0.17,
    angle_epsilon_deg: float = 10.0,
    fontsize: float = 7,
) -> float:
    """Draw feature-vector labels for a biplot, fanning labels out to avoid overlap.

    Biplot arrows for correlated features (common with e.g. lab values that
    move together) often point in nearly the same direction -- placing every
    label at the same fixed offset from its arrow tip stacks them into
    unreadable overlapping text. Pushing labels further along the *same*
    direction doesn't fix this: label text extends sideways (e.g. left-to-
    right for a roughly horizontal group of arrows), so a label placed
    further out along the shared direction still collides with the text of
    the label before it. Instead, this groups arrows by angle and, within
    each group, fans labels out *perpendicular* to the group's shared
    direction (alternating sides, growing distance), which is exactly the
    direction the shared direction's own labels don't already extend into.
    The most prominent (largest-magnitude) arrow keeps its label undisplaced;
    offset labels are connected back to their arrow tip with a thin leader
    line.

    Args:
        ax (Axes): Matplotlib axes with the arrows already drawn.
        x (np.ndarray): X-coordinates of the arrow tips (loading values).
        y (np.ndarray): Y-coordinates of the arrow tips.
        names (Sequence[str]): Feature name for each arrow tip.
        base_offset (float): Radial multiplier applied to every label's
            distance from the origin along its own arrow's direction.
            Defaults to 1.08.
        perp_step_frac (float): Spacing between fanned-out labels in a
            group, as a fraction of the largest loading magnitude in the
            plot. Defaults to 0.12.
        angle_epsilon_deg (float): Arrows within this many degrees of each
            other are treated as one group. Defaults to 10.0.
        fontsize (float): Label font size. Defaults to 7.

    Returns:
        float: The largest absolute coordinate (x or y) used by any label,
            so callers can size axis limits to keep every label inside the
            frame.
    """
    n = len(names)
    if n == 0:
        return base_offset

    # Points with a non-finite coordinate have no sensible label position --
    # skip them here rather than letting a single NaN/Inf loading poison
    # overall_max (and therefore every other label's offset and the caller's
    # axis-limit calculation) for the whole plot.
    finite_mask = np.isfinite(x) & np.isfinite(y)
    valid_idx = np.flatnonzero(finite_mask)
    if len(valid_idx) == 0:
        return base_offset

    magnitudes = np.hypot(x, y)
    overall_max = magnitudes[valid_idx].max()
    if not np.isfinite(overall_max) or overall_max <= 0:
        overall_max = 1.0
    perp_step = overall_max * perp_step_frac
    angles_deg = np.degrees(np.arctan2(y, x))

    order = valid_idx[np.argsort(angles_deg[valid_idx])]
    sorted_angles = angles_deg[order]
    m = len(order)

    if m > 1:
        # Find the circle's emptiest point and cut there, rather than always
        # cutting at the arbitrary -180/180 seam -- two arrows at e.g. -179
        # and +179 degrees are only 2 degrees apart on the circle, but a
        # naive numeric sort puts them at opposite ends of the array, so
        # comparing only consecutive *sorted* neighbors would never group
        # them (finding #1: angle-wraparound bug).
        gaps = np.diff(sorted_angles)
        wrap_gap = (sorted_angles[0] + 360.0) - sorted_angles[-1]
        cut = int(np.argmax(np.concatenate([gaps, [wrap_gap]])))
        order = np.roll(order, -(cut + 1))
        sorted_angles = np.roll(sorted_angles, -(cut + 1))
        # Unwrap so the sequence (now broken at its emptiest point instead of
        # at -180/180) is monotonically non-decreasing -- chain grouping
        # below assumes each step only moves forward.
        for i in range(1, m):
            if sorted_angles[i] < sorted_angles[i - 1]:
                sorted_angles[i:] += 360.0

    # Chain consecutive (sorted-by-angle) points into a group whenever each
    # is within angle_epsilon_deg of its immediate neighbor -- comparing only
    # to a group's first member (as an earlier version of this function did)
    # misses long fans of arrows where cumulative drift across many points
    # exceeds the threshold even though each neighboring pair is close.
    groups: list[list[int]] = []
    current_group: list[int] = []
    prev_angle = None
    for pos in range(m):
        idx = order[pos]
        angle = sorted_angles[pos]
        if current_group and abs(angle - prev_angle) > angle_epsilon_deg:
            groups.append(current_group)
            current_group = []
        current_group.append(idx)
        prev_angle = angle
    if current_group:
        groups.append(current_group)

    max_extent = base_offset * overall_max
    for group in groups:
        # Longest (most important) loading keeps the undisplaced label.
        group_sorted = sorted(group, key=lambda i: magnitudes[i], reverse=True)
        # Circular mean via unit-vector sum (not a plain mean of angles_deg)
        # -- a group straddling the -180/180 seam (e.g. members at -179 and
        # +179 degrees) would otherwise average to ~0 degrees, pointing the
        # fan in the opposite direction from where the arrows actually are.
        unit_vecs = [(x[i] / magnitudes[i], y[i] / magnitudes[i])
                     for i in group_sorted if magnitudes[i] > 1e-9]
        if unit_vecs:
            mean_vx = sum(v[0] for v in unit_vecs) / len(unit_vecs)
            mean_vy = sum(v[1] for v in unit_vecs) / len(unit_vecs)
            mean_angle = np.arctan2(mean_vy, mean_vx)
        else:
            mean_angle = np.radians(angles_deg[group_sorted[0]])
        perp_x, perp_y = -np.sin(mean_angle), np.cos(mean_angle)
        for rank, i in enumerate(group_sorted):
            base_x, base_y = x[i] * base_offset, y[i] * base_offset
            # Fan out alternating sides: 0, +1, -1, +3, -3, +5, -5, ... (in
            # units of perp_step) -- using odd multiples rather than 1, 2,
            # 3, ... keeps same-side neighbors 2 full steps apart instead of
            # 1, which a single step is too tight for typical label widths.
            side = 1 if rank % 2 == 1 else -1
            step_count = (rank + 1) // 2
            depth = 2 * step_count - 1 if step_count > 0 else 0
            perp_offset = side * depth * perp_step
            label_x = base_x + perp_x * perp_offset
            label_y = base_y + perp_y * perp_offset
            max_extent = max(max_extent, abs(label_x), abs(label_y))
            if perp_offset != 0:
                ax.plot([x[i], label_x], [y[i], label_y],
                        color="gray", linewidth=0.6, alpha=0.6, zorder=1)
            # Aligned so a fanned-out label's text extends away from the
            # group's center rather than back into it: whichever axis the
            # fan spreads along (perp_x vs perp_y) gets its alignment keyed
            # off the offset's sign; the other axis falls back to the
            # label's own position (the fan barely moves it on that axis).
            if abs(perp_x) >= abs(perp_y):
                _ha = "left" if perp_offset >= 0 else "right"
                _va = "bottom" if label_y >= 0 else "top"
            else:
                _ha = "left" if label_x >= 0 else "right"
                _va = "bottom" if perp_offset >= 0 else "top"
            ax.text(label_x, label_y, names[i], fontsize=fontsize, ha=_ha, va=_va,
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7, edgecolor="none"),
                    zorder=2)

    return max_extent / overall_max if overall_max > 0 else base_offset


def declutter_point_labels(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    labels: Sequence[str],
    *,
    cluster_frac: float = 0.05,
    vertical_step_frac: float = 0.045,
    fontsize: float = 8,
) -> None:
    """Label a scatter of arbitrary points, staggering labels within close clusters.

    Unlike a biplot's arrows (all sharing the origin), points here can sit
    anywhere -- two unrelated categories that happen to land at nearly the
    same coordinates (a real occurrence e.g. in an MCA category map) would
    otherwise get their text labels drawn on top of each other and become
    unreadable. This groups points by proximity (relative to the data's own
    spread) and stacks each group's labels vertically, connecting offset
    labels back to their point with a thin leader line.

    Args:
        ax (Axes): Matplotlib axes with the points already scattered.
        x (np.ndarray): X-coordinates of the points to label.
        y (np.ndarray): Y-coordinates of the points to label.
        labels (Sequence[str]): Label text for each point.
        cluster_frac (float): Points within this fraction of the data's
            diagonal extent are treated as one group. Defaults to 0.05.
        vertical_step_frac (float): Vertical spacing between stacked labels
            in a group, as a fraction of the data's y-extent. Defaults to
            0.045.
        fontsize (float): Label font size. Defaults to 8.
    """
    n = len(labels)
    if n == 0:
        return

    # Points with a non-finite coordinate have no sensible label position --
    # np.ptp propagates NaN if any single point is NaN, which would silently
    # disable decluttering for every OTHER point in the plot too.
    finite_mask = np.isfinite(x) & np.isfinite(y)
    valid_idx = np.flatnonzero(finite_mask)
    if len(valid_idx) == 0:
        return
    vx, vy = x[valid_idx], y[valid_idx]

    x_range = np.ptp(vx) or 1.0
    y_range = np.ptp(vy) or 1.0
    diag = np.hypot(x_range, y_range)
    cluster_radius = diag * cluster_frac
    vertical_step = y_range * vertical_step_frac

    # Join a point to the *closest* group whose running centroid is within
    # cluster_radius, not the first group with any member in range -- the
    # latter is single-linkage clustering, which chain-collapses a long line
    # of evenly-spaced points into one giant group (each consecutive pair is
    # close, so transitively "everything" joins) and is sensitive to input
    # order (a bridging point between two distant clusters joins whichever
    # cluster happens to be checked first). Centroid distance keeps each
    # group's own spread bounded and picks the same grouping regardless of
    # the order points are processed in.
    groups: list[list[int]] = []
    centroids: list[tuple[float, float]] = []
    for i in valid_idx:
        best_group, best_dist = None, None
        for gi, (cx, cy) in enumerate(centroids):
            d = np.hypot(x[i] - cx, y[i] - cy)
            if d <= cluster_radius and (best_dist is None or d < best_dist):
                best_group, best_dist = gi, d
        if best_group is not None:
            groups[best_group].append(i)
            members = groups[best_group]
            centroids[best_group] = (
                sum(x[j] for j in members) / len(members),
                sum(y[j] for j in members) / len(members),
            )
        else:
            groups.append([i])
            centroids.append((x[i], y[i]))

    for group in groups:
        if len(group) == 1:
            i = group[0]
            ax.annotate(labels[i], (x[i], y[i]), fontsize=fontsize, ha="center", va="bottom")
            continue
        # Stack the group's labels vertically around its centroid, most
        # distinctive (furthest from origin) first.
        group_sorted = sorted(group, key=lambda i: np.hypot(x[i], y[i]), reverse=True)
        cx = np.mean([x[i] for i in group])
        cy = max(y[i] for i in group)
        for rank, i in enumerate(group_sorted):
            label_x, label_y = cx, cy + (rank + 1) * vertical_step
            ax.plot([x[i], label_x], [y[i], label_y],
                    color="gray", linewidth=0.6, alpha=0.6, zorder=1)
            ax.text(label_x, label_y, labels[i], fontsize=fontsize, ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7, edgecolor="none"),
                    zorder=2)


def _merge_small_slices(
    sizes: np.ndarray,
    labels: list,
    threshold_pct: float,
) -> tuple[np.ndarray, list]:
    """Merge slices below threshold_pct of the total into a single "Other" entry."""
    total = sizes.sum()
    if total == 0:
        return sizes, labels
    mask = (sizes / total * 100) >= threshold_pct
    other = sizes[~mask].sum()
    sizes_out = list(sizes[mask]) + ([other] if other > 0 else [])
    labels_out = (
        [l for l, m in zip(labels, mask) if m]
        + (["Other"] if other > 0 else [])
    )
    return np.asarray(sizes_out, dtype=float), labels_out
