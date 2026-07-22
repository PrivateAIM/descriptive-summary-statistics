"""Self-contained Hub entry point for the FLAME/privateAIM federated data-report
pipeline.

Bundles all project modules (data loading, statistics, inferential analysis,
PCA/MCA, plotting, PDF reports, JSON summaries) into a single file for
deployment as a FLAME StarModel.

Architecture:
    * ``DataReportAnalyzer`` runs on each hospital node: loads the node's CSV,
      computes descriptive and inferential statistics, encodes all df-dependent
      plots as base64 strings, and returns a JSON-serialisable result dict.
    * ``DataReportAggregator`` runs on the central coordinator: collects result
      dicts from all nodes, builds per-node and federated outputs (CSVs, PNGs,
      PDFs, summary.json), packs them into a ``.tar.gz`` archive, and returns
      it as a base64-encoded string.

Heavy libraries (seaborn, plotly, prince, missingno, sklearn, scipy, PIL,
reportlab) are lazy-loaded after the FLAME SDK handshake to avoid exceeding
the platform's container startup timeout.

Output: a base64-encoded ``.tar.gz`` (``results.tar.gz.b64.txt``) containing::

    federated/
        overview/ numeric/ categorical/ temporal/
        summary.json  report_short.pdf  report_full.pdf
    local/
        node<N>/
            overview/ numeric/ categorical/ temporal/ comparison/
            pca/  mca/
            inferential/
            summary.json  report_short.pdf  report_full.pdf
"""
from __future__ import annotations
import ast
import base64
import itertools
import json
import logging
import math
import re
import tarfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Sequence, Literal

# matplotlib must be set to headless backend before pyplot is imported.
# numpy and pandas are used throughout and import fast enough to keep at
# module level. Everything else is lazy-loaded in _load_analysis_dependencies()
# and _load_reporting_dependencies() to keep container startup below the
# FLAME platform's timeout.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from flame.star import StarAnalyzer, StarAggregator, StarModel


# ============================================================================
# Lazy-loaded heavy dependencies
# ============================================================================
# PIL, reportlab, seaborn, plotly, prince, missingno, pingouin, scikit-learn
# and the extra scipy submodules are all slow to import. Loading them at
# module level (before StarModel() is called) pushes container startup past
# the FLAME platform's startup timeout (ANALYSISSTARTUPERROR). They are
# loaded once on first use via _load_analysis_dependencies() and
# _load_reporting_dependencies() — called after the SDK handshake.
PILImage = None
colors = getSampleStyleSheet = ParagraphStyle = None
inch = stringWidth = None
BaseDocTemplate = Frame = KeepTogether = NextPageTemplate = None
PageBreak = PageTemplate = Paragraph = Spacer = Table = TableStyle = Image = None
sns = px = prince = msno = pg = mpatches = colormaps = None
PCA = StandardScaler = None
stats = chi2_contingency = fisher_exact = linregress = None
find_peaks = _linear_detrend = blackman = rfft = rfftfreq = None
_analysis_deps_loaded = False
_reporting_deps_loaded = False




def _load_analysis_dependencies() -> None:
    """Load libraries needed for data analysis (scipy, sklearn, seaborn, etc).

    These are moderately heavy but must be available when analysis_method runs.
    Separated from reporting deps so PDF/reportlab loading can be deferred
    to the very end of the aggregator when reports are actually being built.
    """
    global sns, px, prince, msno, pg, mpatches, colormaps
    global PCA, StandardScaler
    global stats, chi2_contingency, fisher_exact, linregress
    global find_peaks, _linear_detrend, blackman, rfft, rfftfreq
    global _analysis_deps_loaded

    if _analysis_deps_loaded:
        return

    print("Loading analysis dependencies...", flush=True)

    import seaborn as _sns
    import plotly.express as _px
    import prince as _prince
    import missingno as _msno
    import pingouin as _pg
    import matplotlib.patches as _mpatches
    from matplotlib import colormaps as _colormaps
    sns = _sns; px = _px; prince = _prince; msno = _msno
    pg = _pg; mpatches = _mpatches; colormaps = _colormaps

    from sklearn.decomposition import PCA as _PCA
    from sklearn.preprocessing import StandardScaler as _SS
    PCA = _PCA; StandardScaler = _SS

    from scipy import stats as _stats
    from scipy.stats import chi2_contingency as _c2, fisher_exact as _fe, linregress as _lr
    from scipy.signal import find_peaks as _fp, detrend as _ltd
    from scipy.signal.windows import blackman as _bm
    from scipy.fft import rfft as _rfft, rfftfreq as _rfftfreq
    stats = _stats
    chi2_contingency = _c2; fisher_exact = _fe; linregress = _lr
    find_peaks = _fp; _linear_detrend = _ltd; blackman = _bm
    rfft = _rfft; rfftfreq = _rfftfreq

    _analysis_deps_loaded = True
    print("Analysis dependencies loaded.", flush=True)


def _load_reporting_dependencies() -> None:
    """Load PIL and reportlab — only needed when generating PDFs/reports.

    Called lazily just before PDF generation in the aggregator, so the
    heavy reportlab startup cost doesn't affect analysis_method timing.
    """
    global PILImage
    global colors, getSampleStyleSheet, ParagraphStyle
    global inch, stringWidth
    global BaseDocTemplate, Frame, KeepTogether, NextPageTemplate
    global PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle, Image
    global _reporting_deps_loaded
    global STYLES, MAX_W, MAX_H, PAGE_MARGIN

    if _reporting_deps_loaded:
        return

    print("Loading reporting dependencies...", flush=True)

    from PIL import Image as _PILImage
    PILImage = _PILImage
    PILImage.MAX_IMAGE_PIXELS = None

    from reportlab.lib import colors as _colors
    from reportlab.lib.styles import getSampleStyleSheet as _gss, ParagraphStyle as _PS
    from reportlab.lib.units import inch as _inch
    from reportlab.pdfbase.pdfmetrics import stringWidth as _sw
    from reportlab.platypus import (
        BaseDocTemplate as _BDT, Frame as _Frame, KeepTogether as _KT,
        NextPageTemplate as _NPT, PageBreak as _PB, PageTemplate as _PT,
        Paragraph as _Para, Spacer as _Spacer, Table as _Table,
        TableStyle as _TS, Image as _Image,
    )
    colors = _colors; getSampleStyleSheet = _gss; ParagraphStyle = _PS
    inch = _inch; stringWidth = _sw
    BaseDocTemplate = _BDT; Frame = _Frame; KeepTogether = _KT
    NextPageTemplate = _NPT; PageBreak = _PB; PageTemplate = _PT
    Paragraph = _Para; Spacer = _Spacer; Table = _Table
    TableStyle = _TS; Image = _Image

    STYLES = getSampleStyleSheet()
    MAX_W = _MAX_W_IN * inch
    MAX_H = _MAX_H_IN * inch
    PAGE_MARGIN = _MARGIN_IN * inch

    _reporting_deps_loaded = True
    print("Reporting dependencies loaded.", flush=True)

def _make_serializable(obj):
    """Recursively convert any non-JSON-native type to a plain Python type.

    Everything returned by analysis_method travels over the network as JSON.
    numpy scalars/arrays, pd.Period, pd.Timestamp, float NaN/Inf and custom
    objects will all silently break that serialization — this function catches
    them all in one place.
    """
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(i) for i in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, pd.Period):
        return str(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (pd.NA.__class__,)):  # pd.NA type
        return None
    return obj


def _sanitize_temporal_statistics(ts: dict) -> dict:
    """Convert pd.Period keys and values inside temporal statistics to strings.

    compute_temporal_statistics returns dicts keyed by pd.Period objects and
    may store pd.Period/pd.Timestamp values — none of which survive JSON
    serialization.
    """
    out = {}
    for col, metrics in ts.items():
        m = {}
        for k, v in metrics.items():
            if k == "observations_per_period":
                m[k] = {
                    str(period): int(count)
                    for period, count in v.items()
                }
            elif isinstance(v, pd.Period):
                m[k] = str(v)
            elif isinstance(v, pd.Timestamp):
                m[k] = v.isoformat()
            else:
                m[k] = v
        out[col] = m
    return out


# ============================================================================
# In-memory plot helpers  (used by the analyzer to encode df-dependent plots)
# ============================================================================

def _fig_to_base64(fig) -> str:
    """Render a matplotlib Figure to a PNG and return it as a base64 string."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _fig_to_bytes(fig) -> bytes:
    """Render a matplotlib Figure to PNG bytes (for the aggregator tar)."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialise a DataFrame to CSV bytes."""
    buf = BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


# ============================================================================
# Tar-archive builder  (used by the aggregator)
# ============================================================================

def _build_tar(file_dict: dict) -> bytes:
    """Pack {relative_path: bytes} into a gzip-compressed tar archive.

    Args:
        file_dict: mapping of archive-internal path strings to raw bytes.

    Returns:
        The complete .tar.gz archive as bytes, ready to be returned by
        aggregation_method.
    """
    buf = BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel_path, content in file_dict.items():
            if content is None:
                continue
            info = tarfile.TarInfo(name=rel_path)
            info.size = len(content)
            tar.addfile(info, BytesIO(content))
    return buf.getvalue()


# ============================================================================
# Source: generate_json_summary.py  (adapted for in-memory output_files)
# ============================================================================
# In-memory versions of generate_local_json_summary / generate_global_json_summary.
# The Hub pipeline keeps everything in output_files: dict[str, bytes] rather
# than on disk, so these helpers read CSV bytes from that dict and store the
# resulting JSON bytes back into it.

# CSV paths relative to the node's archive base (local/node<N>/)
_LOCAL_JSON_CSV_MAP: dict[str, str] = {
    "overview":                "overview/overview.csv",
    "numeric_summary":         "numeric/numeric_summary.csv",
    "categorical_summary":     "categorical/categorical_summary.csv",
    "temporal_summary":        "temporal/temporal_summary.csv",
    "numeric_comparison":      "comparison/numeric_comparison.csv",
    "association_screening":   "inferential/association_screening.csv",
    "significant_associations":"inferential/significant_associations.csv",
}

# CSV paths relative to the federated archive base (federated/)
_GLOBAL_JSON_CSV_MAP: dict[str, str] = {
    "overview":               "overview/overview.csv",
    "numeric_statistics":     "numeric/federated_numeric_statistics.csv",
    "categorical_statistics": "categorical/federated_categorical_statistics.csv",
    "temporal_statistics":    "temporal/federated_temporal_statistics.csv",
    "sex_distribution":       "categorical/sex_distribution_federated.csv",
    "age_distribution":       "numeric/age_distribution_federated.csv",
}


def _json_sanitize(value):
    """Recursively convert numpy/pandas scalars to JSON-safe types; NaN/NaT → null."""
    if isinstance(value, dict):
        return {k: _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if (math.isnan(value) or math.isinf(value)) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _json_parse_cell(value):
    """Parse stringified dict/list cells (e.g. from category_counts columns) into Python objects."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                return ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                return value
    return value


def _csv_bytes_to_records(csv_bytes: bytes | None) -> list | None:
    """Parse CSV bytes into a list of JSON-safe record dicts; returns None if csv_bytes is None."""
    if csv_bytes is None:
        return None
    try:
        df = pd.read_csv(BytesIO(csv_bytes))
    except Exception:
        return []
    if df.empty:
        return []
    return [
        {k: _json_sanitize(_json_parse_cell(v)) for k, v in record.items()}
        for record in df.to_dict(orient="records")
    ]


def _build_json_from_output_files(output_files: dict, base: str, csv_map: dict) -> dict:
    """Read each CSV in csv_map from output_files[base/rel_path] and return a summary dict."""
    return {
        key: _csv_bytes_to_records(output_files.get(f"{base}/{rel_path}"))
        for key, rel_path in csv_map.items()
    }


# ============================================================================
# Source: config.py
# ============================================================================
# Generic outcome-column detection for the inferential statistics section.
# Hospital datasets on the hub vary in naming, so candidates are matched by
# keyword in priority order (mortality/survival ranks above demographics-like
# labels) and then validated for usability (not constant, low cardinality,
# enough observations per class) before being accepted -- see
# detect_outcome_column in inferential_analysis.py.
OUTCOME_KEYWORD_GROUPS: List[List[str]] = [
    ["death", "died", "deceased", "mortality", "survival"],
    ["icu", "intensive_care", "ventilation", "intubation"],
    ["readmission", "re-admission", "admission"],
    ["outcome", "status", "diagnosis", "condition", "label", "pasc"],
    ["complication", "adverse"],
]

# ============================================================================
# Source: figures_style.py
# ============================================================================

"""
Global visual style for all generated figures.

Changing PALETTE here (or calling set_palette) before any plots
are generated will propagate everywhere, because primitives.py accesses
style.PALETTE as a module attribute at call time — not at import time.
"""
# ---------------------------------------------------------------------------
# Default palette
# ---------------------------------------------------------------------------
PALETTE: list[str] = [
    "#4C9BE8",  # blue
    "#F28B30",  # orange
    "#3BB37E",  # green
    "#E85C4C",  # red
    "#A77FD3",  # purple
    "#F7C948",  # yellow
    "#72C9CF",  # teal
    "#D98CB0",  # pink
]

# ---------------------------------------------------------------------------
# Default resolution
# ---------------------------------------------------------------------------
DPI: int = 200

# ---------------------------------------------------------------------------
# Named themes
# ---------------------------------------------------------------------------
THEMES: dict[str, list[str]] = {
    "default": [
        "#4C9BE8", "#F28B30", "#3BB37E", "#E85C4C",
        "#A77FD3", "#F7C948", "#72C9CF", "#D98CB0",
    ],
    # Wong (2011) colorblind-safe palette — safe for deuteranopia and protanopia
    "colorblind": [
        "#0072B2", "#E69F00", "#009E73", "#D55E00",
        "#CC79A7", "#56B4E9", "#F0E442", "#000000",
    ],
    "grayscale": [
        "#222222", "#555555", "#888888", "#AAAAAA",
        "#BBBBBB", "#CCCCCC", "#DDDDDD", "#EEEEEE",
    ],
}

# ---------------------------------------------------------------------------
# Theme switching
# ---------------------------------------------------------------------------

def set_palette(new_palette: list[str]) -> None:
    """Replace the global palette. Affects all plots generated after this call."""
    global PALETTE
    PALETTE = list(new_palette)





# ============================================================================
# Source: figures_primitives.py
# ============================================================================

"""
Axis-level rendering primitives.

Every function takes a matplotlib Axes object as its first argument and
mutates it in place — it never creates a figure or calls plt.savefig.
Domain modules (local_descriptive_plots, etc.) are responsible for figure
layout; primitives only handle rendering and styling.

Palette and DPI are read from py at call time (not at import time),
so set_palette() called before report generation propagates to all plots
automatically.
"""





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
    """
    Create a grid of `n` axes laid out in up to `ncols` columns.

    Unused axes at the end are hidden so callers can zip(axes, items)
    without worrying about leftovers. Returns (fig, flat axes array of
    length n).
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
    """tight_layout + savefig + close. Uses DPI when dpi is not given."""
    fig.tight_layout()
    fig.savefig(path, dpi=dpi or DPI, bbox_inches="tight")
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
    """
    Bar histogram with auto-binning.

    Bin count is computed via Freedman-Diaconis when IQR > 0, with a
    Sturges fallback for constant / near-constant data. Capped at 50 bins
    so wide-range variables stay readable.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        _decorate(ax, title, xlabel, ylabel)
        return
    n_bins = bins if bins is not None else _auto_bins(values)
    ax.hist(values, bins=n_bins, color=color or PALETTE[0], edgecolor="white")
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
    """
    Histogram rendered from pre-computed bin edges and counts.
    Used for federated data where raw values are unavailable.
    """
    edges = np.asarray(edges, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if len(edges) < 2:
        _decorate(ax, title, xlabel, ylabel)
        return
    centers = (edges[:-1] + edges[1:]) / 2
    widths = (edges[1:] - edges[:-1]) * 0.9
    ax.bar(centers, counts, width=widths, color=color or PALETTE[0], edgecolor="white")
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
    """
    Box-and-whisker plot. `data` is a list of 1-D arrays (one per group).
    Outliers rendered as small dots to reduce visual clutter.
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
        patch.set_facecolor(PALETTE[0])
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
    """Violin plot with median line shown. `data` is a list of 1-D arrays."""
    if not data or all(len(d) == 0 for d in data):
        return
    parts = ax.violinplot(data, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(PALETTE[0])
        pc.set_alpha(0.6)
    for part_name in ("cbars", "cmins", "cmaxes", "cmedians"):
        if part_name in parts:
            parts[part_name].set_color(PALETTE[3])
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
    """
    Bar chart with automatic truncation to the top max_n categories by value
    when more than max_n are supplied. Callers that already truncated
    categories/values themselves (e.g. to build a title like "top N") should
    pass max_n=len(categories) so this doesn't silently re-truncate below
    what the title claims is shown.
    """
    categories, values = _maybe_truncate(list(categories), list(values), max_n=max_n)
    c = colors if colors is not None else PALETTE[:len(categories)]
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
    """
    Stacked bar chart. `df` rows are x-axis positions, columns are stack layers.
    Index values are used as x-axis labels.
    """
    if df.empty:
        return
    c = colors if colors is not None else PALETTE[:len(df.columns)]
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
    """
    Scatter plot with optional regression line overlay.
    `hue` can be a Series / array of labels for colour-coding points.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if hue is not None:
        sns.scatterplot(x=x, y=y, hue=hue, alpha=alpha, ax=ax)
    else:
        ax.scatter(x, y, alpha=alpha, color=PALETTE[0])
    if reg_line:
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() > 2:
            m, b = np.polyfit(x[valid], y[valid], 1)
            xs = np.linspace(np.nanmin(x), np.nanmax(x), 200)
            ax.plot(xs, m * xs + b, color=PALETTE[1], linewidth=1.5, linestyle="--")
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
    """
    Line chart. Set fill=True for an area chart (shaded region under the line).
    """
    c = color or PALETTE[0]
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
    """
    Heatmap rendered via seaborn. `matrix` can be a DataFrame or 2-D array.
    Set annotate=True to display cell values (use fmt to control formatting).

    `tick_fontsize`/`annot_fontsize` control the axis label and cell-value
    text size, which otherwise default to a size too small to read once the
    figure is scaled down to fit a report page.
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
    """
    Pie chart with small-slice merging. Slices below `min_slice_pct` of the
    total are collapsed into a single "Other" slice to prevent label overlap.
    """
    sizes_arr = np.asarray(sizes, dtype=float)
    labels_list = list(labels)
    if sizes_arr.sum() == 0:
        _decorate(ax, title)
        return
    sizes_arr, labels_list = _merge_small_slices(sizes_arr, labels_list, min_slice_pct)
    c = colors if colors is not None else PALETTE[:len(sizes_arr)]
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
    """
    Bar chart of value counts. Shows the top `top_n` categories by frequency.
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
    """
    Freedman-Diaconis bin count, with Sturges fallback for zero-IQR data.
    Result is clamped to [5, 50].
    """
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
    """
    Draw feature-vector labels for a biplot, fanning labels out to avoid overlap.

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
    line. Returns the largest absolute coordinate (x or y) used by any
    label as a multiple of the largest loading magnitude, so callers can
    size axis limits to keep every label inside the frame.
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
    """
    Label a scatter of arbitrary points, staggering labels within close clusters.

    Unlike a biplot's arrows (all sharing the origin), points here can sit
    anywhere -- two unrelated categories that happen to land at nearly the
    same coordinates (a real occurrence e.g. in an MCA category map) would
    otherwise get their text labels drawn on top of each other and become
    unreadable. This groups points by proximity (relative to the data's own
    spread) and stacks each group's labels vertically, connecting offset
    labels back to their point with a thin leader line.
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


def _maybe_truncate(
    categories: list,
    values: list,
    max_n: int = 20,
) -> tuple[list, list]:
    """
    Keep only the top `max_n` categories by value (descending).
    Returns the original lists unchanged when len <= max_n.
    """
    if len(categories) <= max_n:
        return categories, values
    paired = sorted(zip(values, categories), reverse=True)[:max_n]
    vals, cats = zip(*paired)
    return list(cats), list(vals)


def _merge_small_slices(
    sizes: np.ndarray,
    labels: list,
    threshold_pct: float,
) -> tuple[np.ndarray, list]:
    """
    Merge slices that represent less than `threshold_pct` percent of the
    total into a single "Other" slice. Returns updated (sizes, labels).
    """
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



# ============================================================================
# Source: get_data_utils.py
# ============================================================================




# function to deal with multi-modal data
# function to list the types of files in each dataset folder
# function to store the sources of each dataset and maybe other metadata



# ============================================================================
# Source: comparison_utils.py
# ============================================================================

# computes column distribution:
# columns that exist in all hospitals
# columns that exist in only a few (partially common)
# columns that are unique, that means exist only in one dataset
def compute_column_distribution(analysis_results, total_sites):
    """Count how many federation nodes contain each column and build a coverage label per node."""
    column_node_counts = {}

    for r in analysis_results:
        node_columns = set()
        node_columns.update(r.get("numeric_statistics", {}).keys())
        node_columns.update(r.get("categorical_statistics", {}).keys())
        node_columns.update(r.get("temporal_statistics", {}).keys())

        for col in node_columns:
            column_node_counts[col] = column_node_counts.get(col, 0) + 1

    column_distribution_summary = {
        "common_all": [],
        "common_partial": [],
        "unique": []
    }

    for col, count in column_node_counts.items():
        if count == total_sites:
            column_distribution_summary["common_all"].append(col)
        elif count > 1:
            column_distribution_summary["common_partial"].append(col)
        else:
            column_distribution_summary["unique"].append(col)

    return column_node_counts, column_distribution_summary

def classify_local_columns(local_columns, total_sites, column_node_counts):
    """
    Label each column in a node as common_all / common_partial / unique_local.

      Args:
          local_columns: columns present in this node.
          total_sites: total number of federation nodes.
          column_node_counts: {column: count_of_nodes_containing_it}.

      Returns:
          {column: label} where label is one of common_all / common_partial /
          unique_local.
    """

    column_labels = {}
    for col in local_columns:
        count = column_node_counts.get(col, 0)
        if count == total_sites:
            column_labels[col] = "common_all"
        elif count > 1:
            column_labels[col] = "common_partial"
        else:
            column_labels[col] = "unique_local"

    return column_labels



# ============================================================================
# Source: local_data_quality.py
# ============================================================================

def compute_missing_by_column(df: pd.DataFrame) -> Dict[str, int]:
    """Return {column: missing_count} for every column in df."""
    return df.isna().sum().astype(int).to_dict()


def compute_total_missing(df: pd.DataFrame) -> int:
    """Return total number of missing cells across all columns in df."""
    return int(df.isna().sum().sum())



# ============================================================================
# Source: local_compute_statistics.py
# ============================================================================

# detect numeric, categorical and temporal columns
def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Classify every column in df into numeric / categorical / binary / temporal.

      Binary columns are a sub-type of categorical: two-value columns that can be
      treated as indicator variables. Temporal columns are those already carrying
      a datetime64 dtype -- this function does not attempt to parse object/string
      columns as dates, and does not itself exclude identifier columns (callers
      are expected to filter id_columns out of the result separately).

      Returns:
          dict with keys ``numeric``, ``categorical``, ``binary``, ``temporal``,
          each mapping to a list of column names.
    """

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    temporal_columns = df.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns.tolist()
    candidate_columns = [
        col for col in df.columns
        if col not in temporal_columns
    ]
    binary_columns = [col for col in candidate_columns if is_binary(df[col], column_name=col)]
    categorical_columns = [
        col for col in df.columns
        if col not in numeric_columns and col not in temporal_columns
    ]
    numeric_columns = [col for col in numeric_columns if col not in binary_columns]
    categorical_columns = categorical_columns + [
        col for col in binary_columns if col not in categorical_columns
    ]

    # Columns with no non-missing values carry no information for any
    # statistic (compute_*_statistics already skip them) -- drop them here
    # too so they don't pollute downstream statistics with all-NaN values
    # (e.g. showing up as a nonsensical "numeric" variable).
    non_empty = {col for col in df.columns if df[col].notna().any()}
    numeric_columns = [c for c in numeric_columns if c in non_empty]
    categorical_columns = [c for c in categorical_columns if c in non_empty]
    temporal_columns = [c for c in temporal_columns if c in non_empty]
    binary_columns = [c for c in binary_columns if c in non_empty]

    return {
        "numeric": numeric_columns,
        "categorical": categorical_columns,
        "temporal": temporal_columns,
        "binary": binary_columns
    }


def detect_quasi_numeric_categorical_columns(
    df: pd.DataFrame, categorical_columns: List[str], threshold: float = 0.5,
) -> List[str]:
    """Identify categorical columns where most (but not all) values look numeric.

    A column like lab results recorded as ["12.5", "8.1", "<5", "300.2"] is
    routed to the categorical bucket by detect_column_types because the
    censored value "<5" keeps the whole column at object dtype -- with no
    indication anywhere that this happened. This function flags such columns
    so a report notice can explain the gap, without silently coercing the
    unparseable values to NaN (which would discard the below/above-detection-
    limit signal without a human deciding that's the right tradeoff for that
    specific column).
    """
    flagged = []
    for col in categorical_columns:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        fraction_numeric = pd.to_numeric(s, errors="coerce").notna().mean()
        if threshold <= fraction_numeric < 1.0:
            flagged.append(col)
    return flagged

# checks if the column is binary
# Medical/clinical terms that strongly imply a column is a binary flag
# (diagnosis present/absent, symptom present/absent, treatment given/not given).
# Used by is_binary to rescue legitimately binary columns that are very sparse
# (fewer than _BINARY_MIN_COUNT non-null values).
_MEDICAL_BINARY_KEYWORDS = {
    # Comorbidities / chronic conditions
    "hypertension", "diabetes", "cardiovasc", "cardiac", "cardio", "coronary", "chd",
    "asthma", "copd", "crd", "ckd", "cld", "renal", "hepatic", "pulmonary",
    "neuro", "psych", "psychiatric", "neurological",
    "tumor", "cancer", "oncol", "malignant", "lesion",
    "immuno", "transplant", "comorb",
    "metabol", "obesity", "obese",
    "hiv", "tb", "tuberculosis", "infectious",
    # Acute symptoms
    "fever", "cough", "dyspnea", "fatigue", "malaise", "anosmia",
    "pain", "confusion", "diarrhea", "nausea", "rash", "fainting",
    "headache", "myalgia", "arthralgia", "ageusia", "anorexia",
    "tinnitus", "dizziness", "seizures", "tremors", "constipation",
    "dismobility", "depressive", "parasthesia",
    "memory", "acute",
    # Hospital events / outcomes
    "hospitalization", "icu", "admission", "discharge",
    "mortality", "death",
    # Treatments / interventions
    "treated", "treatment", "therapy", "procedure", "surgery",
    "oxygen", "ventil",
    "viral", "antiviral", "anti", "antibiotic",
    "glucocorticoid", "steroid", "corticoid",
    "monoclonal", "antibody", "antagonist", "inhibitor",
    "lopinavir", "remdesivir", "molnupiravir", "paxlovid", "ribavirin",
    "il1", "il6",
    "vaccination", "vaccin", "vac",
    # Generic binary-indicator suffixes / prefixes
    "flag", "indicator",
}

# Minimum non-null observations required to classify a numeric {0, 1} column as
# binary on value evidence alone. Below this threshold the column name must
# contain a medical keyword (see _MEDICAL_BINARY_KEYWORDS) to be classified as
# binary -- this prevents a very sparse continuous column that happens to show
# only {0, 1} from being misclassified.
_BINARY_MIN_COUNT = 3


def _name_suggests_binary(column_name: str) -> bool:
    """Return True if the column name contains a medical or binary-indicator keyword."""
    tokens = set(re.split(r"[^a-z0-9]+", column_name.lower())) - {""}
    return bool(tokens & _MEDICAL_BINARY_KEYWORDS)


def is_binary(series: pd.Series, allow_bool: bool = True, column_name: str = "") -> bool:
    """
    Return True if series has at most two distinct non-null values.

      Recognises numeric {0, 1} / {0.0, 1.0}, bool dtype, and string pairs
      like yes/no, true/false, y/n, ja/nein. For sparse numeric {0, 1} columns
      (fewer than _BINARY_MIN_COUNT non-null rows) the column name must
      contain a medical keyword from _MEDICAL_BINARY_KEYWORDS, guarding
      against continuous columns that happen to show only {0, 1} by
      coincidence.
    """

    s = series.dropna()
    if s.empty:
        return False
    if s.nunique() > 2:
        return False
    # numeric binary: 0/1 or 0.0/1.1
    numeric = pd.to_numeric(series, errors="coerce")
    # only look at VALID values
    valid_values = numeric.dropna()
    if len(valid_values) > 0:
        unique_vals = set(valid_values.unique())
        if unique_vals.issubset({0, 1}):
            # Require sufficient observations OR a medical keyword in the name.
            if len(valid_values) >= _BINARY_MIN_COUNT or _name_suggests_binary(column_name):
                return True
    # boolean: True or False
    if allow_bool and pd.api.types.is_bool_dtype(series):
        return True
    # semantic binary: yes/no, y/n
    # convert all letters to lower case
    # normalization so that YES and yes aren't interpreted as different categories
    str_series = series.dropna().astype(str).str.strip().str.lower()
    unique_vals = set(str_series.dropna().unique())
    valid_sets = [
        {"yes", "no"},
        {"y", "n"},
        {"true", "false"},
        {"t", "f"},
        { "ja", "nein"}
    ]

    for valid_set in valid_sets:
        if unique_vals.issubset(valid_set):
            return True

    return False

def detect_id_column(series: pd.Series, column_name: str) -> bool:
    """Return True if the column is likely a patient or record identifier.

    Uses token-split keyword matching: strong keywords (id, name, email …)
    trigger an unconditional True; weak keywords (patient, subject …)
    require uniqueness > 0.9. Datetime columns are never identifiers.

    The bare uniqueness heuristic (ratio ≥ 0.95 for any column) has been
    removed because it produced false positives on date and high-cardinality
    clinical columns.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return False

    name = column_name.lower()
    tokens = set(re.split(r'[^a-z0-9]+', name)) - {""}

    strong_keywords = {
        "id", "identifier", "identifikator", "pid",
        "name", "surname", "firstname", "lastname", "rownames",
        "first_name", "last_name", "family_name",
        "vorname", "nachname", "familienname",
        "telefon", "phone", "email", "adresse", "address",
        "postcode", "zip", "ssn", "dob", "birthdate", "geburtsdatum",
    }
    weak_keywords = {"patient", "person", "subject", "record", "case"}

    if bool(tokens & strong_keywords):
        return True

    valid_values = series.dropna()
    n_valid = len(valid_values)
    if n_valid == 0:
        return False

    # Value-based check: string values matching "patient_001", "subject_02", etc.
    # Sample up to 50 rows to keep this fast.
    if pd.api.types.is_object_dtype(valid_values):
        _id_value_re = re.compile(
            r'^(patient|pat|subject|sub|person|record|case|participant)[_\-]?\d+$',
            re.IGNORECASE,
        )
        sample = valid_values.iloc[:50]
        if sample.apply(lambda v: bool(_id_value_re.match(str(v)))).all():
            return True

    uniqueness_ratio = valid_values.nunique() / n_valid

    if bool(tokens & weak_keywords) and uniqueness_ratio > 0.9:
        return True
    return False

def compute_numeric_statistics(numeric_df: pd.DataFrame) -> dict:
    """
    Compute descriptive statistics for each numeric column.

      Returns:
          {column: {mean, median, mode, min, max, variance, std_dev, q25, q75,
          iqr, count, frequency, relative_frequency, outliers, skewness,
          kurtosis, missing_values}}.
    """
    statistics = {}
    for col in numeric_df.columns:

        # number of missing values in the column
        missing_values = int(numeric_df[col].isna().sum())

        # s = pandas Series
        s = pd.to_numeric(numeric_df[col], errors="coerce").dropna()
        if len(s) == 0:
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = int(((s < lower_bound) | (s > upper_bound)).sum())
        vc = s.value_counts()
        statistics[col] = {
            "mean": round(float(s.mean()), 3),
            "median": round(float(s.median()), 3),
            "mode": float(s.mode().iloc[0]) if not s.mode().empty else None,
            "min": float(s.min()),
            "max": float(s.max()),
            "variance": round(float(s.var()), 3),
            "std_dev": round(float(s.std()), 3),
            "q25": round(float(q1), 3),
            "q75": round(float(q3), 3),
            "iqr": round(float(iqr), 3),
            "count": int(s.count()),
            'frequency': int(vc.iloc[0]),
            'relative_frequency': round(vc.iloc[0] / len(s) * 100, 2),
            'outliers': outlier_count,
            'skewness': round(s.skew(), 3),
            'kurtosis': round(s.kurtosis(), 3),
            'missing_values': missing_values
        }
    return statistics


def compute_categorical_statistics(categorical_df: pd.DataFrame) -> Dict:
    """
    Compute frequency statistics for each categorical column.

      Returns:
          {column: {count, number_of_categories, most_frequent_category,
          least_frequent_category, category_counts, relative_frequencies,
          class_imbalance_ratio, missing_values}}.
    """
    statistics = {}

    for col in categorical_df.columns:
        # number of missing values
        missing_values = int(categorical_df[col].isna().sum())

        # drop missing values *before* converting to string -- otherwise
        # NaN becomes the literal string "nan" and is counted as its own
        # category below.
        s = categorical_df[col].dropna().astype(str)
        if s.empty:
            continue
        # count occurrences of each category
        # vc = the number of observations belonging to each category(frequency of each category)
        # for example: if we have 3 possible categories: A, B and C for a column:
        # we want to know how many times do we have A in the rows?
        # how many rows habe B? how many rows have C?
        vc = s.value_counts()
        if vc.empty:
            continue
        # relative frequency (%) of each category
        rel_freq = (vc / len(s) * 100).round(2)
        # most frequent category
        most_frequent = vc.index[0]
        # least frequent category
        least_frequent = vc.index[-1]
        # simple class imbalance indicator
        # ratio between most frequent and least frequent
        # (vc.iloc[-1] is the smallest non-zero count, so this is always > 0)
        imbalance_ratio = float(vc.iloc[0] / vc.iloc[-1])

        statistics[col] = {
            # total number of valid observations
            "count": int(s.count()),
            # number of distinct categories
            "number_of_categories": int(s.nunique()),
            "most_frequent_category": most_frequent,
            "least_frequent_category": least_frequent,
            # counts of each category
            "category_counts": vc.to_dict(),
            # relative frequency (%) of each category
            "relative_frequencies": {k: round(v, 3) for k, v in rel_freq.items()},
            # imbalance ratio (large value = strong imbalance)
            "class_imbalance_ratio": round(imbalance_ratio, 4),
            "missing_values": missing_values
        }
    return statistics

def compute_temporal_statistics(temporal_df: pd.DataFrame, patient_series: Optional[pd.Series] = None,
    freq: str = "M") -> Dict:
    """
    freq:
        "D" = daily
        "W" = weekly
        "M" = monthly
        "Y" = yearly
    """
    statistics = {}
    # validate alignment if patient_series is provided
    if patient_series is not None:
        if not patient_series.index.equals(temporal_df.index):
            raise ValueError("patient_series must have the same index as temporal_df")

    for col in temporal_df.columns:

        # count missing values
        missing_values = int(temporal_df[col].isna().sum())

        # convert to datetime
        s = pd.to_datetime(temporal_df[col], errors="coerce").dropna()

        if len(s) == 0:
            continue

        start_date = s.min()
        end_date = s.max()
        # observations per time unit
        obs_per_period = s.dt.to_period(freq).value_counts().sort_index()
        # most active period
        most_active_period = obs_per_period.idxmax()
        # detect missing periods
        full_range = pd.period_range(
            start=start_date.to_period(freq),
            end=end_date.to_period(freq),
            freq=freq
        )
        missing_periods = list(set(full_range) - set(obs_per_period.index))

        # observations per patient over time

        # initialize the variable
        # if the dataset does not contain a patient ID column, we leave it None
        # ->  this avoids errors later and clearly signals no patient-level analysis available
        obs_per_patient = None
        if patient_series is not None:
            # create a temporary dataframe with only the columns needed
            tmp = pd.DataFrame({
                "patient": patient_series,
                "time": pd.to_datetime(temporal_df[col], errors="coerce") # convert the column to datetime
                # if invalid values exist,  they become NaT = Not a Time
            }).dropna() # remove rows with missing values (rows where patient is missing or date is missing)

            if not tmp.empty:
                obs_per_patient = (
                # group all rows belonging to the same patient
                    tmp.groupby("patient")["time"]
                    .count()
                    .to_dict()
                )

        statistics[col] = {
            # number of valid timestamps
            "count": int(s.count()),
            # start and end of the dataset timeline
            "time_range": {
                "start": str(start_date),
                "end": str(end_date)
            },
            # number of days between start and end
            "range_days": int((end_date - start_date).days),
            # observations per time unit (month by default)
            "observations_per_period": obs_per_period.to_dict(),
            "most_active_period": str(most_active_period),
            "missing_periods": [str(p) for p in missing_periods],
            # "observations_per_patient": obs_per_patient,
            "missing_values": missing_values
        }
    return statistics


def compute_age_histogram(
    df: pd.DataFrame,
    age_col: str = "age",
    bin_size: int = 5,
    max_age: int = 100,
) -> Tuple[Optional[list], Optional[list]]:
    """
    Compute a fixed-width age histogram from a DataFrame.

    Returns (counts_list, edges_list) suitable for federated aggregation
    (counts are summed across nodes; edges are identical so they stay aligned).
    Returns (None, None) when the age column is absent or entirely non-numeric.
    """
    if age_col not in df.columns:
        return None, None
    ages = pd.to_numeric(df[age_col], errors="coerce").dropna()
    if ages.empty:
        return None, None
    bins = np.arange(0, max_age + bin_size, bin_size)
    hist, edges = np.histogram(ages, bins=bins)
    return hist.astype(int).tolist(), edges.astype(float).tolist()


def count_out_of_range_ages(
    df: pd.DataFrame, age_col: str = "age", max_age: int = 100,
) -> int:
    """
    Count parseable age values that fall outside the plausible [0, max_age] range.

    compute_age_histogram's fixed-width bins silently drop values below 0 or
    above max_age (e.g. a negative age or a data-entry typo like 999) --
    they simply don't appear in any bin. This count lets the report surface
    those values instead of letting them vanish from the age distribution
    with no explanation.
    """
    if age_col not in df.columns:
        return 0
    ages = pd.to_numeric(df[age_col], errors="coerce").dropna()
    if ages.empty:
        return 0
    return int(((ages < 0) | (ages > max_age)).sum())



# ============================================================================
# Source: local_inferential_analysis.py
# ============================================================================

logger = logging.getLogger(__name__)

# compare_two_groups' variance/effect-size math (ddof=1) is undefined for a
# group of size 1 (division by zero), so screen_associations skips any
# num-cat pair where a group falls below this size rather than surfacing a
# NaN effect size.
_MIN_GROUP_SIZE_FOR_COMPARISON = 2

# helper functions
def _group_values(df, value_col, group_col):
    """Split df[value_col] into a list of per-group arrays and matching group labels."""
    groups = []
    labels = []
    for name, g in df.groupby(group_col):
        vals = g[value_col].dropna().values
        if len(vals) > 0:
            groups.append(vals)
            labels.append(name)
    return groups, labels

def _distribution_diagnostics(groups):
    """
    Compute normality and outlier diagnostics for each group array.

      Uses Shapiro-Wilk for n ≤ 5000, then falls back to skew/kurtosis bounds
      (|skew| < 1, |excess kurtosis| < 2) for larger samples.
      Returns {overall_normal, groups: [{n, shapiro_p, skewness, kurtosis,
      approx_normal, has_outliers}]}.
    """

    diagnostics = []

    for g in groups:

        g = np.asarray(g)
        n = len(g)

        if n < 3:
            diagnostics.append({
                "n": n,
                "shapiro_p": np.nan,
                "skewness": np.nan,
                "kurtosis": np.nan,
                "approx_normal": False,
                "has_outliers": False
            })
            continue

        # Avoid Shapiro on very large samples
        shapiro_p = stats.shapiro(g)[1] if n <= 5000 else np.nan

        skewness = stats.skew(g, bias=False)
        kurt = stats.kurtosis(g, fisher=True, bias=False)

        # IQR outlier detection
        q1, q3 = np.percentile(g, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = np.sum((g < lower) | (g > upper))

        has_outliers = outliers > 0

        if n <= 50 and not np.isnan(shapiro_p):
            approx_normal = shapiro_p > 0.05
        else:
            # Fallback for n > 50 (Shapiro-Wilk over-rejects at large n): rule-of-thumb
            # bounds on excess kurtosis (Fisher, 0 = normal) and skewness, common in
            # applied normality screening (e.g. George & Mallery's |skew|, |kurt| < 2).
            approx_normal = (
                    abs(skewness) < 1 and
                    abs(kurt) < 2
            )

        diagnostics.append({
            "n": n,
            "shapiro_p": shapiro_p,
            "skewness": skewness,
            "kurtosis": kurt,
            "approx_normal": approx_normal,
            "has_outliers": has_outliers
        })

    overall_normal = all(d["approx_normal"] for d in diagnostics)

    return {
        "overall_normal": overall_normal,
        "groups": diagnostics
    }

def _cohens_d(g1, g2):
    """Compute Cohen's d (pooled-SD standardised mean difference) for two independent groups."""

    n1, n2 = len(g1), len(g2)

    s1 = np.var(g1, ddof=1)
    s2 = np.var(g2, ddof=1)

    pooled_sd = np.sqrt(
        ((n1 - 1) * s1 + (n2 - 1) * s2) /
        (n1 + n2 - 2)
    )

    if pooled_sd == 0:
        return 0.0 if np.mean(g1) == np.mean(g2) else np.inf

    return (np.mean(g1) - np.mean(g2)) / pooled_sd


def _hedges_g(g1, g2):
    """Compute Hedges' g (bias-corrected Cohen's d) for two independent groups."""

    d = _cohens_d(g1, g2)

    n = len(g1) + len(g2)

    denom = 4 * n - 9
    correction = 1 - (3 / denom) if denom > 0 else 1.0

    return d * correction

def _rank_biserial(u, n1, n2):
    """Compute rank-biserial correlation from a Mann-Whitney U statistic and group sizes."""
    # u: Mann-Whitney U statistic for g1 relative to g2 (as returned by
    # stats.mannwhitneyu(g1, g2, ...)) -- passed in so the test isn't run twice
    return 1 - (2 * u) / (n1 * n2)

def _should_use_nonparametric(diagnostics):
    """Return True if severe non-normality or outliers indicate a nonparametric test is preferable."""

    min_n = min(d["n"] for d in diagnostics["groups"])

    severe_skew = any(
        abs(d["skewness"]) > 2
        for d in diagnostics["groups"]
        if not np.isnan(d["skewness"])
    )

    severe_kurtosis = any(
        abs(d["kurtosis"]) > 4
        for d in diagnostics["groups"]
        if not np.isnan(d["kurtosis"])
    )

    outliers = any(
        d["has_outliers"]
        for d in diagnostics["groups"]
    )

    overall_normal = diagnostics["overall_normal"]

    return (
        (severe_skew or severe_kurtosis or outliers) and
        not (overall_normal and min_n >= 30)
    )

def _check_variance(groups):
    """Run Brown-Forsythe (median Levene) test and return {equal_variance, p_value}."""
    #  Brown-Forsythe (median-centered Levene)
    stat, p = stats.levene(*groups, center="median")
    return {
        "equal_variance": p > 0.05,
        "p_value": p
    }

def _diagnose_groups(groups):
    """Run distribution diagnostics and variance check in one call; used by compare_two_groups and one_way_group_comparison."""
    # Shared by compare_two_groups and one_way_group_comparison so the
    # distribution/variance checks that drive "auto" method selection are
    # computed in exactly one place.
    diagnostics = _distribution_diagnostics(groups)
    variance = _check_variance(groups)
    return diagnostics, variance
#----------------------------------------------------------------
def compare_two_groups(df, value_col, group_col, method="auto"):
    """
    Compare two independent groups on a numeric variable.

      Auto-selects Student t-test (equal variance), Welch t-test (unequal
      variance), or Mann-Whitney U (non-parametric) based on distribution
      diagnostics.  Returns method, statistic, p_value, effect_size, and
      assumption metadata.
    """

    groups, labels = _group_values(df, value_col, group_col)
    if len(groups) != 2:
        raise ValueError("Exactly 2 groups required")

    g1, g2 = groups

    diagnostics, variance = _diagnose_groups(groups)

    if method == "auto":
        if _should_use_nonparametric(diagnostics):
            method = "mannwhitney"
        elif variance["equal_variance"]:
            method = "student_ttest"
        else:
            method = "welch_ttest"

    if method == "student_ttest":
        stat, p = stats.ttest_ind(g1, g2, equal_var=True)
        effect_size = {
            "cohens_d": _cohens_d(g1, g2),
            "hedges_g": _hedges_g(g1, g2)
        }

    elif method == "welch_ttest":
        stat, p = stats.ttest_ind(g1, g2, equal_var=False)
        effect_size = {
            "cohens_d": _cohens_d(g1, g2),
            "hedges_g": _hedges_g(g1, g2)
        }

    elif method == "mannwhitney":
        stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        effect_size = {
            "rank_biserial": _rank_biserial(stat, len(g1), len(g2))
        }
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "method": method,
        "statistic": stat,
        "p_value": p,
        "effect_size": effect_size,
        "assumptions": {
            "distribution_diagnostics": diagnostics,
            "equal_variance": variance["equal_variance"],
            "variance_test_pvalue": variance["p_value"]  # Brown-Forsythe (median-centered Levene) test p-value for equal variances
        }
    }

#--------------------------------------------------------
# ANOVA/Welch:compare means
# Kruskal: compare distributions/ranks, often interpreted as medians
# one-way comparison of 2 or more independent groups
def one_way_group_comparison(df, value_col, group_col):
    """
    Compare ≥2 independent groups on a numeric variable.

      Auto-selects Welch ANOVA (parametric) or Kruskal-Wallis (non-parametric)
      based on distribution diagnostics.  Returns method, statistic, p_value, and
      assumption metadata.
    """

    groups, labels = _group_values(df, value_col, group_col)
    if len(groups) < 2:
        raise ValueError("Need at least 2 groups")

    diagnostics, variance = _diagnose_groups(groups)
    if _should_use_nonparametric(diagnostics):
        method = "kruskal"
    else:
        method = "welch"

    result = {
        "method": method,
        "statistic": None,
        "p_value": None,
        "assumptions": {
            "distribution_diagnostics": diagnostics,
            "equal_variance": variance["equal_variance"],
            "variance_test_pvalue": variance["p_value"]  # Brown-Forsythe (median-centered Levene) test p-value for equal variances
        }
    }

    if method == "welch":
        welch = pg.welch_anova(data=df, dv=value_col, between=group_col)
        stat = welch["F"].iloc[0]
        # pingouin names this column "p-unc" in some versions and "p_unc" in
        # others (e.g. 0.6.x) -- support both so the call doesn't KeyError.
        p_col = "p-unc" if "p-unc" in welch.columns else "p_unc"
        p = welch[p_col].iloc[0]

    elif method == "kruskal":
        stat, p = stats.kruskal(*groups)

    else:
        raise ValueError(f"Unknown method: {method}")

    result["statistic"] = stat
    result["p_value"] = p
    return result

#--------------------------
# Post-hoc tests
def posthoc_test(df, value_col, group_col, method):
    """
    Run the appropriate post-hoc test for the given primary comparison method.

      welch → Games-Howell, kruskal → pairwise MWU + Holm-Bonferroni.
      Returns a DataFrame of pairwise comparisons.
    """
    # NOTE: `method` names the *primary* group-comparison test that was run
    # (matches the "method" returned by one_way_group_comparison), not the
    # post-hoc test itself. Each primary test maps to its standard companion:
    #   "welch"   -> Games-Howell       (pg.pairwise_gameshowell)
    #   "kruskal" -> pairwise MWU + Holm-Bonferroni (scipy + statsmodels only)
    if method == "welch":
        return pg.pairwise_gameshowell(
            data=df,
            dv=value_col,
            between=group_col
        )
    elif method == "kruskal":
        from statsmodels.stats.multitest import multipletests
        groups = sorted(df[group_col].dropna().unique())
        pairs = [(a, b) for i, a in enumerate(groups) for b in groups[i + 1:]]
        raw_p = []
        for a, b in pairs:
            g1 = df.loc[df[group_col] == a, value_col].dropna().values
            g2 = df.loc[df[group_col] == b, value_col].dropna().values
            _, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            raw_p.append(p)
        _, adj_p, _, _ = multipletests(raw_p, method="holm")
        mat = pd.DataFrame(np.nan, index=groups, columns=groups)
        for (a, b), p in zip(pairs, adj_p):
            mat.loc[a, b] = p
            mat.loc[b, a] = p
        return mat
    else:
        raise ValueError(f"Unsupported method: {method}")

# ------------
# Correlation between numeric variables (Pearson / Spearman):
# Correlation analysis is a statistical technique that provides information about the relationship between 2 variables.
# The strength of the correlation is determined by the correlation coefficient, which ranges from -1 to +1.
# Correlation analyses can therefore be used to determine the strength and direction of the correlation.

# For Pearson's correlation to be used, the variables must be normally distributed, and there must be a linear relationship between them.
# If these conditions are not met, Spearman's correlation is used.

def correlation_between_two_variables(df, var1, var2, method = "auto", return_summary=False):
    """
    Compute Pearson or Spearman correlation, auto-selected by normality and outliers.

      Uses Pearson when n ≥ 30, |skew| < 2 for both variables, and no IQR
      outliers; otherwise falls back to Spearman.  Returns method, correlation
      coefficient, and p-value.
    """
    data = df[[var1, var2]].dropna()
    if len(data) < 3:
        raise ValueError("Need at least 3 observations")
    x = data[var1]
    y = data[var2]

    skew_x = stats.skew(x)
    skew_y = stats.skew(y)
    # IQR-based outlier detection (does not assume normality, unlike z-scores --
    # which would be circular here since the result feeds the Pearson/Spearman decision)
    q1_x, q3_x = np.percentile(x, [25, 75])
    iqr_x = q3_x - q1_x
    outlier_x = np.any((x < q1_x - 1.5 * iqr_x) | (x > q3_x + 1.5 * iqr_x))
    q1_y, q3_y = np.percentile(y, [25, 75])
    iqr_y = q3_y - q1_y
    outlier_y = np.any((y < q1_y - 1.5 * iqr_y) | (y > q3_y + 1.5 * iqr_y))

    if method == "auto":
        if (
                len(data) >= 30 and
                abs(skew_x) < 2 and
                abs(skew_y) < 2 and
                not outlier_x and
                not outlier_y
        ):
            method = "pearson"
        else:
            method = "spearman"

    if method == "pearson":
        r, p = stats.pearsonr(x, y)

    elif method == "spearman":
        r, p = stats.spearmanr(x, y)

    else:
        raise ValueError(f"Unknown method: {method}")

    result = {
        "method": method,
        "correlation": r,
        "p_value": p,
        "diagnostics": {
            "skewness": {
                var1: skew_x,
                var2: skew_y
            },
            "outliers": {
                var1: outlier_x,
                var2: outlier_y
            }
        }
    }

    if return_summary:
        result["summary_table"] = pg.corr(x, y, method=method)

    return result

# -------

def categorical_association(df, col1, col2, alpha=0.05):
    """
    Test association between two categorical columns.

      Uses chi-squared when all expected cell counts ≥ 5; falls back to
      Fisher's exact test for 2×2 tables with small expected counts.
      Returns method, statistic, p_value, cramers_v, and assumption flags.
    """

    # alpha: the risk that you are willing to take in drawing the wrong conclusion.
    # for example: α = 0.05 -> means you are undertaking a 5 %risk of concluding that two variables are independent when in reality they are not.

    # hypothesis testing:
    # H0: variables are independent
    # HA (alternative): variables are associated

    # contingency table: displays the frequnecy distribution of the two categorical columns
    # contingency table is
    contingency_table = pd.crosstab(df[col1], df[col2])
    rows, cols = contingency_table.shape

    # chi-square test

    # p = p-value
    # p-value is calculated from the chi-square score.
    # p-value  will tell you if your test results are significant or not.
    # p < 0.05 – this means the two categorical variables are correlated.
    # p > 0.05 – this means the two categorical variables are not correlated.

    # to calculate the p-value, you need:
    # 1. degrees of freedom:  number of categories - 1
    # 2. Chi - square score

    # dof = degree of freedom
    # Degrees of freedom are the number of independent variables that can be estimated in a statistical analysis,
    # and tell you how many items can be randomly selected before constraints must be put in place.

    # expected =  expected counts if independent
    # these are theoretical counts we would expect if no relationship existed.

    chi2, p, dof, expected = chi2_contingency(contingency_table)

    # assumption checks
    # all expected counts should be >= 1
    # no more than 20% of expected counts should be below 5
    # if assumptions are violated, Fisher's Exact Test may be better for small 2x2 tables.

    # count how many expected values are below 5
    expected_lt5 = (expected < 5).sum()
    # total number of cells
    total_cells = expected.size
    assumptions_ok = ( (expected >= 1).all() and (expected_lt5 / total_cells) <= 0.20)

    # fisher test for 2x2 tables
    # fisher vs. chi square:
    # fisher computes the EXACT probability,
    # while chi square uses an approximation!
    fisher_p = None
    if rows == 2 and cols == 2:
        _, fisher_p = fisher_exact(contingency_table)
    # total sample size
    n = contingency_table.to_numpy().sum()

    # p-values only tell us if a relationship exists statistically
    # they do not tell us how string the relation is
    # soe we use Cramer's V because it measures effect size, which is the strength of association
    #  0.1 = weak, 0.3 = moderate, 0.5 = strong

    min_dim = min(rows, cols) - 1
    # effect size: Cramer's V
    # effect size = strength of the relationship between variables
    # Cramer's V should be computed from the uncorrected chi-square statistic --
    # chi2_contingency applies Yates' continuity correction by default for 2x2
    # tables, which would bias V. A 1xk / kx1 table (min_dim == 0) has no
    # defined V.
    if min_dim > 0:
        chi2_uncorrected, _, _, _ = chi2_contingency(contingency_table, correction=False)
        cramers_v = np.sqrt(chi2_uncorrected / (n * min_dim))
    else:
        cramers_v = np.nan

    # decide which p-value to use
    if rows == 2 and cols == 2 and not assumptions_ok:
        final_p = fisher_p
        test_used = "Fisher Exact Test"
    else:
        final_p = p
        test_used = "Chi-Square Test"

    # Statistical decision
    reject_h0 = final_p < alpha
    # interpretation
    if reject_h0:
        interpretation = (f"Evidence of association between {col1} and {col2}")
    else:
        interpretation = (f"No statistically significant association detected between {col1} and {col2}")

    return {
        "var1": col1,
        "var2": col2,
        "test_used": test_used,
        "chi2_statistic": chi2,
        "p_value": final_p,
        "degrees_of_freedom": dof,
        "cramers_v": cramers_v,
        "fisher_p_value": fisher_p,
        "assumptions_ok": assumptions_ok,
        "reject_h0": reject_h0,
        # "expected_counts_below_5": int(expected_lt5),
        # "total_cells": int(total_cells),
        # expected frequencies
        "expected_frequencies": pd.DataFrame(expected, index=contingency_table.index,
                                             columns=contingency_table.columns),
        # observed frequencies
        "contingency_table": contingency_table,
        "interpretation": interpretation
    }




# longitudinal: many observations per patient (several rows)
# same subjects followed over time
# example: same patient across multiple visits

# event-based
# discrete events like surgeries ...


# panel data
# multiple entities over time


# ---------------------------------------------------------------------------
# Outcome-column detection
#
# The report runs unattended on hub nodes -- nobody can point it at "the"
# outcome variable. Hospital datasets vary in column naming, so candidates are
# matched by keyword in priority order (mortality/survival ranks above generic
# status/diagnosis labels) and only accepted if they are actually usable as a
# group/target variable: not constant, low cardinality, and every level has
# enough observations to support a stable comparison. Mirrors the keyword +
# validation pattern already used for sex-column detection in analyze.py.
# ---------------------------------------------------------------------------
def detect_outcome_column(df, column_types, keyword_groups, min_class_size=20, max_levels=5):
    """
    Find the first categorical column matching clinical outcome keywords.

      Iterates over keyword groups in priority order; within each group checks
      column names for a keyword match.  A candidate is accepted only when it has
      2–max_levels levels and every level has ≥ min_class_size observations.
      Returns the column name or None.
    """
    candidates = column_types.get("categorical", [])

    for group in keyword_groups:
        for col in candidates:
            name = str(col).lower()
            if not any(keyword in name for keyword in group):
                continue

            counts = df[col].value_counts(dropna=True)
            # not constant, not too many levels for a group comparison / logistic target
            if len(counts) < 2 or len(counts) > max_levels:
                continue
            # every class needs enough observations for a stable comparison
            if counts.min() < min_class_size:
                continue

            return col

    return None


# ---------------------------------------------------------------------------
# Association screening
#
# Tests every numeric-numeric pair and every (cardinality-limited)
# categorical-categorical pair without a human picking them, rather than
# restricting to a data-driven subset -- with ~200 columns the full pairwise
# set is tens of thousands of tests, but each test is cheap, and
# Benjamini-Hochberg FDR correction (below) is specifically designed to stay
# valid regardless of how many tests are run, so there is no statistical need
# to pre-filter which pairs get tested. Every p-value is corrected for
# running many tests at once (BH-FDR -- standard for exploratory screening,
# less conservative than Bonferroni), and a pair is only flagged
# "significant" when BOTH the corrected p-value clears alpha AND the effect
# size clears a conventional small-effect threshold (Cohen's conventions):
# with large n, trivial differences become "significant" by p-value alone.
# ---------------------------------------------------------------------------
def screen_associations(df, column_types, alpha=0.05,
                        effect_size_thresholds=None, max_group_levels=6):
    """
    Run FDR-corrected association screening over all typed variable pairs.

      Tests numeric–numeric pairs (correlation), categorical–categorical pairs
      (Cramér's V / chi-squared), and numeric–categorical pairs (group
      comparison). Categorical-categorical testing is restricted to a
      cardinality shortlist to avoid testing tens of thousands of
      statistically meaningless high-cardinality combinations.  Applies
      Benjamini-Hochberg FDR correction and requires both a corrected p-value
      and a minimum effect size for a pair to be flagged significant.

      Returns a DataFrame with one row per tested pair.
    """
    if effect_size_thresholds is None:
        # "small effect" cutoffs by Cohen's conventions
        effect_size_thresholds = {
            "correlation": 0.2,
            "cramers_v": 0.1,
            "hedges_g": 0.2,
            "rank_biserial": 0.2,
        }

    numeric_cols = column_types.get("numeric", [])
    categorical_cols = column_types.get("categorical", [])
    rows = []

    # numeric ~ numeric: every pair of numeric columns. BH-FDR correction
    # below is what keeps this valid at scale, not a pre-filter on which
    # pairs look promising.
    for var1, var2 in itertools.combinations(numeric_cols, 2):
        try:
            corr = correlation_between_two_variables(df, var1, var2)
        except Exception:
            logger.warning("Correlation screening failed for (%s, %s)", var1, var2, exc_info=True)
            continue
        rows.append({
            "var1": var1, "var2": var2, "pair_type": "num-num",
            "test": corr["method"], "statistic": corr["correlation"],
            "p_value": corr["p_value"],
            "effect_size": abs(corr["correlation"]), "effect_size_metric": "correlation",
        })

    # categorical ~ categorical: every pair among low-cardinality categorical
    # columns. A chi-square test between two high-cardinality columns is both
    # expensive and statistically meaningless (most expected counts fall below
    # 5), so the cardinality shortlist stays -- but every pair within it is
    # tested, with BH-FDR correcting for the resulting number of tests.
    shortlist = [c for c in categorical_cols if df[c].nunique(dropna=True) <= max_group_levels]
    for col1, col2 in itertools.combinations(shortlist, 2):
        try:
            assoc = categorical_association(df, col1, col2)
        except Exception:
            logger.warning("Categorical association screening failed for (%s, %s)", col1, col2, exc_info=True)
            continue
        rows.append({
            "var1": col1, "var2": col2, "pair_type": "cat-cat",
            "test": assoc["test_used"], "statistic": assoc["chi2_statistic"],
            "p_value": assoc["p_value"],
            "effect_size": assoc["cramers_v"], "effect_size_metric": "cramers_v",
        })

    # numeric ~ categorical: every numeric column against binary categorical
    # columns (group_col has exactly 2 levels). Restricted to the binary case
    # because compare_two_groups is the only group-comparison function that
    # already returns a standardized effect size (Cohen's d / Hedges' g /
    # rank-biserial) regardless of which test it auto-selects -- reusing it
    # keeps the screen built entirely on already-tested outputs instead of
    # inventing new effect-size formulas for the 3+-group case.
    for value_col in numeric_cols:
        for group_col in categorical_cols:
            if df[group_col].nunique(dropna=True) != 2:
                continue
            group_sizes = df.groupby(group_col)[value_col].apply(
                lambda s: s.dropna().shape[0]
            )
            if (group_sizes < _MIN_GROUP_SIZE_FOR_COMPARISON).any():
                logger.warning(
                    "Skipping group comparison for (%s, %s): a group has "
                    "fewer than %d observations",
                    value_col, group_col, _MIN_GROUP_SIZE_FOR_COMPARISON,
                )
                continue
            try:
                cmp = compare_two_groups(df, value_col, group_col)
            except Exception:
                logger.warning("Group comparison screening failed for (%s, %s)", value_col, group_col, exc_info=True)
                continue
            if "hedges_g" in cmp["effect_size"]:
                metric, effect = "hedges_g", cmp["effect_size"]["hedges_g"]
            else:
                metric, effect = "rank_biserial", cmp["effect_size"]["rank_biserial"]
            rows.append({
                "var1": value_col, "var2": group_col, "pair_type": "num-cat",
                "test": cmp["method"], "statistic": cmp["statistic"],
                "p_value": cmp["p_value"],
                "effect_size": abs(effect), "effect_size_metric": metric,
            })

    columns = ["var1", "var2", "pair_type", "test", "statistic", "p_value",
               "effect_size", "effect_size_metric", "p_adj", "significant"]
    if not rows:
        return pd.DataFrame(columns=columns)

    screening = pd.DataFrame(rows)

    # Benjamini-Hochberg FDR correction across every test run in this screen --
    # without it, running hundreds of tests at alpha=0.05 would flag ~5% of
    # them as "significant" by chance alone.
    #
    # A handful of tests can come back with a NaN p-value (e.g. Welch's ANOVA
    # is undefined for degenerate group variances) -- multipletests propagates
    # a single NaN to *every* adjusted value, silently breaking the correction
    # for the whole screen. Correct only over the valid p-values and leave the
    # NaN rows as NaN/not-significant.
    from statsmodels.stats.multitest import multipletests
    screening["p_adj"] = np.nan
    valid = screening["p_value"].notna()
    if valid.any():
        _, p_adj, _, _ = multipletests(screening.loc[valid, "p_value"], alpha=alpha, method="fdr_bh")
        screening.loc[valid, "p_adj"] = p_adj

    thresholds = screening["effect_size_metric"].map(effect_size_thresholds)
    screening["significant"] = (
        (screening["p_adj"] < alpha)
        & screening["effect_size"].notna()
        & (screening["effect_size"] >= thresholds)
    )

    return screening[columns]



# ============================================================================
# Source: figures_local_descriptive_plots.py (categorical distributions helper;
# unused elsewhere in this file, kept as-is)
# ============================================================================


def save_categorical_distributions_local(
    df: pd.DataFrame,
    categorical_cols: list,
    output_dir,
    *,
    batch_size: int = 6,
    top_n: int = 20,
    node_label: str = None,
):
    """Save batched bar charts for all categorical columns with at least 2 distinct values.

    Single-value columns are excluded (no distribution to show). Charts are
    packed in batches of batch_size per image file named
    categorical_distributions_01.png, _02.png, etc.
    """
    multi_cat = [
        c for c in categorical_cols
        if c in df.columns and df[c].nunique(dropna=True) >= 2
    ]
    if not multi_cat:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_batches = max(1, math.ceil(len(multi_cat) / batch_size))
    written = []
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


# ============================================================================
# Source: figures_pca_plots.py
# ============================================================================

"""
Standalone descriptive PCA visualizations.

PCA here is a descriptive aid for understanding the structure of the
*numeric* feature space (e.g. "how many dimensions are needed to summarize
these variables, and which variables drive them").
"""




@dataclass
class PCAResult:
    """Bundles the fitted model with the data needed to render every plot."""

    pca: PCA
    components: np.ndarray
    feature_names: List[str]
    loadings: np.ndarray
    recommended_n_components: int

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Explained variance ratios of the fitted PCA components."""
        return self.pca.explained_variance_ratio_


def run_pca(df: pd.DataFrame, features: List[str], variance_threshold: float = 0.9) -> PCAResult:
    """
    Standardize the numeric columns in ``features`` and fit PCA on every
    available component (``min(n_samples, n_features)``).

    ``recommended_n_components`` is the smallest number of components whose
    cumulative explained variance reaches ``variance_threshold``. It is
    informational -- the visualizations always use a fixed 2-3 dimensions,
    since their job is to show structure, not to define "the" retained
    subspace for downstream analysis.
    """
    numeric_df = df[features].select_dtypes(include="number")

    dropped = [f for f in features if f not in numeric_df.columns]
    if dropped:
        raise ValueError(
            f"run_pca received non-numeric feature(s) that cannot be used: {dropped}. "
            "Pass only numeric columns, or encode them before calling run_pca."
        )

    # Columns that are entirely missing carry no information to impute from or
    # to project (their variance is undefined). This is common in federated
    # health data, where a given site may not record a particular field at
    # all -- so we drop such columns rather than failing the whole analysis,
    # and only fail if nothing usable remains.
    empty_columns = numeric_df.columns[numeric_df.isna().all()].tolist()
    if empty_columns:
        numeric_df = numeric_df.drop(columns=empty_columns)

    if numeric_df.shape[1] == 0:
        raise ValueError(
            "run_pca requires at least one numeric feature with non-missing values "
            f"(all of {features} were either non-numeric or entirely empty)."
        )

    # Mean imputation is a simple, defensible default for descriptive PCA: it
    # preserves each column's mean/variance contribution without dropping rows,
    # which would shrink an already-numeric-only sample. It is not appropriate
    # for inferential analysis -- this module is visualization-only.
    X = numeric_df.fillna(numeric_df.mean())

    max_components = min(X.shape)
    if max_components < 2:
        raise ValueError(
            "PCA requires at least 2 samples and 2 numeric features; "
            f"got {X.shape[0]} samples and {X.shape[1]} features."
        )

    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=max_components)
    components = pca.fit_transform(X_scaled)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    recommended_n_components = int(np.searchsorted(cumulative_variance, variance_threshold) + 1)
    recommended_n_components = min(recommended_n_components, max_components)

    return PCAResult(
        pca=pca,
        components=components,
        feature_names=list(numeric_df.columns),
        loadings=loadings,
        recommended_n_components=recommended_n_components,
    )


def _pc_label(result: PCAResult, index: int) -> str:
    """Return a formatted axis label for a principal component, e.g. 'PC1 (42.3%)'."""
    return f"PC{index + 1} ({result.explained_variance_ratio[index] * 100:.1f}%)"


def _with_target(components: np.ndarray, columns: List[str], df: pd.DataFrame,
                  target: Optional[str]) -> tuple[pd.DataFrame, Optional[str]]:
    """Build a plot DataFrame from PCA components and optionally attach a target column for colouring."""
    plot_df = pd.DataFrame(components, columns=columns, index=df.index)
    if target is None:
        return plot_df, None
    plot_df[target] = df[target]
    return plot_df, target


def save_explained_variance_plot(result: PCAResult, output_path: Path, figsize=(8, 5)):
    """
    Per-component and cumulative explained variance, with the
    variance-threshold-based recommended component count marked. This is the
    plot that should drive "how many components matter" -- separate from the
    fixed 2-3 dimensions the other plots use purely for visualization.
    """
    ratios = result.explained_variance_ratio
    cumulative = np.cumsum(ratios)
    n = len(ratios)
    k = result.recommended_n_components

    plt.figure(figsize=figsize)
    plt.bar(range(1, n + 1), ratios * 100, alpha=0.6, label="Per-component variance")
    plt.plot(range(1, n + 1), cumulative * 100, marker="o", color="firebrick", label="Cumulative variance")
    plt.axvline(
        k, color="gray", linestyle="--",
        label=f"{k} component{'s' if k != 1 else ''} reach {cumulative[k - 1] * 100:.1f}%",
    )
    plt.title("Explained Variance by Principal Component")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance (%)")
    plt.xticks(range(1, n + 1))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_pca_scatter(result: PCAResult, df: pd.DataFrame, output_path: Path,
                     target: Optional[str] = None, figsize=(8, 6)):
    """2D projection of samples onto the first two principal components."""
    plot_df, hue = _with_target(result.components[:, :2], ["PC1", "PC2"], df, target)

    plt.figure(figsize=figsize)
    sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue=hue, alpha=0.8)
    plt.title("PCA Projection (first two components)")
    plt.xlabel(_pc_label(result, 0))
    plt.ylabel(_pc_label(result, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_pca_scatter_matrix(result: PCAResult, df: pd.DataFrame, output_path: Path,
                            n_dims: int = 3, target: Optional[str] = None):
    """
    Pairwise grid of the first ``n_dims`` components.

    This is *not* equivalent to the 2D scatter -- it is a superset that also
    shows PC1xPC3 and PC2xPC3, useful for checking whether structure visible
    in the PC1/PC2 plane persists (or is hidden) along further components.
    Kept as a secondary/exploratory view alongside the primary 2D scatter.
    """
    n_dims = min(n_dims, result.components.shape[1])
    pc_cols = [f"PC{i + 1}" for i in range(n_dims)]
    plot_df, hue = _with_target(result.components[:, :n_dims], pc_cols, df, target)

    grid = sns.pairplot(plot_df, vars=pc_cols, hue=hue, corner=True, plot_kws={"alpha": 0.7})
    grid.figure.suptitle("PCA Scatter Matrix", y=1.02)
    grid.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(grid.figure)


def save_pca_loadings_biplot(result: PCAResult, output_path: Path,
                             pc_x: int = 0, pc_y: int = 1,
                             top_n: Optional[int] = None, figsize=(8, 8)):
    """
    Biplot-style view of how strongly each original variable contributes to a
    pair of components -- the most directly interpretable PCA output, since it
    ties abstract axes back to real variable names that report readers know.
    """
    x = result.loadings[:, pc_x]
    y = result.loadings[:, pc_y]
    names = np.array(result.feature_names)

    if top_n is not None and top_n < len(names):
        magnitude = np.hypot(x, y)
        keep = np.argsort(magnitude)[-top_n:]
        x, y, names = x[keep], y[keep], names[keep]

    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)
    for xi, yi in zip(x, y):
        ax.annotate(
            "", xy=(xi, yi), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="steelblue", alpha=0.7),
        )
    max_radius_factor = declutter_radial_labels(ax, x, y, names, fontsize=8)

    # Labels for angularly-clustered arrows are staggered out to a larger
    # radius (see declutter_radial_labels), sized from the actual radius
    # used (plus room for the text itself) so outer labels stay inside the frame.
    limit = max(np.abs(x).max(), np.abs(y).max()) * (max_radius_factor + 0.3)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_title("PCA Loadings (Feature Contributions)")
    ax.set_xlabel(_pc_label(result, pc_x))
    ax.set_ylabel(_pc_label(result, pc_y))
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_pca_3d_html(result: PCAResult, df: pd.DataFrame, output_path: Path,
                     target: Optional[str] = None):
    """
    Interactive 3D projection exported to a standalone HTML file.

    A static 3D scatter is hard to read (fixed angle, occlusion) and easy to
    misjudge -- exporting an interactive plot instead lets the reader rotate
    it themselves, which is far more useful for spotting real structure.
    """
    pc_cols = ["PC1", "PC2", "PC3"]
    plot_df, color = _with_target(result.components[:, :3], pc_cols, df, target)

    fig = px.scatter_3d(
        plot_df, x="PC1", y="PC2", z="PC3", color=color,
        title="PCA 3D Projection (interactive)",
        labels={
            "PC1": _pc_label(result, 0),
            "PC2": _pc_label(result, 1),
            "PC3": _pc_label(result, 2),
        },
    )
    fig.write_html(str(output_path))


def save_pca_overview(result: PCAResult, df: pd.DataFrame, output_path: Path,
                      target: Optional[str] = None, figsize=(15, 4.5)):
    """
    Single composite figure (variance + 2D scatter + top loadings) intended
    as a one-glance summary panel for non-expert report readers.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    ratios = result.explained_variance_ratio
    cumulative = np.cumsum(ratios)
    n = len(ratios)
    axes[0].bar(range(1, n + 1), ratios * 100, alpha=0.6)
    axes[0].plot(range(1, n + 1), cumulative * 100, marker="o", color="firebrick")
    axes[0].set_title("Explained Variance")
    axes[0].set_xlabel("Component")
    axes[0].set_ylabel("Variance (%)")

    plot_df, hue = _with_target(result.components[:, :2], ["PC1", "PC2"], df, target)
    sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue=hue, alpha=0.8, ax=axes[1], legend=False)
    axes[1].set_title("Sample Projection")
    axes[1].set_xlabel(_pc_label(result, 0))
    axes[1].set_ylabel(_pc_label(result, 1))

    x, y = result.loadings[:, 0], result.loadings[:, 1]
    top = np.argsort(np.hypot(x, y))[-10:]
    top_x, top_y = x[top], y[top]
    top_names = [result.feature_names[i] for i in top]
    axes[2].axhline(0, color="gray", linewidth=0.8)
    axes[2].axvline(0, color="gray", linewidth=0.8)
    for xi, yi in zip(top_x, top_y):
        axes[2].annotate("", xy=(xi, yi), xytext=(0, 0),
                         arrowprops=dict(arrowstyle="->", color="steelblue", alpha=0.7))
    # A larger perp_step_frac than the standalone biplot's default -- this
    # panel is ~1/3 the width, so the same data-unit gap buys less pixel
    # space between fanned-out labels.
    max_radius_factor = declutter_radial_labels(
        axes[2], top_x, top_y, top_names, perp_step_frac=0.2,
    )
    # Annotation arrows don't participate in matplotlib's autoscale, so the
    # limits must be set explicitly from the plotted points (mirrors
    # save_pca_loadings_biplot), sized from the actual radius declutter_radial_labels
    # used (plus room for the text itself) so outer labels stay inside the frame.
    limit = max(np.abs(top_x).max(), np.abs(top_y).max()) * (max_radius_factor + 0.3)
    axes[2].set_xlim(-limit, limit)
    axes[2].set_ylim(-limit, limit)
    axes[2].set_title("Top Feature Contributions")
    axes[2].set_xlabel(_pc_label(result, 0))
    axes[2].set_ylabel(_pc_label(result, 1))

    fig.suptitle("PCA Overview")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_pca_outputs(df: pd.DataFrame, features: List[str], output_dir: Path,
                     target: Optional[str] = None) -> PCAResult:
    """
    Run PCA on ``features`` and save the full set of descriptive
    visualizations into ``output_dir``.

    Standalone entry point for PCA output generation -- the only function
    report generation needs to call.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if target is not None and target not in df.columns:
        target = None

    result = run_pca(df, features)
    n_dims = result.components.shape[1]

    save_explained_variance_plot(result, output_dir / "pca_explained_variance.png")
    save_pca_scatter(result, df, output_dir / "pca_scatter_2d.png", target=target)
    save_pca_loadings_biplot(result, output_dir / "pca_loadings_pc1_pc2.png", pc_x=0, pc_y=1)

    if n_dims >= 3:
        save_pca_scatter_matrix(result, df, output_dir / "pca_scatter_matrix.png", target=target)
        save_pca_loadings_biplot(result, output_dir / "pca_loadings_pc1_pc3.png", pc_x=0, pc_y=2)
        save_pca_3d_html(result, df, output_dir / "pca_scatter_3d.html", target=target)

    save_pca_overview(result, df, output_dir / "pca_overview.png", target=target)

    return result



# ============================================================================
# Source: figures_mca_plots.py
# ============================================================================

"""
Standalone descriptive MCA (Multiple Correspondence Analysis) visualizations.

MCA is the categorical-data analog of PCA used in :mod:`pca_plots`: instead of
projecting numeric variables into a reduced variance-maximizing space, it
projects categorical *levels* (and the samples that hold them) into a space
that captures association between categories. It is a more honest tool for
healthcare categorical data (symptoms, diagnoses, demographics) than coercing
one-hot-encoded categories through PCA's Euclidean/variance assumptions.

Like :mod:`pca_plots`, this module is a descriptive/exploratory aid only.
"""




@dataclass
class MCAResult:
    """Bundles the fitted model with the data needed to render every plot."""

    mca: prince.MCA
    row_coordinates: np.ndarray
    column_coordinates: pd.DataFrame
    column_variable: pd.Series  # maps each "variable__level" label back to its source variable
    feature_names: List[str]
    explained_inertia_ratio: np.ndarray
    recommended_n_components: int


def _variable_for_label(label: str, feature_names: List[str], sep: str) -> str:
    """Extract the variable name prefix from an MCA coordinate index label (strips the category suffix).

    Picks the longest (most specific) matching feature name, not the first
    one found -- a first-match search would mis-attribute every category of
    "site__region" to "site" if "site" happens to come first in
    feature_names and is itself a prefix of the other feature's name.
    """
    matches = [name for name in feature_names if label.startswith(f"{name}{sep}")]
    return max(matches, key=len) if matches else label


def run_mca(df: pd.DataFrame, features: List[str], variance_threshold: float = 0.9,
            max_levels_per_variable: int = 30) -> MCAResult:
    """
    Fit MCA on the categorical columns in ``features``.

    Columns with more than ``max_levels_per_variable`` distinct values are
    rejected: MCA one-hot-encodes every category, so high-cardinality columns
    (free-text fields, raw codes, identifiers) would dominate the geometry,
    blow up runtime, and make the category map unreadable. Curate ``features``
    to the categorical variables you actually want to compare.

    Columns that are entirely missing are dropped silently rather than
    failing the whole analysis (mirrors ``run_pca``'s handling of
    entirely-missing numeric columns), and only fail if fewer than 2 usable
    columns remain.

    ``recommended_n_components`` mirrors :func:`pca_plots.run_pca`'s convention: the
    smallest number of dimensions whose cumulative explained inertia reaches
    ``variance_threshold``. Visualizations always use a fixed 2-3 dimensions
    regardless, since their job is to show structure, not define a retained
    subspace.
    """
    # Columns that are entirely missing have no categories to one-hot-encode
    # and would otherwise reach prince.MCA.fit() and fail unpredictably. This
    # mirrors run_pca's handling of entirely-missing numeric columns. Checked
    # on the raw requested columns *before* the dtype filter below, since an
    # all-NaN column can be inferred as a non-object dtype (e.g. float64),
    # which would otherwise misclassify it as "non-categorical" instead of
    # recognising it as simply empty.
    requested_df = df[features]
    empty_columns = [f for f in features if requested_df[f].isna().all()]
    usable_features = [f for f in features if f not in empty_columns]

    categorical_df = df[usable_features].select_dtypes(include=["object", "category", "bool"])

    dropped = [f for f in usable_features if f not in categorical_df.columns]
    if dropped:
        raise ValueError(
            f"run_mca received non-categorical feature(s) that cannot be used: {dropped}. "
            "Pass only categorical/object/bool columns, or recode them before calling run_mca."
        )

    if categorical_df.shape[1] == 0:
        raise ValueError("run_mca requires at least one categorical feature.")
    if categorical_df.shape[1] < 2:
        raise ValueError(
            "MCA requires at least 2 categorical features to relate to each other; "
            f"got {categorical_df.shape[1]}."
        )

    high_cardinality = {
        col: int(categorical_df[col].nunique(dropna=True))
        for col in categorical_df.columns
        if categorical_df[col].nunique(dropna=True) > max_levels_per_variable
    }
    if high_cardinality:
        raise ValueError(
            f"Column(s) exceed max_levels_per_variable={max_levels_per_variable}: "
            f"{high_cardinality}. Exclude them or raise the limit explicitly -- "
            "high-cardinality columns make the MCA category map unreadable."
        )

    sep = "__"
    total_categories = sum(categorical_df[col].nunique(dropna=True) for col in categorical_df.columns)
    max_components = max(total_categories - categorical_df.shape[1], 1)

    mca = prince.MCA(
        n_components=max_components,
        one_hot_prefix_sep=sep,
        random_state=42,
        engine="sklearn",
    )
    mca = mca.fit(categorical_df)

    row_coordinates = mca.row_coordinates(categorical_df).to_numpy()
    column_coordinates = mca.column_coordinates(categorical_df)
    column_variable = pd.Series(
        [_variable_for_label(label, list(categorical_df.columns), sep) for label in column_coordinates.index],
        index=column_coordinates.index,
    )

    explained_inertia_ratio = np.asarray(mca.percentage_of_variance_) / 100.0
    cumulative = np.cumsum(explained_inertia_ratio)
    recommended_n_components = int(np.searchsorted(cumulative, variance_threshold) + 1)
    recommended_n_components = min(recommended_n_components, len(explained_inertia_ratio))

    return MCAResult(
        mca=mca,
        row_coordinates=row_coordinates,
        column_coordinates=column_coordinates,
        column_variable=column_variable,
        feature_names=list(categorical_df.columns),
        explained_inertia_ratio=explained_inertia_ratio,
        recommended_n_components=recommended_n_components,
    )


def _dim_label(result: MCAResult, index: int) -> str:
    """Return a formatted axis label for an MCA dimension, e.g. 'Dim 1 (12.4%)'."""
    return f"Dim {index + 1} ({result.explained_inertia_ratio[index] * 100:.1f}%)"


def _with_target_mca(coordinates: np.ndarray, columns: List[str], df: pd.DataFrame,
                  target: Optional[str]) -> tuple[pd.DataFrame, Optional[str]]:
    """Build a plot DataFrame from MCA row coordinates and optionally attach a target column for colouring."""
    plot_df = pd.DataFrame(coordinates, columns=columns, index=df.index)
    if target is None:
        return plot_df, None
    plot_df[target] = df[target]
    return plot_df, target


def save_explained_inertia_plot(result: MCAResult, output_path: Path, figsize=(8, 5)):
    """
    Per-dimension and cumulative explained inertia, with the
    variance-threshold-based recommended dimension count marked. Mirrors
    :func:`pca_plots.save_explained_variance_plot` -- "inertia" is MCA's analog of
    PCA's "explained variance".
    """
    ratios = result.explained_inertia_ratio
    cumulative = np.cumsum(ratios)
    n = len(ratios)
    k = result.recommended_n_components

    plt.figure(figsize=figsize)
    plt.bar(range(1, n + 1), ratios * 100, alpha=0.6, label="Per-dimension inertia")
    plt.plot(range(1, n + 1), cumulative * 100, marker="o", color="firebrick", label="Cumulative inertia")
    plt.axvline(
        k, color="gray", linestyle="--",
        label=f"{k} dimension{'s' if k != 1 else ''} reach {cumulative[k - 1] * 100:.1f}%",
    )
    plt.title("Explained Inertia by MCA Dimension")
    plt.xlabel("MCA Dimension")
    plt.ylabel("Explained Inertia (%)")
    plt.xticks(range(1, n + 1))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_mca_row_scatter(result: MCAResult, df: pd.DataFrame, output_path: Path,
                         target: Optional[str] = None, figsize=(8, 6)):
    """2D projection of samples onto the first two MCA dimensions."""
    plot_df, hue = _with_target_mca(result.row_coordinates[:, :2], ["Dim1", "Dim2"], df, target)

    plt.figure(figsize=figsize)
    sns.scatterplot(data=plot_df, x="Dim1", y="Dim2", hue=hue, alpha=0.8)
    plt.title("MCA Projection (first two dimensions)")
    plt.xlabel(_dim_label(result, 0))
    plt.ylabel(_dim_label(result, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_mca_scatter_matrix(result: MCAResult, df: pd.DataFrame, output_path: Path,
                            n_dims: int = 3, target: Optional[str] = None):
    """
    Pairwise grid of the first ``n_dims`` row-coordinate dimensions --
    mirrors :func:`pca_plots.save_pca_scatter_matrix`: a superset of the 2D
    projection, useful for checking whether structure visible in Dim1/Dim2
    persists along further dimensions.
    """
    n_dims = min(n_dims, result.row_coordinates.shape[1])
    dim_cols = [f"Dim{i + 1}" for i in range(n_dims)]
    plot_df, hue = _with_target_mca(result.row_coordinates[:, :n_dims], dim_cols, df, target)

    grid = sns.pairplot(plot_df, vars=dim_cols, hue=hue, corner=True, plot_kws={"alpha": 0.7})
    grid.figure.suptitle("MCA Scatter Matrix", y=1.02)
    grid.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(grid.figure)


def save_mca_column_map(result: MCAResult, output_path: Path,
                        dim_x: int = 0, dim_y: int = 1, figsize=(9, 8)):
    """
    Map of category-level coordinates, colored by source variable and
    labeled by category -- the MCA equivalent of a PCA loadings/biplot. This
    is the most directly interpretable MCA output: it shows which category
    *levels* (e.g. "smoker=yes", "symptom_x=present") sit close together,
    i.e. tend to co-occur across samples.
    """
    coords = result.column_coordinates
    x = coords.iloc[:, dim_x].to_numpy()
    y = coords.iloc[:, dim_y].to_numpy()
    variables = result.column_variable.to_numpy()
    labels = [idx.split("__", 1)[-1] if "__" in idx else idx for idx in coords.index]

    unique_variables = list(dict.fromkeys(variables))
    palette = colormaps["tab10"].resampled(max(len(unique_variables), 1))
    color_for_variable = {var: palette(i) for i, var in enumerate(unique_variables)}

    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)
    for var in unique_variables:
        mask = variables == var
        ax.scatter(x[mask], y[mask], color=color_for_variable[var], label=var, s=40, alpha=0.85)
    declutter_point_labels(ax, x, y, labels)

    ax.set_title("MCA Category Map (Column Coordinates)")
    ax.set_xlabel(_dim_label(result, dim_x))
    ax.set_ylabel(_dim_label(result, dim_y))
    ax.legend(title="Variable", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_mca_3d_html(result: MCAResult, df: pd.DataFrame, output_path: Path,
                     target: Optional[str] = None):
    """
    Interactive 3D projection of samples onto the first three MCA dimensions,
    exported to a standalone HTML file -- mirrors :func:`pca_plots.save_pca_3d_html`
    for the same reason: a static 3D scatter is hard to read, an interactive
    one lets the reader rotate it themselves.
    """
    dim_cols = ["Dim1", "Dim2", "Dim3"]
    plot_df, color = _with_target_mca(result.row_coordinates[:, :3], dim_cols, df, target)

    fig = px.scatter_3d(
        plot_df, x="Dim1", y="Dim2", z="Dim3", color=color,
        title="MCA 3D Projection (interactive)",
        labels={
            "Dim1": _dim_label(result, 0),
            "Dim2": _dim_label(result, 1),
            "Dim3": _dim_label(result, 2),
        },
    )
    fig.write_html(str(output_path))


def save_mca_overview(result: MCAResult, df: pd.DataFrame, output_path: Path,
                      target: Optional[str] = None, figsize=(15, 4.5)):
    """
    Single composite figure (inertia + row projection + category map)
    intended as a one-glance summary panel for non-expert report readers --
    mirrors :func:`pca_plots.save_pca_overview`.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    ratios = result.explained_inertia_ratio
    cumulative = np.cumsum(ratios)
    n = len(ratios)
    axes[0].bar(range(1, n + 1), ratios * 100, alpha=0.6)
    axes[0].plot(range(1, n + 1), cumulative * 100, marker="o", color="firebrick")
    axes[0].set_title("Explained Inertia")
    axes[0].set_xlabel("Dimension")
    axes[0].set_ylabel("Inertia (%)")

    plot_df, hue = _with_target_mca(result.row_coordinates[:, :2], ["Dim1", "Dim2"], df, target)
    sns.scatterplot(data=plot_df, x="Dim1", y="Dim2", hue=hue, alpha=0.8, ax=axes[1], legend=False)
    axes[1].set_title("Sample Projection")
    axes[1].set_xlabel(_dim_label(result, 0))
    axes[1].set_ylabel(_dim_label(result, 1))

    coords = result.column_coordinates
    x = coords.iloc[:, 0].to_numpy()
    y = coords.iloc[:, 1].to_numpy()
    labels = [idx.split("__", 1)[-1] if "__" in idx else idx for idx in coords.index]
    axes[2].axhline(0, color="gray", linewidth=0.8)
    axes[2].axvline(0, color="gray", linewidth=0.8)
    axes[2].scatter(x, y, alpha=0.8, s=30, color="steelblue")
    # This panel is ~1/3 the width of the standalone category map -- with
    # dozens of categories (common once a dataset has several categorical
    # variables) there simply isn't room to label every point, however
    # cleverly staggered, so only the most distinctive (furthest from
    # origin) categories are shown here.
    _OVERVIEW_MAX_LABELS = 15
    if len(labels) > _OVERVIEW_MAX_LABELS:
        top_idx = np.argsort(np.hypot(x, y))[-_OVERVIEW_MAX_LABELS:]
        declutter_point_labels(axes[2], x[top_idx], y[top_idx],
                                [labels[i] for i in top_idx], fontsize=7)
    else:
        declutter_point_labels(axes[2], x, y, labels, fontsize=7)
    axes[2].set_title("Category Map")
    axes[2].set_xlabel(_dim_label(result, 0))
    axes[2].set_ylabel(_dim_label(result, 1))

    fig.suptitle("MCA Overview")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


_MCA_COLUMN_MAP_BATCH = 5


def _subset_mca_result_hub(result: MCAResult, batch_vars: list) -> MCAResult:
    """Return a copy of result restricted to the category levels of batch_vars."""
    kept = [
        lbl for lbl in result.column_coordinates.index
        if any(lbl.startswith(f"{v}_") or lbl == v for v in batch_vars)
    ]
    if not kept:
        kept = list(result.column_coordinates.index)
    import copy
    sub = copy.copy(result)
    sub.column_coordinates = result.column_coordinates.loc[kept]
    return sub


def save_mca_outputs(df: pd.DataFrame, features: List[str], output_dir: Path,
                     target: Optional[str] = None) -> MCAResult:
    """Run MCA on features and save descriptive visualizations into output_dir.

    Column maps are batched at 5 source variables per image so legend entries
    remain readable regardless of how many categorical variables are present.
    Also saves an overview panel and a row scatter plot.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if target is not None and target not in df.columns:
        target = None

    result = run_mca(df, features)
    n_dims = result.row_coordinates.shape[1]

    save_explained_inertia_plot(result, output_dir / "mca_explained_inertia.png")
    save_mca_row_scatter(result, df, output_dir / "mca_row_scatter_2d.png", target=target)

    # Batched column maps (5 source variables per image)
    all_vars = list(features)
    n_batches = max(1, math.ceil(len(all_vars) / _MCA_COLUMN_MAP_BATCH))
    for b in range(n_batches):
        batch_vars = all_vars[b * _MCA_COLUMN_MAP_BATCH:(b + 1) * _MCA_COLUMN_MAP_BATCH]
        try:
            batch_result = _subset_mca_result_hub(result, batch_vars)
            fname = f"mca_column_map_batch_{b + 1:02d}.png"
            save_mca_column_map(batch_result, output_dir / fname, dim_x=0, dim_y=1)
        except Exception:
            save_mca_column_map(result, output_dir / f"mca_column_map_batch_{b + 1:02d}.png",
                                dim_x=0, dim_y=1)

    if n_dims >= 3:
        save_mca_scatter_matrix(result, df, output_dir / "mca_scatter_matrix.png", target=target)
        save_mca_3d_html(result, df, output_dir / "mca_scatter_3d.html", target=target)

    save_mca_overview(result, df, output_dir / "mca_overview.png", target=target)

    return result



# ============================================================================
# Source: figures_data_quality_plots.py
# ============================================================================

"""
Data quality visualizations.

Internal helpers used by the analyzer and aggregator:
  - _make_missing_bar_fig        missingno bar chart (column completeness overview)
  - _make_missing_heatmap_fig    missingno heatmap (nullity correlation between columns)
  - _make_missing_by_column_fig  stacked horizontal bar: present vs. missing % per column
"""













# ============================================================================
# Source: figures_local_descriptive_plots.py
#
# NOTE: this section is a MIX of (a) shared helpers genuinely used elsewhere
# in this file (e.g. _periods_to_timestamps) and (b) orchestrator functions
# (save_numeric_histograms/save_numeric_boxplots/save_categorical_distributions/
# save_temporal_activity_batched and others) that are NOT called from the live
# analyzer/aggregator path -- that path builds equivalent charts inline,
# directly from serialized statistics, elsewhere in this file (search for
# "Numeric histograms grid", "Numeric boxplots grid", "Categorical
# distribution plots", "Batched temporal activity grid"). Do not assume this
# block is a 1:1 mirror of local_descriptive_plots.py; diff the live inline
# blocks instead when checking for drift.
# ============================================================================

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





logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _periods_to_timestamps(obs: dict) -> tuple[list, list]:
    """
    Convert a period-keyed observations dict to (timestamps, counts) lists,
    ready for matplotlib.

    Keys may be pd.Period objects or strings. Any key that cannot be parsed
    is silently dropped. The returned lists are sorted chronologically.
    """
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













# ===========================================================================
# Categorical section
# ===========================================================================









# ===========================================================================
# Temporal section
# ===========================================================================










# ============================================================================
# Source: figures_federated_descriptive_plots.py
# ============================================================================

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





logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Overview section
# ---------------------------------------------------------------------------







# ---------------------------------------------------------------------------
# Numeric section
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Categorical section
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Temporal section
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Federated inferential: simple trend analysis
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def save_all_federated_plots(federated_results: dict, output_files: dict,
                              archive_base: str = "federated") -> None:
    """
    In-memory version: generate all federated plots and store PNG bytes
    directly into output_files under archive_base/…/filename.png.

    Replaces the old disk-writing signature
    save_all_federated_plots(federated_results, output_dir).
    """
    global_numeric     = federated_results.get("global_numeric", {})
    global_categorical = federated_results.get("global_categorical", {})
    global_temporal    = federated_results.get("global_temporal", {})

    # --- overview ---
    counts = [len(global_numeric), len(global_categorical), len(global_temporal)]
    if sum(counts) > 0:
        try:
            fig, ax = plt.subplots(figsize=(6, 6))
            pie_chart(
                ax, counts,
                ["Numerical", "Categorical", "Temporal"],
                colors=["#A6D8A8", "#86C5F9", "#FFCC80"],
                title="Federated Data Type Distribution",
                # Only 3 possible slices here -- no label-overlap risk to
                # guard against, so a small category shouldn't be renamed
                # "Other" instead of shown by its real name.
                min_slice_pct=0,
            )
            output_files[f"{archive_base}/overview/data_type_distribution.png"] = (
                _fig_to_bytes(fig)
            )
        except Exception as e:
            print(f"Federated data-type pie error: {e}", flush=True)

    age_edges = federated_results.get("age_edges")
    age_hist  = federated_results.get("age_hist")
    if age_edges is not None and age_hist is not None:
        try:
            edges = np.asarray(age_edges, dtype=float)
            hist  = np.asarray(age_hist,  dtype=float)
            if len(edges) >= 2 and hist.sum() > 0:
                fig, ax = plt.subplots(figsize=(10, 5))
                histogram_from_bins(
                    ax, edges, hist,
                    title="Federated Age Distribution",
                    xlabel="Age", ylabel="Count (federated)",
                )
                output_files[f"{archive_base}/numeric/age_distribution_federated.png"] = (
                    _fig_to_bytes(fig)
                )
        except Exception as e:
            print(f"Federated age distribution error: {e}", flush=True)

    sex_counts = federated_results.get("sex_counts", {})
    if sex_counts:
        try:
            keys = [k for k in sex_counts if pd.notna(k)]
            vals = [sex_counts[k] for k in keys]
            if keys:
                fig, ax = plt.subplots(figsize=(6, 5))
                bar_chart(
                    ax, keys, vals,
                    colors=[PALETTE[i % len(PALETTE)] for i in range(len(keys))],
                    title="Federated Sex Distribution",
                    xlabel="Sex", ylabel="Count (federated)",
                )
                output_files[f"{archive_base}/categorical/sex_distribution_federated.png"] = (
                    _fig_to_bytes(fig)
                )
        except Exception as e:
            print(f"Federated sex distribution error: {e}", flush=True)

    # --- numeric ---
    if global_numeric:
        try:
            cols  = list(global_numeric.keys())
            means = [global_numeric[c].get("mean", 0) for c in cols]
            stds  = [global_numeric[c].get("std",  0) for c in cols]
            fig, ax = plt.subplots(figsize=(10, max(5, len(cols) * 0.4)))
            y_pos = range(len(cols))
            ax.barh(list(y_pos), means, xerr=stds,
                    color=PALETTE[0], ecolor=PALETTE[3], capsize=4, alpha=0.8)
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(cols)
            ax.set_xlabel("Mean ± Std (federated)")
            ax.set_title("Federated Numeric Feature Summary")
            ax.grid(axis="x", alpha=0.3)
            output_files[f"{archive_base}/numeric/numeric_summary_bars.png"] = (
                _fig_to_bytes(fig)
            )
        except Exception as e:
            print(f"Federated numeric summary bars error: {e}", flush=True)

    # --- categorical (batched, multi-category only, 6 per image) ---
    if global_categorical:
        try:
            # Only multi-category columns with > 2 distinct non-null values --
            # binary columns are already visible in the summary table and sex
            # distribution chart (matches the modular project's
            # save_federated_categorical_distributions).
            multi_cat_cols = [
                c for c in global_categorical
                if len([k for k in global_categorical[c].get("counts", {}) if pd.notna(k)]) > 2
            ]
            batch_size_cat = 6
            n_batches_cat = max(1, math.ceil(len(multi_cat_cols) / batch_size_cat)) if multi_cat_cols else 0
            for b in range(n_batches_cat):
                batch_cols = multi_cat_cols[b * batch_size_cat:(b + 1) * batch_size_cat]
                fig, axes = make_subplots(len(batch_cols), ncols=2, width=7, height=5)
                for ax, col in zip(axes, batch_cols):
                    counts_d = global_categorical[col]["counts"]
                    sorted_items = sorted(
                        counts_d.items(), key=lambda x: x[1], reverse=True
                    )[:20]
                    if not sorted_items:
                        continue
                    cats, vals = zip(*sorted_items)
                    bar_chart(ax, list(cats), list(vals), horizontal=True,
                              title=col, xlabel="Count (federated)")
                title = "Federated Categorical Distributions (multi-category)"
                if n_batches_cat > 1:
                    title += f" ({b + 1}/{n_batches_cat})"
                fig.suptitle(title, fontsize=13, y=1.01)
                fig.tight_layout(h_pad=3.0)
                fname = f"categorical_distributions_{b + 1:02d}.png"
                output_files[f"{archive_base}/categorical/{fname}"] = _fig_to_bytes(fig)
        except Exception as e:
            print(f"Federated categorical distributions error: {e}", flush=True)

    # --- temporal ---
    for feature, feat_stats in global_temporal.items():
        counts_per_period = feat_stats.get("counts_per_period")
        if not counts_per_period:
            continue
        try:
            timestamps, cnt = _periods_to_timestamps(counts_per_period)
            if not timestamps:
                continue
            fig, ax = plt.subplots(figsize=(10, 5))
            line_chart(ax, timestamps, cnt,
                       title=f"{feature} — Federated Temporal Activity",
                       xlabel="Time", ylabel="Observations (federated)")
            most_active = feat_stats.get("most_active_period")
            if most_active is not None:
                ma_ts = pd.to_datetime(most_active, errors="coerce")
                if not pd.isna(ma_ts) and ma_ts in timestamps:
                    idx = timestamps.index(ma_ts)
                    ax.scatter([ma_ts], [cnt[idx]], s=100, zorder=5,
                               color=PALETTE[1], label="Most Active Period")
                    ax.legend()
            output_files[
                f"{archive_base}/temporal/{feature}_activity_federated.png"
            ] = _fig_to_bytes(fig)
        except Exception as e:
            print(f"Federated temporal chart error ({feature}): {e}", flush=True)

    # --- temporal trend summary ---
    try:
        from scipy.stats import linregress as _lr
        slopes, r2s, labels = [], [], []
        for feature, feat_stats in global_temporal.items():
            cpp = feat_stats.get("counts_per_period")
            if not cpp:
                continue
            try:
                _, cnt = _periods_to_timestamps(cpp)
                if len(cnt) < 3:
                    continue
                x = np.arange(len(cnt), dtype=float)
                y = np.asarray(cnt, dtype=float)
                slope, _, r, _, _ = _lr(x, y)
                slopes.append(slope); r2s.append(r ** 2); labels.append(feature)
            except Exception:
                continue
        if slopes:
            order  = np.argsort(np.abs(slopes))
            slopes = [slopes[i] for i in order]
            r2s    = [r2s[i]    for i in order]
            labels = [labels[i] for i in order]
            clrs   = [PALETTE[0] if s >= 0 else PALETTE[3] for s in slopes]
            fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.5)))
            bars = ax.barh(labels, slopes, color=clrs, alpha=0.85)
            for bar, r2 in zip(bars, r2s):
                if r2 >= 0.1:
                    ax.text(bar.get_width() / 2,
                            bar.get_y() + bar.get_height() / 2,
                            f"R²={r2:.2f}", ha="center", va="center",
                            fontsize=8, color="white")
            ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
            ax.set_xlabel("Linear Trend Slope (observations / period)")
            ax.set_title("Federated Temporal Trend Summary")
            import matplotlib.patches as _mp
            ax.legend(handles=[
                _mp.Patch(color=PALETTE[0], label="Increasing ↑"),
                _mp.Patch(color=PALETTE[3], label="Decreasing ↓"),
            ], loc="lower right")
            ax.grid(axis="x", alpha=0.3)
            output_files[f"{archive_base}/temporal/temporal_trend_summary.png"] = (
                _fig_to_bytes(fig)
            )
    except Exception as e:
        print(f"Federated trend summary error: {e}", flush=True)


def _make_missing_by_column_fig(missing_by_col: dict, n_rows: int,
                                  node_label: str = "") -> "Figure | None":
    """Return a bar chart of per-column missing-value counts, or None on error."""
    try:
        if not missing_by_col:
            return None
        cols   = list(missing_by_col.keys())
        counts = [missing_by_col[c] for c in cols]
        fig, ax = plt.subplots(figsize=(max(8, len(cols) * 0.5), 5))
        ax.bar(range(len(cols)), counts, color=PALETTE[3], alpha=0.8)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right")
        ax.set_ylabel("Missing count")
        ax.set_title(
            f"Missing Values by Column"
            + (f" — {node_label}" if node_label else "")
        )
        ax.axhline(n_rows * 0.5, color="gray", linestyle="--",
                   label="50% threshold")
        ax.legend()
        fig.tight_layout()
        return fig
    except Exception as e:
        print(f"_make_missing_by_column_fig error: {e}", flush=True)
        return None



# ---------------------------------------------------------------------------
# In-memory PDF generators (wrappers around the existing report builders)
# ---------------------------------------------------------------------------

def generate_local_report_bytes(node_result: dict, node_comp: dict,
                                  mode: str = "full",
                                  node_number: int = 1,
                                  output_files: dict | None = None,
                                  federated_results: dict | None = None) -> bytes | None:
    """
    Generate a local-node PDF report entirely in memory and return it as bytes.
    Writes both CSVs and PNG plots to a temp directory so the PDF builder
    can find all assets via disk paths.
    """
    import tempfile, shutil
    tmp = Path(tempfile.mkdtemp())
    try:
        node_name = f"node{node_number}"
        node_dir  = tmp / "local" / node_name
        fed_dir   = tmp / "federated"

        # Write CSVs
        _write_node_csvs_to_dir(node_result, node_comp, node_dir)

        # Write federated CSVs so add_categorical_comparison can read global stats
        if federated_results:
            _write_federated_csvs_to_dir(federated_results, fed_dir)

        # Write PNG and CSV files from output_files that belong to this node
        if output_files:
            prefix = f"local/node{node_number}/"
            for rel_path, content in output_files.items():
                if rel_path.startswith(prefix) and (
                    rel_path.endswith(".png") or rel_path.endswith(".csv")
                ):
                    dest = tmp / "local" / rel_path[len("local/"):]
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(content)

        pdf_path = generate_local_report(
            node_dir, node_dir, mode=mode, results_dir=fed_dir
        )
        return pdf_path.read_bytes() if pdf_path and pdf_path.exists() else None
    except Exception as e:
        print(f"generate_local_report_bytes error ({mode}): {e}", flush=True)
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def generate_global_report_bytes(federated_results: dict,
                                   mode: str = "full",
                                   output_files: dict | None = None) -> bytes | None:
    """Generate the federated PDF report and return it as bytes.
    Writes both CSVs and PNG plots to a temp directory so the PDF builder
    can find all assets via disk paths.
    """
    import tempfile, shutil
    tmp = Path(tempfile.mkdtemp())
    try:
        fed_dir = tmp / "federated"
        _write_federated_csvs_to_dir(federated_results, fed_dir)

        # Write PNG plots from output_files that belong to federated section
        if output_files:
            for rel_path, content in output_files.items():
                if rel_path.startswith("federated/") and rel_path.endswith(".png"):
                    dest = tmp / rel_path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(content)

        pdf_path = generate_global_report(fed_dir, fed_dir, mode=mode)
        return pdf_path.read_bytes() if pdf_path and pdf_path.exists() else None
    except Exception as e:
        print(f"generate_global_report_bytes error ({mode}): {e}", flush=True)
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _write_node_csvs_to_dir(node_result: dict, node_comp: dict,
                              node_dir: Path) -> None:
    """Write the CSVs that generate_local_report() reads from node_dir."""
    numeric_statistics    = node_result.get("numeric_statistics", {})
    categorical_statistics = node_result.get("categorical_statistics", {})
    temporal_statistics   = node_result.get("temporal_statistics", {})
    column_comparison     = node_comp.get("column_comparison", {})
    numeric_comparison    = node_comp.get("numeric_comparison", {})
    overview_comparison   = node_comp.get("overview_comparison", {})

    (node_dir / "overview").mkdir(parents=True, exist_ok=True)
    (node_dir / "numeric").mkdir(parents=True, exist_ok=True)
    (node_dir / "categorical").mkdir(parents=True, exist_ok=True)
    (node_dir / "temporal").mkdir(parents=True, exist_ok=True)
    (node_dir / "inferential").mkdir(parents=True, exist_ok=True)

    # Overview
    relative_missing = overview_comparison.get("relative_missing", 0)
    if isinstance(relative_missing, str):
        try: relative_missing = float(relative_missing)
        except ValueError: relative_missing = 0.0
    missing_label = (
        "above federation average" if relative_missing > 1
        else "below federation average" if relative_missing < 1
        else "equal to federation average"
    )
    _n_analytical = node_result.get("n_analytical_cols", node_result["n_cols"])
    _id_cols = node_result.get("id_columns", [])
    _id_suffix = (
        f" + {len(_id_cols)} identifier column{'s' if len(_id_cols) != 1 else ''}"
        f" detected ({', '.join(_id_cols)})"
        if _id_cols else ""
    )
    ov = {
        "Patients": f"{node_result['n_rows']}",
        "Features": f"{_n_analytical} analytical columns{_id_suffix}",
        "Missing Values": f"{node_result['total_missing']:,} ({round(node_result['missing_values_percentage'], 2)}%)",
        "Missingness vs Federation": f"{round(relative_missing, 2)}× ({missing_label})",
        "Duplicates": node_result["n_duplicates"],
    }
    pd.DataFrame([{"metric": k, "value": v} for k, v in ov.items()]).to_csv(
        node_dir / "overview" / "overview.csv", index=False
    )

    # Numeric
    if numeric_statistics:
        rows = [{"feature": f, "availability": column_comparison.get(f, "unknown"), **m}
                for f, m in numeric_statistics.items()]
        pd.DataFrame(rows).to_csv(node_dir / "numeric" / "numeric_summary.csv", index=False)

    # Categorical
    if categorical_statistics:
        rows = []
        for f, m in categorical_statistics.items():
            row = {"feature": f, "availability": column_comparison.get(f, "unknown")}
            for k, v in m.items():
                if k != "category_counts":
                    row[k] = v
            rows.append(row)
        pd.DataFrame(rows).to_csv(
            node_dir / "categorical" / "categorical_summary.csv", index=False
        )

    # Temporal
    if temporal_statistics:
        rows = []
        for f, m in temporal_statistics.items():
            row = {"feature": f, "availability": column_comparison.get(f, "unknown")}
            for k, v in m.items():
                if k not in ("observations_per_period", "time_range", "range_days"):
                    row[k] = v
            rows.append(row)
        pd.DataFrame(rows).to_csv(
            node_dir / "temporal" / "temporal_summary.csv", index=False
        )

    # Numeric comparison
    if numeric_comparison:
        pd.DataFrame([
            {"column": col, "comparison": info["comparison_category"]}
            for col, info in numeric_comparison.items()
        ]).to_csv(node_dir / "comparison" / "numeric_comparison.csv"
                  if (node_dir / "comparison").exists()
                  else node_dir / "numeric" / "numeric_comparison.csv",
                  index=False)


def _write_federated_csvs_to_dir(federated_results: dict, fed_dir: Path) -> None:
    """Write the CSVs that generate_global_report() reads from fed_dir."""
    (fed_dir / "overview").mkdir(parents=True, exist_ok=True)
    (fed_dir / "numeric").mkdir(parents=True, exist_ok=True)
    (fed_dir / "categorical").mkdir(parents=True, exist_ok=True)
    (fed_dir / "temporal").mkdir(parents=True, exist_ok=True)

    overview = {
        "number of hospitals": federated_results.get("n_nodes"),
        "total number of patients": federated_results.get("total_rows"),
        "total number of features": federated_results.get("n_cols"),
        "total number of values": federated_results.get("n_total_values"),
        "total missing values": federated_results.get("total_missing"),
        "total missing values percentage": (
            f"{round(federated_results.get('total_missing_percentage', 0), 3)}%"
        ),
    }
    pd.DataFrame([{"metric": k, "value": v} for k, v in overview.items()]).to_csv(
        fed_dir / "overview" / "overview.csv", index=False
    )

    global_numeric = federated_results.get("global_numeric", {})
    if global_numeric:
        pd.DataFrame([{"feature": f, **m} for f, m in global_numeric.items()]).to_csv(
            fed_dir / "numeric" / "federated_numeric_statistics.csv", index=False
        )

    global_categorical = federated_results.get("global_categorical", {})
    if global_categorical:
        pd.DataFrame([{"feature": f, **m} for f, m in global_categorical.items()]).to_csv(
            fed_dir / "categorical" / "federated_categorical_statistics.csv", index=False
        )

    global_temporal = federated_results.get("global_temporal", {})
    if global_temporal:
        rows = []
        for f, s in global_temporal.items():
            row = {"feature": f, **s}
            row["counts_per_period"] = json.dumps(row.get("counts_per_period", {}))
            row["most_active_period"] = str(row.get("most_active_period"))
            rows.append(row)
        pd.DataFrame(rows).to_csv(
            fed_dir / "temporal" / "federated_temporal_statistics.csv", index=False
        )



# ============================================================================
# Source: figures_inferential_plots.py
# ============================================================================

"""
Inferential statistics visualizations.

All functions take pre-computed result dicts (from inferential_analysis.py)
and/or raw DataFrames. No statistical tests are performed here — plotting only.

Sections:
  Group comparison   — two-group boxplots/violins, one-way comparison,
                       effect-size summary bars, association screening
  Correlation        — scatter + regression overlay, correlation matrix
  Chi-square         — annotated contingency heatmap, Cramer's V bar chart
  Regression         — coefficient plots, residuals, predicted vs. actual
  Time series        — power spectrum, peak annotation (moved from inferential_analysis.py)
"""






# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sig_label(p_value: float) -> str:
    """Map a p-value to a conventional significance label."""
    if p_value is None or np.isnan(p_value):
        return "n.s."
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def _annotate_significance(ax, x1: float, x2: float, y: float, p_value: float,
                            bar_height: float = 0.03) -> None:
    """
    Draw a significance bracket between two box positions.

    x1, x2 are the x-axis positions of the two groups; y is the top of the
    bracket in data coordinates; bar_height is the height of the vertical
    ticks as a fraction of the y range.
    """
    ylim = ax.get_ylim()
    h = (ylim[1] - ylim[0]) * bar_height
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            lw=1.2, color="black")
    label = _sig_label(p_value)
    ax.text((x1 + x2) / 2, y + h * 1.2, label,
            ha="center", va="bottom", fontsize=11)


# ===========================================================================
# Group comparison section
# ===========================================================================

def save_two_group_comparison(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    result: dict,
    output_dir,
    *,
    node_label: Optional[str] = None,
) -> None:
    """
    Boxplot with significance bracket for a two-group comparison.

    `result` is the dict returned by `compare_two_groups`. The test method
    and p-value are shown in the title; a bracket annotates significance level
    (*** / ** / * / n.s.) above the two boxes.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups, labels = [], []
    for name, g in df.groupby(group_col):
        vals = g[value_col].dropna().values
        if len(vals) > 0:
            groups.append(vals)
            labels.append(str(name))
    if len(groups) != 2:
        return

    method = result.get("method", "")
    p = result.get("p_value", np.nan)

    # Pick the most prominent effect size available
    es = result.get("effect_size", {})
    es_label = ""
    for key in ("hedges_g", "cohens_d", "rank_biserial"):
        if key in es and es[key] is not None:
            val = es[key]
            if isinstance(val, (int, float)) and not np.isnan(val):
                es_label = f"  |{key}| = {abs(val):.3f}"
                break

    fig, ax = plt.subplots(figsize=(6, 6))
    boxplot(ax, groups, labels=labels)

    # Significance bracket
    y_top = max(g.max() for g in groups)
    ylim = ax.get_ylim()
    bracket_y = y_top + (ylim[1] - ylim[0]) * 0.05
    _annotate_significance(ax, 1, 2, bracket_y, p)
    ax.set_ylim(ylim[0], bracket_y + (ylim[1] - ylim[0]) * 0.15)

    title = f"{value_col} by {group_col}"
    if node_label:
        title += f" — {node_label}"
    p_str = f"{p:.4f}" if p is not None and not np.isnan(p) else "n/a"
    ax.set_title(f"{title}\n{method}  p = {p_str}{es_label}", fontsize=10)

    fname = f"{value_col}_vs_{group_col}_two_group.png"
    save_fig(fig, output_dir / fname)




def save_one_way_comparison(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    result: dict,
    output_dir,
    *,
    node_label: Optional[str] = None,
) -> None:
    """
    One boxplot per group for a one-way comparison (ANOVA / Welch / Kruskal).

    Shows the test method and p-value in the title. For 3+ groups there is no
    single significance bracket — the title carries the omnibus result and
    post-hoc details should be stored separately as a CSV.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups, labels = [], []
    for name, g in df.groupby(group_col):
        vals = g[value_col].dropna().values
        if len(vals) > 0:
            groups.append(vals)
            labels.append(str(name))
    if len(groups) < 2:
        return

    method = result.get("method", "")
    p = result.get("p_value", np.nan)
    p_str = f"{p:.4f}" if p is not None and not np.isnan(p) else "n/a"

    title = f"{value_col} by {group_col}"
    if node_label:
        title += f" — {node_label}"

    fig, ax = plt.subplots(figsize=(max(6, len(groups) * 1.2), 6))
    boxplot(ax, groups, labels=labels,
            title=f"{title}\n{method}  p = {p_str}")

    # Two-group special case: add significance bracket
    if len(groups) == 2:
        y_top = max(g.max() for g in groups)
        ylim = ax.get_ylim()
        bracket_y = y_top + (ylim[1] - ylim[0]) * 0.05
        _annotate_significance(ax, 1, 2, bracket_y, p)
        ax.set_ylim(ylim[0], bracket_y + (ylim[1] - ylim[0]) * 0.15)

    fname = f"{value_col}_vs_{group_col}_oneway.png"
    save_fig(fig, output_dir / fname)


def save_group_comparisons_summary(
    comparison_df: pd.DataFrame,
    output_dir,
    *,
    alpha: float = 0.05,
    node_label: Optional[str] = None,
) -> None:
    """
    Horizontal bar chart of effect sizes from a group-comparison summary table.

    `comparison_df` matches the shape written to `comparisons_by_<outcome>.csv`
    in analyze.py: one row per (value_col, outcome_col) pair with columns
    method, statistic, p_value, effect_size_*.

    Bars are sorted by effect size descending. Significant comparisons
    (p_value < alpha) are highlighted with a distinct colour.
    """
    if comparison_df is None or comparison_df.empty:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Each row only has ONE populated effect_size_* column -- whichever metric
    # its auto-selected test produced (t-test rows get effect_size_cohens_d/
    # hedges_g, Mann-Whitney rows get effect_size_rank_biserial). Reading a
    # single column and dropping its NaNs would silently omit every row that
    # happened to use a different test/metric, so coalesce across all of them.
    es_cols = [c for c in comparison_df.columns if c.startswith("effect_size_")]
    if not es_cols or "value_col" not in comparison_df.columns:
        return

    plot_df = comparison_df.copy()
    plot_df["_effect_size"] = plot_df[es_cols].bfill(axis=1).iloc[:, 0]
    plot_df = plot_df.dropna(subset=["_effect_size"]).copy()
    plot_df["_es_abs"] = plot_df["_effect_size"].abs()
    plot_df = plot_df.sort_values("_es_abs", ascending=True)

    colors = [
        PALETTE[0] if row["p_value"] < alpha else PALETTE[6]
        for _, row in plot_df.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(10, max(5, len(plot_df) * 0.4)))
    ax.barh(plot_df["value_col"].astype(str), plot_df["_es_abs"], color=colors)
    ax.set_xlabel("Effect Size (absolute)")

    title = "Group Comparison Effect Sizes"
    if "outcome_col" in plot_df.columns and not plot_df["outcome_col"].empty:
        outcome = plot_df["outcome_col"].iloc[0]
        title += f" — outcome: {outcome}"
    if node_label:
        title += f"  ({node_label})"
    ax.set_title(title)

    # Legend
    sig_patch = mpatches.Patch(color=PALETTE[0], label=f"p < {alpha}")
    ns_patch = mpatches.Patch(color=PALETTE[6], label=f"p ≥ {alpha}")
    ax.legend(handles=[sig_patch, ns_patch], loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    save_fig(fig, output_dir / "group_comparisons_summary.png")


def _posthoc_to_pvalue_matrix(posthoc_df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Normalise three possible posthoc_test output shapes into a square symmetric p-value matrix."""
    if posthoc_df.index.equals(posthoc_df.columns):
        return posthoc_df.astype(float)
    if {"A", "B", "pval"}.issubset(posthoc_df.columns):
        groups = sorted(set(posthoc_df["A"]) | set(posthoc_df["B"]))
        mat = pd.DataFrame(index=groups, columns=groups, dtype=float)
        for _, row in posthoc_df.iterrows():
            mat.loc[row["A"], row["B"]] = row["pval"]
            mat.loc[row["B"], row["A"]] = row["pval"]
        return mat
    if {"group1", "group2", "p-unc"}.issubset(posthoc_df.columns):
        groups = sorted(set(posthoc_df["group1"]) | set(posthoc_df["group2"]))
        mat = pd.DataFrame(index=groups, columns=groups, dtype=float)
        for _, row in posthoc_df.iterrows():
            mat.loc[row["group1"], row["group2"]] = row["p-unc"]
            mat.loc[row["group2"], row["group1"]] = row["p-unc"]
        return mat
    return posthoc_df.astype(float)


def save_posthoc_heatmap(
    posthoc_df: pd.DataFrame,
    method: str,
    value_col: str,
    group_col: str,
    output_dir,
    *,
    node_label: Optional[str] = None,
) -> None:
    """Save a heatmap of pairwise post-hoc p-values for a significant one-way comparison."""
    if posthoc_df is None or posthoc_df.empty:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mat = _posthoc_to_pvalue_matrix(posthoc_df, method)
    if mat is None or mat.empty:
        return
    test_label = {
        "kruskal": "Dunn (Holm-adjusted)",
        "welch": "Games-Howell",
        "anova": "Tukey HSD",
    }.get(method, method)
    n = len(mat)
    size = max(4, n * 0.9)
    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    title = f"Post-hoc p-values: {value_col} by {group_col}\n{test_label}"
    if node_label:
        title += f"  ({node_label})"
    heatmap(ax, mat, cmap="Blues_r", vmin=0, vmax=1, annotate=True, fmt=".3f", title=title)
    ax.set_xlabel(group_col)
    ax.set_ylabel(group_col)
    save_fig(fig, output_dir / f"posthoc_{value_col}_vs_{group_col}.png")


def save_association_screening(
    screening_df: pd.DataFrame,
    output_dir,
    *,
    top_n: int = 30,
    node_label: Optional[str] = None,
) -> None:
    """
    Two-panel figure from the association screening DataFrame:
      Left  — effect-size bar chart for significant pairs (top_n by effect size)
      Right — effect size vs. –log10(p_adj) scatter (volcano-style)

    `screening_df` is the DataFrame returned by `screen_associations`.
    If no significant associations exist, only the volcano panel is drawn.
    """
    if screening_df is None or screening_df.empty:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sig = screening_df[screening_df["significant"] == True].copy()
    sig = sig.sort_values("effect_size", ascending=False).head(top_n)

    has_sig = not sig.empty
    ncols = 2 if has_sig else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, max(5, len(sig) * 0.35 + 4)))

    # --- Volcano panel (always drawn) ---
    # plt.subplots(1,1) returns a bare Axes, not an array; plt.subplots(1,2) returns array.
    volcano_ax = axes[1] if has_sig else axes
    valid = screening_df[["p_adj", "effect_size"]].dropna()
    if not valid.empty:
        log_p = -np.log10(valid["p_adj"].clip(lower=1e-30))
        colors_v = [
            PALETTE[0] if s else PALETTE[6]
            for s in screening_df.loc[valid.index, "significant"]
        ]
        volcano_ax.scatter(valid["effect_size"], log_p, c=colors_v, alpha=0.7, s=25)
        volcano_ax.axhline(-np.log10(0.05), color="gray", linestyle="--", linewidth=0.8,
                           label="p_adj = 0.05")
        volcano_ax.set_xlabel("Effect Size")
        volcano_ax.set_ylabel("−log₁₀(p_adj)")
        title = "Association Screening — Volcano"
        if node_label:
            title += f"  ({node_label})"
        volcano_ax.set_title(title)
        volcano_ax.legend(fontsize=8)
        volcano_ax.grid(alpha=0.3)

    # --- Effect-size bar chart for significant pairs ---
    if has_sig:
        bar_ax = axes[0]
        labels = [f"{r.var1} × {r.var2}" for r in sig.itertuples()]
        vals = sig["effect_size"].values
        # sig is already truncated to top_n above -- bar_chart's own default
        # cap (20) would otherwise silently re-truncate below top_n (default
        # 30), hiding real significant associations the title claims are shown.
        bar_chart(
            bar_ax, labels, list(vals),
            horizontal=True,
            title=f"Significant Associations (top {len(sig)})",
            xlabel="Effect Size",
            max_n=len(sig),
        )

    fig.tight_layout()
    save_fig(fig, output_dir / "association_screening.png")


# ===========================================================================
# Correlation section
# ===========================================================================



def save_correlation_matrix(
    screening_df: pd.DataFrame,
    output_dir,
    *,
    node_label: Optional[str] = None,
) -> None:
    """
    Heatmap of pairwise correlations found in the screening DataFrame.

    Only num-num pairs (pair_type == "num-num") are used. Variables are
    extracted from the var1 / var2 columns; missing pairs default to NaN.
    Requires at least 2 distinct variables to draw.
    """
    if screening_df is None or screening_df.empty:
        return
    num_pairs = screening_df[screening_df["pair_type"] == "num-num"].copy()
    if num_pairs.empty:
        return

    vars_ = sorted(set(num_pairs["var1"]) | set(num_pairs["var2"]))
    if len(vars_) < 2:
        return

    matrix = pd.DataFrame(np.nan, index=vars_, columns=vars_)
    np.fill_diagonal(matrix.values, 1.0)
    for _, row in num_pairs.iterrows():
        v = row.get("statistic", np.nan)
        matrix.loc[row["var1"], row["var2"]] = v
        matrix.loc[row["var2"], row["var1"]] = v

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    size = max(6, len(vars_) * 0.7)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    heatmap(
        ax, matrix,
        cmap="coolwarm", vmin=-1, vmax=1,
        annotate=True, fmt=".2f",
        title="Pairwise Correlations (from screening)"
        + (f" — {node_label}" if node_label else ""),
    )
    save_fig(fig, output_dir / "correlation_matrix.png")


# ===========================================================================
# Chi-square / categorical association section
# ===========================================================================





# ===========================================================================
# Regression section
# ===========================================================================







# ===========================================================================
# Time-series / spectral section
# ===========================================================================

def peak_annotation_inferential(x, y, k, ax=None, min_height=None, fft_labels=True):
    """
    Annotate the top-k peaks on an existing axes.

    Moved from inferential_analysis.py to keep that module free of plotting
    code. Signature is unchanged so existing callers are not broken.
    """
    from scipy.signal import find_peaks as _find_peaks

    if ax is None:
        ax = plt.gca()
    x = np.asarray(x)
    y = np.asarray(y)

    peaks, _ = _find_peaks(y, height=min_height)
    if len(peaks) == 0:
        peaks = [np.argmax(y)]

    if k == 1:
        peaks = [np.argmax(y)]
    else:
        peaks, _ = _find_peaks(y, height=min_height)
        if len(peaks) > 0:
            peak_heights = y[peaks]
            sorted_idx = np.argsort(peak_heights)[::-1]
            peaks = peaks[sorted_idx[:min(k, len(sorted_idx))]]
        else:
            peaks = [np.argmax(y)]

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(arrowstyle="->")
    for peak in peaks:
        xmax, ymax = x[peak], y[peak]
        text = (
            f"freq={xmax:.3f}\npower={ymax:.3f}"
            if fft_labels
            else f"x={xmax:.3f}\ny={ymax:.3f}"
        )
        if k == 1:
            ax.annotate(text, xy=(xmax, ymax), xytext=(0.94, 0.96),
                        textcoords="axes fraction", bbox=bbox_props,
                        arrowprops=arrowprops, ha="right", va="top")
        else:
            ax.annotate(text, xy=(xmax, ymax), xytext=(20, 20),
                        textcoords="offset points", bbox=bbox_props,
                        arrowprops=arrowprops)





# ============================================================================
# Source: section_definitions.py
# ============================================================================

"""Declarative configuration shared by generate_local_report.py and
generate_global_report.py.

This holds the small pieces of per-subsection metadata that would
otherwise be duplicated/hardcoded across the two report builders
(directory names, short-mode plot choices, narrative labels). The
report-building logic itself lives in report_utils.py and the two
generate_*_report.py modules.
"""



@dataclass
class ReductionSubsection:
    """PCA / MCA subsection.

    Short mode: a single summary plot (`short_plot`).
    Full mode: every plot in `subdir`.
    """
    title: str
    subdir: str        # relative to node_dir, e.g. "pca"
    short_plot: str    # e.g. "pca_explained_variance.png"


# short_plot deliberately differs from the modular project's section_definitions.py
# (which uses "pca_overview.png"/"mca_overview.png"): this file's live aggregator
# path never generates a combined overview panel, only individual plots, so
# pointing short mode at the overview filename would always come up empty.
LOCAL_PCA = ReductionSubsection(title="PCA", subdir="pca", short_plot="pca_explained_variance.png")
LOCAL_MCA = ReductionSubsection(title="MCA", subdir="mca", short_plot="mca_explained_inertia.png")

# Short-mode table truncation
SHORT_TABLE_MAX_ROWS = 10

# Short-mode: number of per-feature plots (e.g. temporal activity charts) shown
SHORT_PLOT_MAX = 5



# ============================================================================
# Source: report_utils.py
# ============================================================================

"""Shared helpers for local and global report generation.

Layered like the rest of the reporting stack: low-level ReportLab/Pillow
helpers (tables, images, narrative boxes) at the bottom, comparison-column
and ranking helpers in the middle, and privacy-notice / narrative-summary
helpers at the top. Report builders (`generate_local_report.py`,
`generate_global_report.py`) compose these via `section_definitions.py`.
"""



STYLES = None  # populated after lazy load; accessed only inside report functions
MAX_W = MAX_H = PAGE_MARGIN = None  # populated after lazy load (require inch)
TABLE_FONT_SIZE = 7
TABLE_MIN_FONT_SIZE = 5

SMALL_GROUP_THRESHOLD = 5
ID_KEYWORDS = ["id", "identifier", "patient_id", "identifikator"]


# ---------------------------------------------------------------------------
# Header / footer
# ---------------------------------------------------------------------------

def make_header(section_title):
    """Return a ReportLab canvas callback that draws a page header, footer, and generation timestamp."""
    def header(canvas, doc):
        """ReportLab canvas callback; draws section title, page number, and generation date."""
        canvas.saveState()
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        canvas.drawString(40, 20, f"Generated: {date_str}")
        canvas.drawRightString(doc.pagesize[0] - 40, 20, f"Page {canvas.getPageNumber()}")
        canvas.drawString(40, doc.pagesize[1] - 30, section_title)
        canvas.restoreState()
    return header


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def add_section_heading(elements, title, level=1):
    """Append a Heading1 or Heading2 paragraph to a ReportLab elements list."""
    style = STYLES["Heading1"] if level == 1 else STYLES["Heading2"]
    elements.append(Paragraph(title, style))
    elements.append(Spacer(1, 10))


def auto_fit_image(path, max_width, max_height):
    """Largest size that fits within max_width x max_height, preserving aspect ratio."""
    try:
        with PILImage.open(path) as img:
            w, h = img.size
    except Exception:
        return max_width, max_height

    aspect = w / h if h else 1
    width, height = max_width, max_width / aspect
    if height > max_height:
        height = max_height
        width = max_height * aspect
    return width, height


def add_figure(elements, path, max_width=None, max_height=None, caption=None):
    """Append a figure from disk to elements, auto-fitting it within max_width × max_height."""
    if max_width is None: max_width = 6.5 * inch
    if max_height is None: max_height = 4 * inch
    path = Path(path)
    if not path.exists():
        return
    width, height = auto_fit_image(path, max_width, max_height)
    title = path.stem.replace("_", " ").title()
    group = [
        Paragraph(title, STYLES["Heading3"]),
        Spacer(1, 6),
        Image(str(path), width=width, height=height),
    ]
    if caption:
        group.append(Spacer(1, 4))
        group.append(Paragraph(f"<i>{caption}</i>", STYLES["BodyText"]))
    group.append(Spacer(1, 14))
    elements.append(KeepTogether(group))


def add_plots_from_dir(elements, directory, max_width=None, max_height=None,
                        only=None, exclude=None):
    """Append all PNG files from a directory to elements, with optional name-based filtering."""
    if max_width is None: max_width = 6.5 * inch
    if max_height is None: max_height = 4 * inch
    directory = Path(directory)
    if not directory.exists():
        return
    files = sorted(directory.glob("*.png"))
    if only is not None:
        files = [f for f in files if f.name in only]
    if exclude is not None:
        files = [f for f in files if f.name not in exclude]
    for f in files:
        add_figure(elements, f, max_width, max_height)


def add_heading_and_plots(elements, title, paths, level=1,
                           max_width=None, max_height=None):
    """
    Append a section heading followed by one or more figures to elements.

      Keeps the heading and the first figure together on the same page to avoid
      an orphaned heading at the bottom of a page.  Returns True if any figure
      was added.
    """
    if max_width is None: max_width = 6.5 * inch
    if max_height is None: max_height = 4 * inch
    """Section heading followed by figure(s), keeping the heading and the
    first figure on the same page (avoids an orphaned heading at the
    bottom of a page with its figure starting on the next).

    `paths` may contain non-existent files; only existing ones are shown.
    Returns True if at least one figure was added.
    """
    paths = [Path(p) for p in paths if Path(p).exists()]
    if not paths:
        add_section_heading(elements, title, level=level)
        return False

    group = []
    add_section_heading(group, title, level=level)
    add_figure(group, paths[0], max_width, max_height)
    elements.append(KeepTogether(group))
    for p in paths[1:]:
        add_figure(elements, p, max_width, max_height)
    return True


def _longest_word_widths(df, font_size, padding=6):
    """For each column, the render width (at `font_size`) of its longest
    unsplittable "word" (whitespace-separated token) across the header and
    all cells, plus cell padding."""
    widths = []
    for col in df.columns:
        cells = [str(col)] + [str(v) for v in df[col]]
        longest = max(
            (stringWidth(tok, "Helvetica", font_size) for c in cells for tok in c.split()),
            default=0,
        )
        widths.append(longest + padding)
    return widths


def create_table(df, available_width, max_rows=None):
    """DataFrame -> ReportLab Table, column widths proportional to content length.

    Each column is also given a minimum width wide enough to fit its longest
    unsplittable "word" (whitespace-separated token), so ReportLab doesn't
    fall back to breaking long values mid-word/hyphenating to make them fit.
    The font size shrinks (down to TABLE_MIN_FONT_SIZE) for tables with many
    or wide columns, where TABLE_FONT_SIZE would make those minimums exceed
    the available width.
    """
    if max_rows is not None:
        df = df.head(max_rows)

    # Format p-value columns before general float rounding
    _PVALUE_COLS = {"p_value", "p_adj", "p-val", "p-unc", "pval"}
    df = df.copy()
    for col in df.columns:
        if str(col).lower() in _PVALUE_COLS and pd.api.types.is_float_dtype(df[col]):
            def _fmt_p(v):
                if pd.isna(v):
                    return ""
                f = float(v)
                if f == 0.0:
                    return "0"
                if f < 0.001:
                    return "< 0.001"
                return f"{f:.3f}"
            df[col] = df[col].apply(_fmt_p)

    float_cols = df.select_dtypes(include="float").columns
    df[float_cols] = df[float_cols].round(3)

    font_size = TABLE_FONT_SIZE
    min_widths = _longest_word_widths(df, font_size)
    while sum(min_widths) > available_width and font_size > TABLE_MIN_FONT_SIZE:
        font_size -= 0.5
        min_widths = _longest_word_widths(df, font_size)

    raw_widths = [
        max([len(str(col))] + [len(str(v)) for v in df[col]])
        for col in df.columns
    ]
    total_raw = sum(raw_widths) or 1
    extra = max(0.0, available_width - sum(min_widths))
    col_widths = [m + extra * (rw / total_raw) for m, rw in zip(min_widths, raw_widths)]
    if sum(col_widths) > available_width:
        scale = available_width / sum(col_widths)
        col_widths = [w * scale for w in col_widths]

    cell_style = ParagraphStyle(
        "TableCell", parent=STYLES["BodyText"],
        fontSize=font_size, leading=font_size + 2,
    )
    _avail_readable = {
        "common_all": "common in all",
        "common_partial": "common in some",
        "unique_local": "unique to this node",
        "not_common_all": "not common in all",
        "unknown": "unknown",
    }
    _SYSTEM_VALUE_COLS = {"vs_global", "comparison", "test", "effect_size_metric"}

    def _cell_text(col_name, value):
        s = str(value)
        if str(col_name) == "availability":
            return _avail_readable.get(s, s.replace("_", " "))
        if str(col_name) in _SYSTEM_VALUE_COLS:
            return s.replace("_", " ")
        return s

    header = [Paragraph(str(col).replace("_", " "), cell_style) for col in df.columns]
    body = [
        [Paragraph(_cell_text(col, cell), cell_style)
         for col, cell in zip(df.columns, row)]
        for row in df.values
    ]
    table = Table([header] + body, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 0), (-1, -1), font_size),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
    ]))
    return table


# ---------------------------------------------------------------------------
# Narrative / warning system
# ---------------------------------------------------------------------------

@dataclass
class NarrativeMessage:
    """
    Typed callout box inserted into PDF reports.

      Attributes:
          level: Visual style — ``info`` (blue), ``warning`` (amber), or
              ``insight`` (green).
          text: Message body; may contain basic ReportLab XML markup.
    """
    level: Literal["info", "warning", "insight"]
    text: str


_NARRATIVE_STYLE = None  # populated on first use by _get_narrative_style()


def _get_narrative_style():
    """Build _NARRATIVE_STYLE on first call (after lazy deps are loaded)."""
    global _NARRATIVE_STYLE
    if _NARRATIVE_STYLE is None:
        _NARRATIVE_STYLE = {
            "info":    {"bg": colors.HexColor("#EAF2FB"), "border": colors.HexColor("#5B9BD5"), "label": "Note"},
            "warning": {"bg": colors.HexColor("#FDF3E7"), "border": colors.HexColor("#E0A458"), "label": "Warning"},
            "insight": {"bg": colors.HexColor("#EAF7EC"), "border": colors.HexColor("#6FBF73"), "label": "Insight"},
        }
    return _NARRATIVE_STYLE


def render_narrative(elements, msg: NarrativeMessage):
    """Append a styled NarrativeMessage callout box (coloured border and background) to elements."""
    style = _get_narrative_style()[msg.level]
    p_style = ParagraphStyle(
        f"Narrative_{msg.level}",
        parent=STYLES["BodyText"],
        backColor=style["bg"],
        borderColor=style["border"],
        borderWidth=1,
        borderPadding=6,
        spaceBefore=4,
        spaceAfter=4,
    )
    elements.append(Paragraph(f"<b>{style['label']}:</b> {msg.text}", p_style))
    elements.append(Spacer(1, 8))


def truncation_note(shown, total, criterion, csv_filename):
    """Build an info NarrativeMessage noting that a table was truncated to the top-N rows."""
    return NarrativeMessage(
        level="info",
        text=(
            f"Showing the {shown} most {criterion} of {total} total variables "
            f"(ranked by {criterion}). Full results: <code>{csv_filename}</code>."
        ),
    )


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def safe_read_csv(path) -> Optional[pd.DataFrame]:
    """Read a CSV into a DataFrame; return None on missing file, read error, or empty result."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    return df if not df.empty else None


def drop_internal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns whose names start with '_' (internal annotation columns added by comparison helpers)."""
    return df.loc[:, [c for c in df.columns if not str(c).startswith("_")]]


def _format_range_value(value) -> str:
    """Format a numeric range endpoint as an integer string when whole, otherwise 3 significant figures."""
    if pd.isna(value):
        return "n/a"
    value = float(value)
    return str(int(value)) if value.is_integer() else f"{value:.3g}"


def prepare_numeric_display(df: pd.DataFrame) -> pd.DataFrame:
    """Curate a numeric summary table for display: drop skewness/kurtosis
    (not actionable for a general audience) and collapse min/max into a
    single 'range' column ([min, max]) to save horizontal space."""
    df = df.drop(columns=["skewness", "kurtosis"], errors="ignore")
    if "min" in df.columns and "max" in df.columns:
        df = df.copy()
        ranges = df.apply(
            lambda r: f"[{_format_range_value(r['min'])}, {_format_range_value(r['max'])}]",
            axis=1,
        )
        min_pos = df.columns.get_loc("min")
        df = df.drop(columns=["min", "max"])
        df.insert(min_pos, "range", ranges)
    return df


# ---------------------------------------------------------------------------
# Comparison columns (computed at report-generation time)
# ---------------------------------------------------------------------------

def add_comparison_column(local_df, global_df, key_col, value_col, threshold=0.1,
                           global_value_col=None):
    """Join local_df against global_df on key_col and label each row
    'above_average' / 'below_average' / 'similar' / 'n/a' based on the
    relative difference of value_col vs. the federated value.
    """
    global_value_col = global_value_col or value_col
    global_slim = global_df[[key_col, global_value_col]].rename(
        columns={global_value_col: f"{value_col}_global"}
    )
    merged = local_df.merge(global_slim, on=key_col, how="left")

    global_vals = merged[f"{value_col}_global"].replace(0, np.nan)
    rel_diff = (merged[value_col] - merged[f"{value_col}_global"]) / global_vals

    merged["vs_global"] = np.select(
        [rel_diff > threshold, rel_diff < -threshold],
        ["above_average", "below_average"],
        default="similar",
    )
    merged.loc[merged[f"{value_col}_global"].isna(), "vs_global"] = "n/a"
    merged["_rel_diff_abs"] = rel_diff.abs()
    return merged


def add_numeric_comparison(local_df, global_df, threshold=0.1):
    """Add a vs_global comparison label to a local numeric summary DataFrame by joining against the federated stats."""
    return add_comparison_column(local_df, global_df, key_col="feature",
                                   value_col="mean", threshold=threshold)


def _parse_dict(value):
    """Parse a stringified dict cell value using ast.literal_eval; return {} on failure."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError, TypeError):
        return {}


def add_categorical_comparison(local_df, global_df, threshold=0.1):
    """Compares each feature's local top-category share against the global
    share of that *same* category (apples-to-apples)."""
    global_lookup = {}
    for _, grow in global_df.iterrows():
        global_lookup[grow["feature"]] = _parse_dict(grow.get("relative_freq", "{}"))

    local = local_df.copy()
    local_shares, global_shares, vs_global, rel_diffs = [], [], [], []
    for _, row in local.iterrows():
        local_freqs = _parse_dict(row.get("relative_frequencies", "{}"))
        top_cat = row.get("most_frequent_category")
        local_share = local_freqs.get(top_cat, np.nan)

        g_freqs = global_lookup.get(row["feature"], {})
        global_share = g_freqs.get(top_cat, np.nan)
        if global_share is not np.nan and not pd.isna(global_share):
            global_share = global_share * 100  # global stored as fraction, local as %

        local_shares.append(local_share)
        global_shares.append(global_share)

        if pd.isna(local_share) or pd.isna(global_share) or global_share == 0:
            vs_global.append("n/a")
            rel_diffs.append(np.nan)
            continue

        rel_diff = (local_share - global_share) / global_share
        rel_diffs.append(abs(rel_diff))
        if rel_diff > threshold:
            vs_global.append("above_average")
        elif rel_diff < -threshold:
            vs_global.append("below_average")
        else:
            vs_global.append("similar")

    local["top cat % (local)"] = local_shares
    local["top cat % (global)"] = global_shares
    local["vs_global"] = vs_global
    local["_rel_diff_abs"] = rel_diffs
    return local


def add_temporal_comparison(local_df, global_df, threshold=0.1):
    """Compares each feature's local record count ('activity level') against
    the federated total derived from its counts_per_period."""
    global_copy = global_df.copy()

    def _total_count(value):
        """Return the total valid observation count by summing all category counts in a categorical stats row."""
        counts = _parse_dict(value)
        return sum(counts.values()) if counts else np.nan

    global_copy["count"] = global_copy["counts_per_period"].apply(_total_count)
    return add_comparison_column(local_df, global_copy, key_col="feature",
                                   value_col="count", threshold=threshold)


# ---------------------------------------------------------------------------
# Short-mode ranking
# ---------------------------------------------------------------------------

def rank_by_deviation(df, max_rows=10):
    """Numeric/categorical/temporal: rank by |local - global| relative deviation."""
    if "_rel_diff_abs" in df.columns:
        return df.sort_values("_rel_diff_abs", ascending=False, na_position="last").head(max_rows)
    return df.head(max_rows)


def rank_by_imbalance(df, max_rows=10):
    """Categorical: rank by class imbalance ratio (most skewed first)."""
    if "class_imbalance_ratio" in df.columns:
        return df.sort_values("class_imbalance_ratio", ascending=False, na_position="last").head(max_rows)
    return df.head(max_rows)


def rank_by_activity(df, max_rows=10):
    """Temporal: rank by record count (most complete/active first)."""
    if "count" in df.columns:
        return df.sort_values("count", ascending=False, na_position="last").head(max_rows)
    return df.head(max_rows)


def rank_by_effect_size(df, max_rows=10):
    """Inferential: rank by absolute effect size (strongest associations first)."""
    if "effect_size" in df.columns and not df.empty:
        order = df["effect_size"].abs().sort_values(ascending=False).index
        return df.reindex(order).head(max_rows)
    return df.head(max_rows)


# ---------------------------------------------------------------------------
# Narrative summarizers
# ---------------------------------------------------------------------------

def summarise_numeric(numeric_df) -> NarrativeMessage:
    """Build and append the numeric statistics section (table + plots) to a PDF elements list."""
    if numeric_df is None or numeric_df.empty:
        return NarrativeMessage(level="info", text="No numeric variables available.")
    text = f"{len(numeric_df)} numeric variable(s) analyzed."
    if "vs_global" in numeric_df.columns:
        n_above = int((numeric_df["vs_global"] == "above_average").sum())
        n_below = int((numeric_df["vs_global"] == "below_average").sum())
        text += (f" {n_above} above and {n_below} below the federated average "
                 f"(>10% relative difference).")
    return NarrativeMessage(level="info", text=text)


def summarise_categorical(categorical_df) -> NarrativeMessage:
    """Build and append the categorical statistics section (table + plots) to a PDF elements list."""
    if categorical_df is None or categorical_df.empty:
        return NarrativeMessage(level="info", text="No categorical variables available.")
    text = f"{len(categorical_df)} categorical variable(s) analyzed."
    if "vs_global" in categorical_df.columns:
        n_above = int((categorical_df["vs_global"] == "above_average").sum())
        n_below = int((categorical_df["vs_global"] == "below_average").sum())
        text += (f" {n_above} variable(s) have a more dominant top category than "
                 f"the federation average, {n_below} less dominant.")
    return NarrativeMessage(level="info", text=text)


def summarise_temporal(temporal_df) -> NarrativeMessage:
    """Build and append the temporal statistics section (table + plots) to a PDF elements list."""
    if temporal_df is None or temporal_df.empty:
        return NarrativeMessage(level="info", text="No temporal variables available.")
    return NarrativeMessage(level="info", text=f"{len(temporal_df)} temporal variable(s) analyzed.")


def summarise_inferential(significant_df, pair_type=None) -> NarrativeMessage:
    """Build and append the inferential statistics section (association table + plots) to a PDF elements list."""
    if significant_df is None or significant_df.empty:
        return NarrativeMessage(level="info", text="No statistically significant associations detected.")
    df = significant_df
    if pair_type is not None:
        df = df[df["pair_type"] == pair_type]
    if df.empty:
        return NarrativeMessage(level="info", text="No statistically significant associations detected.")
    top = df.reindex(df["effect_size"].abs().sort_values(ascending=False).index).iloc[0]
    text = (
        f"{len(df)} significant association(s) found (FDR-adjusted p < 0.05). "
        f"Strongest: {top['var1']} vs {top['var2']} "
        f"({top['test']}, {top['effect_size_metric']} = {top['effect_size']:.2f})."
    )
    return NarrativeMessage(level="insight", text=text)


# ---------------------------------------------------------------------------
# Privacy & data governance
# ---------------------------------------------------------------------------

def _looks_like_identifier(feature_name) -> bool:
    """Return True if a column name contains keywords that suggest it is an identifier."""
    name = str(feature_name).lower()
    return any(k in name for k in ID_KEYWORDS)


def detect_identifier_features(*dataframes) -> list:
    """Return the list of columns in df that look like identifier columns by name or high uniqueness."""
    found = []
    for df in dataframes:
        if df is None or "feature" not in df.columns:
            continue
        for f in df["feature"]:
            if _looks_like_identifier(f):
                found.append(f)
    return sorted(set(found))


def compute_categorical_group_sizes(categorical_df, threshold=SMALL_GROUP_THRESHOLD):
    """Returns (min_group_size, min_feature, min_category, flagged) where
    flagged is a list of (feature, category, size) below threshold."""
    min_size, min_feature, min_category = None, None, None
    flagged = []
    if categorical_df is None or categorical_df.empty:
        return min_size, min_feature, min_category, flagged

    for _, row in categorical_df.iterrows():
        freqs = _parse_dict(row.get("relative_frequencies", "{}"))
        count = row.get("count")
        if not freqs or count is None:
            continue
        for cat, pct in freqs.items():
            size = pct / 100 * count
            if min_size is None or size < min_size:
                min_size, min_feature, min_category = size, row["feature"], cat
            if size < threshold:
                flagged.append((row["feature"], cat, size))
    return min_size, min_feature, min_category, flagged


def categorical_small_group_warnings(categorical_df, threshold=SMALL_GROUP_THRESHOLD) -> list:
    """Generate NarrativeMessages for categorical columns whose smallest group falls below a minimum count."""
    _, _, _, flagged = compute_categorical_group_sizes(categorical_df, threshold)
    messages = []
    seen_features = set()
    for feature, cat, size in flagged:
        if feature in seen_features:
            continue
        seen_features.add(feature)
        messages.append(NarrativeMessage(
            level="warning",
            text=(
                f"\"{feature}\" contains a category (\"{cat}\") with an estimated "
                f"{size:.0f} individuals - below the reporting threshold of "
                f"{threshold}. Small groups, especially combined with other "
                f"displayed variables, may carry re-identification risk and "
                f"should be interpreted with caution."
            ),
        ))
    return messages


def categorical_excluded_from_distributions_notice(categorical_df) -> Optional[NarrativeMessage]:
    """Note which categorical columns have only 1 observed category and so have no distribution plot.

    save_categorical_distributions (and its hub equivalent) only plots columns
    with >= 2 distinct non-null values -- a column with a single observed
    category has nothing to show. Without this notice such a column would
    simply be absent from the distributions section with no explanation.
    """
    if categorical_df is None or categorical_df.empty or "number_of_categories" not in categorical_df.columns:
        return None
    single_valued = categorical_df.loc[categorical_df["number_of_categories"] <= 1, "feature"]
    if single_valued.empty:
        return None
    names = ", ".join(str(f) for f in single_valued)
    return NarrativeMessage(
        level="info",
        text=(
            f"The following categorical variable(s) have only a single observed "
            f"category in this dataset and are not shown as distribution plots: {names}."
        ),
    )


def quasi_numeric_categorical_notice(categorical_dir) -> Optional[NarrativeMessage]:
    """Note categorical columns where most values look numeric but a few couldn't be parsed.

    A column that is almost entirely numeric-looking (e.g. lab results) but
    contains a few censored values like "<5" fails numeric coercion and is
    classified as categorical -- without this notice that would look like an
    unexplained mis-classification in the report.
    """
    flagged_df = safe_read_csv(Path(categorical_dir) / "quasi_numeric_columns.csv")
    if flagged_df is None or "feature" not in flagged_df.columns:
        return None
    names = ", ".join(str(f) for f in flagged_df["feature"])
    return NarrativeMessage(
        level="info",
        text=(
            f"The following categorical variable(s) have mostly numeric-looking "
            f"values but could not be fully parsed as numbers (e.g. a censored "
            f"lab value like \"<5\"), so they are treated as categorical rather "
            f"than numeric: {names}."
        ),
    )


def out_of_range_age_notice(numeric_dir) -> Optional[NarrativeMessage]:
    """Note age values outside the plausible [0, 100] range excluded from the age histogram.

    compute_age_histogram bins ages into fixed 0-100 (step 5) bins -- a
    negative age or a data-entry typo like 999 falls outside every bin and
    is silently dropped from the plot with no indication anywhere why the
    counts don't match the full patient count.
    """
    flagged_df = safe_read_csv(Path(numeric_dir) / "age_out_of_range.csv")
    if flagged_df is None or "count" not in flagged_df.columns or flagged_df.empty:
        return None
    count = int(flagged_df["count"].iloc[0])
    if count <= 0:
        return None
    return NarrativeMessage(
        level="info",
        text=(
            f"{count} age value(s) fall outside the plausible 0-100 year range "
            f"and are excluded from the age distribution plot."
        ),
    )


def build_privacy_notice(report_type, n_nodes=None, numeric_df=None,
                          categorical_df=None, temporal_df=None,
                          threshold=SMALL_GROUP_THRESHOLD) -> list:
    """Returns a list of flowables for the title-page privacy/governance block."""
    elements = [Paragraph("Data & Privacy Notice", STYLES["Heading2"]), Spacer(1, 6)]
    lines = []

    if report_type == "local":
        lines.append(
            "This report is generated from this node's local data only. No "
            "row-level data was transmitted to other nodes or to a central server."
        )
    else:
        lines.append(
            f"This report contains only aggregated statistics computed across "
            f"{n_nodes if n_nodes else 'all participating'} nodes. No row-level "
            f"or patient-level data is shared or displayed."
        )
        lines.append(
            "Means, standard deviations, and category counts are computed via "
            "federated aggregation; trend slopes via federated OLS regression on "
            "per-period counts. Raw values are never transmitted between nodes."
        )

    identifiers = detect_identifier_features(numeric_df, categorical_df, temporal_df)
    if identifiers:
        lines.append(
            f"Identifier-like columns ({', '.join(identifiers)}) are excluded "
            f"from all tables and plots in this report."
        )
    else:
        lines.append(
            "Identifier columns (e.g. patient/record IDs) are excluded from all "
            "tables and plots in this report."
        )

    if report_type == "local" and categorical_df is not None and not categorical_df.empty:
        min_size, min_feature, min_category, flagged = compute_categorical_group_sizes(
            categorical_df, threshold
        )
        if min_size is not None:
            lines.append(
                f"Smallest displayed group size: {min_size:.0f} "
                f"(category \"{min_category}\" of \"{min_feature}\")."
            )
        n_flagged_features = len(set(f for f, _, _ in flagged))
        if n_flagged_features:
            lines.append(
                f"{n_flagged_features} of {len(categorical_df)} categorical "
                f"breakdowns contain at least one group below the reporting "
                f"threshold (k={threshold}) - see warnings in the relevant sections."
            )

    if report_type == "local" and n_nodes is not None and n_nodes < 5:
        lines.append(
            f"With only {n_nodes} participating node(s), 'above/below average' "
            f"comparisons against federated values may indirectly reveal "
            f"information about other individual nodes."
        )

    for line in lines:
        elements.append(Paragraph(line, STYLES["BodyText"]))
        elements.append(Spacer(1, 4))

    elements.append(Spacer(1, 12))
    return elements





# ============================================================================
# Source: generate_local_report.py
# ============================================================================

"""Local node report generator.

generate_local_report(node_dir, output_dir, mode="full") -> Path

Builds a per-node PDF report from the analysis outputs under
results/local_results/<node>/. Two modes:
  - "short": narrative summaries, key plots, ranked top-N tables
  - "full":  full per-variable detail
"""




# inch is lazy-loaded — use float literals and multiply inside functions.
_MAX_W_IN  = 6.8
_MAX_H_IN  = 4.0
_MARGIN_IN = 0.4


def generate_local_report(node_dir, output_dir, mode="full",
                           results_dir=None, export_comparison_csv=False) -> Path:
    """
    Generate a per-node PDF report from on-disk CSV and PNG outputs.

      Args:
          node_dir: Root directory of the node's analysis outputs.
          output_dir: Directory where the PDF will be written.
          mode: ``'short'`` or ``'full'``; controls which sections are included.
          results_dir: Optional path to the federated results directory.
          export_comparison_csv: Write per-column comparison CSVs alongside the PDF.

      Returns:
          Path to the written PDF file.
    """
    node_dir = Path(node_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    node_name = node_dir.name

    global_dir = Path(results_dir) if results_dir else node_dir.parent.parent / "federated_results"
    global_numeric = safe_read_csv(global_dir / "numeric" / "federated_numeric_statistics.csv")
    global_categorical = safe_read_csv(global_dir / "categorical" / "federated_categorical_statistics.csv")
    global_temporal = safe_read_csv(global_dir / "temporal" / "federated_temporal_statistics.csv")
    n_nodes = _extract_n_nodes(safe_read_csv(global_dir / "overview" / "overview.csv"))

    overview_df = safe_read_csv(node_dir / "overview" / "overview.csv")
    numeric_df = safe_read_csv(node_dir / "numeric" / "numeric_summary.csv")
    categorical_df = safe_read_csv(node_dir / "categorical" / "categorical_summary.csv")
    temporal_df = safe_read_csv(node_dir / "temporal" / "temporal_summary.csv")
    significant_df = safe_read_csv(node_dir / "inferential" / "significant_associations.csv")

    if numeric_df is not None and global_numeric is not None:
        numeric_df = add_numeric_comparison(numeric_df, global_numeric)
    if categorical_df is not None and global_categorical is not None:
        categorical_df = add_categorical_comparison(categorical_df, global_categorical)
    if temporal_df is not None and global_temporal is not None:
        temporal_df = add_temporal_comparison(temporal_df, global_temporal)

    if export_comparison_csv:
        _export_comparison_csvs(node_dir, numeric_df, categorical_df, temporal_df)

    output_path = output_dir / f"local_report_{node_name}_{mode}.pdf"
    doc = BaseDocTemplate(str(output_path), leftMargin=PAGE_MARGIN, rightMargin=PAGE_MARGIN)
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([
        PageTemplate(id="title", frames=[frame], onPage=make_header(f"Local Report - {node_name}")),
        PageTemplate(id="overview", frames=[frame], onPage=make_header("Overview")),
        PageTemplate(id="numeric", frames=[frame], onPage=make_header("Numeric Section")),
        PageTemplate(id="categorical", frames=[frame], onPage=make_header("Categorical Section")),
        PageTemplate(id="temporal", frames=[frame], onPage=make_header("Temporal Section")),
        PageTemplate(id="cross", frames=[frame], onPage=make_header("Cross-Variable Associations")),
    ])

    elements = []
    _build_title_page(elements, node_name, mode, n_nodes, numeric_df, categorical_df, temporal_df)

    elements.append(NextPageTemplate("overview"))
    elements.append(PageBreak())
    _build_overview_section(elements, doc, node_dir, overview_df, mode=mode)

    elements.append(NextPageTemplate("numeric"))
    elements.append(PageBreak())
    _build_numeric_section(elements, doc, node_dir, mode, numeric_df, significant_df)

    elements.append(NextPageTemplate("categorical"))
    elements.append(PageBreak())
    _build_categorical_section(elements, doc, node_dir, mode, categorical_df, significant_df)

    elements.append(NextPageTemplate("temporal"))
    elements.append(PageBreak())
    _build_temporal_section(elements, doc, node_dir, mode, temporal_df)

    elements.append(NextPageTemplate("cross"))
    elements.append(PageBreak())
    _build_cross_variable_section(elements, doc, node_dir, mode, significant_df)

    elements.append(PageBreak())
    _build_glossary_section(elements)

    doc.build(elements)
    return output_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_n_nodes(global_overview_df):
    """Extract the number of federation nodes from the global overview CSV; return None if not found."""
    if global_overview_df is None:
        return None
    matches = global_overview_df[global_overview_df["metric"].str.contains("hospital", case=False, na=False)]
    if matches.empty:
        return None
    try:
        return int(matches.iloc[0]["value"])
    except (ValueError, TypeError):
        return None


def _export_comparison_csvs(node_dir, numeric_df, categorical_df, temporal_df):
    """Write the comparison-enhanced numeric/categorical/temporal DataFrames to CSV files in node_dir."""
    if numeric_df is not None:
        drop_internal_columns(numeric_df).to_csv(
            node_dir / "numeric" / "numeric_summary_with_comparison.csv", index=False)
    if categorical_df is not None:
        drop_internal_columns(categorical_df).to_csv(
            node_dir / "categorical" / "categorical_summary_with_comparison.csv", index=False)
    if temporal_df is not None:
        drop_internal_columns(temporal_df).to_csv(
            node_dir / "temporal" / "temporal_summary_with_comparison.csv", index=False)


# ---------------------------------------------------------------------------
# Title page
# ---------------------------------------------------------------------------

def _build_title_page(elements, node_name, mode, n_nodes, numeric_df, categorical_df, temporal_df):
    """Build the title page and summary statistics elements for a local PDF report."""
    elements.append(Paragraph(f"Local Statistical Report - {node_name}", STYLES["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Report type: {mode.title()}", STYLES["Normal"]))
    now = datetime.now().strftime("%d %B %Y, %H:%M")
    elements.append(Paragraph(f"Generated on: {now}", STYLES["Normal"]))
    elements.append(Spacer(1, 20))
    elements.extend(build_privacy_notice(
        report_type="local", n_nodes=n_nodes,
        numeric_df=numeric_df, categorical_df=categorical_df, temporal_df=temporal_df,
    ))


# ---------------------------------------------------------------------------
# 1. Overview
# ---------------------------------------------------------------------------

def _build_overview_section(elements, doc, node_dir, overview_df, *, mode="full"):
    """Build the overview and data quality section elements for a local PDF report."""
    add_section_heading(elements, "1. Overview")
    overview_dir = node_dir / "overview"

    if overview_df is None:
        render_narrative(elements, NarrativeMessage("info", "No overview data available."))
    else:
        elements.append(create_table(overview_df, doc.width))
        elements.append(Spacer(1, 14))

    add_figure(elements, overview_dir / "data_type_distribution.png", MAX_W, MAX_H)

    missing_by_col_paths = sorted(
        p for p in overview_dir.glob("missing_values_by_column*.png")
        if not p.name.endswith("_short.png")
    )
    if mode == "short":
        quality_plots = [overview_dir / "missing_values_by_column_short.png"]
    else:
        quality_plots = [*missing_by_col_paths, overview_dir / "missingno_heatmap.png"]
    has_quality_plots = add_heading_and_plots(elements, "1.2 Data Quality", quality_plots,
                                               level=2, max_width=MAX_W, max_height=MAX_H)
    if not has_quality_plots:
        render_narrative(elements, NarrativeMessage(
            "info",
            "All columns are fully complete for this node — "
            "no missing-value patterns to display.",
        ))
    elif mode == "full" and not (overview_dir / "missingno_heatmap.png").exists():
        render_narrative(elements, NarrativeMessage(
            "info",
            "Nullity correlation heatmap is not available "
            "(no columns have sufficient missing data).",
        ))

    availability_chart = node_dir / "comparison" / "column_availability.png"
    if availability_chart.exists():
        add_heading_and_plots(elements, "1.3 Column Availability Across Nodes",
                               [availability_chart], level=2, max_width=MAX_W, max_height=MAX_H)
        render_narrative(elements, NarrativeMessage(
            "info",
            "Per-column availability (common to all nodes / common in some nodes / "
            "unique to this node) is also shown in the 'availability' column "
            "of each descriptive table below.",
        ))


# ---------------------------------------------------------------------------
# 2. Numeric
# ---------------------------------------------------------------------------

def _build_numeric_section(elements, doc, node_dir, mode, numeric_df, significant_df):
    """Build the numeric statistics section elements for a local PDF report."""
    add_section_heading(elements, "2. Numeric Section")
    render_narrative(elements, NarrativeMessage(
        "info",
        "Numeric (continuous) variables — e.g. age, blood pressure, lab values. "
        "The summary table gives the mean, median, standard deviation, interquartile range "
        "and outlier count per feature. PCA reduces all numeric features to a small set of "
        "dimensions capturing the main directions of variation across patients.",
    ))
    numeric_dir = node_dir / "numeric"

    add_section_heading(elements, "2.1 Descriptive Statistics", level=2)
    if numeric_df is None:
        render_narrative(elements, NarrativeMessage("info", "No numeric variables available."))
    else:
        render_narrative(elements, summarise_numeric(numeric_df))
        display_df = drop_internal_columns(numeric_df)
        if mode == "short" and len(numeric_df) > SHORT_TABLE_MAX_ROWS:
            ranked = rank_by_deviation(numeric_df, SHORT_TABLE_MAX_ROWS)
            render_narrative(elements, truncation_note(
                len(ranked), len(numeric_df),
                "deviating from the federated average", "numeric_summary.csv",
            ))
            display_df = drop_internal_columns(ranked)
        display_df = prepare_numeric_display(display_df)
        elements.append(create_table(display_df, doc.width))
        elements.append(Spacer(1, 14))

    _dist_plots = [numeric_dir / "age_distribution.png"]
    _dist_plots += sorted(numeric_dir.glob("numeric_histograms_*.png"))
    _dist_plots += sorted(numeric_dir.glob("numeric_boxplots_*.png"))
    add_heading_and_plots(elements, "2.2 Distributions", _dist_plots,
                           level=2, max_width=MAX_W, max_height=MAX_H)
    age_range_notice = out_of_range_age_notice(numeric_dir)
    if age_range_notice is not None:
        render_narrative(elements, age_range_notice)

    _render_reduction_subsection(elements, node_dir, mode, LOCAL_PCA, level=2, prefix="2.3")

    add_section_heading(elements, "2.4 Correlations", level=2)
    _render_pairwise_table(elements, doc, significant_df, pair_type="num-num", mode=mode)


# ---------------------------------------------------------------------------
# 3. Categorical
# ---------------------------------------------------------------------------

def _build_categorical_section(elements, doc, node_dir, mode, categorical_df, significant_df):
    """Build the categorical statistics section elements for a local PDF report."""
    add_section_heading(elements, "3. Categorical Section")
    render_narrative(elements, NarrativeMessage(
        "info",
        "Categorical variables take values from a fixed set of categories "
        "(e.g. sex, diagnosis codes, yes/no flags). "
        "The summary table shows the count of valid observations, number of distinct categories, "
        "most and least frequent category, class imbalance ratio, and number of missing values. "
        "\"top cat % (local)\" is the share of local patients in the most frequent category; "
        "\"top cat % (global)\" is the same category's share in the federation — a large "
        "difference flags a representativeness concern. "
        "MCA maps category levels into a low-dimensional space to show which categories "
        "tend to co-occur across patients.",
    ))
    categorical_dir = node_dir / "categorical"

    add_section_heading(elements, "3.1 Descriptive Statistics", level=2)
    if categorical_df is None:
        render_narrative(elements, NarrativeMessage("info", "No categorical variables available."))
    else:
        render_narrative(elements, summarise_categorical(categorical_df))
        for warning in categorical_small_group_warnings(categorical_df):
            render_narrative(elements, warning)

        display_df = drop_internal_columns(categorical_df)
        if mode == "short" and len(categorical_df) > SHORT_TABLE_MAX_ROWS:
            ranked = rank_by_imbalance(categorical_df, SHORT_TABLE_MAX_ROWS)
            render_narrative(elements, truncation_note(
                len(ranked), len(categorical_df), "imbalanced", "categorical_summary.csv",
            ))
            display_df = drop_internal_columns(ranked)
        display_df = display_df.drop(columns=["relative_frequencies"], errors="ignore")
        elements.append(create_table(display_df, doc.width))
        elements.append(Spacer(1, 14))

    quasi_numeric_notice = quasi_numeric_categorical_notice(categorical_dir)
    if quasi_numeric_notice is not None:
        render_narrative(elements, quasi_numeric_notice)

    dist_paths = [categorical_dir / "sex_distribution.png"]
    dist_paths += sorted(categorical_dir.glob("categorical_distributions_*.png"))
    if not any(p.exists() for p in dist_paths):
        dist_paths = [categorical_dir / "categorical_distributions.png"]
    add_heading_and_plots(elements, "3.2 Distributions", dist_paths, level=2,
                           max_width=MAX_W, max_height=MAX_H)
    if categorical_df is not None:
        excluded_notice = categorical_excluded_from_distributions_notice(categorical_df)
        if excluded_notice is not None:
            render_narrative(elements, excluded_notice)

    _render_reduction_subsection(elements, node_dir, mode, LOCAL_MCA, level=2, prefix="3.3")

    add_section_heading(elements, "3.4 Associations", level=2)
    _render_pairwise_table(elements, doc, significant_df, pair_type="cat-cat", mode=mode)


# ---------------------------------------------------------------------------
# 4. Temporal
# ---------------------------------------------------------------------------

def _build_temporal_section(elements, doc, node_dir, mode, temporal_df):
    """Build the temporal statistics section elements for a local PDF report."""
    add_section_heading(elements, "4. Temporal Section")
    render_narrative(elements, NarrativeMessage(
        "info",
        "Temporal variables are date or timestamp columns (e.g. admission date, discharge date). "
        "Each is analyzed as a time series of observation counts per period. "
        "The summary table shows the overall time range, number of valid timestamps, and missing "
        "periods. The line charts visualize activity over time and highlight the most active period.",
    ))
    temporal_dir = node_dir / "temporal"

    add_section_heading(elements, "4.1 Descriptive Statistics", level=2)
    feature_order = []
    if temporal_df is None:
        render_narrative(elements, NarrativeMessage("info", "No temporal variables available."))
    else:
        render_narrative(elements, summarise_temporal(temporal_df))
        ranked_df = temporal_df
        display_df = drop_internal_columns(temporal_df)
        if mode == "short" and len(temporal_df) > SHORT_TABLE_MAX_ROWS:
            ranked_df = rank_by_activity(temporal_df, SHORT_TABLE_MAX_ROWS)
            render_narrative(elements, truncation_note(
                len(ranked_df), len(temporal_df), "active", "temporal_summary.csv",
            ))
            display_df = drop_internal_columns(ranked_df)
        display_df = display_df.drop(columns=["missing_periods"], errors="ignore")
        elements.append(create_table(display_df, doc.width))
        elements.append(Spacer(1, 14))
        feature_order = list(ranked_df["feature"]) if mode == "short" else list(temporal_df["feature"])

    if mode == "short":
        line_chart_paths = [temporal_dir / f"{feature}_activity.png" for feature in feature_order[:SHORT_PLOT_MAX]]
    else:
        line_chart_paths = (
            sorted(temporal_dir.glob("temporal_activity_batch_*.png")) if temporal_dir.exists() else []
        )
    add_heading_and_plots(elements, "4.2 Line Charts", line_chart_paths, level=2,
                           max_width=MAX_W, max_height=MAX_H)


# ---------------------------------------------------------------------------
# 5. Cross-Variable Associations & Outcome Comparisons
# ---------------------------------------------------------------------------

def _build_cross_variable_section(elements, doc, node_dir, mode, significant_df):
    """Build the cross-variable analysis section (inferential) elements for a local PDF report."""
    inferential_dir = node_dir / "inferential"
    comparisons_dir = inferential_dir / "comparisons"

    add_heading_and_plots(elements, "5. Cross-Variable Associations & Outcome Comparisons",
                           [inferential_dir / "association_screening.png"], level=1,
                           max_width=MAX_W, max_height=MAX_H)
    render_narrative(elements, NarrativeMessage(
        "info",
        "This section reports statistical associations between variables in this dataset. "
        "Tests are chosen automatically based on variable type: t-test or Mann-Whitney U for "
        "numeric vs. categorical; Pearson or Spearman for numeric vs. numeric; chi-square or "
        "Fisher exact for categorical vs. categorical. "
        "Only associations that pass FDR-corrected significance (p < 0.05) and have a "
        "meaningful effect size are included. "
        "A smaller p-value means the result is less likely to be due to chance; "
        "a larger effect size means the difference is more pronounced in practice.",
    ))

    add_section_heading(elements, "5.1 Pairwise Associations", level=2)
    render_narrative(elements, NarrativeMessage(
        "info",
        "Pairwise associations compare one numeric variable against an outcome with exactly "
        "two groups (e.g. survived vs. died, treated vs. control). "
        "The association screening heatmap above shows the overall strength of detected "
        "associations — darker cells indicate stronger associations. "
        "The table below lists only the statistically significant ones. "
        "Each boxplot shows the distribution of a numeric variable split by the two-group "
        "outcome; the bracket annotation indicates the significance level "
        "(*** p < 0.001, ** p < 0.01, * p < 0.05, n.s. not significant) and the effect "
        "size (Hedges' g or rank-biserial r). "
        "Note: p-values in the table are displayed as '< 0.001' when they are very small "
        "but non-zero — this is the standard convention in medical reporting.",
    ))
    _render_pairwise_table(elements, doc, significant_df, pair_type="num-cat", mode=mode)

    if mode == "full":
        add_figure(elements, inferential_dir / "group_comparisons_summary.png", MAX_W, MAX_H)
        two_group_plots = (sorted(comparisons_dir.glob("*_two_group.png"))
                           + sorted(comparisons_dir.glob("*_violin.png"))
                           if comparisons_dir.exists() else [])
        for p in sorted(two_group_plots, key=lambda f: f.name):
            add_figure(elements, p, MAX_W, MAX_H)

    add_section_heading(elements, "5.2 Multi-Group Outcome Comparisons (3+ Groups)", level=2)
    render_narrative(elements, NarrativeMessage(
        "info",
        "When the detected outcome column has 3 or more groups, a one-way omnibus test is "
        "used instead of a two-group test. "
        "Welch's ANOVA is applied when the data is approximately normally distributed; "
        "Kruskal-Wallis is used when the distribution is skewed or variances are unequal.",
    ))
    oneway_plots = (sorted(comparisons_dir.glob("*_oneway.png"))
                    if comparisons_dir.exists() else [])
    if not oneway_plots:
        render_narrative(elements, NarrativeMessage(
            "info",
            "No outcome column with 3 or more groups was detected in this dataset.",
        ))
    elif mode == "full":
        for p in oneway_plots:
            add_figure(elements, p, MAX_W, MAX_H)

    if oneway_plots:
        add_section_heading(elements, "5.3 Post-Hoc Pairwise Tests", level=2)
        render_narrative(elements, NarrativeMessage(
            "info",
            "When the omnibus test is significant (p < 0.05), pairwise post-hoc tests "
            "identify which specific group pairs differ from each other. "
            "Games-Howell is used after Welch's ANOVA; pairwise Mann-Whitney U with "
            "Holm-Bonferroni correction is used after Kruskal-Wallis.",
        ))
        posthoc_plots = (sorted(comparisons_dir.glob("posthoc_*.png"))
                         if comparisons_dir.exists() else [])
        if not posthoc_plots:
            render_narrative(elements, NarrativeMessage(
                "info",
                "No significant multi-group associations were found (omnibus p ≥ 0.05 for all variables).",
            ))
        elif mode == "full":
            for p in posthoc_plots:
                add_figure(elements, p, MAX_W, MAX_H)


# ---------------------------------------------------------------------------
# Shared subsection renderers
# ---------------------------------------------------------------------------

def _render_pairwise_table(elements, doc, significant_df, pair_type, mode):
    """Render an inferential pairwise association table into report elements."""
    render_narrative(elements, summarise_inferential(significant_df, pair_type=pair_type))
    if significant_df is None or significant_df.empty:
        return
    df = significant_df[significant_df["pair_type"] == pair_type]
    if df.empty:
        return
    df = drop_internal_columns(df)
    if mode == "short" and len(df) > SHORT_TABLE_MAX_ROWS:
        ranked = rank_by_effect_size(df, SHORT_TABLE_MAX_ROWS)
        render_narrative(elements, truncation_note(
            len(ranked), len(df), "significant", "significant_associations.csv",
        ))
        df = ranked
    elements.append(create_table(df, doc.width))
    elements.append(Spacer(1, 14))


def reduction_excluded_columns_notice(subdir, title) -> Optional[NarrativeMessage]:
    """Note which columns run_pca/run_mca silently dropped for being entirely missing.

    Both functions drop columns with no non-missing values rather than
    failing the whole analysis (their variance/categories are undefined).
    Without this notice such a column would just be absent from the
    projection with no explanation.
    """
    excluded_df = safe_read_csv(subdir / "excluded_columns.csv")
    if excluded_df is None or "feature" not in excluded_df.columns:
        return None
    names = ", ".join(str(f) for f in excluded_df["feature"])
    return NarrativeMessage(
        level="info",
        text=(
            f"The following column(s) are entirely missing in this dataset and "
            f"were excluded from {title}: {names}."
        ),
    )


def _render_reduction_subsection(elements, node_dir, mode, spec, level, prefix):
    """Render a dimensionality reduction (PCA or MCA) subsection into report elements."""
    subdir = node_dir / spec.subdir
    if not subdir.exists():
        add_section_heading(elements, f"{prefix} {spec.title}", level=level)
        render_narrative(elements, NarrativeMessage(
            "info", f"{spec.title} was not computed for this node.",
        ))
        return

    if mode == "short":
        plot_paths = [subdir / spec.short_plot]
    else:
        # Full mode: individual plots only — exclude the combined overview panel
        plot_paths = [p for p in sorted(subdir.glob("*.png")) if "overview" not in p.name]
    has_plots = add_heading_and_plots(elements, f"{prefix} {spec.title}", plot_paths, level=level,
                                       max_width=MAX_W, max_height=MAX_H)
    if mode == "short" and not has_plots:
        render_narrative(elements, NarrativeMessage(
            "info",
            f"{spec.title} plots are available in the full version of this report.",
        ))
    excluded_notice = reduction_excluded_columns_notice(subdir, spec.title)
    if excluded_notice is not None:
        render_narrative(elements, excluded_notice)


_GLOSSARY = [
    ("p-value", "Probability of observing a result this extreme by chance if there is no real effect. A small p-value (< 0.05 after correction) supports rejecting the null hypothesis."),
    ("FDR correction", "False Discovery Rate: adjusts p-values when many tests are run simultaneously, reducing the chance of false positives."),
    ("Effect size", "Measure of how large an association or difference is, independent of sample size. Examples: Cohen's d (standardized mean difference), Cramér's V (categorical association strength), rank-biserial r (non-parametric effect)."),
    ("Cohen's d", "Standardized mean difference between two groups: d = (mean1 – mean2) / pooled SD. |d| < 0.2 small, 0.5 medium, 0.8+ large."),
    ("Hedges' g", "Bias-corrected version of Cohen's d, preferred for small samples."),
    ("Cramér's V", "Effect size for chi-square tests between categorical variables. Ranges 0 (no association) to 1 (perfect association)."),
    ("IQR", "Interquartile Range: the range from the 25th to the 75th percentile. Robust measure of spread that is not affected by extreme outliers."),
    ("SD / Std Dev", "Standard Deviation: average distance of observations from the mean. A larger SD means more variability in the data."),
    ("PCA", "Principal Component Analysis: linear method that projects numeric variables into orthogonal dimensions ordered by how much variance they explain."),
    ("Explained variance", "In PCA: the proportion of total variance captured by each principal component."),
    ("MCA", "Multiple Correspondence Analysis: the categorical analog of PCA. Projects category levels and samples into a low-dimensional space to reveal co-occurrence patterns."),
    ("ICU", "Intensive Care Unit: a hospital ward for patients requiring close monitoring and life-support equipment."),
    ("BMI", "Body Mass Index: weight (kg) / height² (m²). < 18.5 underweight, 18.5–24.9 normal, 25–29.9 overweight, ≥ 30 obese."),
    ("LOS", "Length of Stay: number of days a patient is hospitalized."),
    ("PASC", "Post-Acute Sequelae of SARS-CoV-2 (Long COVID): persistent symptoms after acute COVID-19 infection."),
    ("NaN / NA", "Not a Number / Not Available: a placeholder indicating a missing or undefined value in the dataset."),
]


def _build_glossary_section(elements):
    """Append a static glossary of terms and abbreviations to the report."""
    add_section_heading(elements, "Appendix: Terms and Abbreviations")
    render_narrative(elements, NarrativeMessage(
        "info",
        "This glossary covers statistical terms used in the report and common hospital/clinical "
        "abbreviations. It is intended to help readers from different backgrounds "
        "(clinicians, data scientists, engineers) interpret the results consistently.",
    ))
    glossary_df = pd.DataFrame(_GLOSSARY, columns=["Term", "Definition"])
    elements.append(create_table(glossary_df, MAX_W))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------



# ============================================================================
# Source: generate_global_report.py
# ============================================================================

"""Federated (global) report generator.

generate_global_report(results_dir, output_dir, mode="full") -> Path

Builds a PDF report from the aggregated outputs under
results/federated_results/. Two modes:
  - "short": narrative summaries, key plots, condensed tables
  - "full":  full per-variable detail
"""


def generate_global_report(results_dir, output_dir, mode="full") -> Path:
    """
    Generate the federated PDF report from on-disk federated CSV and PNG outputs.

      Args:
          federated_dir: Root directory of the federated analysis outputs.
          output_dir: Directory where the PDF will be written.
          mode: ``'short'`` or ``'full'``; controls which sections are included.

      Returns:
          Path to the written PDF file.
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overview_df = safe_read_csv(results_dir / "overview" / "overview.csv")
    numeric_df = safe_read_csv(results_dir / "numeric" / "federated_numeric_statistics.csv")
    categorical_df = safe_read_csv(results_dir / "categorical" / "federated_categorical_statistics.csv")
    temporal_df = safe_read_csv(results_dir / "temporal" / "federated_temporal_statistics.csv")
    n_nodes = _extract_n_nodes_global(overview_df)

    output_path = output_dir / f"global_report_{mode}.pdf"
    doc = BaseDocTemplate(str(output_path), leftMargin=PAGE_MARGIN, rightMargin=PAGE_MARGIN)
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([
        PageTemplate(id="title", frames=[frame], onPage=make_header("Global Report")),
        PageTemplate(id="overview", frames=[frame], onPage=make_header("Overview")),
        PageTemplate(id="numeric", frames=[frame], onPage=make_header("Numeric Section")),
        PageTemplate(id="categorical", frames=[frame], onPage=make_header("Categorical Section")),
        PageTemplate(id="temporal", frames=[frame], onPage=make_header("Temporal Section")),
    ])

    elements = []
    _build_title_page_global(elements, mode, n_nodes, numeric_df, categorical_df, temporal_df)

    elements.append(NextPageTemplate("overview"))
    elements.append(PageBreak())
    _build_overview_section_global(elements, doc, results_dir, overview_df)

    elements.append(NextPageTemplate("numeric"))
    elements.append(PageBreak())
    _build_numeric_section_global(elements, doc, results_dir, mode, numeric_df)

    elements.append(NextPageTemplate("categorical"))
    elements.append(PageBreak())
    _build_categorical_section_global(elements, doc, results_dir, mode, categorical_df)

    elements.append(NextPageTemplate("temporal"))
    elements.append(PageBreak())
    _build_temporal_section_global(elements, doc, results_dir, mode, temporal_df)

    elements.append(PageBreak())
    _build_glossary_section(elements)

    doc.build(elements)
    return output_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_n_nodes_global(overview_df):
    """Extract the number of federation nodes from the global overview CSV for the federated report."""
    if overview_df is None:
        return None
    matches = overview_df[overview_df["metric"].str.contains("hospital", case=False, na=False)]
    if matches.empty:
        return None
    try:
        return int(matches.iloc[0]["value"])
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Title page
# ---------------------------------------------------------------------------

def _build_title_page_global(elements, mode, n_nodes, numeric_df, categorical_df, temporal_df):
    """Build the title page elements for the federated PDF report."""
    elements.append(Paragraph("Global Federated Statistical Report", STYLES["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Report type: {mode.title()}", STYLES["Normal"]))
    now = datetime.now().strftime("%d %B %Y, %H:%M")
    elements.append(Paragraph(f"Generated on: {now}", STYLES["Normal"]))
    elements.append(Spacer(1, 20))
    elements.extend(build_privacy_notice(
        report_type="global", n_nodes=n_nodes,
        numeric_df=numeric_df, categorical_df=categorical_df, temporal_df=temporal_df,
    ))


# ---------------------------------------------------------------------------
# 1. Overview
# ---------------------------------------------------------------------------

def _build_overview_section_global(elements, doc, results_dir, overview_df):
    """Build the overview section elements for the federated PDF report."""
    add_section_heading(elements, "1. Overview")
    overview_dir = results_dir / "overview"

    if overview_df is None:
        render_narrative(elements, NarrativeMessage("info", "No overview data available."))
    else:
        elements.append(create_table(overview_df, doc.width))
        elements.append(Spacer(1, 14))

    add_figure(elements, overview_dir / "data_type_distribution.png", MAX_W, MAX_H)


# ---------------------------------------------------------------------------
# 2. Numeric
# ---------------------------------------------------------------------------

def _build_numeric_section_global(elements, doc, results_dir, mode, numeric_df):
    """Build the numeric statistics section elements for the federated PDF report."""
    numeric_dir = results_dir / "numeric"

    add_heading_and_plots(elements, "2. Numeric Section", [
        numeric_dir / "numeric_summary_bars.png",
        numeric_dir / "age_distribution_federated.png",
    ], level=1, max_width=MAX_W, max_height=MAX_H)
    age_range_notice = out_of_range_age_notice(numeric_dir)
    if age_range_notice is not None:
        render_narrative(elements, age_range_notice)
    render_narrative(elements, NarrativeMessage(
        "info",
        "Federated numeric statistics: the mean, variance, and standard deviation are "
        "computed using the exact Chan-Golub-LeVeque parallel decomposition — these are "
        "equivalent to computing the statistics on the full concatenated dataset. "
        "Min and max are the global extremes across all nodes.",
    ))

    if numeric_df is None:
        render_narrative(elements, NarrativeMessage("info", "No numeric variables available."))
        return

    render_narrative(elements, summarise_numeric(numeric_df))
    display_df = drop_internal_columns(numeric_df)
    if mode == "short" and len(numeric_df) > SHORT_TABLE_MAX_ROWS:
        ranked = rank_by_activity(numeric_df, SHORT_TABLE_MAX_ROWS)
        render_narrative(elements, truncation_note(
            len(ranked), len(numeric_df), "complete (highest record count)",
            "federated_numeric_statistics.csv",
        ))
        display_df = drop_internal_columns(ranked)
    display_df = prepare_numeric_display(display_df)
    elements.append(create_table(display_df, doc.width))
    elements.append(Spacer(1, 14))


# ---------------------------------------------------------------------------
# 3. Categorical
# ---------------------------------------------------------------------------

def _build_categorical_section_global(elements, doc, results_dir, mode, categorical_df):
    """Build the categorical statistics section elements for the federated PDF report."""
    categorical_dir = results_dir / "categorical"

    cat_dist_paths = sorted(categorical_dir.glob("categorical_distributions_*.png"))
    if not cat_dist_paths:
        cat_dist_paths = [categorical_dir / "categorical_distributions.png"]
    cat_dist_paths.append(categorical_dir / "sex_distribution_federated.png")
    add_heading_and_plots(elements, "3. Categorical Section", cat_dist_paths,
                           level=1, max_width=MAX_W, max_height=MAX_H)
    render_narrative(elements, NarrativeMessage(
        "info",
        "Federated categorical statistics: category counts are summed across all nodes. "
        "Relative frequencies reflect the global distribution. "
        "All categorical columns are shown in batched charts above (6 per image).",
    ))

    if categorical_df is None:
        render_narrative(elements, NarrativeMessage("info", "No categorical variables available."))
        return

    render_narrative(elements, summarise_categorical(categorical_df))
    display_df = drop_internal_columns(categorical_df).drop(
        columns=["counts", "relative_freq"], errors="ignore")
    if mode == "short" and len(display_df) > SHORT_TABLE_MAX_ROWS:
        render_narrative(elements, truncation_note(
            SHORT_TABLE_MAX_ROWS, len(display_df), "first listed",
            "federated_categorical_statistics.csv",
        ))
        display_df = display_df.head(SHORT_TABLE_MAX_ROWS)
    elements.append(create_table(display_df, doc.width))
    elements.append(Spacer(1, 14))


# ---------------------------------------------------------------------------
# 4. Temporal
# ---------------------------------------------------------------------------

def _build_temporal_section_global(elements, doc, results_dir, mode, temporal_df):
    """Build the temporal statistics section elements for the federated PDF report."""
    temporal_dir = results_dir / "temporal"

    add_heading_and_plots(elements, "4. Temporal Section",
                           [temporal_dir / "temporal_trend_summary.png"], level=1,
                           max_width=MAX_W, max_height=MAX_H)
    render_narrative(elements, NarrativeMessage(
        "info",
        "Federated inferential analysis is currently limited to trend-slope "
        "estimation above. Per-pair correlation and association tests "
        "(e.g. correlations, Cramer's V, group comparisons) are computed "
        "locally only and are not aggregated across nodes.",
    ))

    if temporal_df is None:
        render_narrative(elements, NarrativeMessage("info", "No temporal variables available."))
    else:
        render_narrative(elements, summarise_temporal(temporal_df))
        display_df = drop_internal_columns(temporal_df).drop(
            columns=["counts_per_period"], errors="ignore")
        if mode == "short" and len(display_df) > SHORT_TABLE_MAX_ROWS:
            display_df = display_df.head(SHORT_TABLE_MAX_ROWS)
            render_narrative(elements, truncation_note(
                SHORT_TABLE_MAX_ROWS, len(temporal_df), "first listed",
                "federated_temporal_statistics.csv",
            ))
        elements.append(create_table(display_df, doc.width))
        elements.append(Spacer(1, 14))

    if mode == "full":
        add_plots_from_dir(elements, temporal_dir, MAX_W, MAX_H,
                            exclude={"temporal_trend_summary.png"})


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------



# ============================================================================
# Source: analyze.py
# ============================================================================


logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
LOCAL_RESULTS_DIR = Path("results/local_results")
FEDERATED_RESULTS_DIR = Path("results/federated_results")

# Optional label column used to colour PCA/MCA plots.
# Set to a column name string if your dataset has a known label column,
# or leave as None to skip label-based colouring.
LABEL_COL = None


# ---------------------------------------------------------------------------
# Reduction gating
# ---------------------------------------------------------------------------

def should_apply_reductions(df: pd.DataFrame, column_types: dict) -> dict:
    """
    Return a dict of bool flags indicating which dimensionality-reduction
    operations have enough data to be meaningful on this dataset.

    Centralises all threshold decisions in one place so they are easy to find,
    adjust, and reason about — instead of being scattered as ad-hoc ``>= 2``
    guards throughout analysis_method.

    Thresholds (chosen conservatively for interpretable results):
      pca  n_numeric  >= 10  fewer components than this give trivial decompositions
      mca  n_cat_usable >= 8   low-cardinality categorical vars (≤ 30 levels)
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


# ---------------------------------------------------------------------------
# Figure helpers used by the aggregator for per-node overview plots.
# ---------------------------------------------------------------------------

def _make_missing_bar_fig(df: pd.DataFrame, max_cols: int = 50):
    """Return a missingno bar chart figure with column-count subtitle, or None on failure."""
    try:
        total_cols = df.shape[1]
        if total_cols > max_cols:
            completeness = df.notna().mean()
            worst_cols = completeness.nsmallest(max_cols).index
            subset = df[worst_cols]
            subtitle = (
                f"Showing {max_cols} most incomplete of {total_cols} total columns "
                f"(sorted by ascending completeness)"
            )
        else:
            subset = df
            subtitle = f"Showing all {total_cols} columns"
        fig, ax = plt.subplots(figsize=(12, 6))
        msno.bar(subset, ax=ax, color=PALETTE[0], fontsize=12)
        ax.set_title(
            f"Column Completeness (non-null values per column)\n{subtitle}",
            fontsize=13,
        )
        fig.tight_layout()
        return fig
    except Exception as e:
        print(f"_make_missing_bar_fig error: {e}", flush=True)
        return None


def _make_missing_heatmap_fig(df: pd.DataFrame, max_cols: int = 50):
    """Return a missingno heatmap figure, or None when no missing values or on failure.

    Returns None (no file written) when the DataFrame is fully complete so
    the report builder can show a narrative message instead of a blank chart.
    """
    if df.isnull().sum().sum() == 0:
        return None
    try:
        subset = df.iloc[:, :max_cols] if df.shape[1] > max_cols else df
        fig, ax = plt.subplots(figsize=(12, 9))
        msno.heatmap(subset, ax=ax, fontsize=12)
        ax.set_title("Nullity Correlation Between Columns", fontsize=14)
        fig.tight_layout()
        return fig
    except Exception as e:
        print(f"_make_missing_heatmap_fig error: {e}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Analyzer  (runs on each node)
# ---------------------------------------------------------------------------
class DataReportAnalyzer(StarAnalyzer):
    """FLAME analyzer that runs per-node descriptive and inferential statistics.

    Executes on each hospital node.  The raw DataFrame never leaves the node;
    instead, all results are serialised into a JSON-compatible dict that is
    returned to the aggregator.

    Return dict keys (all JSON-serializable):
        node_id, n_rows, n_cols, n_analytical_cols, id_columns, total_values —
            basic metadata.
        numeric_statistics, categorical_statistics, temporal_statistics, means —
            per-column descriptive stats.
        missing_by_col, total_missing, missing_values_percentage, n_duplicates —
            data quality metrics.
        age_hist, age_edges, sex_counts — demographic aggregates.
        column_types — detected dtype categories for each column.
        inferential_data — association screening records, significant association
            records, outcome comparison records, and posthoc p-value matrices.
            All are structured data (no base64 PNGs); the aggregator generates
            all plots from these records.
    """

    def __init__(self, flame):
        super().__init__(flame)

    def analysis_method(self, data, aggregator_results):
        """FLAME entry point: delegate to _analysis_method_impl with full traceback logging on failure."""
        import traceback as _tb, sys
        print("analysis_method called", flush=True)
        sys.stderr.write("analysis_method called\n")
        sys.stderr.flush()
        try:
            return self._analysis_method_impl(data, aggregator_results)
        except Exception as _exc:
            msg = _tb.format_exc()
            print("ANALYSIS_METHOD FAILED:\n" + msg, flush=True)
            sys.stderr.write("ANALYSIS_METHOD FAILED:\n" + msg + "\n")
            sys.stderr.flush()
            raise

    def _analysis_method_impl(self, data, aggregator_results):
        """Load data, run all analyses, encode df-dependent plots as base64, and return the result dict."""
        import sys
        print("_analysis_method_impl started", flush=True)
        _load_analysis_dependencies()
        print("analysis dependencies loaded", flush=True)
        # --- Parse CSV bytes ------------------------------------------------
        # data[0] is a dict of {filename: bytes} supplied by the FLAME platform.
        # We search by filename suffix so the code is robust to different naming
        # conventions across hospital sites (mirrors the FLAME example script).
        if not (isinstance(data, list) and data and isinstance(data[0], dict)):
            raise ValueError(
                f"Unexpected data format received by analyzer: {type(data)}. "
                "Expected list[dict[str, bytes]]."
            )
        files = data[0]
        print(f"Available data files: {list(files.keys())}", flush=True)

        # Priority-based file selection — first match wins:
        #   1. Explicitly unlabeled file (datasets that ship both labeled/unlabeled)
        #   2. Any CSV that is NOT the labeled version (single-file or multi-file datasets)
        #   3. Any CSV at all (last-resort fallback — also handles datasets with only
        #      a labeled file, e.g. the FLAME example script's 'labeled.csv')
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
        print(f"Selected file: {csv_key!r}  ({len(file_bytes):,} bytes)", flush=True)
        if isinstance(file_bytes, str):
            file_bytes = file_bytes.encode("utf-8")
        # sep=None tells pandas to figure out the delimiter automatically
        df = pd.read_csv(BytesIO(file_bytes), sep=None, engine="python")
        # clean
        print("Columns:", df.columns.tolist(), flush=True)
        print("Shape:", df.shape, flush=True)
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

        # Use exact API recommended by privateAIM support team.
        # Real node IDs are UUIDs — self.id is the correct attribute.
        node_id = self.id
        all_n_ids = self.partner_node_ids.copy()
        all_n_ids.append(self.id)
        node_index = sorted(all_n_ids).index(self.id)
        node_number = node_index + 1
        print(f"node_id={node_id}, node_number={node_number}", flush=True)
        n_rows, n_cols = df.shape

        # Convert date-like object columns to datetime64 BEFORE identifier
        # detection so that detect_id_column's datetime64 guard correctly
        # skips temporal columns (e.g. discharge_date columns would otherwise
        # fire on the high-uniqueness heuristic).
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

        # Detect ALL identifier columns (not just the first match).
        # Running after date conversion so datetime64 columns are correctly skipped.
        id_columns = [col for col in df.columns if detect_id_column(df[col], col)]
        if id_columns:
            print(f"Identifier columns detected: {id_columns}", flush=True)
        else:
            print("No identifier columns detected", flush=True)

        # Rename the first identifier column to patient_id for temporal analysis.
        if id_columns:
            primary_id = id_columns[0]
            df = df.rename(columns={primary_id: "patient_id"})
            id_columns[0] = "patient_id"
            patient_series = df["patient_id"]
        else:
            patient_series = None

        # Analytical column count excludes identifier columns so that
        # missingness metrics are not diluted by always-present ID fields.
        n_analytical_cols = n_cols - len(id_columns)

        column_types = detect_column_types(df)

        # All identifier columns are kept in df for record counts/joins but
        # must not enter PCA/MCA, descriptive statistics, or inferential
        # screening.
        for dtype in column_types:
            column_types[dtype] = [c for c in column_types[dtype] if c not in id_columns]

        # Note columns that ended up categorical only because a few values
        # (e.g. a censored lab result like "<5") couldn't be parsed as
        # numbers -- serialized for the aggregator to write as a notice
        # source file, rather than silently coercing those values to NaN.
        quasi_numeric_categorical_columns = detect_quasi_numeric_categorical_columns(
            df, column_types["categorical"]
        )

        # ----------------------------------
        # Reduction thresholds — same as local analyze.py
        _reductions = should_apply_reductions(df, column_types)

        # ----------------------------------
        # Inferential statistics (automatic association screening + optional
        # outcome-driven comparisons). Statistical results are serialized into
        # inferential_data and sent to the aggregator, which generates all
        # plots from those records — no PNG encoding happens on the node.
        inferential_data: dict = {}
        try:
            screening = screen_associations(df, column_types)

            inferential_data["association_screening_records"] = _make_serializable(
                screening.to_dict(orient="records")
            )
            inferential_data["significant_associations_records"] = _make_serializable(
                screening[screening["significant"]].to_dict(orient="records")
            )

            outcome_col = detect_outcome_column(df, column_types, OUTCOME_KEYWORD_GROUPS)
            inferential_data["outcome_col"] = outcome_col
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
                    inferential_data["comparisons_by_outcome_records"] = (
                        _make_serializable(comparison_df.to_dict(orient="records"))
                    )

                    # Serialize posthoc p-value matrices (normalized to square
                    # group×group form) so the aggregator can regenerate heatmaps
                    # and write posthoc CSVs without needing the raw dataframe.
                    posthoc_data: dict = {}
                    for num_col, cmp in cmp_results.items():
                        posthoc_df = cmp.get("posthoc")
                        if posthoc_df is None:
                            continue
                        try:
                            mat = _posthoc_to_pvalue_matrix(posthoc_df, cmp["method"])
                            if mat is not None and not mat.empty:
                                posthoc_data[f"{num_col}_vs_{outcome_col}"] = {
                                    "method": cmp["method"],
                                    "value_col": num_col,
                                    "outcome_col": outcome_col,
                                    "groups": [str(g) for g in mat.index.tolist()],
                                    "matrix": _make_serializable(mat.values.tolist()),
                                }
                        except Exception:
                            logger.debug(
                                "Posthoc serialization failed for %s vs %s",
                                num_col, outcome_col, exc_info=True,
                            )
                    inferential_data["posthoc_data"] = posthoc_data
        except Exception as e:
            print(f"Inferential statistics section error: {e}", flush=True)

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

        # Exclude identifier columns from the total-values count: they are
        # always filled so including them would understate the missingness rate.
        total_values = n_rows * n_analytical_cols
        missing_values_percentage = (total_missing / total_values * 100) if total_values else 0.0
        # check how many duplicates there are
        n_duplicates = int(df.duplicated().sum())

        # age histogram
        age_hist, age_edges = compute_age_histogram(df)
        age_out_of_range_count = count_out_of_range_ages(df)

        # numeric histograms for all numeric columns (compact: bin counts + edges)
        numeric_histograms: dict = {}
        for _col in numeric_cols:
            try:
                _vals = pd.to_numeric(df[_col], errors="coerce").dropna()
                if len(_vals) >= 2:
                    _hist, _edges = np.histogram(_vals, bins="auto")
                    numeric_histograms[_col] = {
                        "hist": _hist.astype(int).tolist(),
                        "edges": _edges.astype(float).tolist(),
                    }
            except Exception:
                pass

        # MCA column coordinates + explained inertia
        # Matches local analyze.py: include ALL categorical columns (incl. binary)
        # with ≤30 unique values; recode numeric-dtype columns to category.
        mca_column_coordinates: dict | None = None
        mca_explained_inertia: list | None = None
        mca_dropped_features: list = []
        mca_feature_names: list | None = None
        try:
            _mca_feats = [
                c for c in categorical_cols
                if df[c].nunique(dropna=True) <= 30
            ]
            if _reductions.get("mca") and len(_mca_feats) >= 2:
                _mca_df = df[_mca_feats].copy()
                for _c in _mca_feats:
                    if pd.api.types.is_numeric_dtype(_mca_df[_c]):
                        _mca_df[_c] = _mca_df[_c].astype("category")
                _mca_result = run_mca(_mca_df, _mca_feats)
                mca_column_coordinates = _mca_result.column_coordinates.to_dict()
                mca_explained_inertia = _mca_result.explained_inertia_ratio.tolist()
                mca_dropped_features = [c for c in _mca_feats if c not in _mca_result.feature_names]
                mca_feature_names = list(_mca_result.feature_names)
        except Exception as _mca_err:
            print(f"MCA skipped (node {node_id}): {_mca_err}", flush=True)

        # Nullity correlation matrix (columns that have ≥1 missing value, Pearson on binary mask)
        nullity_correlation: dict | None = None
        try:
            _missing_mask = df.isnull()
            _cols_with_missing = [
                c for c in df.columns
                if c not in id_columns and _missing_mask[c].any()
            ]
            if len(_cols_with_missing) >= 2:
                _corr = _missing_mask[_cols_with_missing].corr()
                nullity_correlation = {
                    "columns": _corr.columns.tolist(),
                    "values": _corr.values.tolist(),
                }
        except Exception as _nc_err:
            print(f"Nullity correlation skipped (node {node_id}): {_nc_err}", flush=True)

        # PCA loadings + explained variance (column-level stats, no per-patient data)
        pca_loadings: list | None = None
        pca_feature_names: list | None = None
        pca_explained_variance: list | None = None
        pca_recommended_n_components: int | None = None
        pca_dropped_features: list = []
        try:
            _pca_feats = [c for c in numeric_cols if c in df.columns]
            if _reductions.get("pca") and len(_pca_feats) >= 2:
                _pca_result = run_pca(df[_pca_feats], _pca_feats)
                pca_loadings = _pca_result.loadings.tolist()
                pca_feature_names = _pca_result.feature_names
                pca_explained_variance = _pca_result.explained_variance_ratio.tolist()
                pca_recommended_n_components = int(_pca_result.recommended_n_components)
                pca_dropped_features = [c for c in _pca_feats if c not in pca_feature_names]
        except Exception as _pca_err:
            print(f"PCA skipped (node {node_id}): {_pca_err}", flush=True)

        # ------------------------------------------------------------------
        # Build return dict — every value must be JSON-serializable.
        # All plot generation is deferred to the aggregator, which works
        # entirely from these summary statistics — no PNGs cross the boundary.
        # _make_serializable does a final sweep to catch anything missed.
        # ------------------------------------------------------------------
        result = {
            "node_id": node_id,
            "all_columns": all_columns,
            "n_rows": int(n_rows),
            "n_cols": int(n_cols),
            "n_analytical_cols": int(n_analytical_cols),
            "id_columns": id_columns,
            "total_values": int(total_values),
            "numeric_statistics": numeric_statistics,
            "categorical_statistics": categorical_statistics,
            # pd.Period keys/values sanitized to strings before serialization
            "temporal_statistics": _sanitize_temporal_statistics(temporal_statistics),
            "means": means,
            "missing_by_col": {str(k): int(v) for k, v in missing_by_col.items()},
            "total_missing": int(total_missing),
            "missing_values_percentage": float(missing_values_percentage),
            "n_duplicates": int(n_duplicates),
            "age_edges": list(age_edges) if age_edges is not None else None,
            "age_hist": list(age_hist) if age_hist is not None else None,
            "age_out_of_range_count": age_out_of_range_count,
            "sex_counts": {str(k): int(v) for k, v in sex_counts.items()},
            "column_types": column_types,
            "quasi_numeric_categorical_columns": quasi_numeric_categorical_columns,
            # inferential records and posthoc matrices — aggregator builds plots from these
            "inferential_data": inferential_data,
            # new compact summaries for aggregator plot generation
            "numeric_histograms": numeric_histograms,
            "mca_column_coordinates": mca_column_coordinates,
            "mca_explained_inertia": mca_explained_inertia,
            "mca_dropped_features": mca_dropped_features,
            "mca_feature_names": mca_feature_names,
            "nullity_correlation": nullity_correlation,
            "pca_loadings": pca_loadings,
            "pca_feature_names": pca_feature_names,
            "pca_explained_variance": pca_explained_variance,
            "pca_recommended_n_components": pca_recommended_n_components,
            "pca_dropped_features": pca_dropped_features,
        }

        # Final safety sweep: convert any remaining numpy/Period/Timestamp types
        result = _make_serializable(result)

        # Validate serializability before sending — surfaces problems immediately
        # in the node log rather than as a silent hang on the aggregator.
        try:
            json.dumps(result)
            print("Serialization check: OK", flush=True)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"analysis_method return value is not JSON-serializable: {exc}"
            ) from exc

        return result

def _combine_node_variances(node_stats, global_mean: float) -> float:
    """Pool per-node sample variances using Chan-Golub-LeVeque parallel decomposition.

    Args:
        node_stats: Iterable of (n, mean, var) tuples, var is local sample variance (ddof=1).
        global_mean: The already-computed weighted global mean across all nodes.

    Returns:
        float: Pooled sample variance of the combined dataset (ddof=1).
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
# Aggregator  (central coordinator)
# ---------------------------------------------------------------------------

class DataReportAggregator(StarAggregator):
    """FLAME aggregator that combines per-node results into a federated report.

    Executes on the central coordinator after all analyzer nodes have returned.
    No raw patient data is available here — only the serialised dicts from each
    ``DataReportAnalyzer``.

    Output pipeline:
        1. All outputs (CSVs, PNGs, PDFs) are collected into an in-memory dict
           ``output_files: dict[str, bytes]`` keyed by archive-relative path.
        2. Four collector methods populate ``output_files``:
             ``_collect_federated_files`` / ``_collect_federated_plots`` —
                 federated overview, numeric, categorical, temporal outputs.
             ``_collect_local_node_files`` — per-node CSVs and plots generated
                 from the summary statistics in each analyzer result dict.
                 No PNG data crosses the node boundary; all plots are produced
                 here from records and numeric aggregates.
             ``_collect_pdf_reports`` — PDF reports for each node and the
                 federation.
        3. ``_collect_json_summaries`` builds per-node and federated
           ``summary.json`` files from the CSV bytes already in ``output_files``.
        4. ``_build_tar(output_files)`` packs everything into a ``.tar.gz``
           written to disk; the file is read back, base64-encoded, and returned
           as ``[result_str]`` with ``output_type="str"``.
    """

    def __init__(self, flame):
        super().__init__(flame)
    # --- main aggregation ---------------------------------------------------
    def aggregation_method(self, analysis_results: list):
        """FLAME entry point: delegate to _aggregation_method_impl with full traceback logging on failure."""
        import traceback as _tb
        try:
            return self._aggregation_method_impl(analysis_results)
        except Exception as _exc:
            print("=" * 60, flush=True)
            print("AGGREGATION_METHOD FAILED — full traceback:", flush=True)
            _tb.print_exc()
            print("=" * 60, flush=True)
            raise

    def _aggregation_method_impl(self, analysis_results: list):
        """Collect all outputs into output_files, build the tar archive, and return it as a base64 string."""
        _load_analysis_dependencies()
        print("aggregation_method started", flush=True)
        if not analysis_results:
            raise ValueError(
                "aggregation_method received no analysis results from any node"
            )

        n_nodes = len(analysis_results)

        # Aggregate descriptive statistics
        total_rows = sum(r["n_rows"] for r in analysis_results)
        n_cols = max(r["n_cols"] for r in analysis_results)
        total_missing = sum(r["total_missing"] for r in analysis_results)
        # Uses n_analytical_cols (excludes identifier columns), matching every
        # other quantity derived from this total (total_values, total_missing)
        # -- identifier columns are never missing and carry no completeness
        # signal, so counting their cells here would inflate the denominator
        # relative to every numerator derived from it.
        n_total_values = sum(r["n_rows"] * r["n_analytical_cols"] for r in analysis_results)
        total_missing_percentage = (total_missing / n_total_values * 100) if n_total_values else 0.0
        global_missing_rate = (total_missing / n_total_values) if n_total_values else 0

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
            weighted_mean_sum = 0
            node_stats_for_var = []
            global_min = None
            global_max = None
            # loop over each node to aggregate statistics for this column
            for r in analysis_results:
                col_stats = r.get("numeric_statistics", {}).get(col)
                if not col_stats:
                    continue
                n = col_stats.get("count", 0)
                mean = col_stats.get("mean", 0)
                var = col_stats.get("variance", 0)
                col_min = col_stats.get("min")
                col_max = col_stats.get("max")

                # skip empty nodes
                if n == 0:
                    continue
                # update totals for federated mean/variance
                total_n += n
                # accumulate weighted mean
                weighted_mean_sum += n * mean
                node_stats_for_var.append((n, mean, var))
                # min / max
                if global_min is None or col_min < global_min:
                    global_min = col_min
                if global_max is None or col_max > global_max:
                    global_max = col_max
            # if we have data we compute the statistics
            if total_n > 0:
                # weighted mean
                global_mean = weighted_mean_sum / total_n
                global_var = _combine_node_variances(node_stats_for_var, global_mean)
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
                col_stats = r.get("categorical_statistics", {}).get(col)
                if not col_stats:
                    continue
                counts = col_stats.get("category_counts", {})
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
                col_stats = r.get("temporal_statistics", {}).get(col)
                if not col_stats:
                    continue
                obs = col_stats.get("observations_per_period", {})
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

            for col, col_stats in local_numeric.items():
                if col not in global_numeric:
                    continue

                local_mean = col_stats.get("mean")
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

        # ------------------------------------------------------------------
        # Generate all output files, pack into tar.gz, write to disk,
        # then read back — mirroring the HALTA example pattern exactly.
        # ------------------------------------------------------------------
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        output_files: dict[str, bytes] = {}
        self._collect_federated_files(federated_results, output_files)
        self._collect_federated_plots(federated_results, output_files)
        self._collect_local_node_files(
            analysis_results, comparison_results_per_node, output_files
        )
        self._collect_json_summaries(analysis_results, output_files)
        _load_reporting_dependencies()
        self._collect_pdf_reports(federated_results, analysis_results,
                                  comparison_results_per_node, output_files)

        # Pack into tar.gz and write to disk
        tar_bytes = _build_tar(output_files)
        tar_path = RESULTS_DIR / "results.tar.gz"
        tar_path.write_bytes(tar_bytes)
        print(f"Written {len(tar_bytes):,} bytes to {tar_path}", flush=True)

        with open(str(tar_path), "rb") as f:
            result_bytes = f.read()
        result_str = base64.b64encode(result_bytes).decode("utf-8")
        print(f"Returning base64 str ({len(result_str):,} chars)", flush=True)
        return [result_str]

    # --- convergence check --------------------------------------------------

    def has_converged(self, result, last_result) -> bool:
        """Return True always — this is a single-pass analysis with no iterative convergence."""
        # Single-pass analysis — always converged after one round.
        # When result is a list, we are on the final return (mirrors HALTA example).
        if isinstance(result, list):
            return True
        return True

    # =========================================================================
    # In-memory file collectors
    # Everything below populates output_files: dict[str, bytes] instead of
    # writing to disk.  The keys are the paths inside the final .tar.gz.
    # =========================================================================

    # -------------------------------------------------------------------------
    # Federated CSVs
    # -------------------------------------------------------------------------
    def _collect_federated_files(self, federated_results: dict,
                                  output_files: dict) -> None:
        """Write all federated summary tables to output_files as CSV bytes."""
        base = "federated"

        # Overview
        overview = {
            "number of hospitals":             federated_results.get("n_nodes"),
            "total number of patients":        federated_results.get("total_rows"),
            "total number of features":        federated_results.get("n_cols"),
            "total number of values":          federated_results.get("n_total_values"),
            "total missing values":            federated_results.get("total_missing"),
            "total missing values percentage": (
                f"{round(federated_results.get('total_missing_percentage', 0), 3)}%"
            ),
        }
        df_overview = pd.DataFrame([
            {"metric": k, "value": v} for k, v in overview.items()
        ])
        output_files[f"{base}/overview/overview.csv"] = _df_to_csv_bytes(df_overview)

        # Numeric
        global_numeric = federated_results.get("global_numeric", {})
        num_rows = [{"feature": f, **m} for f, m in global_numeric.items()]
        if num_rows:
            output_files[f"{base}/numeric/federated_numeric_statistics.csv"] = (
                _df_to_csv_bytes(pd.DataFrame(num_rows))
            )

        # Categorical
        global_categorical = federated_results.get("global_categorical", {})
        cat_rows = [{"feature": f, **m} for f, m in global_categorical.items()]
        if cat_rows:
            output_files[f"{base}/categorical/federated_categorical_statistics.csv"] = (
                _df_to_csv_bytes(pd.DataFrame(cat_rows))
            )

        # Temporal
        global_temporal = federated_results.get("global_temporal", {})
        temp_rows = []
        for feature, s in global_temporal.items():
            row = {"feature": feature, **s}
            row["counts_per_period"] = json.dumps(row.get("counts_per_period", {}))
            row["most_active_period"] = str(row.get("most_active_period"))
            temp_rows.append(row)
        if temp_rows:
            output_files[f"{base}/temporal/federated_temporal_statistics.csv"] = (
                _df_to_csv_bytes(pd.DataFrame(temp_rows))
            )

        # Sex distribution
        sex_counts = federated_results.get("sex_counts", {})
        if sex_counts:
            df_sex = pd.DataFrame(
                list(sex_counts.items()), columns=["sex", "count"]
            )
            output_files[f"{base}/categorical/sex_distribution_federated.csv"] = (
                _df_to_csv_bytes(df_sex)
            )

        # Age distribution
        age_hist  = federated_results.get("age_hist")
        age_edges = federated_results.get("age_edges")
        if age_hist is not None and age_edges is not None:
            bins = [
                f"{int(age_edges[i])}-{int(age_edges[i + 1])}"
                for i in range(len(age_edges) - 1)
            ]
            df_age = pd.DataFrame({"age_bin": bins, "count": age_hist})
            output_files[f"{base}/numeric/age_distribution_federated.csv"] = (
                _df_to_csv_bytes(df_age)
            )

        age_out_of_range_count = federated_results.get("age_out_of_range_count")
        if age_out_of_range_count:
            output_files[f"{base}/numeric/age_out_of_range.csv"] = _df_to_csv_bytes(
                pd.DataFrame({"count": [age_out_of_range_count]})
            )

    # -------------------------------------------------------------------------
    # Federated plots
    # -------------------------------------------------------------------------
    def _collect_federated_plots(self, federated_results: dict,
                                  output_files: dict) -> None:
        """Generate federated plots in memory and add PNG bytes to output_files."""
        base = "federated"
        try:
            save_all_federated_plots(federated_results, output_files, base)
        except Exception as e:
            print(f"Federated plot generation error: {e}", flush=True)

    # -------------------------------------------------------------------------
    # Per-node files
    # -------------------------------------------------------------------------
    def _collect_local_node_files(self, analysis_results: list,
                                   comparison_results: list,
                                   output_files: dict) -> None:
        """Produce per-node CSVs and plots and add them to output_files."""
        # Build a stable sorted list of all node UUIDs once, then index into it.
        all_n_ids = sorted(r["node_id"] for r in analysis_results)
        for r in analysis_results:
            raw_node_id  = r["node_id"]
            node_index   = all_n_ids.index(raw_node_id)
            node_number  = node_index + 1
            base         = f"local/node{node_number}"

            node_comp          = next(
                (c for c in comparison_results if c["node_id"] == raw_node_id), {}
            )
            column_comparison  = node_comp.get("column_comparison", {})
            column_status_map  = column_comparison
            overview_comparison = node_comp.get("overview_comparison", {})
            numeric_comparison  = node_comp.get("numeric_comparison", {})

            numeric_statistics    = r.get("numeric_statistics", {})
            categorical_statistics = r.get("categorical_statistics", {})
            temporal_statistics   = r.get("temporal_statistics", {})

            # -- column counts for pie chart --
            all_common = partially_common = unique = 0
            for status in column_comparison.values():
                if status == "common_all":      all_common += 1
                elif status == "common_partial": partially_common += 1
                elif status == "unique_local":   unique += 1

            # Overview CSV
            patient_contribution    = overview_comparison.get("patient_contribution", 0)
            completeness            = overview_comparison.get("completeness", 0)
            usable_data_contribution = overview_comparison.get("usable_data_contribution", 0)
            relative_missing        = overview_comparison.get("relative_missing", 0)
            total_value_contribution = overview_comparison.get("total_value_contribution", 0)

            if isinstance(relative_missing, str):
                try:
                    relative_missing = float(relative_missing)
                except ValueError:
                    relative_missing = 0.0

            missing_label = (
                "above federation average" if relative_missing > 1
                else "below federation average" if relative_missing < 1
                else "equal to federation average"
            )
            _n_analytical = r.get("n_analytical_cols", r["n_cols"])
            _id_cols = r.get("id_columns", [])
            _id_suffix = (
                f" + {len(_id_cols)} identifier column{'s' if len(_id_cols) != 1 else ''}"
                f" detected ({', '.join(_id_cols)})"
                if _id_cols else ""
            )
            overview_data = {
                "Patients": (
                    f"{r['n_rows']} "
                    f"({patient_contribution}% of all patients in the federation)"
                ),
                "Features": f"{_n_analytical} analytical columns{_id_suffix}",
                "Total Values": (
                    f"{r['total_values']:,} "
                    f"({round(float(str(total_value_contribution).rstrip('%')) * 100 if isinstance(total_value_contribution, str) else total_value_contribution * 100, 2)}%"
                    " of all values in the federation)"
                ),
                "Missing Values": (
                    f"{r['total_missing']:,} "
                    f"({round(r['missing_values_percentage'], 2)}%)"
                ),
                "Missingness Compared to Federation": (
                    f"{round(relative_missing, 2)}× ({missing_label})"
                ),
                "Completeness Contribution": (
                    f"{round(completeness * 100, 2)}% of all values in the federation"
                ),
                "Usable Data Contribution": (
                    f"{round(usable_data_contribution * 100, 2)}%"
                    " of all non-missing values in the federation"
                ),
                "Duplicates": r["n_duplicates"],
            }
            df_ov = pd.DataFrame([
                {"metric": k, "value": v} for k, v in overview_data.items()
            ])
            output_files[f"{base}/overview/overview.csv"] = _df_to_csv_bytes(df_ov)

            # Numeric comparison CSV
            df_num_comp = pd.DataFrame([
                {"column": col, "comparison": info["comparison_category"]}
                for col, info in numeric_comparison.items()
            ])
            output_files[f"{base}/comparison/numeric_comparison.csv"] = (
                _df_to_csv_bytes(df_num_comp)
            )

            # Numeric summary CSV
            if numeric_statistics:
                rows = [
                    {"feature": feat,
                     "availability": column_status_map.get(feat, "unknown"),
                     **metrics}
                    for feat, metrics in numeric_statistics.items()
                ]
                output_files[f"{base}/numeric/numeric_summary.csv"] = (
                    _df_to_csv_bytes(pd.DataFrame(rows))
                )

            # Categorical summary CSV
            if categorical_statistics:
                rows = []
                for feat, metrics in categorical_statistics.items():
                    row = {"feature": feat,
                           "availability": column_status_map.get(feat, "unknown")}
                    for k, v in metrics.items():
                        if k == "category_counts":
                            continue
                        row[k] = v
                    rows.append(row)
                output_files[f"{base}/categorical/categorical_summary.csv"] = (
                    _df_to_csv_bytes(pd.DataFrame(rows))
                )

            # Temporal summary CSV
            if temporal_statistics:
                rows = []
                for feat, metrics in temporal_statistics.items():
                    row = {"feature": feat,
                           "availability": column_status_map.get(feat, "unknown")}
                    for k, v in metrics.items():
                        if k in ("observations_per_period", "time_range", "range_days"):
                            continue
                        row[k] = v
                    rows.append(row)
                output_files[f"{base}/temporal/temporal_summary.csv"] = (
                    _df_to_csv_bytes(pd.DataFrame(rows))
                )

            # Age distribution CSV
            age_hist  = r.get("age_hist")
            age_edges = r.get("age_edges")
            if age_hist is not None and age_edges is not None:
                bins = [
                    f"{int(age_edges[i])}-{int(age_edges[i + 1])}"
                    for i in range(len(age_edges) - 1)
                ]
                df_age = pd.DataFrame({"age_bin": bins, "count": age_hist})
                output_files[f"{base}/numeric/age_distribution.csv"] = (
                    _df_to_csv_bytes(df_age)
                )

                # Per-node age distribution plot -- local's per-node report
                # always shows this as the first numeric distribution plot;
                # hub previously only rendered the federated-wide version,
                # so this chart silently never appeared in hub's per-node
                # PDF reports even though the data was already computed.
                try:
                    _edges_arr = np.asarray(age_edges, dtype=float)
                    _hist_arr = np.asarray(age_hist, dtype=float)
                    if len(_edges_arr) >= 2 and _hist_arr.sum() > 0:
                        _fig_age, _ax_age = plt.subplots(figsize=(7, 5))
                        histogram_from_bins(
                            _ax_age, _edges_arr, _hist_arr,
                            title=f"Age Distribution — node{node_number}",
                            xlabel="Age", ylabel="Count",
                        )
                        output_files[f"{base}/numeric/age_distribution.png"] = (
                            _fig_to_bytes(_fig_age)
                        )
                        plt.close(_fig_age)
                except Exception as e:
                    print(f"Age distribution plot error (node{node_number}): {e}", flush=True)

            _age_out_of_range_count = r.get("age_out_of_range_count")
            if _age_out_of_range_count:
                output_files[f"{base}/numeric/age_out_of_range.csv"] = _df_to_csv_bytes(
                    pd.DataFrame({"count": [_age_out_of_range_count]})
                )

            # -- Numeric histograms grid (all numeric columns, batched 6 per image) --
            try:
                _num_hists = r.get("numeric_histograms", {})
                if _num_hists:
                    _batch_size_nh = 6
                    _cols_list  = list(_num_hists.keys())
                    _n_batches_nh = max(1, math.ceil(len(_cols_list) / _batch_size_nh))
                    for _b_nh in range(_n_batches_nh):
                        _batch_cols = _cols_list[_b_nh * _batch_size_nh:(_b_nh + 1) * _batch_size_nh]
                        _fig_nh, _axes_nh = make_subplots(len(_batch_cols), ncols=3, width=5, height=4)
                        for _idx, (_ax, _col) in enumerate(zip(_axes_nh, _batch_cols)):
                            _h  = np.array(_num_hists[_col]["hist"])
                            _e  = np.array(_num_hists[_col]["edges"])
                            _cx = (_e[:-1] + _e[1:]) / 2
                            _ax.bar(_cx, _h, width=np.diff(_e), edgecolor="white",
                                    color=PALETTE[_idx % len(PALETTE)])
                            _ax.set_title(_col, fontsize=9)
                            _ax.set_ylabel("Count", fontsize=8)
                        _title_nh = f"Numeric Distributions — node{node_number}"
                        if _n_batches_nh > 1:
                            _title_nh += f" ({_b_nh + 1}/{_n_batches_nh})"
                        _fig_nh.suptitle(_title_nh, fontsize=12)
                        _fig_nh.tight_layout()
                        output_files[
                            f"{base}/numeric/numeric_histograms_{_b_nh + 1:02d}.png"
                        ] = _fig_to_bytes(_fig_nh)
                        plt.close(_fig_nh)
            except Exception as e:
                print(f"Numeric histograms plot error (node{node_number}): {e}", flush=True)

            # -- Numeric boxplots grid (from Q25/median/Q75/min/max in numeric_statistics, batched 6 per image) --
            try:
                _num_stats = r.get("numeric_statistics", {})
                _bp_cols = [
                    c for c, s in _num_stats.items()
                    if all(k in s for k in ("q25", "median", "q75", "min", "max"))
                ]
                if _bp_cols:
                    _batch_size_bp = 6
                    _n_batches_bp = max(1, math.ceil(len(_bp_cols) / _batch_size_bp))
                    for _b_bp in range(_n_batches_bp):
                        _batch_bp_cols = _bp_cols[_b_bp * _batch_size_bp:(_b_bp + 1) * _batch_size_bp]
                        _fig_bp, _axes_bp = make_subplots(len(_batch_bp_cols), ncols=3, width=5, height=4)
                        for _ax, _col in zip(_axes_bp, _batch_bp_cols):
                            _s  = _num_stats[_col]
                            _q1, _med, _q3 = _s["q25"], _s["median"], _s["q75"]
                            _lo, _hi       = _s["min"], _s["max"]
                            _iqr = _q3 - _q1
                            _whi_lo = max(_lo, _q1 - 1.5 * _iqr)
                            _whi_hi = min(_hi, _q3 + 1.5 * _iqr)
                            _bp = _ax.bxp(
                                [{
                                    "med": _med, "q1": _q1, "q3": _q3,
                                    "whislo": _whi_lo, "whishi": _whi_hi,
                                    "fliers": [],
                                }],
                                showfliers=False,
                                patch_artist=True,
                            )
                            for _patch in _bp["boxes"]:
                                _patch.set_facecolor(PALETTE[1])
                            _ax.set_title(_col, fontsize=9)
                            _ax.set_xticks([])
                        _title_bp = f"Numeric Boxplots — node{node_number}"
                        if _n_batches_bp > 1:
                            _title_bp += f" ({_b_bp + 1}/{_n_batches_bp})"
                        _fig_bp.suptitle(_title_bp, fontsize=12)
                        _fig_bp.tight_layout()
                        output_files[
                            f"{base}/numeric/numeric_boxplots_{_b_bp + 1:02d}.png"
                        ] = _fig_to_bytes(_fig_bp)
                        plt.close(_fig_bp)
            except Exception as e:
                print(f"Numeric boxplots plot error (node{node_number}): {e}", flush=True)

            # -- Nullity correlation heatmap --
            try:
                _nullity = r.get("nullity_correlation")
                if _nullity:
                    _nc_cols = _nullity["columns"]
                    _nc_vals = np.array(_nullity["values"])
                    _nc_df   = pd.DataFrame(_nc_vals, index=_nc_cols, columns=_nc_cols)
                    _fig_nc, _ax_nc = plt.subplots(
                        figsize=(max(6, len(_nc_cols) * 0.7), max(5, len(_nc_cols) * 0.7))
                    )
                    sns.heatmap(
                        _nc_df, ax=_ax_nc, cmap="coolwarm", vmin=-1, vmax=1,
                        annot=len(_nc_cols) <= 15, fmt=".2f", linewidths=0.5,
                        square=True,
                    )
                    _ax_nc.set_title(f"Nullity Correlation — node{node_number}")
                    _fig_nc.tight_layout()
                    output_files[f"{base}/overview/missingno_heatmap.png"] = _fig_to_bytes(_fig_nc)
                    plt.close(_fig_nc)
            except Exception as e:
                print(f"Nullity correlation heatmap error (node{node_number}): {e}", flush=True)

            # -- MCA excluded-columns notice (entirely-missing columns dropped by run_mca) --
            try:
                _mca_dropped = r.get("mca_dropped_features") or []
                if _mca_dropped:
                    output_files[f"{base}/mca/excluded_columns.csv"] = _df_to_csv_bytes(
                        pd.DataFrame({"feature": _mca_dropped})
                    )
            except Exception as e:
                print(f"MCA excluded-columns CSV error (node{node_number}): {e}", flush=True)

            # -- MCA column map batches --
            try:
                _mca_coords_raw = r.get("mca_column_coordinates")
                _mca_inertia    = r.get("mca_explained_inertia")
                if _mca_coords_raw and _mca_inertia:
                    _mca_coords_df = pd.DataFrame.from_dict(_mca_coords_raw)
                    _mca_inertia_arr = np.array(_mca_inertia)
                    # Prefer the real feature-name list (serialized by the analyzer)
                    # over a naive split on the first "__" -- a variable whose own
                    # name contains "__" (e.g. "site__region") would otherwise have
                    # its labels mis-attributed to a shorter, unrelated prefix.
                    _mca_feat_names = r.get("mca_feature_names") or []
                    if _mca_feat_names:
                        _label_to_var = {
                            lbl: _variable_for_label(lbl, _mca_feat_names, "__")
                            for lbl in _mca_coords_df.index
                        }
                    else:
                        _label_to_var = {
                            lbl: (lbl.split("__")[0] if "__" in lbl else lbl)
                            for lbl in _mca_coords_df.index
                        }
                    _all_vars = list(dict.fromkeys(_label_to_var.values()))
                    _n_batches_mca = max(1, math.ceil(len(_all_vars) / _MCA_COLUMN_MAP_BATCH))
                    for _bi in range(_n_batches_mca):
                        _batch_vars = _all_vars[_bi * _MCA_COLUMN_MAP_BATCH: (_bi + 1) * _MCA_COLUMN_MAP_BATCH]
                        _kept = [
                            lbl for lbl in _mca_coords_df.index
                            if any(lbl.startswith(f"{v}__") or lbl == v for v in _batch_vars)
                        ] or list(_mca_coords_df.index)
                        _sub_coords = _mca_coords_df.loc[_kept]
                        _sub_col_var = pd.Series(
                            [_label_to_var[lbl] for lbl in _sub_coords.index],
                            index=_sub_coords.index,
                        )
                        _sub_labels = [
                            lbl[len(_label_to_var[lbl]) + len("__"):] if _label_to_var[lbl] != lbl else lbl
                            for lbl in _sub_coords.index
                        ]
                        _x = _sub_coords.iloc[:, 0].to_numpy()
                        _y = _sub_coords.iloc[:, 1].to_numpy() if _sub_coords.shape[1] > 1 else np.zeros(len(_x))
                        _unique_vars = list(dict.fromkeys(_sub_col_var))
                        _palette = colormaps["tab10"].resampled(max(len(_unique_vars), 1))
                        _color_map = {v: _palette(i) for i, v in enumerate(_unique_vars)}
                        _fig_mca, _ax_mca = plt.subplots(figsize=(9, 8))
                        _ax_mca.axhline(0, color="gray", linewidth=0.8)
                        _ax_mca.axvline(0, color="gray", linewidth=0.8)
                        for _v in _unique_vars:
                            _mask = _sub_col_var.to_numpy() == _v
                            _ax_mca.scatter(_x[_mask], _y[_mask],
                                            color=_color_map[_v], label=_v, s=40, alpha=0.85)
                        # Cap labels (not points -- every point still gets its dot
                        # scattered above) to the 20 most distinctive categories
                        # by distance from origin, matching local's
                        # save_mca_column_map. Without this, a batch can carry
                        # up to _MCA_COLUMN_MAP_BATCH variables' worth of
                        # categories (unbounded per-variable cardinality) with
                        # every single one labeled, which both looks unreadable
                        # and is far more prone to declutter_point_labels'
                        # cross-group collision edge case than local's capped version.
                        _MCA_MAX_LABELS_PER_BATCH = 20
                        _n_sub_labels = len(_sub_labels)
                        if _n_sub_labels > _MCA_MAX_LABELS_PER_BATCH:
                            _dist_mca = np.hypot(_x, _y)
                            _top_idx_mca = np.argsort(_dist_mca)[-_MCA_MAX_LABELS_PER_BATCH:]
                            _label_mask_mca = np.zeros(_n_sub_labels, dtype=bool)
                            _label_mask_mca[_top_idx_mca] = True
                        else:
                            _label_mask_mca = np.ones(_n_sub_labels, dtype=bool)
                        declutter_point_labels(
                            _ax_mca, _x[_label_mask_mca], _y[_label_mask_mca],
                            [l for l, show in zip(_sub_labels, _label_mask_mca) if show],
                        )
                        _dim1_pct = _mca_inertia_arr[0] * 100 if len(_mca_inertia_arr) > 0 else 0
                        _dim2_pct = _mca_inertia_arr[1] * 100 if len(_mca_inertia_arr) > 1 else 0
                        _ax_mca.set_xlabel(f"Dim 1 ({_dim1_pct:.1f}%)")
                        _ax_mca.set_ylabel(f"Dim 2 ({_dim2_pct:.1f}%)")
                        _title_mca = f"MCA Category Map — node{node_number} (batch {_bi + 1}/{_n_batches_mca})"
                        if _n_sub_labels > _MCA_MAX_LABELS_PER_BATCH:
                            _title_mca += f" — top {_MCA_MAX_LABELS_PER_BATCH} of {_n_sub_labels} categories shown"
                        _ax_mca.set_title(_title_mca)
                        _ax_mca.legend(title="Variable", bbox_to_anchor=(1.02, 1), loc="upper left")
                        _fig_mca.tight_layout()
                        _fname_mca = f"mca_column_map_batch_{_bi + 1:02d}.png"
                        output_files[f"{base}/mca/{_fname_mca}"] = _fig_to_bytes(_fig_mca)
                        plt.close(_fig_mca)
            except Exception as e:
                print(f"MCA column map error (node{node_number}): {e}", flush=True)

            # -- MCA explained inertia bar chart --
            try:
                _mca_inertia = r.get("mca_explained_inertia")
                if _mca_inertia:
                    _inertia_arr = np.array(_mca_inertia)
                    _cumulative = np.cumsum(_inertia_arr)
                    _n_dims = len(_inertia_arr)
                    _step = max(1, _n_dims // 12)
                    _fig_ei, _ax_ei = plt.subplots(figsize=(8, 5))
                    _ax_ei.bar(range(1, _n_dims + 1), _inertia_arr * 100, alpha=0.6, label="Per-dimension inertia")
                    _ax_ei.plot(range(1, _n_dims + 1), _cumulative * 100, marker="o", color="firebrick", label="Cumulative inertia")
                    _ax_ei.set_xticks(range(1, _n_dims + 1, _step))
                    _ax_ei.set_title(f"Explained Inertia by MCA Dimension — node{node_number}")
                    _ax_ei.set_xlabel("MCA Dimension")
                    _ax_ei.set_ylabel("Inertia (%)")
                    _ax_ei.legend()
                    _ax_ei.grid(True, alpha=0.3)
                    _fig_ei.tight_layout()
                    output_files[f"{base}/mca/mca_explained_inertia.png"] = _fig_to_bytes(_fig_ei)
                    plt.close(_fig_ei)
            except Exception as e:
                print(f"MCA inertia plot error (node{node_number}): {e}", flush=True)

            # -- PCA excluded-columns notice (entirely-missing columns dropped by run_pca) --
            try:
                _pca_dropped = r.get("pca_dropped_features") or []
                if _pca_dropped:
                    output_files[f"{base}/pca/excluded_columns.csv"] = _df_to_csv_bytes(
                        pd.DataFrame({"feature": _pca_dropped})
                    )
            except Exception as e:
                print(f"PCA excluded-columns CSV error (node{node_number}): {e}", flush=True)

            # -- PCA loadings biplot and scree plot --
            try:
                _pca_loadings = r.get("pca_loadings")
                _pca_feat_names = r.get("pca_feature_names")
                _pca_ev = r.get("pca_explained_variance")
                _pca_rec = r.get("pca_recommended_n_components")
                if _pca_loadings and _pca_feat_names and _pca_ev:
                    _ld = np.array(_pca_loadings)   # shape: (n_features, n_components)
                    _ev = np.array(_pca_ev)
                    _names = np.array(_pca_feat_names)
                    _n_comps = _ld.shape[1]

                    # Scree / explained variance plot
                    _cumev = np.cumsum(_ev)
                    _n_ev = len(_ev)
                    _step_ev = max(1, _n_ev // 12)
                    _fig_scree, _ax_scree = plt.subplots(figsize=(8, 5))
                    _ax_scree.bar(range(1, _n_ev + 1), _ev * 100, alpha=0.6, label="Per-component variance")
                    _ax_scree.plot(range(1, _n_ev + 1), _cumev * 100, marker="o", color="firebrick", label="Cumulative variance")
                    if _pca_rec:
                        _ax_scree.axvline(_pca_rec, color="gray", linestyle="--",
                                          label=f"{_pca_rec} component{'s' if _pca_rec != 1 else ''} reach {_cumev[_pca_rec - 1] * 100:.1f}%")
                    _ax_scree.set_xticks(range(1, _n_ev + 1, _step_ev))
                    _ax_scree.set_title(f"Explained Variance by Principal Component — node{node_number}")
                    _ax_scree.set_xlabel("Principal Component")
                    _ax_scree.set_ylabel("Explained Variance (%)")
                    _ax_scree.legend()
                    _ax_scree.grid(True, alpha=0.3)
                    _fig_scree.tight_layout()
                    output_files[f"{base}/pca/pca_explained_variance.png"] = _fig_to_bytes(_fig_scree)
                    plt.close(_fig_scree)

                    # Loadings biplot (PC1 vs PC2)
                    if _n_comps >= 2:
                        _pc_pairs = [(0, 1)]
                        if _n_comps >= 3:
                            _pc_pairs.append((0, 2))
                        for _pcx, _pcy in _pc_pairs:
                            _x = _ld[:, _pcx]
                            _y = _ld[:, _pcy]
                            _nm = _names.copy()
                            _MAX_LABELS = 20
                            if len(_nm) > _MAX_LABELS:
                                _mag = np.hypot(_x, _y)
                                _keep = np.argsort(_mag)[-_MAX_LABELS:]
                                _x, _y, _nm = _x[_keep], _y[_keep], _nm[_keep]
                                _truncated = True
                            else:
                                _truncated = False
                            _fig_ld, _ax_ld = plt.subplots(figsize=(8, 8))
                            _ax_ld.axhline(0, color="gray", linewidth=0.8)
                            _ax_ld.axvline(0, color="gray", linewidth=0.8)
                            for _xi, _yi in zip(_x, _y):
                                _ax_ld.annotate("", xy=(_xi, _yi), xytext=(0, 0),
                                                arrowprops=dict(arrowstyle="->", color="steelblue", alpha=0.7))
                            _max_radius_factor_ld = (
                                declutter_radial_labels(_ax_ld, _x, _y, _nm, fontsize=7)
                                if len(_x) else 1.08
                            )
                            _title_ld = f"PCA Loadings — node{node_number} (PC{_pcx+1} vs PC{_pcy+1})"
                            if _truncated:
                                _title_ld += f" — top {_MAX_LABELS} features"
                            _ax_ld.set_title(_title_ld)
                            _ax_ld.set_xlabel(f"PC{_pcx+1} ({_ev[_pcx]*100:.1f}%)")
                            _ax_ld.set_ylabel(f"PC{_pcy+1} ({_ev[_pcy]*100:.1f}%)")
                            # Labels for angularly-clustered arrows are staggered out to
                            # a larger radius (see declutter_radial_labels), sized from
                            # the actual radius used (plus room for the text itself) so
                            # outer labels stay inside the frame.
                            _lim = (
                                max(1.0, max(np.abs(_x).max(), np.abs(_y).max()) * (_max_radius_factor_ld + 0.3))
                                if len(_x) else 1.0
                            )
                            _ax_ld.set_xlim(-_lim, _lim)
                            _ax_ld.set_ylim(-_lim, _lim)
                            _fig_ld.tight_layout()
                            _fname_ld = f"pca_loadings_pc{_pcx+1}_pc{_pcy+1}.png"
                            output_files[f"{base}/pca/{_fname_ld}"] = _fig_to_bytes(_fig_ld)
                            plt.close(_fig_ld)
            except Exception as e:
                print(f"PCA loadings/scree plot error (node{node_number}): {e}", flush=True)

            try:
                _sex_counts = r.get("sex_counts", {})
                if _sex_counts:
                    _keys = [k for k in _sex_counts if pd.notna(k)]
                    _vals = [_sex_counts[k] for k in _keys]
                    fig, ax = plt.subplots(figsize=(6, 5))
                    ax.bar(_keys, _vals,
                           color=[PALETTE[i % len(PALETTE)] for i in range(len(_keys))])
                    ax.set_title(f"Sex Distribution — node{node_number}")
                    ax.set_xlabel("Sex")
                    ax.set_ylabel("Count")
                    ax.grid(axis="y", alpha=0.3)
                    output_files[f"{base}/categorical/sex_distribution.png"] = _fig_to_bytes(fig)
            except Exception as e:
                print(f"Sex distribution plot error (node{node_number}): {e}", flush=True)

            try:
                _col_types   = r.get("column_types", {})
                _type_counts = [
                    len(_col_types.get("numeric", [])),
                    len(_col_types.get("categorical", [])),
                    len(_col_types.get("temporal", [])),
                ]
                if any(_type_counts):
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.pie(_type_counts,
                           labels=["Numerical", "Categorical", "Temporal"],
                           autopct="%1.1f%%", startangle=90,
                           colors=["#A6D8A8", "#86C5F9", "#FFCC80"],
                           pctdistance=0.75, labeldistance=1.15)
                    ax.set_title(f"Data Type Distribution — node{node_number}")
                    output_files[f"{base}/overview/data_type_distribution.png"] = _fig_to_bytes(fig)
            except Exception as e:
                print(f"Data type pie chart error (node{node_number}): {e}", flush=True)

            # -- Temporal activity plots from observations_per_period in temporal_statistics --
            for _feat, _metrics in r.get("temporal_statistics", {}).items():
                _obs = _metrics.get("observations_per_period")
                if not _obs:
                    continue
                try:
                    _periods = [pd.to_datetime(p, errors="coerce") for p in _obs.keys()]
                    _df_tmp  = (
                        pd.DataFrame({"period": _periods, "count": list(_obs.values())})
                        .dropna(subset=["period"])
                        .sort_values("period")
                    )
                    if _df_tmp.empty:
                        continue
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(_df_tmp["period"], _df_tmp["count"], marker="o", linewidth=2)
                    _map = _metrics.get("most_active_period")
                    if _map:
                        _map_dt = pd.to_datetime(_map, errors="coerce")
                        _match  = _df_tmp[_df_tmp["period"] == _map_dt]
                        if not _match.empty:
                            ax.scatter(_match["period"], _match["count"], s=100,
                                       label="Most Active Period")
                            ax.legend()
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Number of Observations")
                    ax.set_title(f"{_feat} — Temporal Activity (Node {node_number})")
                    ax.tick_params(axis="x", rotation=45)
                    ax.grid(alpha=0.3)
                    fig.tight_layout()
                    output_files[f"{base}/temporal/{_feat}_activity.png"] = _fig_to_bytes(fig)
                except Exception as e:
                    print(f"Temporal activity plot error ({_feat}, node{node_number}): {e}", flush=True)

            # -- Batched temporal activity grid (several features per image, full-mode only;
            # short mode still uses the individual {feature}_activity.png files above so it
            # can pick a specific top-N-by-activity subset determined later at report time) --
            try:
                _temporal_batch_items = [
                    (_feat, _metrics) for _feat, _metrics in r.get("temporal_statistics", {}).items()
                    if _metrics.get("observations_per_period")
                ]
                if _temporal_batch_items:
                    _batch_size_tp = 6
                    _n_batches_tp = max(1, math.ceil(len(_temporal_batch_items) / _batch_size_tp))
                    for _b_tp in range(_n_batches_tp):
                        _batch_tp = _temporal_batch_items[_b_tp * _batch_size_tp:(_b_tp + 1) * _batch_size_tp]
                        _fig_tp, _axes_tp = make_subplots(len(_batch_tp), ncols=2, width=6, height=4)
                        for _ax_tp, (_feat_tp, _metrics_tp) in zip(_axes_tp, _batch_tp):
                            try:
                                _obs_tp = _metrics_tp.get("observations_per_period")
                                _periods_tp = [pd.to_datetime(p, errors="coerce") for p in _obs_tp.keys()]
                                _df_tp = (
                                    pd.DataFrame({"period": _periods_tp, "count": list(_obs_tp.values())})
                                    .dropna(subset=["period"])
                                    .sort_values("period")
                                )
                                if _df_tp.empty:
                                    continue
                                _ax_tp.plot(_df_tp["period"], _df_tp["count"], marker="o", linewidth=2)
                                _map_tp = _metrics_tp.get("most_active_period")
                                if _map_tp:
                                    _map_tp_dt = pd.to_datetime(_map_tp, errors="coerce")
                                    _match_tp = _df_tp[_df_tp["period"] == _map_tp_dt]
                                    if not _match_tp.empty:
                                        _ax_tp.scatter(_match_tp["period"], _match_tp["count"], s=60)
                                _ax_tp.set_title(_feat_tp, fontsize=9)
                                _ax_tp.tick_params(axis="x", rotation=45, labelsize=7)
                                _ax_tp.grid(alpha=0.3)
                            except Exception as e:
                                # One malformed feature must not blank out the whole batch image --
                                # the per-feature loop above has the same guard.
                                print(f"Temporal batched activity subplot error ({_feat_tp}, node{node_number}): {e}", flush=True)
                        _title_tp = f"Temporal Activity — node{node_number}"
                        if _n_batches_tp > 1:
                            _title_tp += f" ({_b_tp + 1}/{_n_batches_tp})"
                        _fig_tp.suptitle(_title_tp, fontsize=12)
                        _fig_tp.tight_layout()
                        output_files[
                            f"{base}/temporal/temporal_activity_batch_{_b_tp + 1:02d}.png"
                        ] = _fig_to_bytes(_fig_tp)
                        plt.close(_fig_tp)
            except Exception as e:
                print(f"Temporal batched activity plot error (node{node_number}): {e}", flush=True)

            # -- Quasi-numeric categorical columns notice (censored values like "<5") --
            try:
                _quasi_numeric_cols = r.get("quasi_numeric_categorical_columns") or []
                if _quasi_numeric_cols:
                    output_files[f"{base}/categorical/quasi_numeric_columns.csv"] = (
                        _df_to_csv_bytes(pd.DataFrame({"feature": _quasi_numeric_cols}))
                    )
            except Exception as e:
                print(f"Quasi-numeric categorical columns CSV error (node{node_number}): {e}", flush=True)

            # -- Categorical distribution plots from category_counts in categorical_statistics --
            try:
                _cat_stats   = r.get("categorical_statistics", {})
                _cat_plot_cols = [
                    (col, m["category_counts"])
                    for col, m in _cat_stats.items()
                    if isinstance(m.get("category_counts"), dict) and len(m["category_counts"]) >= 2
                ]
                if _cat_plot_cols:
                    _batch_size = 6
                    _n_batches  = max(1, math.ceil(len(_cat_plot_cols) / _batch_size))
                    for _b_i in range(_n_batches):
                        _batch = _cat_plot_cols[_b_i * _batch_size: (_b_i + 1) * _batch_size]
                        fig, axes = make_subplots(len(_batch), ncols=2, width=7, height=5)
                        for ax, (col, counts) in zip(axes, _batch):
                            # Cap at the top 20 categories by frequency, matching
                            # the modular project's count_plot(top_n=20) -- an
                            # uncapped high-cardinality column would otherwise
                            # produce 100+ overlapping bars/labels.
                            _top_counts = dict(
                                sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:20]
                            )
                            _labels = list(_top_counts.keys())
                            _vals   = list(_top_counts.values())
                            _colors = [PALETTE[i % len(PALETTE)] for i in range(len(_labels))]
                            if len(_labels) > 2:
                                ax.barh(_labels, _vals, color=_colors)
                                ax.set_xlabel("Count")
                            else:
                                ax.bar(_labels, _vals, color=_colors)
                                ax.set_ylabel("Count")
                            ax.set_title(col)
                        _title = f"Categorical Distributions — node{node_number}"
                        if _n_batches > 1:
                            _title += f" ({_b_i + 1}/{_n_batches})"
                        fig.suptitle(_title, fontsize=13, y=1.01)
                        fig.tight_layout(h_pad=3.0)
                        output_files[
                            f"{base}/categorical/categorical_distributions_{_b_i + 1:02d}.png"
                        ] = _fig_to_bytes(fig)
            except Exception as e:
                print(f"Categorical distribution plots error (node{node_number}): {e}", flush=True)

            # -- Column-availability pie chart (computed here — no df needed) --
            try:
                n_nodes_total = len(analysis_results)
                fig, ax = plt.subplots()
                pie_chart(
                    ax,
                    [all_common, partially_common, unique],
                    ["Common", "Partial", "Unique"],
                    colors=["#A6D8A8", "#86C5F9", "#FFCC80"],
                    title=(
                        f'Column Availability '
                        f'({n_nodes_total} node{"s" if n_nodes_total > 1 else ""})'
                    ),
                    # Only 3 possible slices here -- no label-overlap risk to
                    # guard against, so a small category shouldn't be renamed
                    # "Other" instead of shown by its real name.
                    min_slice_pct=0,
                )
                output_files[f"{base}/comparison/column_availability.png"] = (
                    _fig_to_bytes(fig)
                )
            except Exception as e:
                print(f"Column availability pie chart error for node{node_number}: {e}", flush=True)

            # -- Missing-values-by-column batched stacked bar charts --
            try:
                _missing_by_col = r.get("missing_by_col", {})
                _n_rows = r.get("n_rows", 1) or 1
                if _missing_by_col:
                    # Sort descending by missing count; include all cols (0-missing included)
                    _sorted_cols = sorted(
                        _missing_by_col.keys(),
                        key=lambda c: _missing_by_col[c],
                        reverse=True,
                    )

                    def _make_missing_batch_fig(cols_batch, n_rows, node_label, title):
                        _counts = [_missing_by_col.get(c, 0) for c in cols_batch]
                        _present = [n_rows - cnt for cnt in _counts]
                        _miss_pct = [cnt / n_rows * 100 for cnt in _counts]
                        _pres_pct = [p / n_rows * 100 for p in _present]
                        _fig, _ax = plt.subplots(
                            figsize=(10, max(4, len(cols_batch) * 0.45))
                        )
                        _ys = range(len(cols_batch))
                        _ax.barh(_ys, _pres_pct, color=PALETTE[2], label="Present")
                        _ax.barh(_ys, _miss_pct, left=_pres_pct, color=PALETTE[3], label="Missing")
                        _ax.set_yticks(list(_ys))
                        _ax.set_yticklabels(
                            [f"{c}  ({_missing_by_col.get(c,0)})" for c in cols_batch],
                            fontsize=8,
                        )
                        _ax.set_xlabel("% of rows")
                        _ax.set_xlim(0, 100)
                        _ax.set_title(title)
                        # Placed outside the axes rather than an inside loc --
                        # with bars spanning most of the 0-100% width (and
                        # rows sorted so the bottom ones are fully "Present"),
                        # any inside corner ends up overlapping a bar's data.
                        _ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1),
                                   fontsize=8, borderaxespad=0)
                        _fig.tight_layout()
                        return _fig

                    # Full mode: batches of 20
                    _batch = 20
                    _n_batches = max(1, -(-len(_sorted_cols) // _batch))
                    _total = len(_sorted_cols)
                    for _bi in range(_n_batches):
                        _batch_cols = _sorted_cols[_bi * _batch: (_bi + 1) * _batch]
                        _start = _bi * _batch + 1
                        _end = _start + len(_batch_cols) - 1
                        _title = (
                            f"Missing Values — node{node_number}"
                            f" (columns {_start}–{_end} of {_total})"
                        )
                        _fig = _make_missing_batch_fig(
                            _batch_cols, _n_rows, f"node{node_number}", _title
                        )
                        _fname = f"missing_values_by_column_{_bi + 1:02d}.png"
                        output_files[f"{base}/overview/{_fname}"] = _fig_to_bytes(_fig)
                        plt.close(_fig)

                    # Short mode: top 10 most missing
                    _short_cols = [c for c in _sorted_cols if _missing_by_col.get(c, 0) > 0][:10]
                    if not _short_cols:
                        _short_cols = _sorted_cols[:10]
                    _short_title = f"Missing Values (top {len(_short_cols)}) — node{node_number}"
                    _fig_s = _make_missing_batch_fig(
                        _short_cols, _n_rows, f"node{node_number}", _short_title
                    )
                    output_files[f"{base}/overview/missing_values_by_column_short.png"] = (
                        _fig_to_bytes(_fig_s)
                    )
                    plt.close(_fig_s)
            except Exception as e:
                print(f"Missing-by-column plot error for node{node_number}: {e}", flush=True)

            # -- Inferential CSVs and plots generated from records sent by the analyzer --
            inf = r.get("inferential_data", {})
            if inf:
                screening_records = inf.get("association_screening_records")
                if screening_records is not None:
                    try:
                        _scr_df = pd.DataFrame(screening_records)
                        output_files[f"{base}/inferential/association_screening.csv"] = (
                            _df_to_csv_bytes(_scr_df)
                        )
                        # Regenerate screening heatmap from the records
                        import tempfile as _tf
                        with _tf.TemporaryDirectory() as _td:
                            save_association_screening(
                                _scr_df, _td, node_label=f"node{node_number}"
                            )
                            _sc_png = Path(_td) / "association_screening.png"
                            if _sc_png.exists():
                                output_files[f"{base}/inferential/association_screening.png"] = (
                                    _sc_png.read_bytes()
                                )
                    except Exception as e:
                        print(f"association_screening error (node{node_number}): {e}", flush=True)

                sig_records = inf.get("significant_associations_records")
                if sig_records is not None:
                    try:
                        df_sig = pd.DataFrame(sig_records) if sig_records else pd.DataFrame()
                        output_files[f"{base}/inferential/significant_associations.csv"] = (
                            _df_to_csv_bytes(df_sig)
                        )
                    except Exception as e:
                        print(f"significant_associations CSV error (node{node_number}): {e}", flush=True)

                outcome_col_inf = inf.get("outcome_col")
                comp_records = inf.get("comparisons_by_outcome_records")
                if comp_records and outcome_col_inf:
                    try:
                        _comp_df = pd.DataFrame(comp_records)
                        output_files[
                            f"{base}/inferential/comparisons_by_{outcome_col_inf}.csv"
                        ] = _df_to_csv_bytes(_comp_df)
                        # Regenerate group comparisons summary plot
                        import tempfile as _tf
                        with _tf.TemporaryDirectory() as _td:
                            save_group_comparisons_summary(
                                _comp_df, _td, node_label=f"node{node_number}"
                            )
                            _gc_png = Path(_td) / "group_comparisons_summary.png"
                            if _gc_png.exists():
                                output_files[
                                    f"{base}/inferential/group_comparisons_summary.png"
                                ] = _gc_png.read_bytes()
                    except Exception as e:
                        print(f"comparisons_by_outcome error (node{node_number}): {e}", flush=True)

                # Posthoc heatmaps and CSVs from serialized p-value matrices
                for _ph_key, _ph_info in inf.get("posthoc_data", {}).items():
                    try:
                        _groups   = _ph_info["groups"]
                        _matrix   = _ph_info["matrix"]
                        _method   = _ph_info["method"]
                        _val_col  = _ph_info["value_col"]
                        _out_col  = _ph_info["outcome_col"]
                        _mat_df   = pd.DataFrame(_matrix, index=_groups, columns=_groups)
                        output_files[
                            f"{base}/inferential/comparisons/posthoc_{_val_col}_vs_{_out_col}.csv"
                        ] = _df_to_csv_bytes(_mat_df)
                        import tempfile as _tf
                        with _tf.TemporaryDirectory() as _td:
                            save_posthoc_heatmap(
                                _mat_df, _method, _val_col, _out_col, _td,
                                node_label=f"node{node_number}",
                            )
                            _ph_png = Path(_td) / f"posthoc_{_val_col}_vs_{_out_col}.png"
                            if _ph_png.exists():
                                output_files[
                                    f"{base}/inferential/comparisons/posthoc_{_val_col}_vs_{_out_col}.png"
                                ] = _ph_png.read_bytes()
                    except Exception as e:
                        print(f"Posthoc heatmap error ({_ph_key}, node{node_number}): {e}", flush=True)

    # -------------------------------------------------------------------------
    # PDF reports
    # -------------------------------------------------------------------------
    def _collect_pdf_reports(self, federated_results: dict,
                              analysis_results: list,
                              comparison_results: list,
                              output_files: dict) -> None:
        """Generate PDF reports in memory and add them to output_files."""
        all_n_ids = sorted(r["node_id"] for r in analysis_results)
        # Per-node reports
        for r in analysis_results:
            node_index  = all_n_ids.index(r["node_id"])
            node_number = node_index + 1
            base        = f"local/node{node_number}"
            node_comp   = next(
                (c for c in comparison_results if c["node_id"] == r["node_id"]), {}
            )
            for mode in ("short", "full"):
                try:
                    pdf_bytes = generate_local_report_bytes(
                        r, node_comp, mode=mode, node_number=node_number,
                        output_files=output_files,
                        federated_results=federated_results,
                    )
                    if pdf_bytes:
                        output_files[f"{base}/report_{mode}.pdf"] = pdf_bytes
                except Exception as e:
                    print(f"Local PDF ({mode}, node{node_number}) error: {e}", flush=True)

        # Federated report
        for mode in ("short", "full"):
            try:
                pdf_bytes = generate_global_report_bytes(
                    federated_results, mode=mode, output_files=output_files,
                )
                if pdf_bytes:
                    output_files[f"federated/report_{mode}.pdf"] = pdf_bytes
            except Exception as e:
                print(f"Global PDF ({mode}) error: {e}", flush=True)

    # -------------------------------------------------------------------------
    # JSON summaries
    # -------------------------------------------------------------------------
    def _collect_json_summaries(self, analysis_results: list,
                                 output_files: dict) -> None:
        """Generate summary.json for each node and for the federation and add to output_files.

        Reads the CSV bytes already stored in output_files by the earlier
        collectors and bundles them into a single JSON document per scope.
        Mirrors generate_local_json_summary / generate_global_json_summary but
        operates on the in-memory output_files dict instead of disk paths.
        """
        # Per-node JSON summaries
        all_n_ids = sorted(r["node_id"] for r in analysis_results)
        for r in analysis_results:
            node_index  = all_n_ids.index(r["node_id"])
            node_number = node_index + 1
            base = f"local/node{node_number}"
            try:
                summary = _build_json_from_output_files(
                    output_files, base, _LOCAL_JSON_CSV_MAP
                )
                # Find comparisons_by_<outcome>.csv dynamically (filename depends
                # on the auto-detected outcome column name).
                _inf_prefix = f"{base}/inferential/comparisons_by_"
                _comp_records = None
                for _key, _val in output_files.items():
                    if _key.startswith(_inf_prefix) and _key.endswith(".csv"):
                        try:
                            _comp_records = _csv_bytes_to_records(_val)
                        except Exception:
                            pass
                        break
                summary["comparisons_by_outcome"] = _comp_records
                output_files[f"{base}/summary.json"] = (
                    json.dumps(summary, indent=2).encode("utf-8")
                )
                print(f"JSON summary written for node{node_number}", flush=True)
            except Exception as e:
                print(f"Local JSON summary error (node{node_number}): {e}", flush=True)

        # Federated JSON summary
        try:
            fed_base = "federated"
            fed_summary = _build_json_from_output_files(
                output_files, fed_base, _GLOBAL_JSON_CSV_MAP
            )
            output_files[f"{fed_base}/summary.json"] = (
                json.dumps(fed_summary, indent=2).encode("utf-8")
            )
            print("Federated JSON summary written", flush=True)
        except Exception as e:
            print(f"Federated JSON summary error: {e}", flush=True)


# ============================================================================
# Source: hub_entry.py
# ============================================================================

def main():
    """Register DataReportAnalyzer and DataReportAggregator with StarModel and start the FLAME pipeline."""
    # Mirrors the working HALTA example:
    # - multiple_results=True, simple_analysis=False
    # - aggregator returns base64-encoded tar as [str]
    # - output_type="str", filename="results.tar.gz.b64.txt"
    StarModel(
        analyzer=DataReportAnalyzer,
        aggregator=DataReportAggregator,
        data_type="s3",
        query=[],
        multiple_results=True,
        simple_analysis=False,
        output_type="str",
        filename="results.tar.gz.b64.txt",
    )


if __name__ == "__main__":
    main()
