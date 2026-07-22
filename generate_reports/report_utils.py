"""Shared helpers for local and global report generation.

Layered like the rest of the reporting stack: low-level ReportLab/Pillow
helpers (tables, images, narrative boxes) at the bottom, comparison-column
and ranking helpers in the middle, and privacy-notice / narrative-summary
helpers at the top. Report builders (`generate_local_report.py`,
`generate_global_report.py`) compose these via `section_definitions.py`.
"""

import ast
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import Image, KeepTogether, Paragraph, Spacer, Table, TableStyle

PILImage.MAX_IMAGE_PIXELS = None

STYLES = getSampleStyleSheet()
TABLE_FONT_SIZE = 7
TABLE_MIN_FONT_SIZE = 5

SMALL_GROUP_THRESHOLD = 5
ID_KEYWORDS = ["id", "identifier", "patient_id", "identifikator"]


# ---------------------------------------------------------------------------
# Header / footer
# ---------------------------------------------------------------------------

def make_header(section_title):
    """Create a ReportLab canvas callback that draws page headers and footers.

    The returned callable is suitable for use as the ``onPage`` argument of a
    ReportLab ``PageTemplate``.  It draws the generation date-time in the
    bottom-left corner, the page number in the bottom-right, and the section
    title at the top-left.

    Args:
        section_title (str): Text to display at the top-left of every page.

    Returns:
        callable: A ``(canvas, doc)`` function compatible with ReportLab's
            ``onPage`` page-template hook.
    """
    def header(canvas, doc):
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
    """Append a heading paragraph and a small vertical spacer to an elements list.

    Args:
        elements (list): Mutable list of ReportLab flowables to append to.
        title (str): Heading text.
        level (int): Heading level; 1 uses ``Heading1`` style, any other value
            uses ``Heading2`` style.
    """
    style = STYLES["Heading1"] if level == 1 else STYLES["Heading2"]
    elements.append(Paragraph(title, style))
    elements.append(Spacer(1, 10))


def auto_fit_image(path, max_width, max_height):
    """Return the largest (width, height) that fits within a bounding box while preserving aspect ratio.

    Args:
        path (str or Path): Path to an image file readable by Pillow.
        max_width (float): Maximum allowed width in ReportLab units.
        max_height (float): Maximum allowed height in ReportLab units.

    Returns:
        tuple[float, float]: ``(width, height)`` scaled to fill the box while
            preserving the source aspect ratio.  Falls back to
            ``(max_width, max_height)`` if the image cannot be opened.
    """
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


def add_figure(elements, path, max_width=6.5 * inch, max_height=4 * inch, caption=None):
    """Append a titled figure block to an elements list.

    The figure title is derived from the file stem with underscores replaced
    by spaces and title-cased.  The block is wrapped in ``KeepTogether`` so
    the title and image stay on the same page.  Does nothing if the file does
    not exist.

    Args:
        elements (list): Mutable list of ReportLab flowables to append to.
        path (str or Path): Path to a PNG image file.
        max_width (float): Maximum image width in ReportLab units.
        max_height (float): Maximum image height in ReportLab units.
        caption (str or None): Optional italic caption rendered below the image.
    """
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


def add_plots_from_dir(elements, directory, max_width=6.5 * inch, max_height=4 * inch,
                        only=None, exclude=None):
    """Scan a directory for PNG files and append each as a titled figure.

    Files are sorted alphabetically before filtering.  Does nothing if the
    directory does not exist.

    Args:
        elements (list): Mutable list of ReportLab flowables to append to.
        directory (str or Path): Directory to scan for PNG files.
        max_width (float): Maximum image width in ReportLab units.
        max_height (float): Maximum image height in ReportLab units.
        only (set or None): If provided, only filenames in this set are
            included.
        exclude (set or None): If provided, filenames in this set are skipped.
    """
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
                           max_width=6.5 * inch, max_height=4 * inch):
    """Append a section heading followed by one or more figure blocks.

    The heading and the first figure are wrapped in a ``KeepTogether`` block
    to prevent an orphaned heading from appearing alone at the bottom of a
    page.  Non-existent paths in ``paths`` are silently skipped.

    Args:
        elements (list): Mutable list of ReportLab flowables to append to.
        title (str): Section heading text.
        paths (list[str or Path]): Ordered list of image paths to render.
        level (int): Heading level passed to ``add_section_heading``.
        max_width (float): Maximum image width in ReportLab units.
        max_height (float): Maximum image height in ReportLab units.

    Returns:
        bool: True if at least one figure was added, False otherwise.
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
    """Convert a DataFrame to a styled ReportLab Table with auto-fitted column widths.

    Column widths are proportional to maximum character count, subject to a
    per-column minimum that is wide enough to fit the longest unsplittable
    word (whitespace-separated token) in the header or any cell.  The font
    size shrinks from ``TABLE_FONT_SIZE`` down to ``TABLE_MIN_FONT_SIZE`` when
    minimum widths would otherwise overflow ``available_width``.

    p-value columns (``p_value``, ``p_adj``, ``p-val``, ``p-unc``, ``pval``)
    are formatted before general float rounding: exact ``0.0`` becomes
    ``"0"``, values below ``0.001`` become ``"< 0.001"``, and values at or
    above ``0.001`` are rounded to three decimal places.  All remaining float
    columns are rounded to three decimal places.  Column headers and selected
    system-value columns have underscores replaced with spaces for readability.

    Args:
        df (pd.DataFrame): Source data to render.
        available_width (float): Total available width in ReportLab units;
            column widths are scaled to fill this space.
        max_rows (int or None): If provided, only the first ``max_rows`` rows
            of ``df`` are rendered.

    Returns:
        reportlab.platypus.Table: Styled table ready to add to a document's
            flowable list.
    """
    if max_rows is not None:
        df = df.head(max_rows)

    # Format p-value columns first, before general rounding collapses small
    # values to 0.0.  Convention (matching medical journal style):
    #   exactly 0.0  → "0"
    #   0 < p < 0.001 → "< 0.001"
    #   p ≥ 0.001    → rounded to 3 decimal places
    _PVALUE_COLS = {"p_value", "p_adj", "p-val", "p-unc", "pval"}
    df = df.copy()
    for col in df.columns:
        if str(col).lower() in _PVALUE_COLS and pd.api.types.is_float_dtype(df[col]):
            def _fmt_p(v, _col=col):
                if pd.isna(v):
                    return ""
                f = float(v)
                if f == 0.0:
                    return "0"
                if f < 0.001:
                    return "< 0.001"
                return f"{f:.3f}"
            df[col] = df[col].apply(_fmt_p)

    # Round all remaining float columns to 3 decimal places.
    # Integer columns are unaffected; object/string columns are left as-is.
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
    # Replace underscores with spaces in column headers for readability,
    # but keep dataset-originated values (e.g. feature names) unchanged.
    header = [Paragraph(str(col).replace("_", " "), cell_style) for col in df.columns]
    # Columns whose values are system-generated descriptors (not dataset values)
    # — safe to replace underscores with spaces for readability.
    _SYSTEM_VALUE_COLS = {"vs_global", "comparison", "test", "effect_size_metric"}
    _avail_readable = {
        "common_all": "common in all",
        "common_partial": "common in some",
        "unique_local": "unique to this node",
        "not_common_all": "not common in all",
        "unknown": "unknown",
    }
    def _cell_text(col_name, value):
        s = str(value)
        if str(col_name) == "availability":
            return _avail_readable.get(s, s.replace("_", " "))
        if str(col_name) in _SYSTEM_VALUE_COLS:
            return s.replace("_", " ")
        return s
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
    """A typed message for inline explanatory or warning boxes in a report.

    Instances are passed to ``render_narrative`` to produce a styled paragraph
    block whose background colour and border colour depend on the level.

    Attributes:
        level (str): Severity or kind of the message.  One of ``"info"``,
            ``"warning"``, or ``"insight"``.
        text (str): Plain or ReportLab XML-escaped paragraph text to display.
    """

    level: Literal["info", "warning", "insight"]
    text: str


_NARRATIVE_STYLE = {
    "info": {"bg": colors.HexColor("#EAF2FB"), "border": colors.HexColor("#5B9BD5"), "label": "Note"},
    "warning": {"bg": colors.HexColor("#FDF3E7"), "border": colors.HexColor("#E0A458"), "label": "Warning"},
    "insight": {"bg": colors.HexColor("#EAF7EC"), "border": colors.HexColor("#6FBF73"), "label": "Insight"},
}


def render_narrative(elements, msg: NarrativeMessage):
    """Append a styled narrative box to an elements list.

    The box uses a coloured background and left border whose hue depends on
    ``msg.level``: blue for info, amber for warning, green for insight.

    Args:
        elements (list): Mutable list of ReportLab flowables to append to.
        msg (NarrativeMessage): Message to render.
    """
    style = _NARRATIVE_STYLE[msg.level]
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
    """Return an info NarrativeMessage explaining that a table was truncated.

    Args:
        shown (int): Number of rows shown in the rendered table.
        total (int): Total number of rows available in the full dataset.
        criterion (str): Description of the ranking criterion used to select
            the displayed rows, e.g. ``"deviating from the federated average"``.
        csv_filename (str): Filename of the CSV containing the full results.

    Returns:
        NarrativeMessage: Info-level message noting the truncation and
            pointing readers to the full CSV.
    """
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
    """Read a CSV file, returning None on any error or if the file is empty.

    Args:
        path (str or Path): Path to the CSV file.

    Returns:
        pd.DataFrame or None: Parsed DataFrame, or None if the file does not
            exist, cannot be parsed, or contains no rows.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    return df if not df.empty else None


def drop_internal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns whose names start with an underscore.

    These are internal bookkeeping columns (e.g. ``_rel_diff_abs``) that
    should not appear in rendered tables or exported CSVs.

    Args:
        df (pd.DataFrame): Input DataFrame, not modified in place.

    Returns:
        pd.DataFrame: View of ``df`` with underscore-prefixed columns removed.
    """
    return df.loc[:, [c for c in df.columns if not str(c).startswith("_")]]


def _format_range_value(value) -> str:
    """Format a numeric range boundary as an integer string or 3-significant-figure float."""
    if pd.isna(value):
        return "n/a"
    value = float(value)
    return str(int(value)) if value.is_integer() else f"{value:.3g}"


def prepare_numeric_display(df: pd.DataFrame) -> pd.DataFrame:
    """Curate a numeric summary DataFrame for display in a report table.

    Drops ``skewness`` and ``kurtosis`` columns (not actionable for a general
    audience) and collapses the ``min`` and ``max`` columns into a single
    ``range`` column formatted as ``[min, max]`` to reduce horizontal space.

    Args:
        df (pd.DataFrame): Numeric summary DataFrame, typically loaded from
            ``numeric_summary.csv``.

    Returns:
        pd.DataFrame: Modified copy with reduced columns.
    """
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
    """Join a local summary against a global summary and add a vs_global label column.

    Each row is labelled ``"above_average"``, ``"below_average"``,
    ``"similar"``, or ``"n/a"`` based on the relative difference between the
    local and federated values of ``value_col``.

    Args:
        local_df (pd.DataFrame): Per-node summary DataFrame.
        global_df (pd.DataFrame): Federated summary DataFrame.
        key_col (str): Column name used to join the two DataFrames.
        value_col (str): Column in ``local_df`` to compare against the
            federated value.
        threshold (float): Relative-difference boundary for above/below labels.
            Defaults to 0.1 (10%).
        global_value_col (str or None): Column in ``global_df`` that
            corresponds to ``value_col``.  Defaults to ``value_col``.

    Returns:
        pd.DataFrame: ``local_df`` extended with the federated value column,
            ``vs_global`` label, and ``_rel_diff_abs`` helper column.
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
    """Add federated comparison columns to a numeric summary DataFrame.

    Convenience wrapper around ``add_comparison_column`` that joins on
    ``"feature"`` and compares the ``"mean"`` column.

    Args:
        local_df (pd.DataFrame): Local numeric summary.
        global_df (pd.DataFrame): Federated numeric summary.
        threshold (float): Relative-difference threshold for above/below labels.

    Returns:
        pd.DataFrame: Extended DataFrame with ``vs_global`` and
            ``_rel_diff_abs`` columns.
    """
    return add_comparison_column(local_df, global_df, key_col="feature",
                                   value_col="mean", threshold=threshold)


def _parse_dict(value):
    """Parse a stringified dict via ast.literal_eval, returning an empty dict on failure."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError, TypeError):
        return {}


def add_categorical_comparison(local_df, global_df, threshold=0.1):
    """Add federated comparison columns to a categorical summary DataFrame.

    Compares the local share of each feature's most frequent category against
    that same category's share in the federation (apples-to-apples).  The
    global relative-frequency dict stores fractions (0–1); the local dict
    stores percentages (0–100).  The function scales the global value before
    comparing.

    Args:
        local_df (pd.DataFrame): Local categorical summary with
            ``relative_frequencies`` (stringified dict of category to pct)
            and ``most_frequent_category`` columns.
        global_df (pd.DataFrame): Federated categorical summary with a
            ``relative_freq`` column (stringified dict of category to fraction).
        threshold (float): Relative-difference threshold for above/below labels.

    Returns:
        pd.DataFrame: Copy of ``local_df`` extended with ``top cat % (local)``,
            ``top cat % (global)``, ``vs_global``, and ``_rel_diff_abs``
            columns.
    """
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
    """Add federated comparison columns to a temporal summary DataFrame.

    Derives a total record count from the global ``counts_per_period``
    stringified dict and compares each feature's local ``count`` against that
    total.

    Args:
        local_df (pd.DataFrame): Local temporal summary DataFrame with a
            ``count`` column.
        global_df (pd.DataFrame): Federated temporal summary DataFrame with a
            ``counts_per_period`` column containing stringified dicts.
        threshold (float): Relative-difference threshold for above/below labels.

    Returns:
        pd.DataFrame: Extended DataFrame with ``vs_global`` and
            ``_rel_diff_abs`` columns.
    """
    global_copy = global_df.copy()

    def _total_count(value):
        counts = _parse_dict(value)
        return sum(counts.values()) if counts else np.nan

    global_copy["count"] = global_copy["counts_per_period"].apply(_total_count)
    return add_comparison_column(local_df, global_copy, key_col="feature",
                                   value_col="count", threshold=threshold)


# ---------------------------------------------------------------------------
# Short-mode ranking
# ---------------------------------------------------------------------------

def rank_by_deviation(df, max_rows=10):
    """Rank rows by descending absolute deviation from the federated average.

    Uses the ``_rel_diff_abs`` helper column produced by the comparison
    functions.  Falls back to the first ``max_rows`` rows if that column is
    absent.

    Args:
        df (pd.DataFrame): DataFrame optionally containing a ``_rel_diff_abs``
            column.
        max_rows (int): Maximum number of rows to return.

    Returns:
        pd.DataFrame: Subset of ``df`` sorted with the most-deviating rows
            first.
    """
    if "_rel_diff_abs" in df.columns:
        return df.sort_values("_rel_diff_abs", ascending=False, na_position="last").head(max_rows)
    return df.head(max_rows)


def rank_by_imbalance(df, max_rows=10):
    """Rank categorical rows by descending class imbalance ratio.

    Uses the ``class_imbalance_ratio`` column if present.  Falls back to the
    first ``max_rows`` rows if that column is absent.

    Args:
        df (pd.DataFrame): Categorical summary DataFrame.
        max_rows (int): Maximum number of rows to return.

    Returns:
        pd.DataFrame: Subset of ``df`` with the most skewed categories first.
    """
    if "class_imbalance_ratio" in df.columns:
        return df.sort_values("class_imbalance_ratio", ascending=False, na_position="last").head(max_rows)
    return df.head(max_rows)


def rank_by_activity(df, max_rows=10):
    """Rank temporal rows by descending record count.

    Uses the ``count`` column if present.  Falls back to the first ``max_rows``
    rows if that column is absent.

    Args:
        df (pd.DataFrame): Temporal summary DataFrame.
        max_rows (int): Maximum number of rows to return.

    Returns:
        pd.DataFrame: Subset of ``df`` with the most active features first.
    """
    if "count" in df.columns:
        return df.sort_values("count", ascending=False, na_position="last").head(max_rows)
    return df.head(max_rows)


def rank_by_effect_size(df, max_rows=10):
    """Rank inferential rows by descending absolute effect size.

    Uses the absolute value of the ``effect_size`` column if present and
    non-empty.  Falls back to the first ``max_rows`` rows otherwise.

    Args:
        df (pd.DataFrame): Significant-associations DataFrame.
        max_rows (int): Maximum number of rows to return.

    Returns:
        pd.DataFrame: Subset of ``df`` with the strongest associations first.
    """
    if "effect_size" in df.columns and not df.empty:
        order = df["effect_size"].abs().sort_values(ascending=False).index
        return df.reindex(order).head(max_rows)
    return df.head(max_rows)


# ---------------------------------------------------------------------------
# Narrative summarizers
# ---------------------------------------------------------------------------

def summarise_numeric(numeric_df) -> NarrativeMessage:
    """Produce a narrative summary of the numeric variables section.

    Args:
        numeric_df (pd.DataFrame or None): Numeric summary DataFrame,
            optionally containing a ``vs_global`` column.

    Returns:
        NarrativeMessage: Info-level message stating the variable count and,
            when comparison data is present, how many variables are above or
            below the federated average.
    """
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
    """Produce a narrative summary of the categorical variables section.

    Args:
        categorical_df (pd.DataFrame or None): Categorical summary DataFrame,
            optionally containing a ``vs_global`` column.

    Returns:
        NarrativeMessage: Info-level message stating the variable count and,
            when comparison data is present, how many top categories are more
            or less dominant than the federation average.
    """
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
    """Produce a narrative summary of the temporal variables section.

    Args:
        temporal_df (pd.DataFrame or None): Temporal summary DataFrame.

    Returns:
        NarrativeMessage: Info-level message stating the variable count.
    """
    if temporal_df is None or temporal_df.empty:
        return NarrativeMessage(level="info", text="No temporal variables available.")
    return NarrativeMessage(level="info", text=f"{len(temporal_df)} temporal variable(s) analyzed.")


def summarise_inferential(significant_df, pair_type=None) -> NarrativeMessage:
    """Produce a narrative summary of the significant associations section.

    Reports the total count of significant associations and names the
    strongest one by absolute effect size.  Optionally filters to a specific
    pair type before summarising.

    Args:
        significant_df (pd.DataFrame or None): Significant-associations
            DataFrame with ``pair_type``, ``var1``, ``var2``, ``test``,
            ``effect_size_metric``, and ``effect_size`` columns.
        pair_type (str or None): If provided, only rows whose ``pair_type``
            matches are considered (e.g. ``"num-cat"``, ``"num-num"``).

    Returns:
        NarrativeMessage: Insight-level message if significant associations
            exist; info-level message otherwise.
    """
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
    """Return True if the feature name contains any known identifier keyword."""
    name = str(feature_name).lower()
    return any(k in name for k in ID_KEYWORDS)


def detect_identifier_features(*dataframes) -> list:
    """Scan summary DataFrames for features whose names resemble identifiers.

    A feature name is considered identifier-like if it contains any keyword
    from ``ID_KEYWORDS`` (case-insensitive substring match).

    Args:
        *dataframes: Variable number of DataFrames or None values.  Each
            DataFrame is expected to have a ``"feature"`` column.

    Returns:
        list[str]: Sorted, deduplicated list of feature names that appear to
            be identifiers.
    """
    found = []
    for df in dataframes:
        if df is None or "feature" not in df.columns:
            continue
        for f in df["feature"]:
            if _looks_like_identifier(f):
                found.append(f)
    return sorted(set(found))


def compute_categorical_group_sizes(categorical_df, threshold=SMALL_GROUP_THRESHOLD):
    """Estimate the size of every category group in a categorical summary.

    Group size is estimated as ``(relative_frequency_pct / 100) * count``.

    Args:
        categorical_df (pd.DataFrame or None): Categorical summary DataFrame
            with ``relative_frequencies``, ``count``, and ``feature`` columns.
        threshold (int): Group size below which a group is considered small.

    Returns:
        tuple: A four-element tuple
            ``(min_size, min_feature, min_category, flagged)`` where
            ``min_size`` is the smallest estimated group size (or None if no
            data), ``min_feature`` and ``min_category`` name the corresponding
            feature and category, and ``flagged`` is a list of
            ``(feature, category, size)`` tuples for groups below
            ``threshold``.
    """
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
    """Return privacy-warning NarrativeMessages for categories with small group sizes.

    At most one warning is emitted per feature (covering its first flagged
    category).

    Args:
        categorical_df (pd.DataFrame or None): Categorical summary DataFrame.
        threshold (int): Minimum group size below which a warning is raised.

    Returns:
        list[NarrativeMessage]: Warning-level messages, one per affected
            feature.  Empty list if no groups fall below the threshold.
    """
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

    ``save_categorical_distributions`` only plots columns with >= 2 distinct
    non-null values -- a column with a single observed category has nothing
    to show. Without this notice such a column would simply be absent from
    the distributions section with no explanation.

    Args:
        categorical_df (pd.DataFrame or None): Categorical summary DataFrame
            with a ``number_of_categories`` column.

    Returns:
        NarrativeMessage or None: Info-level message naming the excluded
            column(s), or None if there are none (or no data).
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

    A lab-result column recorded as e.g. ``["12.5", "8.1", "<5", "300.2"]`` is
    classified as categorical because the censored value ``"<5"`` keeps the
    whole column at object dtype -- with no numeric statistics computed for
    it and no indication anywhere why. This reads
    ``quasi_numeric_columns.csv`` (written by
    ``detect_quasi_numeric_categorical_columns``) to surface the gap.

    Args:
        categorical_dir (Path): The node's categorical output directory,
            which may contain ``quasi_numeric_columns.csv``.

    Returns:
        NarrativeMessage or None: Info-level message naming the affected
            column(s), or None if there are none (or no such file).
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
    counts don't match the full patient count. This reads
    ``age_out_of_range.csv`` (written by ``count_out_of_range_ages``) to
    surface the gap.

    Args:
        numeric_dir (Path): The node's (or federation's) numeric output
            directory, which may contain ``age_out_of_range.csv``.

    Returns:
        NarrativeMessage or None: Info-level message stating the count, or
            None if there are no out-of-range ages (or no such file).
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


def reduction_excluded_columns_notice(subdir, title) -> Optional[NarrativeMessage]:
    """Note which columns run_pca/run_mca silently dropped for being entirely missing.

    Both functions drop columns with no non-missing values rather than
    failing the whole analysis (their variance/categories are undefined).
    Without this notice such a column would just be absent from the
    projection with no explanation.

    Args:
        subdir (Path): The PCA or MCA output directory for this node
            (e.g. ``node_dir / "pca"``), which may contain an
            ``excluded_columns.csv`` written by ``save_pca_outputs`` /
            ``save_mca_outputs``.
        title (str): Human-readable name of the reduction method, used in
            the notice text (e.g. ``"PCA"``).

    Returns:
        NarrativeMessage or None: Info-level message naming the excluded
            column(s), or None if there are none (or no such file).
    """
    excluded_df = safe_read_csv(Path(subdir) / "excluded_columns.csv")
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


def build_privacy_notice(report_type, n_nodes=None, numeric_df=None,
                          categorical_df=None, temporal_df=None,
                          threshold=SMALL_GROUP_THRESHOLD) -> list:
    """Build a list of ReportLab flowables for the report's privacy and governance block.

    The block appears on the title page and describes the data scope, what was
    and was not shared between nodes, any detected identifier-like columns, the
    smallest displayed category group, and a caveat when the federation is small.

    Args:
        report_type (str): Either ``"local"`` (single-node report) or
            ``"global"`` (federated report).
        n_nodes (int or None): Number of participating nodes; used in the
            federated description and small-federation caveats.
        numeric_df (pd.DataFrame or None): Local numeric summary, used for
            identifier detection.
        categorical_df (pd.DataFrame or None): Local categorical summary,
            used for identifier detection and small-group checks.
        temporal_df (pd.DataFrame or None): Local temporal summary, used for
            identifier detection.
        threshold (int): Small-group reporting threshold forwarded to
            ``compute_categorical_group_sizes``.

    Returns:
        list: ReportLab flowables (Paragraph, Spacer) forming the privacy
            notice block.
    """
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


def small_n_nodes_caveat(n_nodes, threshold=5) -> Optional[NarrativeMessage]:
    """Return a warning NarrativeMessage when the federation has too few nodes.

    Args:
        n_nodes (int or None): Number of participating nodes.
        threshold (int): Minimum number of nodes needed to suppress the
            warning.  Defaults to 5.

    Returns:
        NarrativeMessage or None: Warning-level message if ``n_nodes`` is not
            None and is below ``threshold``; None otherwise.
    """
    if n_nodes is None or n_nodes >= threshold:
        return None
    return NarrativeMessage(
        level="warning",
        text=(
            f"This comparison is shown against a federation of only {n_nodes} "
            f"node(s). 'Above/below average' labels may indirectly reveal "
            f"information about other individual nodes."
        ),
    )
