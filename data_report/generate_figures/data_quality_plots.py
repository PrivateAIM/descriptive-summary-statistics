"""
Data quality visualizations.

Three public entry points:
  - save_missing_bar       missingno bar chart (column completeness overview)
  - save_missing_heatmap   missingno heatmap (nullity correlation between columns)
  - save_missing_by_column stacked horizontal bar: present vs. missing % per column
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd

from data_report.generate_figures import style


def save_missing_bar(df: pd.DataFrame, path, *, max_cols: int = 50) -> None:
    """
    Save a missingno bar chart showing the completeness of each column.

    Columns are sorted by ascending completeness (most incomplete first) when
    truncated, so the chart highlights the worst-quality features. The
    subtitle states how many columns are shown out of how many total.

    Args:
        df (pd.DataFrame): Source data to summarise.
        path: Destination path for the PNG file.
        max_cols (int): Maximum number of columns to include; the
            ``max_cols`` least complete columns are selected when the
            DataFrame exceeds this limit.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    total_cols = df.shape[1]
    if total_cols > max_cols:
        # Sort by ascending completeness (most missing first) before taking the subset
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
    msno.bar(subset, ax=ax, color=style.PALETTE[0], fontsize=12)
    ax.set_title(
        f"Column Completeness (non-null values per column)\n{subtitle}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=style.DPI, bbox_inches="tight")
    plt.close(fig)


def save_missing_heatmap(df: pd.DataFrame, path, *, max_cols: int = 50) -> bool:
    """
    Save a missingno heatmap showing nullity correlation between columns.

    High correlation means two columns tend to be missing together.
    Columns are capped at ``max_cols`` for readability.

    Args:
        df (pd.DataFrame): Source data to summarise.
        path: Destination path for the PNG file.
        max_cols (int): Maximum number of columns passed to the heatmap.

    Returns:
        bool: True when the file was written; False when the DataFrame
            has no missing values (no file is written in that case, and
            the caller should show a narrative message instead of a blank
            chart).
    """
    if df.isnull().sum().sum() == 0:
        return False

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    subset = df.iloc[:, :max_cols] if df.shape[1] > max_cols else df
    fig, ax = plt.subplots(figsize=(12, 9))
    msno.heatmap(subset, ax=ax, fontsize=12)
    ax.set_title("Nullity Correlation Between Columns", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=style.DPI, bbox_inches="tight")
    plt.close(fig)
    return True


def save_missing_by_column(
    missing_counts: dict,
    n_rows: int,
    path,
    *,
    node_label: Optional[str] = None,
    chunk_size: int = 20,
) -> list[Path]:
    """
    Save stacked horizontal bar chart(s) of present vs. missing percentage per column.

    Columns are sorted descending by missing percentage so the most
    problematic columns appear at the top. When there are more than
    ``chunk_size`` columns the chart is split into multiple images
    (e.g. columns 1–20, 21–40, …) so each one stays readable — a single
    chart with hundreds of columns becomes an unreadable wall of text when
    scaled to fit a report page.

    Args:
        missing_counts (dict): Mapping from column name to number of missing
            values.
        n_rows (int): Total number of rows in the source DataFrame; used to
            compute missing percentages.
        path: Base destination path for the PNG file(s). Batch suffixes
            (``_01``, ``_02``, …) are appended automatically when
            ``n_chunks > 1``.
        node_label (str, optional): Node identifier appended to the title.
        chunk_size (int): Maximum number of columns shown per image.

    Returns:
        list[Path]: Paths of all written image files, in order.
    """
    if not missing_counts or n_rows == 0:
        return []

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    missing_pct = {k: (v / n_rows) * 100 for k, v in missing_counts.items()}
    present_pct = {k: 100.0 - missing_pct[k] for k in missing_counts}

    plot_df = pd.DataFrame({
        "feature": list(missing_pct.keys()),
        "missing": list(missing_pct.values()),
        "present": list(present_pct.values()),
    })
    plot_df["feature"] = plot_df["feature"].astype(str)
    plot_df = plot_df.dropna(subset=["feature"])
    plot_df = plot_df.sort_values("missing", ascending=False).reset_index(drop=True)

    n_chunks = math.ceil(len(plot_df) / chunk_size)
    written = []
    for i in range(n_chunks):
        chunk = plot_df.iloc[i * chunk_size:(i + 1) * chunk_size]

        fig, ax = plt.subplots(figsize=(10, max(6, len(chunk) * 0.3)))
        ax.barh(chunk["feature"], chunk["present"],
                color=style.PALETTE[2], label="Present")
        ax.barh(chunk["feature"], chunk["missing"],
                left=chunk["present"], color=style.PALETTE[3], label="Missing")
        ax.invert_yaxis()

        ax.set_xlabel("Percentage (%)")
        title = "Missing vs Present Values by Feature"
        if node_label:
            title += f" — {node_label}"
        if n_chunks > 1:
            title += f" (columns {i * chunk_size + 1}-{i * chunk_size + len(chunk)} of {len(plot_df)})"
        ax.set_title(title)
        # Placed outside the axes rather than via loc="best" -- with bars
        # spanning most of the 0-100% width, "best" has nowhere free to put
        # it and ends up overlapping a bar's data instead.
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
        ax.grid(axis="x", alpha=0.3)

        fig.tight_layout()
        if n_chunks == 1:
            out_path = path
        else:
            out_path = path.with_name(f"{path.stem}_{i + 1:02d}{path.suffix}")
        fig.savefig(out_path, dpi=style.DPI, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)

    return written
