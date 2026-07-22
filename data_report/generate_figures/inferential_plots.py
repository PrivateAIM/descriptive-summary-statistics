"""
Inferential statistics visualizations.

All functions accept pre-computed result dicts and/or raw DataFrames.
No statistical tests are performed here — plotting only.

Sections:
  Group comparison   — two-group boxplots/violins, one-way comparison,
                       effect-size summary bars, association screening
  Correlation        — scatter + regression overlay, correlation matrix
  Chi-square         — annotated contingency heatmap, Cramer's V bar chart
  Regression         — coefficient plots, residuals, predicted vs. actual
  Time series        — power spectrum, peak annotation
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from data_report.generate_figures import style
from data_report.generate_figures.primitives import (
    bar_chart,
    boxplot,
    heatmap,
    histogram,
    line_chart,
    make_subplots,
    save_fig,
    scatter,
    violin,
)


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
    """Draw a significance bracket between two group positions on an axes."""
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
    Save a boxplot with significance bracket for a two-group comparison.

    The test method and p-value are shown in the title; a bracket annotates
    the significance level (*** / ** / * / n.s.) above the two boxes.

    Args:
        df (pd.DataFrame): Data containing the value and group columns.
        value_col (str): Name of the numeric column to plot on the y-axis.
        group_col (str): Name of the grouping column; must yield exactly two
            non-empty groups.
        result (dict): Result dict returned by ``compare_two_groups``.
        output_dir: Directory where the figure is saved.
        node_label (str, optional): Node identifier appended to the title.
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


def save_two_group_violins(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    result: dict,
    output_dir,
    *,
    node_label: Optional[str] = None,
) -> None:
    """
    Save a violin plot for a two-group comparison.

    Complements the boxplot view by showing the full distribution shape —
    useful when the two groups have very different spreads or multi-modal
    distributions.

    Args:
        df (pd.DataFrame): Data containing the value and group columns.
        value_col (str): Name of the numeric column to plot on the y-axis.
        group_col (str): Name of the grouping column; must yield at least
            two groups with more than one observation each.
        result (dict): Result dict returned by ``compare_two_groups``.
        output_dir: Directory where the figure is saved.
        node_label (str, optional): Node identifier appended to the title.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups, labels = [], []
    for name, g in df.groupby(group_col):
        vals = g[value_col].dropna().values
        if len(vals) > 1:
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

    fig, ax = plt.subplots(figsize=(6, 6))
    violin(ax, groups, labels=labels,
           title=f"{title}\n{method}  p = {p_str}")

    fname = f"{value_col}_vs_{group_col}_violin.png"
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
    Save one boxplot per group for a one-way comparison (ANOVA / Welch / Kruskal).

    Shows the test method and p-value in the title. For three or more groups
    there is no single significance bracket — the title carries the omnibus
    result and post-hoc details should be stored separately.

    Args:
        df (pd.DataFrame): Data containing the value and group columns.
        value_col (str): Name of the numeric column to plot on the y-axis.
        group_col (str): Name of the grouping column.
        result (dict): Result dict returned by a one-way comparison function.
        output_dir: Directory where the figure is saved.
        node_label (str, optional): Node identifier appended to the title.
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


def _posthoc_to_pvalue_matrix(posthoc_df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Normalise three possible posthoc_test output shapes into a square symmetric p-value matrix."""
    if method == "kruskal":
        return posthoc_df.astype(float)

    if method == "welch":
        groups = sorted(set(posthoc_df["A"]) | set(posthoc_df["B"]))
        mat = pd.DataFrame(np.nan, index=groups, columns=groups)
        for _, row in posthoc_df.iterrows():
            p = float(row["pval"])
            mat.loc[row["A"], row["B"]] = p
            mat.loc[row["B"], row["A"]] = p
        return mat

    if method == "anova":
        groups = sorted(set(posthoc_df["group1"]) | set(posthoc_df["group2"]))
        mat = pd.DataFrame(np.nan, index=groups, columns=groups)
        for _, row in posthoc_df.iterrows():
            p = float(row["p-adj"])
            mat.loc[row["group1"], row["group2"]] = p
            mat.loc[row["group2"], row["group1"]] = p
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
    """
    Save a heatmap of pairwise post-hoc p-values for a one-way comparison.

    Low p-values (dark cells) indicate pairs that differ significantly after
    the omnibus test.

    Args:
        posthoc_df (pd.DataFrame): DataFrame returned by ``posthoc_test``.
        method (str): Omnibus test method; one of ``"welch"``, ``"kruskal"``,
            or ``"anova"``. Controls how ``posthoc_df`` is normalised to a
            square p-value matrix.
        value_col (str): Name of the numeric outcome column; used in the title
            and filename.
        group_col (str): Name of the grouping column; used in the title and
            filename.
        output_dir: Directory where the figure is saved.
        node_label (str, optional): Node identifier appended to the title.
    """
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

    heatmap(
        ax, mat,
        cmap="Blues_r",
        vmin=0, vmax=1,
        annotate=True, fmt=".3f",
        title=title,
    )
    ax.set_xlabel(group_col)
    ax.set_ylabel(group_col)

    fname = f"posthoc_{value_col}_vs_{group_col}.png"
    save_fig(fig, output_dir / fname)


def save_group_comparisons_summary(
    comparison_df: pd.DataFrame,
    output_dir,
    *,
    alpha: float = 0.05,
    node_label: Optional[str] = None,
) -> None:
    """
    Save a horizontal bar chart of effect sizes from a group-comparison summary table.

    Bars are sorted by effect size descending. Significant comparisons
    (``p_value < alpha``) are highlighted with a distinct colour.

    Args:
        comparison_df (pd.DataFrame): Summary table with one row per
            (value_col, outcome_col) pair. Must include a column prefixed
            ``"effect_size_"`` and columns ``"value_col"`` and ``"p_value"``.
        output_dir: Directory where the figure is saved.
        alpha (float): Significance threshold used to colour-code bars.
        node_label (str, optional): Node identifier appended to the title.
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
        style.PALETTE[0] if row["p_value"] < alpha else style.PALETTE[6]
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
    sig_patch = mpatches.Patch(color=style.PALETTE[0], label=f"p < {alpha}")
    ns_patch = mpatches.Patch(color=style.PALETTE[6], label=f"p ≥ {alpha}")
    ax.legend(handles=[sig_patch, ns_patch], loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    save_fig(fig, output_dir / "group_comparisons_summary.png")


def save_association_screening(
    screening_df: pd.DataFrame,
    output_dir,
    *,
    top_n: int = 30,
    node_label: Optional[str] = None,
) -> None:
    """
    Save a two-panel association screening figure.

    The left panel shows an effect-size bar chart for the top significant
    pairs; the right panel shows a volcano-style scatter of effect size
    vs. −log₁₀(p_adj). If no significant associations exist, only the
    volcano panel is drawn.

    Args:
        screening_df (pd.DataFrame): DataFrame returned by
            ``screen_associations``.
        output_dir: Directory where the figure is saved.
        top_n (int): Maximum number of significant pairs shown in the bar
            chart.
        node_label (str, optional): Node identifier appended to the title.
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
            style.PALETTE[0] if s else style.PALETTE[6]
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

def save_correlation_scatter(
    df: pd.DataFrame,
    var1: str,
    var2: str,
    result: dict,
    output_dir,
    *,
    node_label: Optional[str] = None,
) -> None:
    """
    Save a scatter plot with regression line overlay for a two-variable correlation.

    Args:
        df (pd.DataFrame): Data containing both variables.
        var1 (str): Name of the first numeric variable (x-axis).
        var2 (str): Name of the second numeric variable (y-axis).
        result (dict): Result dict returned by
            ``correlation_between_two_variables``.
        output_dir: Directory where the figure is saved.
        node_label (str, optional): Node identifier appended to the title.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = df[[var1, var2]].dropna()
    if len(data) < 3:
        return

    method = result.get("method", "")
    r = result.get("correlation", np.nan)
    p = result.get("p_value", np.nan)
    r_str = f"{r:.3f}" if not np.isnan(r) else "n/a"
    p_str = f"{p:.4f}" if not np.isnan(p) else "n/a"

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter(
        ax,
        data[var1].values,
        data[var2].values,
        reg_line=True,
        title=(
            f"{var1} vs {var2}"
            + (f" — {node_label}" if node_label else "")
            + f"\n{method}  r = {r_str}  p = {p_str}"
        ),
        xlabel=var1,
        ylabel=var2,
    )
    fname = f"corr_{var1}_vs_{var2}.png"
    save_fig(fig, output_dir / fname)


def save_correlation_matrix(
    screening_df: pd.DataFrame,
    output_dir,
    *,
    node_label: Optional[str] = None,
) -> None:
    """
    Save a heatmap of pairwise correlations found in the screening DataFrame.

    Only num-num pairs (``pair_type == "num-num"``) are used. Variables are
    extracted from the ``var1`` and ``var2`` columns; missing pairs default
    to NaN. Requires at least two distinct variables to draw.

    Args:
        screening_df (pd.DataFrame): DataFrame returned by
            ``screen_associations``.
        output_dir: Directory where the figure is saved.
        node_label (str, optional): Node identifier appended to the title.
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

def save_contingency_heatmap(
    result: dict,
    output_dir,
    *,
    node_label: Optional[str] = None,
) -> None:
    """
    Save side-by-side heatmaps of observed and expected counts for a categorical association.

    Shows both the contingency table and expected frequencies so the reader
    can judge where the association is concentrated.

    Args:
        result (dict): Result dict returned by ``categorical_association``.
            Must contain ``contingency_table``, ``expected_frequencies``,
            ``var1``, ``var2``, ``p_value``, ``cramers_v``, and
            ``test_used``.
        output_dir: Directory where the figure is saved.
        node_label (str, optional): Node identifier appended to the figure
            title.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ct = result.get("contingency_table")
    ef = result.get("expected_frequencies")
    if ct is None or ef is None:
        return

    col1 = result.get("var1", "var1")
    col2 = result.get("var2", "var2")
    p = result.get("p_value", np.nan)
    v = result.get("cramers_v", np.nan)
    test = result.get("test_used", "")

    p_str = f"{p:.4f}" if not np.isnan(p) else "n/a"
    v_str = f"{v:.3f}" if not np.isnan(v) else "n/a"

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, ct.shape[0] * 0.6 + 2)))
    heatmap(axes[0], ct, cmap="Blues", annotate=True, fmt=".0f",
            title="Observed Counts")
    heatmap(axes[1], pd.DataFrame(ef, index=ct.index, columns=ct.columns),
            cmap="Oranges", annotate=True, fmt=".1f",
            title="Expected Counts")

    suptitle = (
        f"{col1} × {col2}  |  {test}  p = {p_str}  Cramer's V = {v_str}"
        + (f"  ({node_label})" if node_label else "")
    )
    fig.suptitle(suptitle, fontsize=10)
    fname = f"chi2_{col1}_vs_{col2}.png"
    save_fig(fig, output_dir / fname)


def save_cramers_v_bars(
    screening_df: pd.DataFrame,
    output_dir,
    *,
    top_n: int = 20,
    node_label: Optional[str] = None,
) -> None:
    """
    Save a horizontal bar chart of Cramer's V values for categorical pairs.

    Only cat-cat pairs from the screening DataFrame are shown, sorted
    descending by Cramer's V. Significant pairs are highlighted.

    Args:
        screening_df (pd.DataFrame): DataFrame returned by
            ``screen_associations``.
        output_dir: Directory where the figure is saved.
        top_n (int): Maximum number of pairs to display.
        node_label (str, optional): Node identifier appended to the title.
    """
    if screening_df is None or screening_df.empty:
        return
    cat_pairs = screening_df[
        (screening_df["pair_type"] == "cat-cat") &
        screening_df["effect_size"].notna()
    ].copy()
    if cat_pairs.empty:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cat_pairs = cat_pairs.sort_values("effect_size", ascending=True).tail(top_n)
    labels = [f"{r.var1} × {r.var2}" for r in cat_pairs.itertuples()]
    vals = cat_pairs["effect_size"].values
    colors = [
        style.PALETTE[0] if s else style.PALETTE[6]
        for s in cat_pairs["significant"]
    ]

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.4)))
    ax.barh(labels, vals, color=colors)
    ax.set_xlabel("Cramer's V")
    title = "Categorical Associations — Cramer's V"
    if node_label:
        title += f"  ({node_label})"
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    save_fig(fig, output_dir / "cramers_v_bars.png")


# ===========================================================================
# Regression section
# ===========================================================================

def save_regression_coefficients(
    result: dict,
    output_dir,
    *,
    node_label: Optional[str] = None,
) -> None:
    """
    Save a forest-plot-style coefficient chart for a regression result.

    Each row shows one predictor with its point estimate and 95% confidence
    interval. The intercept row is excluded since its scale is usually
    incomparable to the other coefficients.

    Args:
        result (dict): Result dict returned by ``regression``. Must contain
            ``coefficients`` (a DataFrame with columns ``"coef"``,
            ``"ci_lower"``, and ``"ci_upper"``), ``"type"``, and optionally
            ``"metrics"``.
        output_dir: Directory where the figure is saved.
        node_label (str, optional): Node identifier appended to the title.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    coef_table = result.get("coefficients")
    if coef_table is None or coef_table.empty:
        return

    coefs = coef_table.drop(index="const", errors="ignore").copy()
    if coefs.empty:
        return

    reg_type = result.get("type", "regression")
    metrics = result.get("metrics", {})
    metric_label = ""
    if "r_squared" in metrics:
        metric_label = f"  R² = {metrics['r_squared']:.3f}"
    elif "pseudo_r_squared" in metrics:
        metric_label = f"  pseudo-R² = {metrics['pseudo_r_squared']:.3f}"

    fig, ax = plt.subplots(figsize=(9, max(4, len(coefs) * 0.5)))
    y_pos = np.arange(len(coefs))
    ax.errorbar(
        coefs["coef"].values,
        y_pos,
        xerr=[
            (coefs["coef"] - coefs["ci_lower"]).values,
            (coefs["ci_upper"] - coefs["coef"]).values,
        ],
        fmt="o",
        color=style.PALETTE[0],
        ecolor=style.PALETTE[3],
        capsize=4,
        linewidth=1.5,
    )
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(coefs.index.astype(str))
    ax.set_xlabel("Coefficient (95% CI)")

    title = f"{reg_type.capitalize()} Regression Coefficients{metric_label}"
    if node_label:
        title += f"  ({node_label})"
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)

    save_fig(fig, output_dir / f"{reg_type}_coefficients.png")


def save_regression_residuals(
    result: dict,
    output_dir,
    *,
    node_label: Optional[str] = None,
) -> None:
    """
    Save a two-panel residuals plot for a linear regression result.

    The left panel shows predicted vs. actual values with a 45-degree
    identity line; the right panel shows a residuals histogram.

    Args:
        result (dict): Result dict returned by ``regression``. Must have
            ``type == "linear"`` and contain ``"model"`` and
            ``"predictions"``.
        output_dir: Directory where the figure is saved.
        node_label (str, optional): Node identifier appended to the title.
    """
    if result.get("type") != "linear":
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_result = result.get("model")
    predictions = result.get("predictions")
    if model_result is None or predictions is None:
        return

    try:
        actual = model_result.model.endog
        residuals = actual - np.asarray(predictions)
    except Exception:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Predicted vs. actual
    scatter(
        axes[0],
        np.asarray(predictions), actual,
        title="Predicted vs. Actual" + (f"  ({node_label})" if node_label else ""),
        xlabel="Predicted",
        ylabel="Actual",
    )
    lims = [
        min(actual.min(), np.min(predictions)),
        max(actual.max(), np.max(predictions)),
    ]
    axes[0].plot(lims, lims, color=style.PALETTE[1], linestyle="--", linewidth=1)

    # Residuals histogram
    histogram(
        axes[1], residuals,
        title="Residuals Distribution",
        xlabel="Residual",
    )

    save_fig(fig, output_dir / "linear_residuals.png")


def save_logistic_predicted_proba(
    result: dict,
    df: pd.DataFrame,
    target_col: str,
    output_dir,
    *,
    node_label: Optional[str] = None,
) -> None:
    """
    Save a histogram of predicted probabilities split by true class.

    Shows whether the logistic regression model separates the classes.

    Args:
        result (dict): Result dict returned by ``regression``. Must have
            ``type == "logistic"`` and contain ``"predictions"``.
        df (pd.DataFrame): Data containing the target column.
        target_col (str): Name of the binary target column.
        output_dir: Directory where the figure is saved.
        node_label (str, optional): Node identifier appended to the title.
    """
    if result.get("type") != "logistic":
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = result.get("predictions")
    if predictions is None:
        return

    y = df[target_col].dropna()
    pred = np.asarray(predictions)
    if len(pred) != len(y):
        return

    unique_classes = sorted(y.unique())
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, cls in enumerate(unique_classes):
        mask = (y.values == cls)
        ax.hist(
            pred[mask], bins=30,
            alpha=0.6,
            color=style.PALETTE[i % len(style.PALETTE)],
            label=str(cls),
        )
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Count")
    title = "Logistic Predicted Probabilities by Class"
    if node_label:
        title += f"  ({node_label})"
    ax.set_title(title)
    ax.legend(title=target_col)
    save_fig(fig, output_dir / "logistic_predicted_proba.png")


# ===========================================================================
# Time-series / spectral section
# ===========================================================================

def peak_annotation(x, y, k, ax=None, min_height=None, fft_labels=True):
    """
    Annotate the top-k peaks on an existing axes.

    Args:
        x (array-like): X-axis values (e.g. frequencies).
        y (array-like): Y-axis values (e.g. power).
        k (int): Number of peaks to annotate.
        ax (matplotlib.axes.Axes, optional): Target axes; defaults to the
            current axes.
        min_height (float, optional): Minimum peak height passed to
            ``scipy.signal.find_peaks``.
        fft_labels (bool): When True, annotations read ``freq=…/power=…``;
            otherwise ``x=…/y=…``.

    Note:
        When ``k == 1`` the single highest point is always annotated
        regardless of whether it is a local peak. For ``k > 1``,
        ``scipy.signal.find_peaks`` is called first; if no peaks are found
        the global maximum is used as a fallback.
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


def save_power_spectrum(
    seasonality_result: dict,
    output_dir,
    *,
    feature_name: str = "",
    top_k: int = 3,
    node_label: Optional[str] = None,
) -> None:
    """
    Save a power spectrum plot with peak annotations.

    Args:
        seasonality_result (dict): Result dict returned by
            ``detect_seasonality_fft``. Must contain ``"frequencies"`` and
            ``"power_spectrum"`` arrays.
        output_dir: Directory where the figure is saved.
        feature_name (str): Feature name prepended to the title and used in
            the output filename.
        top_k (int): Number of peaks to annotate via ``peak_annotation``.
        node_label (str, optional): Node identifier appended to the title.
    """
    freqs = np.asarray(seasonality_result.get("frequencies", []))
    power = np.asarray(seasonality_result.get("power_spectrum", []))
    if len(freqs) == 0 or len(power) == 0:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    line_chart(
        ax, freqs, power,
        markers=False,
        title=(
            (f"{feature_name} — " if feature_name else "")
            + "Power Spectrum (FFT)"
            + (f"  ({node_label})" if node_label else "")
        ),
        xlabel="Frequency",
        ylabel="Power",
    )
    peak_annotation(freqs, power, k=top_k, ax=ax, fft_labels=True)
    save_fig(fig, output_dir / f"{feature_name}_power_spectrum.png")
