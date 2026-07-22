"""Local node report generator.

generate_local_report(node_dir, output_dir, mode="full") -> Path

Builds a per-node PDF report from the analysis outputs under
results/local_results/<node>/. Two modes:
  - "short": narrative summaries, key plots, ranked top-N tables
  - "full":  full per-variable detail
"""

from datetime import datetime
from pathlib import Path

from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, Frame, NextPageTemplate, PageBreak, PageTemplate,
    Paragraph, Spacer,
)

from generate_reports.report_utils import (
    STYLES,
    NarrativeMessage,
    add_categorical_comparison,
    add_figure,
    add_heading_and_plots,
    add_numeric_comparison,
    add_plots_from_dir,
    add_section_heading,
    add_temporal_comparison,
    build_privacy_notice,
    categorical_excluded_from_distributions_notice,
    categorical_small_group_warnings,
    create_table,
    drop_internal_columns,
    make_header,
    out_of_range_age_notice,
    prepare_numeric_display,
    quasi_numeric_categorical_notice,
    rank_by_activity,
    rank_by_deviation,
    rank_by_effect_size,
    rank_by_imbalance,
    reduction_excluded_columns_notice,
    render_narrative,
    safe_read_csv,
    summarise_categorical,
    summarise_inferential,
    summarise_numeric,
    summarise_temporal,
    truncation_note,
)
from generate_reports.section_definitions import (
    LOCAL_MCA,
    LOCAL_PCA,
    SHORT_PLOT_MAX,
    SHORT_TABLE_MAX_ROWS,
)

MAX_W = 6.8 * inch
MAX_H = 4 * inch
PAGE_MARGIN = 0.4 * inch


def generate_local_report(node_dir, output_dir, mode="full",
                           results_dir=None, export_comparison_csv=False) -> Path:
    """Build a per-node PDF statistical report and write it to disk.

    Loads per-node CSV outputs from ``node_dir``, optionally joins them
    against federated aggregates for above/below-average annotations, then
    assembles a multi-section ReportLab document covering overview, numeric,
    categorical, temporal, cross-variable, and glossary sections.

    Args:
        node_dir (str or Path): Root directory of the node's analysis outputs
            (e.g. ``results/local_results/<node>/``).
        output_dir (str or Path): Directory where the PDF will be written.
        mode (str): Report detail level.  ``"short"`` includes narrative
            summaries, key plots, and ranked top-N tables; ``"full"`` includes
            complete per-variable detail.
        results_dir (str or Path or None): Override for the federated results
            directory.  Defaults to two levels above ``node_dir`` under
            ``federated_results/``.
        export_comparison_csv (bool): If True, writes augmented CSVs containing
            the ``vs_global`` and ``_rel_diff_abs`` columns alongside the PDF.

    Returns:
        Path: Absolute path of the written PDF file.
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
    """Extract the participating-node count from a global overview DataFrame."""
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
    """Write comparison-augmented summary CSVs alongside the node's analysis outputs."""
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
    """Append title-page flowables (heading, metadata, privacy notice) to the elements list."""
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
    """Append Section 1 (Overview) flowables, including the summary table and data-quality plots."""
    add_section_heading(elements, "1. Overview")
    overview_dir = node_dir / "overview"

    if overview_df is None:
        render_narrative(elements, NarrativeMessage("info", "No overview data available."))
    else:
        elements.append(create_table(overview_df, doc.width))
        elements.append(Spacer(1, 14))

    add_figure(elements, overview_dir / "data_type_distribution.png", MAX_W, MAX_H)

    missing_by_col_paths = sorted(overview_dir.glob("missing_values_by_column*.png"))
    if mode == "short":
        # Compact summary: just the missingno bar (annotated with column count)
        quality_plots = [overview_dir / "missingno_bar.png"]
    else:
        # Full detail: per-column stacked charts + nullity heatmap (no bar — redundant)
        quality_plots = [*missing_by_col_paths, overview_dir / "missingno_heatmap.png"]
    has_quality_plots = add_heading_and_plots(elements, "1.2 Data Quality", quality_plots,
                                               level=2, max_width=MAX_W, max_height=MAX_H)
    # If heatmap is absent in full mode the dataset likely has no missing values at all
    if not has_quality_plots or (mode == "full"
                                 and not (overview_dir / "missingno_heatmap.png").exists()):
        render_narrative(elements, NarrativeMessage(
            "info",
            "All columns are fully complete for this node — "
            "no missing-value patterns to display.",
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
    """Append Section 2 (Numeric) flowables, including descriptive table, PCA, and correlations."""
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
    add_heading_and_plots(elements, "2.2 Distributions", _dist_plots, level=2,
                           max_width=MAX_W, max_height=MAX_H)
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
    """Append Section 3 (Categorical) flowables, including descriptive table, MCA, and associations."""
    add_section_heading(elements, "3. Categorical Section")
    render_narrative(elements, NarrativeMessage(
        "info",
        "Categorical variables take values from a fixed set of categories "
        "(e.g. sex, diagnosis codes, yes/no flags). "
        "The summary table shows the count of valid observations, number of distinct categories, "
        "most and least frequent category, class imbalance ratio (how much more common the top "
        "category is than the rarest one), and number of missing values. "
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

    # Sex distribution + any batched multi-category distributions
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
    """Append Section 4 (Temporal) flowables, including descriptive table and line charts."""
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
    """Append Section 5 (Cross-Variable Associations) flowables, including pairwise, multi-group, and post-hoc subsections."""
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

    # --- 5.1 Pairwise Associations (two-group outcome) -----------------------
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
        "but non-zero — this is the standard convention in medical reporting. "
        "A value of '< 0.001' is not zero; it means the probability of seeing this result "
        "by chance is less than 1 in 1000.",
    ))
    _render_pairwise_table(elements, doc, significant_df, pair_type="num-cat", mode=mode)

    if mode == "full":
        add_figure(elements, inferential_dir / "group_comparisons_summary.png", MAX_W, MAX_H)
        two_group_plots = sorted(comparisons_dir.glob("*_two_group.png")) \
            + sorted(comparisons_dir.glob("*_violin.png")) \
            if comparisons_dir.exists() else []
        for p in sorted(two_group_plots, key=lambda f: f.name):
            add_figure(elements, p, MAX_W, MAX_H)

    # --- 5.2 Multi-Group Outcome Comparisons (3+ groups) ---------------------
    add_section_heading(elements, "5.2 Multi-Group Outcome Comparisons (3+ Groups)", level=2)
    render_narrative(elements, NarrativeMessage(
        "info",
        "When the detected outcome column has 3 or more groups, a one-way omnibus test is "
        "used instead of a two-group test. "
        "Welch's ANOVA is applied when the data is approximately normally distributed; "
        "Kruskal-Wallis is used when the distribution is skewed or variances are unequal. "
        "Each boxplot shows the distribution of a numeric variable split by each outcome "
        "group, with the omnibus test name and p-value in the title. "
        "Variables with a non-significant omnibus result (p ≥ 0.05) still appear here — "
        "they show no meaningful difference across groups.",
    ))

    oneway_plots = sorted(comparisons_dir.glob("*_oneway.png")) \
        if comparisons_dir.exists() else []

    if not oneway_plots:
        render_narrative(elements, NarrativeMessage(
            "info",
            "No outcome column with 3 or more groups was detected in this dataset. "
            "Multi-group comparisons require an outcome variable with at least 3 distinct "
            "groups, each containing a minimum of 20 observations. "
            "See Section 5.1 above for two-group pairwise associations.",
        ))
    elif mode == "full":
        for p in oneway_plots:
            add_figure(elements, p, MAX_W, MAX_H)

    # --- 5.3 Post-Hoc Pairwise Tests -----------------------------------------
    if oneway_plots:
        add_section_heading(elements, "5.3 Post-Hoc Pairwise Tests", level=2)
        render_narrative(elements, NarrativeMessage(
            "info",
            "When the omnibus test is significant (p < 0.05), pairwise post-hoc tests "
            "identify which specific group pairs differ from each other. "
            "Games-Howell post-hoc is used after Welch's ANOVA; "
            "pairwise Mann-Whitney U with Holm-Bonferroni correction is used after "
            "Kruskal-Wallis. "
            "Each heatmap shows the adjusted p-values for every pair of groups: "
            "darker blue (lower p-value) means the two groups are significantly different "
            "from each other. "
            "The diagonal is blank (a group is not compared against itself). "
            "Values near 0.000 indicate very strong evidence of a difference between those "
            "two groups; values close to 1.000 indicate no significant difference.",
        ))

        posthoc_plots = sorted(comparisons_dir.glob("posthoc_*.png")) \
            if comparisons_dir.exists() else []

        if not posthoc_plots:
            render_narrative(elements, NarrativeMessage(
                "info",
                "No significant multi-group associations were found "
                "(omnibus p ≥ 0.05 for all variables tested). "
                "Post-hoc pairwise tests are only run when the omnibus test is significant. "
                "This may indicate that the outcome groups do not differ meaningfully on any "
                "of the numeric variables, or that the sample size per group is too small to "
                "detect a difference reliably.",
            ))
        elif mode == "full":
            for p in posthoc_plots:
                add_figure(elements, p, MAX_W, MAX_H)


# ---------------------------------------------------------------------------
# Shared subsection renderers
# ---------------------------------------------------------------------------

def _render_pairwise_table(elements, doc, significant_df, pair_type, mode):
    """Append a narrative summary and table for one pair-type of significant associations."""
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


def _render_reduction_subsection(elements, node_dir, mode, spec, level, prefix):
    """Append a dimensionality-reduction subsection (PCA or MCA plots) to the elements list."""
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
        # (which is kept for short mode only).
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


# ---------------------------------------------------------------------------
# Glossary appendix
# ---------------------------------------------------------------------------

_GLOSSARY = [
    # Statistical methods
    ("p-value", "Probability of observing a result this extreme by chance if there is no real effect. A small p-value (< 0.05 after correction) supports rejecting the null hypothesis."),
    ("FDR correction", "False Discovery Rate: adjusts p-values when many tests are run simultaneously, reducing the chance of false positives."),
    ("Effect size", "Measure of how large an association or difference is, independent of sample size. Examples: Cohen's d (standardized mean difference), Cramér's V (categorical association strength), rank-biserial r (non-parametric effect)."),
    ("Cohen's d", "Standardized mean difference between two groups: d = (mean1 – mean2) / pooled SD. |d| < 0.2 small, 0.5 medium, 0.8+ large."),
    ("Hedges' g", "Bias-corrected version of Cohen's d, preferred for small samples."),
    ("Cramér's V", "Effect size for chi-square tests between categorical variables. Ranges 0 (no association) to 1 (perfect association)."),
    ("IQR", "Interquartile Range: the range from the 25th to the 75th percentile. Robust measure of spread that is not affected by extreme outliers."),
    ("SD / Std Dev", "Standard Deviation: average distance of observations from the mean. A larger SD means more variability in the data."),
    ("Skewness", "Asymmetry of a distribution. Positive = long tail to the right (more high values); negative = long tail to the left."),
    ("Kurtosis", "Peakedness of a distribution relative to a normal distribution. High kurtosis means heavy tails (more extreme values)."),
    # Dimensionality reduction
    ("PCA", "Principal Component Analysis: linear method that projects numeric variables into orthogonal dimensions (principal components) ordered by how much variance they explain."),
    ("Explained variance", "In PCA: the proportion of total variance captured by each principal component. 'Cumulative explained variance' shows how much is retained by the first k components combined."),
    ("MCA", "Multiple Correspondence Analysis: the categorical analog of PCA. Projects category levels and samples into a low-dimensional space to reveal co-occurrence patterns."),
    ("Explained inertia", "In MCA: the proportion of total inertia (variation in categorical associations) captured by each dimension."),
    # Clinical / hospital terms
    ("ICU", "Intensive Care Unit: a hospital ward for patients who require close monitoring and life-support equipment."),
    ("BP", "Blood Pressure: the pressure of circulating blood against vessel walls, typically measured as systolic/diastolic (e.g. 120/80 mmHg)."),
    ("HR", "Heart Rate: number of heartbeats per minute."),
    ("BMI", "Body Mass Index: weight (kg) / height² (m²). Used as a proxy for body fat: < 18.5 underweight, 18.5–24.9 normal, 25–29.9 overweight, ≥ 30 obese."),
    ("LOS", "Length of Stay: number of days a patient is hospitalized."),
    ("ICD", "International Classification of Diseases: standardized codes used to record diagnoses and procedures (e.g. ICD-10)."),
    ("PASC", "Post-Acute Sequelae of SARS-CoV-2 (Long COVID): persistent symptoms after acute COVID-19 infection."),
    ("NaN / NA", "Not a Number / Not Available: a placeholder indicating a missing or undefined value in the dataset."),
]


def _build_glossary_section(elements):
    """Append a static glossary of terms and abbreviations."""
    add_section_heading(elements, "Appendix: Terms and Abbreviations")
    render_narrative(elements, NarrativeMessage(
        "info",
        "This glossary covers statistical terms used in the report and common hospital/clinical "
        "abbreviations. It is intended to help readers from different backgrounds "
        "(clinicians, data scientists, engineers) interpret the results consistently.",
    ))
    import pandas as pd
    glossary_df = pd.DataFrame(_GLOSSARY, columns=["Term", "Definition"])
    from reportlab.lib.units import inch
    elements.append(create_table(glossary_df, 6.8 * inch))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    local_results_dir = base_dir / "results" / "local_results"

    for node_path in sorted(local_results_dir.iterdir()):
        if not node_path.is_dir():
            continue
        for report_mode in ("short", "full"):
            written = generate_local_report(node_path, node_path, mode=report_mode)
            print(f"Wrote {written}")
