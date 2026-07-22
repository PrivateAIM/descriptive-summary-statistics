"""Federated (global) report generator.

generate_global_report(results_dir, output_dir, mode="full") -> Path

Builds a PDF report from the aggregated outputs under
results/federated_results/. Two modes:
  - "short": narrative summaries, key plots, condensed tables
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
    add_figure,
    add_heading_and_plots,
    add_plots_from_dir,
    add_section_heading,
    build_privacy_notice,
    create_table,
    drop_internal_columns,
    make_header,
    out_of_range_age_notice,
    prepare_numeric_display,
    rank_by_activity,
    render_narrative,
    safe_read_csv,
    summarise_categorical,
    summarise_numeric,
    summarise_temporal,
    truncation_note,
)
from generate_reports.generate_local_report import _build_glossary_section
from generate_reports.section_definitions import SHORT_TABLE_MAX_ROWS

MAX_W = 6.8 * inch
MAX_H = 4 * inch
PAGE_MARGIN = 0.4 * inch


def generate_global_report(results_dir, output_dir, mode="full") -> Path:
    """Build a federated PDF statistical report and write it to disk.

    Loads aggregated CSV outputs from ``results_dir``, then assembles a
    multi-section ReportLab document covering overview, numeric, categorical,
    temporal, and glossary sections.

    Args:
        results_dir (str or Path): Root directory of the federated analysis
            outputs (e.g. ``results/federated_results/``).
        output_dir (str or Path): Directory where the PDF will be written.
        mode (str): Report detail level.  ``"short"`` includes narrative
            summaries, key plots, and condensed tables; ``"full"`` includes
            complete per-variable detail.

    Returns:
        Path: Absolute path of the written PDF file.
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overview_df = safe_read_csv(results_dir / "overview" / "overview.csv")
    numeric_df = safe_read_csv(results_dir / "numeric" / "federated_numeric_statistics.csv")
    categorical_df = safe_read_csv(results_dir / "categorical" / "federated_categorical_statistics.csv")
    temporal_df = safe_read_csv(results_dir / "temporal" / "federated_temporal_statistics.csv")
    n_nodes = _extract_n_nodes(overview_df)

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
    _build_title_page(elements, mode, n_nodes, numeric_df, categorical_df, temporal_df)

    elements.append(NextPageTemplate("overview"))
    elements.append(PageBreak())
    _build_overview_section(elements, doc, results_dir, overview_df)

    elements.append(NextPageTemplate("numeric"))
    elements.append(PageBreak())
    _build_numeric_section(elements, doc, results_dir, mode, numeric_df)

    elements.append(NextPageTemplate("categorical"))
    elements.append(PageBreak())
    _build_categorical_section(elements, doc, results_dir, mode, categorical_df)

    elements.append(NextPageTemplate("temporal"))
    elements.append(PageBreak())
    _build_temporal_section(elements, doc, results_dir, mode, temporal_df)

    elements.append(PageBreak())
    _build_glossary_section(elements)

    doc.build(elements)
    return output_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_n_nodes(overview_df):
    """Extract the participating-node count from a federated overview DataFrame."""
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

def _build_title_page(elements, mode, n_nodes, numeric_df, categorical_df, temporal_df):
    """Append title-page flowables (heading, metadata, privacy notice) to the elements list."""
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

def _build_overview_section(elements, doc, results_dir, overview_df):
    """Append Section 1 (Overview) flowables, including the federated summary table and data-type chart."""
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

def _build_numeric_section(elements, doc, results_dir, mode, numeric_df):
    """Append Section 2 (Numeric) flowables with federated statistics and summary plots."""
    numeric_dir = results_dir / "numeric"

    add_heading_and_plots(elements, "2. Numeric Section", [
        numeric_dir / "numeric_summary_bars.png",
        numeric_dir / "age_distribution_federated.png",
    ], level=1, max_width=MAX_W, max_height=MAX_H)
    age_range_notice = out_of_range_age_notice(numeric_dir)
    if age_range_notice is not None:
        render_narrative(elements, age_range_notice)

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

def _build_categorical_section(elements, doc, results_dir, mode, categorical_df):
    """Append Section 3 (Categorical) flowables with federated statistics and distribution plots."""
    categorical_dir = results_dir / "categorical"

    dist_paths = sorted(categorical_dir.glob("categorical_distributions_*.png"))
    if not dist_paths:
        dist_paths = [categorical_dir / "categorical_distributions.png"]
    add_heading_and_plots(elements, "3. Categorical Section",
                           [*dist_paths, categorical_dir / "sex_distribution_federated.png"],
                           level=1, max_width=MAX_W, max_height=MAX_H)

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

def _build_temporal_section(elements, doc, results_dir, mode, temporal_df):
    """Append Section 4 (Temporal) flowables with federated trend data and per-feature charts."""
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

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    federated_results_dir = base_dir / "results" / "federated_results"

    for report_mode in ("short", "full"):
        written = generate_global_report(federated_results_dir, federated_results_dir, mode=report_mode)
        print(f"Wrote {written}")
