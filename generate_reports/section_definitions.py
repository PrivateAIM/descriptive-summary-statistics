"""Declarative configuration shared by generate_local_report.py and
generate_global_report.py.

This holds the small pieces of per-subsection metadata that would
otherwise be duplicated/hardcoded across the two report builders
(directory names, short-mode plot choices, narrative labels). The
report-building logic itself lives in report_utils.py and the two
generate_*_report.py modules.
"""

from dataclasses import dataclass


@dataclass
class ReductionSubsection:
    """Configuration for a dimensionality-reduction subsection (PCA or MCA).

    In short mode a single overview plot is rendered; in full mode every plot
    in ``subdir`` is included, excluding files whose names contain
    ``"overview"`` to avoid duplication with the short-mode panel.

    Attributes:
        title (str): Section heading text, e.g. ``"PCA"`` or ``"MCA"``.
        subdir (str): Path to the reduction outputs relative to the node
            results directory, e.g. ``"pca"`` or ``"mca"``.
        short_plot (str): Filename of the overview plot shown in short mode,
            e.g. ``"pca_overview.png"``.
    """

    title: str
    subdir: str        # relative to node_dir, e.g. "pca"
    short_plot: str    # e.g. "pca_explained_variance.png"


LOCAL_PCA = ReductionSubsection(title="PCA", subdir="pca", short_plot="pca_overview.png")
LOCAL_MCA = ReductionSubsection(title="MCA", subdir="mca", short_plot="mca_overview.png")

# Short-mode table truncation
SHORT_TABLE_MAX_ROWS = 10

# Short-mode: number of per-feature plots (e.g. temporal activity charts) shown
SHORT_PLOT_MAX = 5
