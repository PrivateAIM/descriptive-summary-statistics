"""Column-availability utilities for federated dataset comparison.

These helpers analyse the set of column names reported by each node and
classify them as universally common (present at every site), partially common
(present at more than one site but not all), or unique (present at only one
site).
"""


def compute_column_distribution(analysis_results, total_sites):
    """Count how many nodes contain each column and categorise coverage.

    Counts are derived from the keys of the ``numeric_statistics``,
    ``categorical_statistics``, and ``temporal_statistics`` sub-dicts in each
    node result.  No raw data leaves the node.

    Args:

        analysis_results (list): List of per-node result dictionaries as
            returned by ``DataReportAnalyzer.analysis_method``.
        total_sites (int): Total number of participating nodes.

    Returns:

        tuple:
            - column_node_counts (dict): Mapping of column name to the number
              of nodes that contain it.
            - column_distribution_summary (dict): Categorised lists of column
              names under keys ``"common_all"``, ``"common_partial"``, and
              ``"unique"``.
    """
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
    """Assign an availability label to each column in a single node's dataset.

    Labels reflect how widely each column appears across the federation:

    * ``"common_all"`` — present at every participating site.
    * ``"common_partial"`` — present at more than one site but not all.
    * ``"unique_local"`` — present at this site only.

    Args:

        local_columns (iterable): Column names present in the node being
            labelled.
        total_sites (int): Total number of participating nodes.
        column_node_counts (dict): Mapping of column name to federation-wide
            node count, as returned by ``compute_column_distribution``.

    Returns:

        dict: Mapping of column name to its availability label string.
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