"""Machine-readable JSON export of per-node and federated summary tables.

Bundles the summary-table CSVs already written by the analysis pipeline
into a single ``summary.json`` per node and one for the federated results.
The JSON keys mirror the CSV directory layout used by the PDF report
builders.  This module does not re-derive any statistics; it only
repackages CSVs that have already been written.

The two public entry points are ``generate_local_json_summary`` for per-node
exports and ``generate_global_json_summary`` for the federated export.  The
``comparisons_by_outcome`` key in the local summary is populated dynamically
by globbing for ``comparisons_by_*.csv`` files whose filename depends on the
detected outcome column name.
"""

import ast
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Section name -> CSV path, relative to a node's results directory
# (results/local_results/<node>/).
LOCAL_CSV_MAP = {
    "overview": "overview/overview.csv",
    "numeric_summary": "numeric/numeric_summary.csv",
    "categorical_summary": "categorical/categorical_summary.csv",
    "temporal_summary": "temporal/temporal_summary.csv",
    "numeric_comparison": "comparison/numeric_comparison.csv",
    "association_screening": "inferential/association_screening.csv",
    "significant_associations": "inferential/significant_associations.csv",
    # comparisons_by_<outcome>.csv — resolved dynamically in generate_local_json_summary
}

# Section name -> CSV path, relative to results/federated_results/.
GLOBAL_CSV_MAP = {
    "overview": "overview/overview.csv",
    "numeric_statistics": "numeric/federated_numeric_statistics.csv",
    "categorical_statistics": "categorical/federated_categorical_statistics.csv",
    "temporal_statistics": "temporal/federated_temporal_statistics.csv",
    "sex_distribution": "categorical/sex_distribution_federated.csv",
    "age_distribution": "numeric/age_distribution_federated.csv",
}


def _sanitize(value: Any) -> Any:
    """Recursively convert numpy/pandas scalars to JSON-serializable types; map NaN/Inf/NaT to null."""
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if (math.isnan(value) or math.isinf(value)) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _parse_cell(value: Any) -> Any:
    """Parse stringified dict or list cell values into Python structures, leaving other values unchanged."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                return ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                return value
    return value


def _csv_to_records(path: Path) -> Optional[list]:
    """Read a CSV into a list of JSON-safe dicts, returning None if the file does not exist."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        # A genuinely empty (0-byte) file raises EmptyDataError; degrade to an
        # empty section instead of aborting the whole summary.json build.
        return []
    if df.empty:
        return []
    return [
        {k: _sanitize(_parse_cell(v)) for k, v in record.items()}
        for record in df.to_dict(orient="records")
    ]


def _build_summary(base_dir: Path, csv_map: dict) -> dict:
    """Build a summary dict by reading each CSV in csv_map relative to base_dir."""
    return {key: _csv_to_records(base_dir / rel_path) for key, rel_path in csv_map.items()}


def generate_local_json_summary(node_dir, output_dir=None) -> Path:
    """Write a machine-readable summary.json for a single node.

    Bundles the node's descriptive and inferential CSVs into a single JSON
    document grouped by section.  The ``comparisons_by_outcome``
    key is populated by globbing for ``comparisons_by_*.csv`` in the node's
    ``inferential/`` directory; the filename depends on the detected outcome
    column name.

    Args:
        node_dir (str or Path): Root directory of the node's analysis outputs.
        output_dir (str or Path or None): Directory where ``summary.json``
            will be written.  Defaults to ``node_dir``.

    Returns:
        Path: Absolute path of the written ``summary.json`` file.
    """
    node_dir = Path(node_dir)
    output_dir = Path(output_dir) if output_dir else node_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = _build_summary(node_dir, LOCAL_CSV_MAP)

    # comparisons_by_<outcome>.csv — filename depends on the detected outcome column
    inf_dir = node_dir / "inferential"
    comparisons_csvs = sorted(inf_dir.glob("comparisons_by_*.csv"))
    summary["comparisons_by_outcome"] = (
        _csv_to_records(comparisons_csvs[0]) if comparisons_csvs else None
    )

    output_path = output_dir / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2))
    return output_path


def generate_global_json_summary(federated_dir, output_dir=None) -> Path:
    """Write a machine-readable summary.json for the federated results.

    Bundles the federated CSVs for overview, numeric, categorical, temporal,
    sex-distribution, and age-distribution sections into a single JSON
    document.

    Args:
        federated_dir (str or Path): Root directory of the federated analysis
            outputs.
        output_dir (str or Path or None): Directory where ``summary.json``
            will be written.  Defaults to ``federated_dir``.

    Returns:
        Path: Absolute path of the written ``summary.json`` file.
    """
    federated_dir = Path(federated_dir)
    output_dir = Path(output_dir) if output_dir else federated_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = _build_summary(federated_dir, GLOBAL_CSV_MAP)

    output_path = output_dir / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2))
    return output_path


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    local_results_dir = base_dir / "results" / "local_results"
    federated_results_dir = base_dir / "results" / "federated_results"

    if federated_results_dir.exists():
        written = generate_global_json_summary(federated_results_dir)
        print(f"Wrote {written}")

    if local_results_dir.exists():
        for node_path in sorted(local_results_dir.iterdir()):
            if node_path.is_dir():
                written = generate_local_json_summary(node_path)
                print(f"Wrote {written}")
