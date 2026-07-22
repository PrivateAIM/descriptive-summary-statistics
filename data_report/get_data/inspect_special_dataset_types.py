"""Standalone inspection script for the longitudinal / event-based / panel
synthetic datasets (``data/dataset4_longitudinal``, ``data/dataset4_event_based``,
``data/dataset4_panel``).

These three dataset types are NOT wired into ``DataReportAnalyzer.analysis_method``
(the StarModel/``dr-analyze`` pipeline only runs the cross-sectional path), so
``detect_dataset_type``, ``analyze_longitudinal``, ``analyze_event_based`` and
``analyze_panel`` (in ``data_report.statistical_analysis.local.inferential_analysis``)
currently have no caller. This script loads each node's CSV directly and runs
those functions so their output can be inspected/compared node-to-node.

Run with:

    python -m data_report.get_data.inspect_special_dataset_types
"""

from pathlib import Path

import pandas as pd

from data_report.statistical_analysis.local.inferential_analysis import (
    detect_dataset_type,
    analyze_longitudinal,
    analyze_event_based,
    analyze_panel,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def _print_detection(label: str, df: pd.DataFrame) -> dict:
    """Run dataset-type detection and print the result to stdout."""
    result = detect_dataset_type(df)
    print(f"  detect_dataset_type -> {result['dataset_type']!r} "
          f"(patient_column={result['patient_column']!r}, "
          f"selected_date_column={result.get('selected_date_column')!r})")
    return result


def inspect_longitudinal() -> None:
    """Load and analyse all nodes from the longitudinal synthetic dataset.

    For each node CSV, prints the detected dataset type and the mean
    per-subject slope of ``weight_kg`` over ``visit_date``.
    """
    print("\n=== dataset4_longitudinal ===")
    for node_dir in sorted((DATA_DIR / "dataset4_longitudinal").iterdir()):
        if not node_dir.is_dir():
            continue
        for csv_path in sorted(node_dir.glob("*.csv")):
            df = pd.read_csv(csv_path, parse_dates=["visit_date"])
            print(f"\n-- {csv_path.relative_to(DATA_DIR)} ({df.shape[0]} rows) --")
            _print_detection(node_dir.name, df)

            results = analyze_longitudinal(df, subject_col="patient_id", time_col="visit_date", value_col="weight_kg")
            n_subjects = len(results)
            avg_slope = sum(r["slope"] for r in results.values()) / n_subjects
            print(f"  analyze_longitudinal(weight_kg) -> {n_subjects} subjects with >=2 visits, "
                  f"mean slope = {avg_slope:.3f} kg/day")


def inspect_event_based() -> None:
    """Load and analyse all nodes from the event-based synthetic dataset.

    For each node CSV, prints the detected dataset type together with
    event-density and peak-day statistics computed by ``analyze_event_based``.
    """
    print("\n=== dataset4_event_based ===")
    for node_dir in sorted((DATA_DIR / "dataset4_event_based").iterdir()):
        if not node_dir.is_dir():
            continue
        for csv_path in sorted(node_dir.glob("*.csv")):
            df = pd.read_csv(csv_path, parse_dates=["event_date"])
            print(f"\n-- {csv_path.relative_to(DATA_DIR)} ({df.shape[0]} rows) --")
            _print_detection(node_dir.name, df)

            result = analyze_event_based(df, time_col="event_date")
            print(f"  analyze_event_based -> event_count={result['event_count']}, "
                  f"event_density_mean={result['event_density_mean']:.3f}/day, "
                  f"peak_day={result['peak_day']} ({result['peak_day_count']} events), "
                  f"avg_time_between_events={result['avg_time_between_events']:.1f}s")


def inspect_panel() -> None:
    """Load and analyse all nodes from the panel synthetic dataset.

    For each node CSV, prints per-entity mean and growth statistics for the
    ``admissions`` column.

    Note:
        ``detect_dataset_type`` currently classifies panel data as
        ``"longitudinal"`` because the panel branch requires at least one
        event column detected by ``detect_event_columns``, which this
        dataset does not have.
    """
    print("\n=== dataset4_panel ===")
    for node_dir in sorted((DATA_DIR / "dataset4_panel").iterdir()):
        if not node_dir.is_dir():
            continue
        for csv_path in sorted(node_dir.glob("*.csv")):
            df = pd.read_csv(csv_path, parse_dates=["month"])
            print(f"\n-- {csv_path.relative_to(DATA_DIR)} ({df.shape[0]} rows) --")
            _print_detection(node_dir.name, df)

            results = analyze_panel(df, entity_col="hospital_id", time_col="month", value_col="admissions")
            for entity, r in results.items():
                print(f"  analyze_panel(admissions)[{entity}] -> "
                      f"n_observations={r['n_observations']}, mean={r['mean']:.1f}, growth={r['growth']:.1f}")


if __name__ == "__main__":
    inspect_longitudinal()
    inspect_event_based()
    inspect_panel()
