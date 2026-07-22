"""Command-line entry point for the ``dr-analyze`` federated data-report tool."""
import shutil
import sys
from pathlib import Path
from data_report.get_data.load_data import load_dataset
from data_report.get_data.load_data import load_all_datasets

# Project root, so `generate_reports` (a sibling of `data_report`, not part of
# the installed package) can be imported regardless of the caller's cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def analyze_main() -> None:
    """Run the full federated data-report pipeline and write all outputs.

    Orchestrates the pipeline in three phases:

    1. **Federated analysis**: loads the target dataset, instantiates
       ``DataReportAnalyzer`` and ``DataReportAggregator``, and runs the
       FLAME ``StarModelTester`` simulation.  Per-node results are written to
       ``results/local_results/`` and federation-wide results to
       ``results/federated_results/`` as a side-effect of aggregation.

    2. **PDF reports**: generates short and full PDF reports for the federation
       as a whole, then short and full PDF reports for each node directory.

    3. **JSON summaries**: generates machine-readable JSON summary files for
       the federation and for each node.

    The output directory (``results/``) is preserved across calls, but
    ``results/local_results/`` and ``results/federated_results/`` are removed
    and recreated on each run so stale outputs from a previous dataset cannot
    accumulate.
    """
    import os
    from flame.star import StarModelTester
    from data_report import analyze
    from data_report.analyze import DataReportAnalyzer, DataReportAggregator
    from generate_reports.generate_global_report import generate_global_report
    from generate_reports.generate_local_report import generate_local_report
    from generate_reports.generate_json_summary import (
        generate_global_json_summary,
        generate_local_json_summary,
    )

    data_dir = Path("data")
    dataset_name = "dataset1"
    dataset_path = data_dir / dataset_name
    results_dir = Path("results")
    # analyze.RESULTS_DIR = results_dir
    os.makedirs(results_dir, exist_ok=True)

    # Clear outputs from any previous run so switching datasets (different
    # node counts/columns) can't leave stale local/federated results or
    # reports behind.
    shutil.rmtree(analyze.LOCAL_RESULTS_DIR, ignore_errors=True)
    shutil.rmtree(analyze.FEDERATED_RESULTS_DIR, ignore_errors=True)

    data_splits = load_dataset(dataset_path)



    StarModelTester(
        data_splits=data_splits,
        analyzer=DataReportAnalyzer,  # Custom analyzer class (must inherit from StarAnalyzer)
        aggregator=DataReportAggregator,  # Custom aggregator class (must inherit from StarAggregator)
        data_type="s3",  # Type of data source ('fhir' or 's3') what is s3 though? -> data is like raw file bytes
        # query='Patient?_summary=count',  # Query or list of queries to retrieve data -> what is this?
        simple_analysis=True,  # True for single-iteration; False for multi-iterative analysis
        output_type="pickle",  # Output format for the final result ('str', 'bytes', or 'pickle')
        result_filepath=str(results_dir / "data_report.pkl"),
        # analyzer_kwargs=None,  # Additional keyword arguments for the custom analyzer constructor (i.e. MyAnalyzer)
        # aggregator_kwargs=None  # Additional keyword arguments for the custom aggregator constructor (i.e. MyAggregator)
    )

    # Build the global (federated) PDF reports for this run.
    for mode in ("short", "full"):
        report_path = generate_global_report(
            analyze.FEDERATED_RESULTS_DIR, analyze.FEDERATED_RESULTS_DIR, mode=mode
        )
        print(f"  {report_path}")

    # Build the machine-readable global summary for this run.
    json_path = generate_global_json_summary(analyze.FEDERATED_RESULTS_DIR, analyze.FEDERATED_RESULTS_DIR)
    print(f"  {json_path}")

    # Build the per-node local PDF reports and JSON summaries for this run.
    for node_dir in sorted(analyze.LOCAL_RESULTS_DIR.iterdir()):
        if not node_dir.is_dir():
            continue
        for mode in ("short", "full"):
            report_path = generate_local_report(node_dir, node_dir, mode=mode)
            print(f"  {report_path}")
        json_path = generate_local_json_summary(node_dir, node_dir)
        print(f"  {json_path}")

    print(f"\nDone. Outputs in {results_dir}/:")
    print(f"  {results_dir}/data_report.pkl                    -- pickled aggregated result")
    # print(f"  {results_dir}/age_distribution_federated.png     -- age histogram")
    # print(f"  {results_dir}/sex_distribution_federated.png     -- sex bar chart")

