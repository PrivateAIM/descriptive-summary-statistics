"""
Local simulation of the FLAME pipeline on dataset1 (2 nodes).

Instantiates DataReportAnalyzer and DataReportAggregator via MockFlameCoreSDK,
runs the full analyze → aggregate → report cycle, and writes all outputs under
results/local_results/ and results/federated_results/.

Usage (from project root):
    python run_local_sim.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flame.utils.mock_flame_core import MockFlameCoreSDK
from data_report.analyze import DataReportAnalyzer, DataReportAggregator
from generate_reports.generate_local_report import generate_local_report
from generate_reports.generate_global_report import generate_global_report
from data_report.analyze import LOCAL_RESULTS_DIR, FEDERATED_RESULTS_DIR

# ── dataset configuration ─────────────────────────────────────────────────────
NODES = [
    Path("data/dataset1/node1/synthetic_eucare_1.csv"),
    Path("data/dataset1/node2/synthetic_eucare_2.csv"),
]
PARTICIPANT_IDS = [f"node_{i}" for i in range(len(NODES))]

# ── step 1: run each analyzer ─────────────────────────────────────────────────
analysis_results = []
for i, csv_path in enumerate(NODES):
    print(f"\n{'='*60}")
    print(f"  Analyzing node_{i}  ({csv_path.name})")
    print(f"{'='*60}")
    file_bytes = csv_path.read_bytes()
    mock = MockFlameCoreSDK({
        "node_id": f"node_{i}",
        "participant_ids": PARTICIPANT_IDS,
        "role": "default",
        "s3_data": [{csv_path.name: file_bytes}],
        "num_iterations": 0,
        "latest_result": None,
    })
    analyzer = DataReportAnalyzer(mock)
    result = analyzer.analysis_method([{csv_path.name: file_bytes}], None)
    result["node_id"] = f"node_{i}"
    analysis_results.append(result)
    print(f"  node_{i} done.")

# ── step 2: aggregate ─────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  Aggregating …")
print(f"{'='*60}")
mock_agg = MockFlameCoreSDK({
    "node_id": "aggregator",
    "participant_ids": PARTICIPANT_IDS,
    "role": "aggregator",
    "s3_data": [],
    "num_iterations": 0,
    "latest_result": None,
})
aggregator = DataReportAggregator(mock_agg)
aggregator.aggregation_method(analysis_results)
print("  Aggregation done.")

# ── step 3: generate PDF reports ──────────────────────────────────────────────
print(f"\n{'='*60}")
print("  Generating reports …")
print(f"{'='*60}")
if LOCAL_RESULTS_DIR.exists():
    for node_dir in sorted(LOCAL_RESULTS_DIR.iterdir()):
        if node_dir.is_dir():
            for mode in ("short", "full"):
                out = generate_local_report(node_dir, node_dir, mode=mode)
                print(f"  {out}")

if FEDERATED_RESULTS_DIR.exists():
    for mode in ("short", "full"):
        out = generate_global_report(FEDERATED_RESULTS_DIR, FEDERATED_RESULTS_DIR, mode=mode)
        print(f"  {out}")

print("\nDone. Check results/ for outputs.")
