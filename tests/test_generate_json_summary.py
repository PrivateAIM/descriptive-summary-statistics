"""Unit tests for generate_reports.generate_json_summary."""

import json

import pandas as pd
import pytest

from generate_reports.generate_json_summary import (
    generate_global_json_summary,
    generate_local_json_summary,
)


@pytest.fixture
def minimal_node_dir(tmp_path):
    node_dir = tmp_path / "local_results" / "node1"
    for sub in ["overview", "numeric", "categorical", "temporal", "inferential"]:
        (node_dir / sub).mkdir(parents=True)

    pd.DataFrame({"metric": ["Patients", "Features"], "value": ["10", "5"]}).to_csv(
        node_dir / "overview" / "overview.csv", index=False)

    pd.DataFrame({
        "feature": ["weight", "waist"],
        "availability": ["common_all", "common_all"],
        "mean": [180.0, 36.0],
        "count": [10, 10],
    }).to_csv(node_dir / "numeric" / "numeric_summary.csv", index=False)

    pd.DataFrame({
        "feature": ["gender", "smoking_status"],
        "availability": ["common_all", "common_all"],
        "count": [10, 10],
        "most_frequent_category": ["Female", "Non-smoker"],
        "relative_frequencies": ["{'Female': 60.0, 'Male': 40.0}", "{'Non-smoker': 90.0, 'Smoker': 10.0}"],
        "class_imbalance_ratio": [1.5, 9.0],
    }).to_csv(node_dir / "categorical" / "categorical_summary.csv", index=False)

    pd.DataFrame({
        "feature": ["admission_date"],
        "availability": ["common_all"],
        "count": [10],
        "most_active_period": ["2023-01"],
        "missing_periods": ["['2023-04', '2023-08']"],
    }).to_csv(node_dir / "temporal" / "temporal_summary.csv", index=False)

    pd.DataFrame({
        "var1": ["weight"], "var2": ["waist"], "pair_type": ["num-num"],
        "test": ["spearman"], "statistic": [0.8], "p_value": [0.001],
        "effect_size": [0.8], "effect_size_metric": ["correlation"],
        "p_adj": [0.005], "significant": [True],
    }).to_csv(node_dir / "inferential" / "significant_associations.csv", index=False)

    return node_dir


@pytest.fixture
def minimal_federated_dir(tmp_path):
    federated_dir = tmp_path / "federated_results"
    for sub in ["overview", "numeric", "categorical", "temporal"]:
        (federated_dir / sub).mkdir(parents=True)

    pd.DataFrame({
        "metric": ["number of hospitals", "total number of patients"],
        "value": ["3", "30"],
    }).to_csv(federated_dir / "overview" / "overview.csv", index=False)

    pd.DataFrame({
        "feature": ["weight", "waist"],
        "mean": [170.0, 36.0],
        "count": [30, 30],
        "availability": ["common_all", "common_all"],
    }).to_csv(federated_dir / "numeric" / "federated_numeric_statistics.csv", index=False)

    pd.DataFrame({
        "feature": ["gender", "smoking_status"],
        "relative_freq": ["{'Female': 0.4, 'Male': 0.6}", "{'Non-smoker': 0.7, 'Smoker': 0.3}"],
        "mode": ["Male", "Non-smoker"],
        "num_categories": [2, 2],
        "missing_count": [0, 0],
        "availability": ["common_all", "common_all"],
    }).to_csv(federated_dir / "categorical" / "federated_categorical_statistics.csv", index=False)

    pd.DataFrame({
        "feature": ["admission_date"],
        "counts_per_period": ['{"2023-01": 10, "2023-02": 20}'],
        "most_active_period": ["2023-02"],
        "missing_count": [0],
        "availability": ["common_all"],
    }).to_csv(federated_dir / "temporal" / "federated_temporal_statistics.csv", index=False)

    return federated_dir


class TestGenerateLocalJsonSummary:

    def test_writes_summary_json_in_node_dir(self, minimal_node_dir):
        output_path = generate_local_json_summary(minimal_node_dir)

        assert output_path == minimal_node_dir / "summary.json"
        assert output_path.exists()

    def test_summary_contains_expected_sections(self, minimal_node_dir):
        output_path = generate_local_json_summary(minimal_node_dir)
        summary = json.loads(output_path.read_text())

        assert summary["overview"] == [
            {"metric": "Patients", "value": 10},
            {"metric": "Features", "value": 5},
        ]
        assert summary["numeric_summary"][0]["feature"] == "weight"
        assert summary["numeric_summary"][0]["mean"] == 180.0

    def test_stringified_dict_and_list_cells_are_parsed(self, minimal_node_dir):
        output_path = generate_local_json_summary(minimal_node_dir)
        summary = json.loads(output_path.read_text())

        gender_row = summary["categorical_summary"][0]
        assert gender_row["relative_frequencies"] == {"Female": 60.0, "Male": 40.0}

        admission_row = summary["temporal_summary"][0]
        assert admission_row["missing_periods"] == ["2023-04", "2023-08"]

    def test_missing_optional_csv_is_none(self, minimal_node_dir):
        output_path = generate_local_json_summary(minimal_node_dir)
        summary = json.loads(output_path.read_text())

        assert summary["comparisons_by_outcome"] is None
        assert summary["numeric_comparison"] is None

    def test_output_dir_override(self, minimal_node_dir, tmp_path):
        output_dir = tmp_path / "json_out"
        output_path = generate_local_json_summary(minimal_node_dir, output_dir=output_dir)

        assert output_path == output_dir / "summary.json"
        assert output_path.exists()


class TestGenerateGlobalJsonSummary:

    def test_writes_summary_json_in_federated_dir(self, minimal_federated_dir):
        output_path = generate_global_json_summary(minimal_federated_dir)

        assert output_path == minimal_federated_dir / "summary.json"
        assert output_path.exists()

    def test_summary_contains_expected_sections(self, minimal_federated_dir):
        output_path = generate_global_json_summary(minimal_federated_dir)
        summary = json.loads(output_path.read_text())

        assert summary["overview"] == [
            {"metric": "number of hospitals", "value": 3},
            {"metric": "total number of patients", "value": 30},
        ]
        assert summary["numeric_statistics"][0]["feature"] == "weight"

    def test_stringified_dict_cells_are_parsed(self, minimal_federated_dir):
        output_path = generate_global_json_summary(minimal_federated_dir)
        summary = json.loads(output_path.read_text())

        gender_row = summary["categorical_statistics"][0]
        assert gender_row["relative_freq"] == {"Female": 0.4, "Male": 0.6}

        admission_row = summary["temporal_statistics"][0]
        assert admission_row["counts_per_period"] == {"2023-01": 10, "2023-02": 20}

    def test_missing_optional_csv_is_none(self, minimal_federated_dir):
        output_path = generate_global_json_summary(minimal_federated_dir)
        summary = json.loads(output_path.read_text())

        assert summary["sex_distribution"] is None
        assert summary["age_distribution"] is None
