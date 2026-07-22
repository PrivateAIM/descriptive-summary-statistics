"""Unit tests for generate_reports.report_utils and smoke tests for the
local/global report builders."""

import pandas as pd
import pytest
from reportlab.platypus import Table

from generate_reports.report_utils import (
    NarrativeMessage,
    add_categorical_comparison,
    add_numeric_comparison,
    add_temporal_comparison,
    auto_fit_image,
    build_privacy_notice,
    categorical_excluded_from_distributions_notice,
    categorical_small_group_warnings,
    compute_categorical_group_sizes,
    create_table,
    detect_identifier_features,
    drop_internal_columns,
    rank_by_activity,
    rank_by_deviation,
    rank_by_effect_size,
    rank_by_imbalance,
    reduction_excluded_columns_notice,
    summarise_categorical,
    summarise_inferential,
    summarise_numeric,
    summarise_temporal,
    truncation_note,
)
from generate_reports.generate_local_report import generate_local_report
from generate_reports.generate_global_report import generate_global_report


# ---------------------------------------------------------------------------
# Comparison columns
# ---------------------------------------------------------------------------

class TestComparisonColumns:

    def test_add_numeric_comparison_labels(self):
        local = pd.DataFrame({"feature": ["weight", "waist"], "mean": [200.0, 30.0]})
        global_df = pd.DataFrame({"feature": ["weight", "waist"], "mean": [170.0, 36.0]})

        result = add_numeric_comparison(local, global_df)

        assert result.loc[result["feature"] == "weight", "vs_global"].iloc[0] == "above_average"
        assert result.loc[result["feature"] == "waist", "vs_global"].iloc[0] == "below_average"
        assert "mean_global" in result.columns

    def test_add_numeric_comparison_missing_global_feature(self):
        local = pd.DataFrame({"feature": ["pulse"], "mean": [70.0]})
        global_df = pd.DataFrame({"feature": ["weight"], "mean": [170.0]})

        result = add_numeric_comparison(local, global_df)

        assert result["vs_global"].iloc[0] == "n/a"

    def test_add_categorical_comparison_compares_top_category_share(self):
        local = pd.DataFrame({
            "feature": ["gender"],
            "most_frequent_category": ["Female"],
            "relative_frequencies": ["{'Female': 60.0, 'Male': 40.0}"],
            "count": [20],
        })
        global_df = pd.DataFrame({
            "feature": ["gender"],
            "relative_freq": ["{'Female': 0.4, 'Male': 0.6}"],
        })

        result = add_categorical_comparison(local, global_df)

        assert result["top cat % (local)"].iloc[0] == 60.0
        assert result["top cat % (global)"].iloc[0] == pytest.approx(40.0)
        assert result["vs_global"].iloc[0] == "above_average"

    def test_add_temporal_comparison_sums_counts_per_period(self):
        local = pd.DataFrame({"feature": ["admission_date"], "count": [15]})
        global_df = pd.DataFrame({
            "feature": ["admission_date"],
            "counts_per_period": ['{"2023-01": 5, "2023-02": 10}'],
        })

        result = add_temporal_comparison(local, global_df)

        assert result["count_global"].iloc[0] == 15
        assert result["vs_global"].iloc[0] == "similar"


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

class TestRanking:

    def test_rank_by_deviation_orders_by_absolute_relative_diff(self):
        df = pd.DataFrame({
            "feature": ["a", "b", "c"],
            "_rel_diff_abs": [0.05, 0.5, 0.2],
        })

        ranked = rank_by_deviation(df, max_rows=2)

        assert list(ranked["feature"]) == ["b", "c"]

    def test_rank_by_imbalance_orders_by_class_imbalance_ratio(self):
        df = pd.DataFrame({
            "feature": ["a", "b"],
            "class_imbalance_ratio": [1.2, 5.0],
        })

        ranked = rank_by_imbalance(df, max_rows=1)

        assert list(ranked["feature"]) == ["b"]

    def test_rank_by_activity_orders_by_count(self):
        df = pd.DataFrame({"feature": ["a", "b"], "count": [3, 9]})

        ranked = rank_by_activity(df, max_rows=1)

        assert list(ranked["feature"]) == ["b"]

    def test_rank_by_effect_size_orders_by_absolute_effect_size(self):
        df = pd.DataFrame({
            "var1": ["a", "b"],
            "effect_size": [-0.9, 0.3],
        })

        ranked = rank_by_effect_size(df, max_rows=1)

        assert list(ranked["var1"]) == ["a"]


# ---------------------------------------------------------------------------
# Privacy / small-group helpers
# ---------------------------------------------------------------------------

class TestPrivacyHelpers:

    def test_detect_identifier_features(self):
        df = pd.DataFrame({"feature": ["patient_id", "weight"]})

        found = detect_identifier_features(df)

        assert found == ["patient_id"]

    def test_compute_categorical_group_sizes_flags_small_groups(self):
        categorical_df = pd.DataFrame({
            "feature": ["smoking_status"],
            "count": [20],
            "relative_frequencies": ["{'Non-smoker': 90.0, 'Smoker': 10.0}"],
        })

        min_size, min_feature, min_category, flagged = compute_categorical_group_sizes(
            categorical_df, threshold=5
        )

        assert min_size == pytest.approx(2.0)
        assert min_feature == "smoking_status"
        assert min_category == "Smoker"
        assert flagged == [("smoking_status", "Smoker", pytest.approx(2.0))]

    def test_categorical_small_group_warnings_emits_warning_level(self):
        categorical_df = pd.DataFrame({
            "feature": ["smoking_status"],
            "count": [20],
            "relative_frequencies": ["{'Non-smoker': 90.0, 'Smoker': 10.0}"],
        })

        warnings = categorical_small_group_warnings(categorical_df, threshold=5)

        assert len(warnings) == 1
        assert warnings[0].level == "warning"
        assert "smoking_status" in warnings[0].text

    def test_categorical_small_group_warnings_empty_when_no_small_groups(self):
        categorical_df = pd.DataFrame({
            "feature": ["gender"],
            "count": [20],
            "relative_frequencies": ["{'Female': 60.0, 'Male': 40.0}"],
        })

        assert categorical_small_group_warnings(categorical_df, threshold=5) == []

    def test_categorical_excluded_from_distributions_notice_names_single_valued_columns(self):
        categorical_df = pd.DataFrame({
            "feature": ["gender", "site"],
            "number_of_categories": [2, 1],
        })

        msg = categorical_excluded_from_distributions_notice(categorical_df)

        assert msg is not None
        assert msg.level == "info"
        assert "site" in msg.text
        assert "gender" not in msg.text

    def test_categorical_excluded_from_distributions_notice_none_when_all_multi_valued(self):
        categorical_df = pd.DataFrame({
            "feature": ["gender", "smoking_status"],
            "number_of_categories": [2, 3],
        })

        assert categorical_excluded_from_distributions_notice(categorical_df) is None

    def test_categorical_excluded_from_distributions_notice_none_for_empty_or_missing_df(self):
        assert categorical_excluded_from_distributions_notice(None) is None
        assert categorical_excluded_from_distributions_notice(pd.DataFrame()) is None
        assert categorical_excluded_from_distributions_notice(
            pd.DataFrame({"feature": ["a"]})
        ) is None

    def test_reduction_excluded_columns_notice_reads_csv_and_names_columns(self, tmp_path):
        subdir = tmp_path / "pca"
        subdir.mkdir()
        pd.DataFrame({"feature": ["empty_col"]}).to_csv(subdir / "excluded_columns.csv", index=False)

        msg = reduction_excluded_columns_notice(subdir, "PCA")

        assert msg is not None
        assert msg.level == "info"
        assert "empty_col" in msg.text
        assert "PCA" in msg.text

    def test_reduction_excluded_columns_notice_none_when_no_csv(self, tmp_path):
        subdir = tmp_path / "mca"
        subdir.mkdir()

        assert reduction_excluded_columns_notice(subdir, "MCA") is None

    def test_build_privacy_notice_local_mentions_min_group_size(self):
        categorical_df = pd.DataFrame({
            "feature": ["smoking_status"],
            "count": [20],
            "relative_frequencies": ["{'Non-smoker': 90.0, 'Smoker': 10.0}"],
        })

        elements = build_privacy_notice(report_type="local", n_nodes=3, categorical_df=categorical_df)
        texts = [getattr(e, "text", "") for e in elements]

        assert any("local data only" in t for t in texts)
        assert any("Smallest displayed group size" in t for t in texts)
        assert any("only 3 participating node(s)" in t for t in texts)

    def test_build_privacy_notice_global_mentions_aggregation(self):
        elements = build_privacy_notice(report_type="global", n_nodes=3)
        texts = [getattr(e, "text", "") for e in elements]

        assert any("aggregated statistics" in t for t in texts)
        assert any("federated aggregation" in t for t in texts)


# ---------------------------------------------------------------------------
# Narrative summarizers
# ---------------------------------------------------------------------------

class TestSummarizers:

    def test_summarise_numeric_counts_comparison_labels(self):
        df = pd.DataFrame({
            "feature": ["a", "b", "c"],
            "vs_global": ["above_average", "below_average", "similar"],
        })

        msg = summarise_numeric(df)

        assert "1 above and 1 below" in msg.text

    def test_summarise_categorical_handles_empty(self):
        msg = summarise_categorical(None)

        assert msg.level == "info"
        assert "No categorical variables" in msg.text

    def test_summarise_temporal_counts_variables(self):
        df = pd.DataFrame({"feature": ["admission_date", "discharge_date"]})

        msg = summarise_temporal(df)

        assert "2 temporal variable(s)" in msg.text

    def test_summarise_inferential_filters_by_pair_type_and_picks_strongest(self):
        df = pd.DataFrame({
            "var1": ["weight", "gender"],
            "var2": ["waist", "smoking_status"],
            "pair_type": ["num-num", "cat-cat"],
            "test": ["spearman", "chi2"],
            "effect_size": [0.8, 0.3],
            "effect_size_metric": ["correlation", "cramers_v"],
        })

        msg = summarise_inferential(df, pair_type="num-num")

        assert msg.level == "insight"
        assert "weight vs waist" in msg.text

    def test_summarise_inferential_no_significant_results(self):
        msg = summarise_inferential(None)

        assert "No statistically significant" in msg.text


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

class TestLayoutHelpers:

    def test_create_table_returns_table_with_correct_column_widths_total(self):
        df = pd.DataFrame({"a": [1, 2], "bb": ["x", "yyyy"]})

        table = create_table(df, available_width=400)

        assert isinstance(table, Table)
        assert sum(table._colWidths) == pytest.approx(400)

    def test_create_table_respects_max_rows(self):
        df = pd.DataFrame({"a": range(20)})

        table = create_table(df, available_width=200, max_rows=5)

        # header row + 5 body rows
        assert len(table._cellvalues) == 6

    def test_auto_fit_image_preserves_aspect_ratio(self, tmp_path):
        from PIL import Image as PILImage

        path = tmp_path / "test.png"
        PILImage.new("RGB", (200, 100)).save(path)

        width, height = auto_fit_image(path, max_width=100, max_height=100)

        assert width == pytest.approx(100)
        assert height == pytest.approx(50)

    def test_truncation_note_format(self):
        msg = truncation_note(5, 20, "deviating from the federated average", "numeric_summary.csv")

        assert msg.level == "info"
        assert "Showing the 5 most deviating from the federated average of 20" in msg.text
        assert "numeric_summary.csv" in msg.text

    def test_drop_internal_columns(self):
        df = pd.DataFrame({"feature": ["a"], "_rel_diff_abs": [0.1]})

        result = drop_internal_columns(df)

        assert list(result.columns) == ["feature"]


# ---------------------------------------------------------------------------
# End-to-end smoke tests
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_node_dir(tmp_path):
    node_dir = tmp_path / "local_results" / "node1"
    for sub in ["overview", "numeric", "categorical", "temporal", "inferential",
                "pca", "mca", "comparison"]:
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
        "missing_periods": ["[]"],
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


class TestGenerateLocalReport:

    @pytest.mark.parametrize("mode", ["short", "full"])
    def test_returns_path_to_non_empty_pdf(self, minimal_node_dir, minimal_federated_dir, mode):
        out = generate_local_report(minimal_node_dir, minimal_node_dir, mode=mode,
                                      results_dir=minimal_federated_dir)

        assert out == minimal_node_dir / f"local_report_node1_{mode}.pdf"
        assert out.exists()
        assert out.stat().st_size > 0

    def test_works_without_federated_data(self, minimal_node_dir, tmp_path):
        empty_federated = tmp_path / "no_federated"
        empty_federated.mkdir()

        out = generate_local_report(minimal_node_dir, minimal_node_dir, mode="short",
                                      results_dir=empty_federated)

        assert out.exists()
        assert out.stat().st_size > 0

    def test_export_comparison_csv(self, minimal_node_dir, minimal_federated_dir):
        generate_local_report(minimal_node_dir, minimal_node_dir, mode="short",
                               results_dir=minimal_federated_dir, export_comparison_csv=True)

        for fname, subdir in [
            ("numeric_summary_with_comparison.csv", "numeric"),
            ("categorical_summary_with_comparison.csv", "categorical"),
            ("temporal_summary_with_comparison.csv", "temporal"),
        ]:
            out_csv = minimal_node_dir / subdir / fname
            assert out_csv.exists()
            df = pd.read_csv(out_csv)
            assert "vs_global" in df.columns
            assert not any(c.startswith("_") for c in df.columns)


class TestGenerateGlobalReport:

    @pytest.mark.parametrize("mode", ["short", "full"])
    def test_returns_path_to_non_empty_pdf(self, minimal_federated_dir, mode):
        out = generate_global_report(minimal_federated_dir, minimal_federated_dir, mode=mode)

        assert out == minimal_federated_dir / f"global_report_{mode}.pdf"
        assert out.exists()
        assert out.stat().st_size > 0
