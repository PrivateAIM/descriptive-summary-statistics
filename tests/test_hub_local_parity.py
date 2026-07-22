"""Parity tests between the modular local project and hub_entrypoint_10.py.

hub_entrypoint_10.py bundles copies of the same functions that live in the
modular ``data_report`` package (same names, same intended behaviour) so it
can be deployed as a single self-contained FLAME artifact. Because the two
copies are maintained by hand, they can silently drift apart -- e.g. the
``is_binary`` bug found in an earlier review, where the hub copy classified
sparse numeric columns differently from the local copy given the exact same
input. These tests catch that class of bug by running both copies on
identical synthetic inputs and asserting identical outputs.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_report.statistical_analysis.local import compute_statistics as local_stats
from data_report.statistical_analysis.local import inferential_analysis as local_inf
from data_report import config as local_config

HUB_PATH = Path(__file__).resolve().parents[1] / "hub_entrypoint_10.py"


@pytest.fixture(scope="module")
def hub():
    """Load hub_entrypoint_10.py as a module once per test session."""
    spec = importlib.util.spec_from_file_location("hub_entrypoint_10_under_test", HUB_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Heavy analysis libraries (scipy, sklearn, seaborn, ...) are lazy-loaded
    # after the FLAME SDK handshake in production; load them now so the
    # module's stats-dependent functions work under test.
    module._load_analysis_dependencies()
    return module


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

class TestSharedConstants:

    def test_outcome_keyword_groups_match(self, hub):
        assert hub.OUTCOME_KEYWORD_GROUPS == local_config.OUTCOME_KEYWORD_GROUPS

    def test_binary_min_count_matches(self, hub):
        assert hub._BINARY_MIN_COUNT == local_stats._BINARY_MIN_COUNT

    def test_medical_binary_keywords_match(self, hub):
        assert hub._MEDICAL_BINARY_KEYWORDS == local_stats._MEDICAL_BINARY_KEYWORDS

    def test_min_group_size_for_comparison_matches(self, hub):
        assert hub._MIN_GROUP_SIZE_FOR_COMPARISON == local_inf._MIN_GROUP_SIZE_FOR_COMPARISON


# ---------------------------------------------------------------------------
# is_binary
# ---------------------------------------------------------------------------

BINARY_TEST_CASES = [
    pytest.param(pd.Series([0, 1, 0, 1, 1]), "flag", id="numeric_01_dense"),
    pytest.param(pd.Series([0, 1]), "some_random_col", id="numeric_01_sparse_no_keyword"),
    pytest.param(pd.Series([0, 1]), "diabetes_present", id="numeric_01_sparse_with_keyword"),
    pytest.param(pd.Series([True, False, True]), "flag", id="bool_series"),
    pytest.param(pd.Series([1, 2, 3, 4, 5]), "age", id="non_binary_numeric"),
    pytest.param(pd.Series(["yes", "no", "yes"]), "smoker", id="string_yes_no"),
    pytest.param(pd.Series([np.nan, np.nan, np.nan]), "empty_col", id="all_nan"),
    pytest.param(pd.Series([0, 1, 2]), "flag", id="numeric_three_levels"),
]


class TestIsBinaryParity:

    @pytest.mark.parametrize("series, column_name", BINARY_TEST_CASES)
    def test_is_binary_matches(self, hub, series, column_name):
        local_result = local_stats.is_binary(series, column_name=column_name)
        hub_result = hub.is_binary(series, column_name=column_name)
        assert local_result == hub_result


# ---------------------------------------------------------------------------
# detect_column_types / detect_id_column
# ---------------------------------------------------------------------------

def make_mixed_dataframe():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "patient_id": [f"P{i:04d}" for i in range(100)],
        "age": rng.integers(0, 90, size=100),
        "sex": rng.choice(["M", "F"], size=100),
        "diabetes": rng.choice([0, 1], size=100),
        "visit_date": pd.date_range("2020-01-01", periods=100, freq="D"),
        "rare_flag": [0] * 98 + [1, 0],
        "all_missing": [np.nan] * 100,
        "lab_value": [str(v) for v in rng.normal(10, 2, size=100)],
    })


class TestColumnTypeDetectionParity:

    def test_detect_column_types_matches(self, hub):
        df = make_mixed_dataframe()
        local_result = local_stats.detect_column_types(df)
        hub_result = hub.detect_column_types(df)
        assert {k: sorted(v) for k, v in local_result.items()} == \
            {k: sorted(v) for k, v in hub_result.items()}

    @pytest.mark.parametrize("column_name", [
        "patient_id", "age", "sex", "visit_date", "subject_code", "name",
    ])
    def test_detect_id_column_matches(self, hub, column_name):
        df = make_mixed_dataframe()
        series = df[column_name] if column_name in df.columns else pd.Series(
            [f"S{i}" for i in range(100)]
        )
        local_result = local_stats.detect_id_column(series, column_name)
        hub_result = hub.detect_id_column(series, column_name)
        assert local_result == hub_result


# ---------------------------------------------------------------------------
# Quasi-numeric categorical detection (this session's fix)
# ---------------------------------------------------------------------------

class TestQuasiNumericParity:

    def test_censored_numeric_column_matches(self, hub):
        df = pd.DataFrame({
            "lab": ["12.5", "8.1", "<5", "300.2", "9.9", "<5", "11.0", "7.7"],
            "clean_numeric_as_str": [str(v) for v in range(8)],
            "clean_categorical": ["a", "b", "a", "b", "a", "b", "a", "b"],
        })
        categorical_cols = ["lab", "clean_numeric_as_str", "clean_categorical"]
        local_result = local_stats.detect_quasi_numeric_categorical_columns(df, categorical_cols)
        hub_result = hub.detect_quasi_numeric_categorical_columns(df, categorical_cols)
        assert sorted(local_result) == sorted(hub_result)


# ---------------------------------------------------------------------------
# Age histogram / out-of-range age count (this session's fix)
# ---------------------------------------------------------------------------

class TestAgeHistogramParity:

    @pytest.mark.parametrize("ages", [
        [5, 15, 25, 35, 45, 55, 65, 75, 85, 95],
        [-5, 5, 999, 40, 50],
        [np.nan, np.nan, 30],
        [],
    ])
    def test_compute_age_histogram_matches(self, hub, ages):
        df = pd.DataFrame({"age": ages}) if ages else pd.DataFrame({"age": pd.Series(dtype=float)})
        local_hist, local_edges = local_stats.compute_age_histogram(df)
        hub_hist, hub_edges = hub.compute_age_histogram(df)
        assert local_hist == hub_hist
        assert local_edges == hub_edges

    @pytest.mark.parametrize("ages", [
        [5, 15, 25, 35, 45, 55, 65, 75, 85, 95],
        [-5, 5, 999, 40, 50],
        [np.nan, np.nan, 30],
        [],
    ])
    def test_count_out_of_range_ages_matches(self, hub, ages):
        df = pd.DataFrame({"age": ages}) if ages else pd.DataFrame({"age": pd.Series(dtype=float)})
        local_count = local_stats.count_out_of_range_ages(df)
        hub_count = hub.count_out_of_range_ages(df)
        assert local_count == hub_count


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

class TestDescriptiveStatisticsParity:

    def test_compute_numeric_statistics_matches(self, hub):
        # hub's copy additionally reports q25/q75 (needed for its boxplot
        # rendering, which happens at the aggregator from serialized stats
        # rather than from a raw df) -- an intentional, reviewed divergence,
        # so only the fields present in both are compared here.
        rng = np.random.default_rng(1)
        numeric_df = pd.DataFrame({
            "age": rng.integers(0, 90, size=200).astype(float),
            "bmi": rng.normal(25, 4, size=200),
        })
        local_result = local_stats.compute_numeric_statistics(numeric_df)
        hub_result = hub.compute_numeric_statistics(numeric_df)
        for col in local_result:
            shared_keys = local_result[col].keys() & hub_result[col].keys()
            assert {k: local_result[col][k] for k in shared_keys} == \
                {k: hub_result[col][k] for k in shared_keys}
            assert local_result[col].keys() <= hub_result[col].keys()

    def test_compute_categorical_statistics_matches(self, hub):
        rng = np.random.default_rng(2)
        categorical_df = pd.DataFrame({
            "sex": rng.choice(["M", "F"], size=200),
            "diagnosis": rng.choice(["a", "b", "c"], size=200),
        })
        local_result = local_stats.compute_categorical_statistics(categorical_df)
        hub_result = hub.compute_categorical_statistics(categorical_df)
        assert local_result == hub_result


# ---------------------------------------------------------------------------
# Effect-size helpers used by screen_associations
# ---------------------------------------------------------------------------

class TestEffectSizeParity:

    def test_cohens_d_matches(self, hub):
        rng = np.random.default_rng(3)
        g1 = rng.normal(0, 1, size=50)
        g2 = rng.normal(0.5, 1, size=50)
        assert local_inf._cohens_d(g1, g2) == hub._cohens_d(g1, g2)

    def test_hedges_g_matches(self, hub):
        rng = np.random.default_rng(4)
        g1 = rng.normal(0, 1, size=50)
        g2 = rng.normal(0.5, 1, size=50)
        assert local_inf._hedges_g(g1, g2) == hub._hedges_g(g1, g2)


# ---------------------------------------------------------------------------
# screen_associations (full pipeline, including the group-size guard)
# ---------------------------------------------------------------------------

def make_screening_dataframe():
    rng = np.random.default_rng(5)
    n = 150
    num_a = rng.normal(0, 1, size=n)
    num_b = num_a * 0.6 + rng.normal(0, 0.8, size=n)
    grp_strong = np.where(num_a + rng.normal(0, 0.3, size=n) > 0, "high", "low")
    cat_x = rng.choice(["a", "b"], size=n)
    flip = rng.random(n) < 0.2
    cat_y = np.where(flip, np.where(cat_x == "a", "b", "a"), cat_x)

    df = pd.DataFrame({
        "num_a": num_a, "num_b": num_b,
        "grp_strong": grp_strong, "cat_x": cat_x, "cat_y": cat_y,
    })
    # Singleton group to exercise the group-size guard identically in both copies.
    df["grp_singleton"] = "common"
    df.loc[df.index[0], "grp_singleton"] = "rare"
    return df


def screening_column_types():
    return {
        "numeric": ["num_a", "num_b"],
        "categorical": ["grp_strong", "cat_x", "cat_y", "grp_singleton"],
        "temporal": [],
    }


class TestScreenAssociationsParity:

    def test_screening_output_matches(self, hub):
        df = make_screening_dataframe()
        column_types = screening_column_types()

        local_screening = local_inf.screen_associations(df, column_types)
        hub_screening = hub.screen_associations(df, column_types)

        sort_cols = ["var1", "var2", "pair_type"]
        local_sorted = local_screening.sort_values(sort_cols).reset_index(drop=True)
        hub_sorted = hub_screening.sort_values(sort_cols).reset_index(drop=True)

        pd.testing.assert_frame_equal(local_sorted, hub_sorted, check_dtype=False)

    def test_singleton_group_excluded_in_both(self, hub):
        df = make_screening_dataframe()
        column_types = screening_column_types()

        local_screening = local_inf.screen_associations(df, column_types)
        hub_screening = hub.screen_associations(df, column_types)

        local_num_cat = local_screening[local_screening["pair_type"] == "num-cat"]
        hub_num_cat = hub_screening[hub_screening["pair_type"] == "num-cat"]
        assert "grp_singleton" not in set(local_num_cat["var2"])
        assert "grp_singleton" not in set(hub_num_cat["var2"])
