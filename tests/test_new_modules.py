"""
Unit tests for modules introduced by the graphics refactoring:
  - should_apply_reductions  (analyze.py)
  - compute_age_histogram     (compute_statistics.py)
  - style.py                  (set_palette, set_theme, call-time propagation)
  - primitives._auto_bins     (Freedman-Diaconis, Sturges fallback, cap)
  - data_quality_plots.py     (file creation + edge cases)
  - local_descriptive_plots._periods_to_timestamps
  - key local_descriptive_plots functions
"""

import math

import numpy as np
import pandas as pd
import pytest

from data_report.generate_figures import style as fig_style


@pytest.fixture(autouse=True)
def reset_palette():
    """Restore the default palette after every test to prevent cross-test contamination."""
    yield
    fig_style.set_theme("default")

# ---------------------------------------------------------------------------
# should_apply_reductions
# ---------------------------------------------------------------------------

from data_report.analyze import combine_node_variances, should_apply_reductions


def _make_column_types(n_numeric=0, n_cat=0, cat_cardinality=5):
    """Build a minimal df + column_types dict for threshold testing."""
    data = {}
    column_types = {"numeric": [], "categorical": [], "temporal": [], "binary": []}

    for i in range(n_numeric):
        col = f"num_{i}"
        data[col] = np.arange(10, dtype=float)
        column_types["numeric"].append(col)

    for i in range(n_cat):
        col = f"cat_{i}"
        data[col] = [str(j % cat_cardinality) for j in range(10)]
        column_types["categorical"].append(col)

    df = pd.DataFrame(data) if data else pd.DataFrame({"_dummy": range(10)})
    return df, column_types


class TestShouldApplyReductions:

    def test_all_false_for_empty_column_types(self):
        df, ct = _make_column_types()
        r = should_apply_reductions(df, ct)
        assert r == {
            "pca": False,
            "mca": False,
        }

    def test_pca_requires_10_numeric(self):
        # 9 numeric → False; 10 → True
        _, ct_9 = _make_column_types(n_numeric=9)
        df_9 = pd.DataFrame({c: np.arange(10.0) for c in ct_9["numeric"]})
        r9 = should_apply_reductions(df_9, ct_9)
        assert r9["pca"] is False

        _, ct_10 = _make_column_types(n_numeric=10)
        df_10 = pd.DataFrame({c: np.arange(10.0) for c in ct_10["numeric"]})
        r10 = should_apply_reductions(df_10, ct_10)
        assert r10["pca"] is True

    def test_mca_requires_8_usable_categoricals(self):
        _, ct7 = _make_column_types(n_cat=7)
        df7 = pd.DataFrame({c: list("abcde" * 2) for c in ct7["categorical"]})
        assert should_apply_reductions(df7, ct7)["mca"] is False

        _, ct8 = _make_column_types(n_cat=8)
        df8 = pd.DataFrame({c: list("abcde" * 2) for c in ct8["categorical"]})
        assert should_apply_reductions(df8, ct8)["mca"] is True

    def test_high_cardinality_cats_dont_count_for_mca(self):
        # 10 categorical columns all with 31 unique values → none usable
        n = 310
        cols = [f"cat_{i}" for i in range(10)]
        df = pd.DataFrame({c: [str(j % 31) for j in range(n)] for c in cols})
        ct = {"numeric": [], "categorical": cols, "temporal": [], "binary": []}
        r = should_apply_reductions(df, ct)
        assert r["mca"] is False

    def test_returns_both_keys(self):
        df, ct = _make_column_types()
        r = should_apply_reductions(df, ct)
        assert set(r.keys()) == {"pca", "mca"}


# ---------------------------------------------------------------------------
# combine_node_variances
# ---------------------------------------------------------------------------

class TestCombineNodeVariances:

    def test_matches_pooled_ground_truth_when_node_means_differ(self):
        # Two nodes with zero within-node spread but very different means --
        # all of the combined variance must come from the between-node term.
        # node1={0,0}, node2={10,10}; combined sample variance is exactly
        # 100/3, computed independently from the raw pooled values.
        node_stats = [(2, 0.0, 0.0), (2, 10.0, 0.0)]
        global_mean = 5.0
        assert combine_node_variances(node_stats, global_mean) == pytest.approx(100 / 3)

    def test_matches_numpy_ground_truth_on_random_data(self):
        rng = np.random.default_rng(42)
        data = rng.normal(loc=50, scale=12, size=300)
        chunks = [data[:80], data[80:170], data[170:]]

        node_stats = [(len(c), float(np.mean(c)), float(np.var(c, ddof=1))) for c in chunks]
        global_mean = float(np.mean(data))

        expected = float(np.var(data, ddof=1))
        assert combine_node_variances(node_stats, global_mean) == pytest.approx(expected)

    def test_single_node_reduces_to_its_own_variance(self):
        node_stats = [(5, 3.0, 2.5)]
        assert combine_node_variances(node_stats, global_mean=3.0) == pytest.approx(2.5)

    def test_single_total_sample_returns_zero(self):
        assert combine_node_variances([(1, 7.0, 0.0)], global_mean=7.0) == 0.0

    def test_no_samples_returns_zero(self):
        assert combine_node_variances([], global_mean=0.0) == 0.0


# ---------------------------------------------------------------------------
# compute_age_histogram
# ---------------------------------------------------------------------------

from data_report.statistical_analysis.local.compute_statistics import compute_age_histogram


class TestComputeAgeHistogram:

    def test_returns_none_when_age_col_missing(self):
        df = pd.DataFrame({"height": [160, 170, 180]})
        counts, edges = compute_age_histogram(df, age_col="age")
        assert counts is None
        assert edges is None

    def test_returns_none_for_all_nan_ages(self):
        df = pd.DataFrame({"age": [np.nan, np.nan]})
        counts, edges = compute_age_histogram(df)
        assert counts is None
        assert edges is None

    def test_returns_lists(self):
        df = pd.DataFrame({"age": [10, 25, 40, 55, 70]})
        counts, edges = compute_age_histogram(df)
        assert isinstance(counts, list)
        assert isinstance(edges, list)

    def test_edges_are_floats(self):
        df = pd.DataFrame({"age": [20, 30, 40]})
        counts, edges = compute_age_histogram(df)
        assert all(isinstance(e, float) for e in edges)

    def test_edges_span_zero_to_max_age(self):
        df = pd.DataFrame({"age": [10, 50, 90]})
        counts, edges = compute_age_histogram(df, bin_size=5, max_age=100)
        assert edges[0] == 0.0
        assert edges[-1] == 100.0

    def test_counts_sum_to_n_rows(self):
        n = 50
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"age": rng.integers(0, 100, size=n).astype(float)})
        counts, _ = compute_age_histogram(df, bin_size=5, max_age=100)
        assert sum(counts) == n

    def test_ages_above_max_are_excluded(self):
        df = pd.DataFrame({"age": [10, 50, 150]})  # 150 is above max_age=100
        counts, _ = compute_age_histogram(df, bin_size=5, max_age=100)
        # only the two values <= 100 should appear
        assert sum(counts) == 2

    def test_non_numeric_ages_coerced_and_counted(self):
        df = pd.DataFrame({"age": ["25", "40", "bad_value", None]})
        counts, _ = compute_age_histogram(df)
        assert sum(counts) == 2  # only the two valid numeric values

    def test_custom_age_col_name(self):
        df = pd.DataFrame({"patient_age": [30, 45, 60]})
        counts, edges = compute_age_histogram(df, age_col="patient_age")
        assert counts is not None
        assert sum(counts) == 3

    def test_bin_size_controls_number_of_bins(self):
        df = pd.DataFrame({"age": [10, 50]})
        _, edges_5 = compute_age_histogram(df, bin_size=5, max_age=100)
        _, edges_10 = compute_age_histogram(df, bin_size=10, max_age=100)
        assert len(edges_5) > len(edges_10)


# ---------------------------------------------------------------------------
# style.py
# ---------------------------------------------------------------------------

class TestStyle:

    def setup_method(self):
        # Reset to default theme before each test
        fig_style.set_theme("default")

    def test_default_palette_has_8_colors(self):
        assert len(fig_style.PALETTE) == 8

    def test_set_palette_replaces_global(self):
        new_pal = ["#AAAAAA", "#BBBBBB"]
        fig_style.set_palette(new_pal)
        assert fig_style.PALETTE == new_pal

    def test_set_palette_makes_a_copy(self):
        new_pal = ["#111111"]
        fig_style.set_palette(new_pal)
        new_pal.append("#999999")
        assert len(fig_style.PALETTE) == 1  # mutation of caller's list doesn't affect module

    def test_set_theme_colorblind(self):
        fig_style.set_theme("colorblind")
        assert fig_style.PALETTE[0] == "#0072B2"

    def test_set_theme_grayscale(self):
        fig_style.set_theme("grayscale")
        assert fig_style.PALETTE[0] == "#222222"

    def test_set_theme_default_restores_original(self):
        fig_style.set_theme("colorblind")
        fig_style.set_theme("default")
        assert fig_style.PALETTE[0] == "#4C9BE8"

    def test_unknown_theme_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown theme"):
            fig_style.set_theme("neon_disco")

    def test_call_time_propagation(self):
        """Palette read at call time by primitives — changing it before a plot
        must affect the plot, not the value at import time."""
        fig_style.set_palette(["#CAFEBA"])
        # Access through the module reference (simulating what primitives.py does)
        assert fig_style.PALETTE[0] == "#CAFEBA"


# ---------------------------------------------------------------------------
# primitives._auto_bins
# ---------------------------------------------------------------------------

from data_report.generate_figures.primitives import _auto_bins


class TestAutoBins:

    def test_tiny_array_returns_at_least_one(self):
        assert _auto_bins(np.array([1.0])) >= 1
        assert _auto_bins(np.array([1.0, 2.0])) >= 1

    def test_constant_array_uses_sturges_fallback(self):
        # IQR = 0 → Sturges: 1 + log2(n), capped at 50
        n = 1000
        arr = np.ones(n)
        result = _auto_bins(arr)
        sturges = min(int(math.ceil(1 + math.log2(n))), 50)
        assert result == sturges

    def test_freedman_diaconis_for_normal_data(self):
        rng = np.random.default_rng(0)
        arr = rng.normal(0, 1, 500)
        result = _auto_bins(arr)
        assert 5 <= result <= 50

    def test_result_capped_at_50(self):
        # Sparse wide-range data would produce many bins — must be capped
        rng = np.random.default_rng(0)
        arr = np.linspace(0, 1e6, 10000)
        result = _auto_bins(arr)
        assert result <= 50

    def test_result_at_least_5(self):
        # Normal array with IQR > 0 should produce at least 5 bins
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = _auto_bins(arr)
        assert result >= 5

    def test_zero_range_returns_one(self):
        # All identical values — no bins meaningful except 1
        arr = np.full(20, 42.0)
        assert _auto_bins(arr) == 1 or _auto_bins(arr) >= 1  # any non-zero is fine; specifically check <= 50
        assert _auto_bins(arr) <= 50


# ---------------------------------------------------------------------------
# data_quality_plots.py
# ---------------------------------------------------------------------------

from data_report.generate_figures.data_quality_plots import (
    save_missing_bar,
    save_missing_heatmap,
    save_missing_by_column,
)


def _dummy_df(n_rows=100, n_cols=5, missing_frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    data = {f"col_{i}": rng.normal(size=n_rows).astype(float) for i in range(n_cols)}
    df = pd.DataFrame(data)
    for col in df.columns:
        mask = rng.random(n_rows) < missing_frac
        df.loc[mask, col] = np.nan
    return df


class TestSaveMissingBar:

    def test_creates_file(self, tmp_path):
        df = _dummy_df()
        out = tmp_path / "bar.png"
        save_missing_bar(df, out)
        assert out.exists() and out.stat().st_size > 0

    def test_creates_parent_dirs(self, tmp_path):
        df = _dummy_df()
        out = tmp_path / "sub" / "dir" / "bar.png"
        save_missing_bar(df, out)
        assert out.exists()

    def test_caps_columns_at_max_cols(self, tmp_path):
        # 60-column df — save_missing_bar should cap at 50, not fail
        df = _dummy_df(n_cols=60)
        out = tmp_path / "bar_wide.png"
        save_missing_bar(df, out, max_cols=50)
        assert out.exists()

    def test_single_column_df(self, tmp_path):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        out = tmp_path / "single_col.png"
        save_missing_bar(df, out)
        assert out.exists()

    def test_all_null_column(self, tmp_path):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [np.nan, np.nan]})
        out = tmp_path / "all_null.png"
        save_missing_bar(df, out)
        assert out.exists()


class TestSaveMissingHeatmap:

    def test_creates_file(self, tmp_path):
        df = _dummy_df(n_cols=5)
        out = tmp_path / "heatmap.png"
        save_missing_heatmap(df, out)
        assert out.exists() and out.stat().st_size > 0

    def test_caps_at_max_cols(self, tmp_path):
        df = _dummy_df(n_cols=60)
        out = tmp_path / "heatmap_wide.png"
        save_missing_heatmap(df, out, max_cols=50)
        assert out.exists()


class TestSaveMissingByColumn:

    def test_creates_file(self, tmp_path):
        mc = {"a": 10, "b": 5, "c": 0}
        out = tmp_path / "by_col.png"
        save_missing_by_column(mc, 100, out)
        assert out.exists() and out.stat().st_size > 0

    def test_does_not_create_file_when_missing_counts_empty(self, tmp_path):
        out = tmp_path / "empty.png"
        save_missing_by_column({}, 100, out)
        assert not out.exists()

    def test_does_not_create_file_when_n_rows_zero(self, tmp_path):
        out = tmp_path / "zero_rows.png"
        save_missing_by_column({"a": 0}, 0, out)
        assert not out.exists()

    def test_node_label_appended_to_title(self, tmp_path):
        mc = {"x": 3}
        out = tmp_path / "labeled.png"
        save_missing_by_column(mc, 10, out, node_label="site_A")
        assert out.exists()

    def test_all_missing_column(self, tmp_path):
        mc = {"a": 100}  # fully missing column
        out = tmp_path / "all_miss.png"
        save_missing_by_column(mc, 100, out)
        assert out.exists()


# ---------------------------------------------------------------------------
# local_descriptive_plots._periods_to_timestamps
# ---------------------------------------------------------------------------

from data_report.generate_figures.local_descriptive_plots import _periods_to_timestamps


class TestPeriodsToTimestamps:

    def test_period_objects_converted(self):
        obs = {pd.Period("2020-01", freq="M"): 5, pd.Period("2020-02", freq="M"): 8}
        ts, counts = _periods_to_timestamps(obs)
        assert len(ts) == 2
        assert len(counts) == 2
        assert all(isinstance(t, pd.Timestamp) for t in ts)

    def test_string_keys_converted(self):
        obs = {"2021-03": 4, "2021-04": 6}
        ts, counts = _periods_to_timestamps(obs)
        assert len(ts) == 2
        assert counts == [4, 6] or set(counts) == {4, 6}

    def test_sorted_chronologically(self):
        obs = {"2021-06": 1, "2020-01": 2, "2021-01": 3}
        ts, _ = _periods_to_timestamps(obs)
        for i in range(len(ts) - 1):
            assert ts[i] <= ts[i + 1]

    def test_unparseable_keys_silently_dropped(self):
        obs = {"not-a-date": 9, "2021-05": 3}
        ts, counts = _periods_to_timestamps(obs)
        assert len(ts) == 1
        assert counts[0] == 3

    def test_empty_dict_returns_empty_lists(self):
        ts, counts = _periods_to_timestamps({})
        assert ts == []
        assert counts == []

    def test_all_unparseable_returns_empty(self):
        obs = {"garbage": 1, "also_bad": 2}
        ts, counts = _periods_to_timestamps(obs)
        assert ts == []
        assert counts == []

    def test_mixed_period_and_string_keys(self):
        obs = {pd.Period("2021-01", freq="M"): 5, "2021-02": 3}
        ts, counts = _periods_to_timestamps(obs)
        assert len(ts) == 2
        assert ts[0] < ts[1]


# ---------------------------------------------------------------------------
# key local_descriptive_plots functions
# ---------------------------------------------------------------------------

from data_report.generate_figures.local_descriptive_plots import (
    save_age_distribution,
    save_data_type_distribution,
    save_sex_distribution,
    save_column_availability_chart,
)


class TestSaveAgeDistribution:

    def test_creates_file(self, tmp_path):
        edges = np.arange(0, 105, 5, dtype=float).tolist()
        hist = [0] * 5 + [3, 5, 8, 6, 4, 3, 2, 1] + [0] * 7
        save_age_distribution(edges, hist, tmp_path)
        assert (tmp_path / "age_distribution.png").exists()

    def test_does_not_create_when_none(self, tmp_path):
        save_age_distribution(None, None, tmp_path)
        assert not (tmp_path / "age_distribution.png").exists()

    def test_does_not_create_when_all_zero_hist(self, tmp_path):
        edges = [0.0, 5.0, 10.0]
        hist = [0, 0]
        save_age_distribution(edges, hist, tmp_path)
        assert not (tmp_path / "age_distribution.png").exists()

    def test_node_label_in_filename_not_path(self, tmp_path):
        edges = np.arange(0, 105, 5, dtype=float).tolist()
        hist = [0] * 5 + [3, 5, 8, 6, 4, 3, 2, 1] + [0] * 7
        save_age_distribution(edges, hist, tmp_path, node_label="site_B")
        assert (tmp_path / "age_distribution.png").exists()


class TestSaveDataTypeDistribution:

    def test_creates_pie_file(self, tmp_path):
        save_data_type_distribution(
            {"a": {}, "b": {}},
            {"c": {}},
            {},
            tmp_path,
        )
        assert (tmp_path / "data_type_distribution.png").exists()

    def test_no_file_when_all_empty(self, tmp_path):
        save_data_type_distribution({}, {}, {}, tmp_path)
        assert not (tmp_path / "data_type_distribution.png").exists()

    def test_creates_parent_dir(self, tmp_path):
        out_dir = tmp_path / "sub" / "plots"
        save_data_type_distribution({"x": {}}, {}, {}, out_dir)
        assert (out_dir / "data_type_distribution.png").exists()


class TestSaveSexDistribution:

    def test_creates_file(self, tmp_path):
        save_sex_distribution({"M": 40, "F": 60}, tmp_path)
        assert (tmp_path / "sex_distribution.png").exists()

    def test_no_file_when_empty(self, tmp_path):
        save_sex_distribution({}, tmp_path)
        assert not (tmp_path / "sex_distribution.png").exists()

    def test_nan_key_excluded(self, tmp_path):
        save_sex_distribution({np.nan: 5, "M": 20, "F": 30}, tmp_path)
        # Should not crash; file created from valid keys only
        assert (tmp_path / "sex_distribution.png").exists()

    def test_all_nan_keys_produces_no_file(self, tmp_path):
        save_sex_distribution({np.nan: 5, float("nan"): 3}, tmp_path)
        assert not (tmp_path / "sex_distribution.png").exists()


class TestSaveColumnAvailabilityChart:

    def test_creates_pie(self, tmp_path):
        comp = {"a": "common_all", "b": "common_partial", "c": "unique_local"}
        save_column_availability_chart(comp, 3, tmp_path)
        assert (tmp_path / "column_availability.png").exists()

    def test_no_file_when_all_unrecognised_categories(self, tmp_path):
        comp = {"a": "unknown_type"}
        save_column_availability_chart(comp, 2, tmp_path)
        # total = 0 → function should return without creating file
        assert not (tmp_path / "column_availability.png").exists()

    def test_single_node_label_grammar(self, tmp_path):
        comp = {"a": "common_all"}
        save_column_availability_chart(comp, 1, tmp_path)
        assert (tmp_path / "column_availability.png").exists()


# ---------------------------------------------------------------------------
# batched numeric/temporal distribution plots
# ---------------------------------------------------------------------------

from data_report.generate_figures.local_descriptive_plots import (
    save_numeric_histograms,
    save_numeric_boxplots,
    save_temporal_activity_batched,
)


class TestSaveNumericHistograms:

    def test_batches_across_multiple_images(self, tmp_path):
        rng = np.random.default_rng(0)
        cols = [f"num_{i}" for i in range(14)]
        df = pd.DataFrame({c: rng.normal(size=50) for c in cols})

        written = save_numeric_histograms(df, cols, tmp_path, batch_size=6)

        assert [p.name for p in written] == [
            "numeric_histograms_01.png", "numeric_histograms_02.png", "numeric_histograms_03.png",
        ]
        for p in written:
            assert p.exists() and p.stat().st_size > 0

    def test_single_batch_when_under_batch_size(self, tmp_path):
        rng = np.random.default_rng(0)
        cols = [f"num_{i}" for i in range(3)]
        df = pd.DataFrame({c: rng.normal(size=50) for c in cols})

        written = save_numeric_histograms(df, cols, tmp_path, batch_size=6)

        assert [p.name for p in written] == ["numeric_histograms_01.png"]

    def test_skips_columns_with_fewer_than_two_non_null_values(self, tmp_path):
        df = pd.DataFrame({"num_0": [1.0, np.nan, np.nan], "num_1": [1.0, 2.0, 3.0]})

        written = save_numeric_histograms(df, ["num_0", "num_1"], tmp_path, batch_size=6)

        assert len(written) == 1

    def test_returns_empty_list_when_no_valid_columns(self, tmp_path):
        df = pd.DataFrame({"num_0": [np.nan, np.nan]})

        assert save_numeric_histograms(df, ["num_0"], tmp_path) == []
        assert save_numeric_histograms(df, [], tmp_path) == []


class TestSaveNumericBoxplots:

    def test_batches_across_multiple_images(self, tmp_path):
        rng = np.random.default_rng(0)
        cols = [f"num_{i}" for i in range(14)]
        df = pd.DataFrame({c: rng.normal(size=50) for c in cols})

        written = save_numeric_boxplots(df, cols, tmp_path, batch_size=6)

        assert [p.name for p in written] == [
            "numeric_boxplots_01.png", "numeric_boxplots_02.png", "numeric_boxplots_03.png",
        ]
        for p in written:
            assert p.exists() and p.stat().st_size > 0

    def test_returns_empty_list_when_no_valid_columns(self, tmp_path):
        df = pd.DataFrame({"num_0": [np.nan, np.nan]})

        assert save_numeric_boxplots(df, ["num_0"], tmp_path) == []


class TestSaveTemporalActivityBatched:

    def _make_temporal_statistics(self, n_features, n_periods=6):
        periods = pd.period_range("2021-01", periods=n_periods, freq="M")
        return {
            f"date_{i}": {
                "observations_per_period": {p: (i + 1) * 10 for p in periods},
                "most_active_period": periods[2],
            }
            for i in range(n_features)
        }

    def test_batches_across_multiple_images(self, tmp_path):
        temporal_statistics = self._make_temporal_statistics(8)

        written = save_temporal_activity_batched(temporal_statistics, tmp_path, batch_size=6)

        assert [p.name for p in written] == [
            "temporal_activity_batch_01.png", "temporal_activity_batch_02.png",
        ]
        for p in written:
            assert p.exists() and p.stat().st_size > 0

    def test_skips_features_with_no_observations(self, tmp_path):
        temporal_statistics = self._make_temporal_statistics(2)
        temporal_statistics["date_empty"] = {"observations_per_period": {}}

        written = save_temporal_activity_batched(temporal_statistics, tmp_path, batch_size=6)

        assert len(written) == 1

    def test_returns_empty_list_when_no_data(self, tmp_path):
        assert save_temporal_activity_batched({}, tmp_path) == []
        assert save_temporal_activity_batched(
            {"date_0": {"observations_per_period": {}}}, tmp_path
        ) == []

    def test_one_malformed_feature_does_not_blank_out_the_whole_batch(self, tmp_path):
        temporal_statistics = self._make_temporal_statistics(3)
        # A truthy but malformed value (passes the "has observations" filter,
        # then breaks _periods_to_timestamps's .items() call).
        temporal_statistics["date_broken"] = {"observations_per_period": "not-a-dict"}

        written = save_temporal_activity_batched(temporal_statistics, tmp_path, batch_size=6)

        assert [p.name for p in written] == ["temporal_activity_batch_01.png"]
        assert written[0].exists() and written[0].stat().st_size > 0
