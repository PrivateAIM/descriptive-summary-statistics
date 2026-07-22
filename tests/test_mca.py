"""
Unit tests for the standalone MCA visualization module.
Covers: run_mca, every save_* renderer, and the save_mca_outputs orchestrator.
"""

import numpy as np
import pandas as pd
import pytest

from data_report.generate_figures.mca_plots import (
    MCAResult,
    run_mca,
    save_explained_inertia_plot,
    save_mca_row_scatter,
    save_mca_scatter_matrix,
    save_mca_column_map,
    save_mca_3d_html,
    save_mca_overview,
    save_mca_outputs,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_categorical_df(n=200, seed=0, with_target=False, with_missing=False, with_high_cardinality=False):
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "symptom_a": rng.choice(["present", "absent"], size=n),
        "symptom_b": rng.choice(["mild", "moderate", "severe"], size=n),
        "sex": rng.choice(["M", "F"], size=n),
        "smoker": rng.choice(["yes", "no", "unknown"], size=n),
    })

    if with_missing:
        df.loc[0:3, "symptom_a"] = np.nan

    if with_high_cardinality:
        df["patient_code"] = [f"id_{i}" for i in range(n)]

    if with_target:
        df["pasc"] = rng.choice(["yes", "no"], size=n)

    return df


def categorical_feature_names(df):
    return [c for c in ["symptom_a", "symptom_b", "sex", "smoker"] if c in df.columns]


# ---------------------------------------------------------------------------
# run_mca
# ---------------------------------------------------------------------------

class TestRunMca:

    def test_returns_mca_result_with_expected_shapes(self):
        df = make_categorical_df(n=200)
        features = categorical_feature_names(df)

        result = run_mca(df, features)

        assert isinstance(result, MCAResult)
        assert result.row_coordinates.shape[0] == 200
        assert result.row_coordinates.shape[1] == result.column_coordinates.shape[1]
        assert result.feature_names == features

    def test_explained_inertia_ratio_sums_to_one(self):
        df = make_categorical_df(n=200)
        result = run_mca(df, categorical_feature_names(df))

        assert result.explained_inertia_ratio.sum() == pytest.approx(1.0)

    def test_dimensions_ordered_by_decreasing_inertia(self):
        df = make_categorical_df(n=300)
        result = run_mca(df, categorical_feature_names(df))

        ratios = result.explained_inertia_ratio
        assert all(ratios[i] >= ratios[i + 1] - 1e-12 for i in range(len(ratios) - 1))

    def test_recommended_n_components_within_bounds(self):
        df = make_categorical_df(n=200)
        result = run_mca(df, categorical_feature_names(df))

        assert 1 <= result.recommended_n_components <= len(result.explained_inertia_ratio)

    def test_recommended_n_components_reaches_threshold(self):
        df = make_categorical_df(n=300)
        result = run_mca(df, categorical_feature_names(df), variance_threshold=0.9)

        cumulative = np.cumsum(result.explained_inertia_ratio)
        k = result.recommended_n_components
        assert cumulative[k - 1] >= 0.9 - 1e-9
        if k > 1:
            assert cumulative[k - 2] < 0.9

    def test_column_coordinates_indexed_by_variable_and_level(self):
        df = make_categorical_df(n=200)
        result = run_mca(df, categorical_feature_names(df))

        # every "variable__level" label should map back to one of the source features
        assert set(result.column_variable.unique()) == set(result.feature_names)
        # every original category level should be represented
        for col in result.feature_names:
            for level in df[col].unique():
                assert f"{col}__{level}" in result.column_coordinates.index

    def test_handles_missing_values(self):
        df = make_categorical_df(n=200, with_missing=True)
        result = run_mca(df, categorical_feature_names(df))

        assert not np.isnan(result.row_coordinates).any()

    def test_raises_on_non_categorical_feature(self):
        df = make_categorical_df(n=100)
        df["num_x"] = np.arange(len(df), dtype=float)

        with pytest.raises(ValueError, match="non-categorical"):
            run_mca(df, categorical_feature_names(df) + ["num_x"])

    def test_raises_on_empty_feature_list(self):
        df = make_categorical_df(n=100)

        with pytest.raises(ValueError, match="at least one categorical feature"):
            run_mca(df, [])

    def test_raises_on_single_feature(self):
        df = make_categorical_df(n=100)

        with pytest.raises(ValueError, match="at least 2 categorical features"):
            run_mca(df, ["symptom_a"])

    def test_raises_on_high_cardinality_column(self):
        df = make_categorical_df(n=100, with_high_cardinality=True)

        with pytest.raises(ValueError, match="exceed max_levels_per_variable"):
            run_mca(df, categorical_feature_names(df) + ["patient_code"])

    def test_drops_entirely_missing_column_instead_of_failing(self):
        df = make_categorical_df(n=100)
        df["empty_col"] = pd.Series([np.nan] * 100, dtype="object")
        features = categorical_feature_names(df) + ["empty_col"]

        result = run_mca(df, features)

        assert "empty_col" not in result.feature_names
        assert set(result.feature_names) == set(categorical_feature_names(df))

    def test_raises_if_missing_columns_leave_fewer_than_two(self):
        df = make_categorical_df(n=100)
        df["empty_col"] = pd.Series([np.nan] * 100, dtype="object")

        with pytest.raises(ValueError, match="at least 2 categorical features"):
            run_mca(df, ["symptom_a", "empty_col"])

    def test_drops_entirely_missing_column_regardless_of_inferred_dtype(self):
        # An all-NaN column built without an explicit dtype is inferred as
        # float64 by pandas, not object -- the empty-column check must catch
        # this case too, not just object-dtype all-NaN columns.
        df = make_categorical_df(n=100)
        df["empty_float"] = pd.Series([np.nan] * 100)
        assert df["empty_float"].dtype == np.float64
        features = categorical_feature_names(df) + ["empty_float"]

        result = run_mca(df, features)

        assert "empty_float" not in result.feature_names
        assert set(result.feature_names) == set(categorical_feature_names(df))


# ---------------------------------------------------------------------------
# save_* renderers (smoke tests: do they produce non-empty files?)
# ---------------------------------------------------------------------------

class TestSaveRenderers:

    def _result(self, with_target=False):
        df = make_categorical_df(n=200, with_target=with_target)
        result = run_mca(df, categorical_feature_names(df))
        return df, result

    def _assert_written(self, path):
        assert path.exists()
        assert path.stat().st_size > 0

    def test_save_explained_inertia_plot(self, tmp_path):
        _, result = self._result()
        out = tmp_path / "inertia.png"
        save_explained_inertia_plot(result, out)
        self._assert_written(out)

    def test_save_mca_row_scatter_without_target(self, tmp_path):
        df, result = self._result()
        out = tmp_path / "row_scatter.png"
        save_mca_row_scatter(result, df, out)
        self._assert_written(out)

    def test_save_mca_row_scatter_with_target(self, tmp_path):
        df, result = self._result(with_target=True)
        out = tmp_path / "row_scatter_target.png"
        save_mca_row_scatter(result, df, out, target="pasc")
        self._assert_written(out)

    def test_save_mca_scatter_matrix(self, tmp_path):
        df, result = self._result(with_target=True)
        out = tmp_path / "matrix.png"
        save_mca_scatter_matrix(result, df, out, target="pasc")
        self._assert_written(out)

    def test_save_mca_column_map(self, tmp_path):
        _, result = self._result()
        out = tmp_path / "column_map.png"
        save_mca_column_map(result, out)
        self._assert_written(out)

    def test_save_mca_3d_html(self, tmp_path):
        df, result = self._result(with_target=True)
        out = tmp_path / "scatter3d.html"
        save_mca_3d_html(result, df, out, target="pasc")
        self._assert_written(out)
        assert out.read_text(encoding="utf-8").lstrip().startswith("<")

    def test_save_mca_overview(self, tmp_path):
        df, result = self._result(with_target=True)
        out = tmp_path / "overview.png"
        save_mca_overview(result, df, out, target="pasc")
        self._assert_written(out)


# ---------------------------------------------------------------------------
# save_mca_outputs (orchestrator)
# ---------------------------------------------------------------------------

class TestSaveMcaOutputs:

    def test_creates_output_directory_and_full_set_of_files(self, tmp_path):
        df = make_categorical_df(n=200, with_target=True)
        out_dir = tmp_path / "mca"

        result = save_mca_outputs(df, categorical_feature_names(df), out_dir, target="pasc")

        assert isinstance(result, MCAResult)
        produced = {p.name for p in out_dir.iterdir()}
        # Column maps are now batched: mca_column_map_batch_01.png, _02.png, …
        # (5 source variables per batch). Old fixed-name files are gone.
        assert "mca_explained_inertia.png" in produced
        assert "mca_row_scatter_2d.png" in produced
        assert "mca_scatter_matrix.png" in produced
        assert "mca_scatter_3d.html" in produced
        assert "mca_overview.png" in produced
        assert any(p.startswith("mca_column_map_batch_") for p in produced)
        assert "mca_column_map_dim1_dim2.png" not in produced
        assert "mca_column_map_dim1_dim3.png" not in produced
        for p in out_dir.iterdir():
            assert p.stat().st_size > 0

    def test_skips_three_dimensional_outputs_with_only_two_binary_features(self, tmp_path):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "a": rng.choice(["x", "y"], size=80),
            "b": rng.choice(["p", "q"], size=80),
        })
        out_dir = tmp_path / "mca_2d"

        save_mca_outputs(df, ["a", "b"], out_dir)

        produced = {p.name for p in out_dir.iterdir()}
        assert "mca_scatter_3d.html" not in produced
        assert "mca_scatter_matrix.png" not in produced
        assert "mca_row_scatter_2d.png" in produced
        # With only 2 source variables there should be exactly one batch file
        batch_files = [p for p in produced if p.startswith("mca_column_map_batch_")]
        assert len(batch_files) == 1

    def test_ignores_target_column_not_present_in_dataframe(self, tmp_path):
        df = make_categorical_df(n=120)
        out_dir = tmp_path / "mca_no_target"

        save_mca_outputs(df, categorical_feature_names(df), out_dir, target="missing_label")

        assert (out_dir / "mca_row_scatter_2d.png").exists()

    def test_writes_excluded_columns_csv_when_a_column_is_entirely_missing(self, tmp_path):
        df = make_categorical_df(n=120)
        df["empty_col"] = pd.Series([np.nan] * 120, dtype="object")
        out_dir = tmp_path / "mca_excluded"

        save_mca_outputs(df, categorical_feature_names(df) + ["empty_col"], out_dir)

        excluded = pd.read_csv(out_dir / "excluded_columns.csv")
        assert list(excluded["feature"]) == ["empty_col"]

    def test_no_excluded_columns_csv_when_nothing_dropped(self, tmp_path):
        df = make_categorical_df(n=120)
        out_dir = tmp_path / "mca_no_excluded"

        save_mca_outputs(df, categorical_feature_names(df), out_dir)

        assert not (out_dir / "excluded_columns.csv").exists()
