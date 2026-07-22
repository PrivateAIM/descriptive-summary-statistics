"""
Unit tests for the standalone PCA visualization module.
Covers: run_pca, every save_* renderer, and the save_pca_outputs orchestrator.
"""

import numpy as np
import pandas as pd
import pytest

from data_report.generate_figures.pca_plots import (
    PCAResult,
    run_pca,
    save_explained_variance_plot,
    save_pca_scatter,
    save_pca_scatter_matrix,
    save_pca_loadings_biplot,
    save_pca_3d_html,
    save_pca_overview,
    save_pca_outputs,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_numeric_df(n=120, n_features=6, seed=0, with_target=False, with_missing=False):
    rng = np.random.default_rng(seed)

    # Build features with decreasing variance so PCA has clear, ordered structure.
    base = rng.normal(size=n)
    data = {}
    for i in range(n_features):
        scale = n_features - i
        data[f"num_{i}"] = base * scale + rng.normal(scale=0.5, size=n)

    df = pd.DataFrame(data)

    if with_missing:
        df.loc[0, "num_0"] = np.nan
        df.loc[1, "num_1"] = np.nan

    if with_target:
        df["pasc"] = rng.choice(["yes", "no"], size=n)

    return df


def numeric_feature_names(df, prefix="num_"):
    return [c for c in df.columns if c.startswith(prefix)]


# ---------------------------------------------------------------------------
# run_pca
# ---------------------------------------------------------------------------

class TestRunPca:

    def test_returns_pca_result_with_expected_shapes(self):
        df = make_numeric_df(n=100, n_features=5)
        features = numeric_feature_names(df)

        result = run_pca(df, features)

        assert isinstance(result, PCAResult)
        assert result.components.shape == (100, 5)
        assert result.loadings.shape == (5, 5)
        assert result.feature_names == features

    def test_explained_variance_ratio_sums_to_one(self):
        df = make_numeric_df(n=100, n_features=4)
        result = run_pca(df, numeric_feature_names(df))

        assert result.explained_variance_ratio.sum() == pytest.approx(1.0)

    def test_components_are_ordered_by_decreasing_variance(self):
        df = make_numeric_df(n=200, n_features=5)
        result = run_pca(df, numeric_feature_names(df))

        ratios = result.explained_variance_ratio
        assert all(ratios[i] >= ratios[i + 1] for i in range(len(ratios) - 1))

    def test_components_are_uncorrelated(self):
        # PCA components are orthogonal in the projected space, so their
        # pairwise (Pearson) correlations should be ~0.
        df = make_numeric_df(n=300, n_features=4)
        result = run_pca(df, numeric_feature_names(df))

        corr = np.corrcoef(result.components, rowvar=False)
        off_diagonal = corr[~np.eye(corr.shape[0], dtype=bool)]
        assert np.allclose(off_diagonal, 0.0, atol=1e-8)

    def test_recommended_n_components_within_bounds(self):
        df = make_numeric_df(n=100, n_features=6)
        result = run_pca(df, numeric_feature_names(df))

        assert 1 <= result.recommended_n_components <= 6

    def test_recommended_n_components_reaches_threshold(self):
        df = make_numeric_df(n=200, n_features=6)
        result = run_pca(df, numeric_feature_names(df), variance_threshold=0.9)

        cumulative = np.cumsum(result.explained_variance_ratio)
        k = result.recommended_n_components
        assert cumulative[k - 1] >= 0.9
        # and it should be the *smallest* such k
        if k > 1:
            assert cumulative[k - 2] < 0.9

    def test_handles_missing_values_via_imputation(self):
        df = make_numeric_df(n=100, n_features=4, with_missing=True)
        result = run_pca(df, numeric_feature_names(df))

        assert not np.isnan(result.components).any()
        assert result.components.shape == (100, 4)

    def test_raises_on_non_numeric_feature(self):
        df = make_numeric_df(n=50, n_features=3, with_target=True)

        with pytest.raises(ValueError, match="non-numeric"):
            run_pca(df, numeric_feature_names(df) + ["pasc"])

    def test_raises_when_feature_list_is_empty(self):
        df = make_numeric_df(n=50, n_features=3, with_target=True)

        with pytest.raises(ValueError, match="at least one numeric feature"):
            run_pca(df, [])

    def test_drops_all_nan_columns_and_continues(self):
        # Entirely-empty columns are common in federated health data (a site
        # may not record a given field at all); they should be excluded
        # rather than failing the whole analysis.
        df = make_numeric_df(n=50, n_features=2)
        df["empty"] = np.nan

        result = run_pca(df, numeric_feature_names(df) + ["empty"])

        assert "empty" not in result.feature_names
        assert result.components.shape == (50, 2)

    def test_raises_when_all_requested_numeric_columns_are_empty(self):
        df = make_numeric_df(n=50, n_features=2)
        df["empty_a"] = np.nan
        df["empty_b"] = np.nan

        with pytest.raises(ValueError, match="entirely empty"):
            run_pca(df, ["empty_a", "empty_b"])

    def test_raises_on_fewer_than_two_features(self):
        df = make_numeric_df(n=50, n_features=1)

        with pytest.raises(ValueError, match="at least 2 samples and 2 numeric features"):
            run_pca(df, numeric_feature_names(df))


# ---------------------------------------------------------------------------
# save_* renderers (smoke tests: do they produce non-empty files?)
# ---------------------------------------------------------------------------

class TestSaveRenderers:

    def _result(self, with_target=False, n_features=6):
        df = make_numeric_df(n=120, n_features=n_features, with_target=with_target)
        result = run_pca(df, numeric_feature_names(df))
        return df, result

    def _assert_written(self, path):
        assert path.exists()
        assert path.stat().st_size > 0

    def test_save_explained_variance_plot(self, tmp_path):
        _, result = self._result()
        out = tmp_path / "variance.png"
        save_explained_variance_plot(result, out)
        self._assert_written(out)

    def test_save_pca_scatter_without_target(self, tmp_path):
        df, result = self._result()
        out = tmp_path / "scatter.png"
        save_pca_scatter(result, df, out)
        self._assert_written(out)

    def test_save_pca_scatter_with_target(self, tmp_path):
        df, result = self._result(with_target=True)
        out = tmp_path / "scatter_target.png"
        save_pca_scatter(result, df, out, target="pasc")
        self._assert_written(out)

    def test_save_pca_scatter_matrix(self, tmp_path):
        df, result = self._result(with_target=True)
        out = tmp_path / "matrix.png"
        save_pca_scatter_matrix(result, df, out, target="pasc")
        self._assert_written(out)

    def test_save_pca_loadings_biplot(self, tmp_path):
        _, result = self._result()
        out = tmp_path / "loadings.png"
        save_pca_loadings_biplot(result, out)
        self._assert_written(out)

    def test_save_pca_loadings_biplot_respects_top_n(self, tmp_path):
        _, result = self._result(n_features=8)
        out = tmp_path / "loadings_top.png"
        save_pca_loadings_biplot(result, out, top_n=3)
        self._assert_written(out)

    def test_save_pca_3d_html(self, tmp_path):
        df, result = self._result(with_target=True)
        out = tmp_path / "scatter3d.html"
        save_pca_3d_html(result, df, out, target="pasc")
        self._assert_written(out)
        assert out.read_text(encoding="utf-8").lstrip().startswith("<")

    def test_save_pca_overview(self, tmp_path):
        df, result = self._result(with_target=True)
        out = tmp_path / "overview.png"
        save_pca_overview(result, df, out, target="pasc")
        self._assert_written(out)


# ---------------------------------------------------------------------------
# save_pca_outputs (orchestrator)
# ---------------------------------------------------------------------------

class TestSavePcaOutputs:

    def test_creates_output_directory_and_full_set_of_files(self, tmp_path):
        df = make_numeric_df(n=120, n_features=6, with_target=True)
        out_dir = tmp_path / "pca"

        result = save_pca_outputs(df, numeric_feature_names(df), out_dir, target="pasc")

        assert isinstance(result, PCAResult)
        produced = {p.name for p in out_dir.iterdir()}
        assert produced == {
            "pca_explained_variance.png",
            "pca_scatter_2d.png",
            "pca_loadings_pc1_pc2.png",
            "pca_scatter_matrix.png",
            "pca_loadings_pc1_pc3.png",
            "pca_scatter_3d.html",
            "pca_overview.png",
        }
        for p in out_dir.iterdir():
            assert p.stat().st_size > 0

    def test_skips_three_dimensional_outputs_with_only_two_features(self, tmp_path):
        df = make_numeric_df(n=80, n_features=2)
        out_dir = tmp_path / "pca_2d"

        save_pca_outputs(df, numeric_feature_names(df), out_dir)

        produced = {p.name for p in out_dir.iterdir()}
        assert "pca_scatter_3d.html" not in produced
        assert "pca_scatter_matrix.png" not in produced
        assert "pca_loadings_pc1_pc3.png" not in produced
        assert "pca_scatter_2d.png" in produced

    def test_ignores_target_column_not_present_in_dataframe(self, tmp_path):
        df = make_numeric_df(n=80, n_features=3)
        out_dir = tmp_path / "pca_no_target"

        # Should not raise even though "missing_label" is absent from df.
        save_pca_outputs(df, numeric_feature_names(df), out_dir, target="missing_label")

        assert (out_dir / "pca_scatter_2d.png").exists()

    def test_writes_excluded_columns_csv_when_a_column_is_entirely_missing(self, tmp_path):
        df = make_numeric_df(n=80, n_features=3)
        df["empty_col"] = np.nan
        out_dir = tmp_path / "pca_excluded"

        save_pca_outputs(df, numeric_feature_names(df) + ["empty_col"], out_dir)

        excluded = pd.read_csv(out_dir / "excluded_columns.csv")
        assert list(excluded["feature"]) == ["empty_col"]

    def test_no_excluded_columns_csv_when_nothing_dropped(self, tmp_path):
        df = make_numeric_df(n=80, n_features=3)
        out_dir = tmp_path / "pca_no_excluded"

        save_pca_outputs(df, numeric_feature_names(df), out_dir)

        assert not (out_dir / "excluded_columns.csv").exists()
