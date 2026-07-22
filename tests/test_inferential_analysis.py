"""
Unit tests for the inferential analysis module.

Covers the pieces that were added or changed while wiring an automatic
inferential-statistics section into the report pipeline:
  - detect_outcome_column (generic, keyword-driven outcome detection)
  - screen_associations (full-pairwise association screening, including the
    Benjamini-Hochberg FDR correction and its NaN-safety fix)
  - posthoc_test (welch / kruskal paths)
  - make_hospital_dataset_5 (dataset structure)
  - report section 5.2 / 5.3 fallback messages
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from data_report.statistical_analysis.local.inferential_analysis import (
    posthoc_test,
    detect_outcome_column,
    screen_associations,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_outcome_df(n=100, seed=0):
    """A small dataset with a usable binary outcome column named 'death'."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "patient_id": [f"P{i:04d}" for i in range(n)],
        "age": rng.normal(60, 12, size=n),
        "death": rng.choice(["yes", "no"], size=n, p=[0.4, 0.6]),
    })


def make_four_group_df(n_per_group=60, seed=42):
    """DataFrame with 4 well-separated groups for posthoc tests."""
    rng = np.random.default_rng(seed)
    groups = ["Mild", "Standard", "Complex", "Deceased"]
    means = [0.0, 1.0, 2.5, 4.5]
    records = []
    for g, m in zip(groups, means):
        records.extend([{"value": rng.normal(m, 0.5), "group": g}
                        for _ in range(n_per_group)])
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# posthoc_test
# ---------------------------------------------------------------------------

class TestPosthocTest:

    def test_welch_path_returns_long_form_with_pval(self):
        df = make_four_group_df()
        result = posthoc_test(df, "value", "group", method="welch")
        assert isinstance(result, pd.DataFrame)
        assert "p-val" in result.columns or "pval" in result.columns or \
               any("p" in c.lower() for c in result.columns)

    def test_kruskal_path_returns_square_matrix(self):
        df = make_four_group_df()
        result = posthoc_test(df, "value", "group", method="kruskal")
        assert isinstance(result, pd.DataFrame)
        groups = df["group"].unique()
        assert result.shape == (len(groups), len(groups))
        assert set(result.index) == set(groups)
        assert set(result.columns) == set(groups)

    def test_kruskal_matrix_is_symmetric(self):
        df = make_four_group_df()
        result = posthoc_test(df, "value", "group", method="kruskal")
        for a in result.index:
            for b in result.columns:
                va, vb = result.loc[a, b], result.loc[b, a]
                if pd.notna(va) and pd.notna(vb):
                    assert va == pytest.approx(vb, abs=1e-10)

    def test_kruskal_diagonal_is_nan(self):
        df = make_four_group_df()
        result = posthoc_test(df, "value", "group", method="kruskal")
        for g in result.index:
            assert pd.isna(result.loc[g, g])

    def test_kruskal_well_separated_groups_have_low_pvalues(self):
        df = make_four_group_df(n_per_group=80)
        result = posthoc_test(df, "value", "group", method="kruskal")
        # Mild vs Deceased are far apart; expect p < 0.01
        assert result.loc["Mild", "Deceased"] < 0.01

    def test_unsupported_method_raises(self):
        df = make_four_group_df()
        with pytest.raises(ValueError, match="Unsupported"):
            posthoc_test(df, "value", "group", method="tukey")


# ---------------------------------------------------------------------------
# make_hospital_dataset_5 structure
# ---------------------------------------------------------------------------

class TestDataset5Structure:
    """Check the generated dataset5 CSVs have the right shape and groups."""

    @pytest.fixture(scope="class")
    def node1_df(self):
        import pathlib
        root = pathlib.Path(__file__).parent.parent
        path = root / "data" / "dataset5" / "node1" / "hospital_node1.csv"
        return pd.read_csv(path)

    def test_mortality_status_has_four_groups(self, node1_df):
        assert node1_df["mortality_status"].nunique() == 4

    def test_expected_group_names_present(self, node1_df):
        expected = {"Mild_Recovery", "Standard_Recovery", "Complex_Recovery", "Deceased"}
        assert set(node1_df["mortality_status"].unique()) == expected

    def test_each_group_has_at_least_20_rows(self, node1_df):
        counts = node1_df["mortality_status"].value_counts()
        assert (counts >= 20).all()

    def test_severity_driven_columns_exist(self, node1_df):
        for col in ["heart_rate", "temperature_c", "crp_level", "mortality_status"]:
            assert col in node1_df.columns

    def test_three_nodes_present(self):
        import pathlib
        root = pathlib.Path(__file__).parent.parent
        nodes = list((root / "data" / "dataset5").iterdir())
        node_dirs = [n for n in nodes if n.is_dir()]
        assert len(node_dirs) == 3


# ---------------------------------------------------------------------------
# Report section 5.2 / 5.3 fallback messages
# ---------------------------------------------------------------------------

class TestReportFallbackMessages:
    """Check that the report builder emits fallback Note blocks when needed."""

    def _call_section(self, has_oneway_plots=False, has_posthoc_plots=False, mode="full"):
        import pathlib, tempfile
        from generate_reports.generate_local_report import _build_cross_variable_section

        with tempfile.TemporaryDirectory() as tmp:
            node_dir = pathlib.Path(tmp)
            comparisons_dir = node_dir / "inferential" / "comparisons"
            comparisons_dir.mkdir(parents=True)

            if has_oneway_plots:
                (comparisons_dir / "age_oneway.png").touch()
            if has_posthoc_plots:
                (comparisons_dir / "posthoc_age_vs_status.png").touch()

            doc = SimpleNamespace(width=400)
            elements = []
            _build_cross_variable_section(
                elements, doc, node_dir, mode, significant_df=pd.DataFrame()
            )
            return elements

    @staticmethod
    def _collect_text(elements):
        texts = []
        for el in elements:
            if hasattr(el, "text"):
                texts.append(el.text)
            if hasattr(el, "_content"):
                texts.append(str(el._content))
            # NarrativeMessage rendered as KeepTogether/Table — walk children
            if hasattr(el, "_flowables"):
                for child in el._flowables:
                    if hasattr(child, "text"):
                        texts.append(child.text)
        return " ".join(str(t) for t in texts)

    def test_no_multigroup_outcome_shows_fallback_note(self):
        """No *_oneway.png files → section 5.2 must emit the 'no 3+ groups' message."""
        elements = self._call_section(has_oneway_plots=False, has_posthoc_plots=False)
        combined = self._collect_text(elements)
        assert "3 or more groups" in combined or "No outcome" in combined

    def test_no_posthoc_plots_shows_fallback_note(self):
        """Oneway plots exist but no posthoc plots → section 5.3 must emit fallback."""
        elements = self._call_section(has_oneway_plots=True, has_posthoc_plots=False)
        combined = self._collect_text(elements)
        assert "No significant multi-group" in combined or "omnibus p" in combined


# ---------------------------------------------------------------------------
# detect_outcome_column
# ---------------------------------------------------------------------------

class TestDetectOutcomeColumn:

    KEYWORD_GROUPS = [
        ["death", "mortality"],
        ["icu"],
        ["outcome", "status", "label"],
    ]

    def test_returns_first_usable_keyword_match(self):
        df = make_outcome_df(n=100)
        column_types = {"categorical": ["death"], "numeric": ["age"]}

        result = detect_outcome_column(df, column_types, self.KEYWORD_GROUPS)

        assert result == "death"

    def test_skips_constant_column_and_falls_through(self):
        df = make_outcome_df(n=100)
        df["death"] = "no"  # constant -> unusable
        df["status"] = np.random.default_rng(1).choice(["alive", "deceased"], size=100)
        column_types = {"categorical": ["death", "status"], "numeric": ["age"]}

        result = detect_outcome_column(df, column_types, self.KEYWORD_GROUPS)

        assert result == "status"

    def test_skips_column_with_too_few_observations_per_class(self):
        df = make_outcome_df(n=100)
        # 95/5 split -> minority class below the default min_class_size=20
        df["death"] = ["yes"] * 5 + ["no"] * 95
        column_types = {"categorical": ["death"], "numeric": ["age"]}

        result = detect_outcome_column(df, column_types, self.KEYWORD_GROUPS, min_class_size=20)

        assert result is None

    def test_skips_column_with_too_many_levels(self):
        df = make_outcome_df(n=100)
        df["status"] = np.random.default_rng(2).choice(
            ["a", "b", "c", "d", "e", "f"], size=100
        )
        column_types = {"categorical": ["status"], "numeric": ["age"]}

        result = detect_outcome_column(df, column_types, self.KEYWORD_GROUPS, max_levels=5)

        assert result is None

    def test_returns_none_when_no_keyword_matches(self):
        df = make_outcome_df(n=100)
        df = df.rename(columns={"death": "unrelated_field"})
        column_types = {"categorical": ["unrelated_field"], "numeric": ["age"]}

        result = detect_outcome_column(df, column_types, self.KEYWORD_GROUPS)

        assert result is None

    def test_priority_order_is_respected(self):
        # "death" group ranks above "outcome"/"status"/"label" -- even though
        # both are usable, the higher-priority group should win.
        df = make_outcome_df(n=100)
        df["status"] = np.random.default_rng(3).choice(["a", "b"], size=100)
        column_types = {"categorical": ["status", "death"], "numeric": ["age"]}

        result = detect_outcome_column(df, column_types, self.KEYWORD_GROUPS)

        assert result == "death"


# ---------------------------------------------------------------------------
# screen_associations
# ---------------------------------------------------------------------------

def make_screening_df(n=200, seed=0):
    """
    Synthetic frame with KNOWN structure:
      - num_a / num_b are correlated (r ~ 0.7)   -> should be flagged
      - num_noise is independent of num_a         -> should NOT be flagged
      - grp_strong splits num_a with a large effect (Hedges' g large) -> flagged
      - grp_weak is unrelated to num_a             -> NOT flagged
      - cat_x / cat_y are associated                -> flagged
      - cat_p / cat_q are independent               -> NOT flagged
    """
    rng = np.random.default_rng(seed)
    num_a = rng.normal(0, 1, size=n)
    num_b = num_a * 0.7 + rng.normal(0, 0.7, size=n)
    num_noise = rng.normal(0, 1, size=n)

    grp_strong = np.where(num_a + rng.normal(0, 0.3, size=n) > 0, "high", "low")
    grp_weak = rng.choice(["x", "y"], size=n)

    cat_x = rng.choice(["a", "b"], size=n)
    # cat_y mostly mirrors cat_x (associated)
    flip = rng.random(n) < 0.15
    cat_y = np.where(flip, np.where(cat_x == "a", "b", "a"), cat_x)
    cat_p = rng.choice(["p1", "p2"], size=n)
    cat_q = rng.choice(["q1", "q2"], size=n)

    return pd.DataFrame({
        "num_a": num_a,
        "num_b": num_b,
        "num_noise": num_noise,
        "grp_strong": grp_strong,
        "grp_weak": grp_weak,
        "cat_x": cat_x,
        "cat_y": cat_y,
        "cat_p": cat_p,
        "cat_q": cat_q,
    })


def base_column_types():
    return {
        "numeric": ["num_a", "num_b", "num_noise"],
        "categorical": ["grp_strong", "grp_weak", "cat_x", "cat_y", "cat_p", "cat_q"],
        "temporal": [],
    }


class TestScreenAssociations:

    def test_returns_empty_frame_with_expected_columns_when_no_pairs(self):
        df = make_screening_df(n=50)
        # A single numeric column and no categorical columns -> no num-num
        # combinations, no cat-cat combinations, no num-cat pairs.
        column_types = {"numeric": ["num_a"], "categorical": [], "temporal": []}

        screening = screen_associations(df, column_types)

        expected_columns = ["var1", "var2", "pair_type", "test", "statistic",
                            "p_value", "effect_size", "effect_size_metric",
                            "p_adj", "significant"]
        assert list(screening.columns) == expected_columns
        assert screening.empty

    def test_tests_every_numeric_and_categorical_pair(self):
        df = make_screening_df(n=200)
        column_types = base_column_types()

        screening = screen_associations(df, column_types)

        # Every numeric pair is tested, including the noise pair -- there is
        # no cluster-based bounding of which pairs get screened.
        num_num = screening[screening["pair_type"] == "num-num"]
        assert set(zip(num_num["var1"], num_num["var2"])) == {
            ("num_a", "num_b"), ("num_a", "num_noise"), ("num_b", "num_noise"),
        }

        # Every low-cardinality categorical pair is tested, including cross
        # pairs between the independent groups.
        cat_cat = screening[screening["pair_type"] == "cat-cat"]
        pairs = set(zip(cat_cat["var1"], cat_cat["var2"]))
        assert ("cat_x", "cat_p") in pairs
        assert ("cat_x", "cat_q") in pairs

    def test_num_cat_only_covers_binary_categoricals(self):
        df = make_screening_df(n=200)
        df["three_levels"] = np.random.default_rng(4).choice(["a", "b", "c"], size=200)
        column_types = base_column_types()
        column_types["categorical"].append("three_levels")

        screening = screen_associations(df, column_types)

        num_cat = screening[screening["pair_type"] == "num-cat"]
        assert "three_levels" not in set(num_cat["var2"])
        assert df["grp_strong"].nunique() == 2
        assert set(num_cat["var2"]) <= {"grp_strong", "grp_weak", "cat_x", "cat_y", "cat_p", "cat_q"}

    def test_skips_num_cat_pair_with_singleton_group(self):
        # grp_singleton has exactly one "rare" row -- compare_two_groups'
        # ddof=1 variance is undefined (NaN) for a group of size 1, which
        # would otherwise surface a NaN effect size/statistic in the screen
        # instead of just omitting the untestable pair.
        df = make_screening_df(n=200)
        df["grp_singleton"] = "common"
        df.loc[df.index[0], "grp_singleton"] = "rare"
        column_types = base_column_types()
        column_types["categorical"].append("grp_singleton")

        screening = screen_associations(df, column_types)

        num_cat = screening[screening["pair_type"] == "num-cat"]
        assert "grp_singleton" not in set(num_cat["var2"])

    def test_p_adj_is_never_smaller_than_raw_p_value(self):
        df = make_screening_df(n=200)
        column_types = base_column_types()

        screening = screen_associations(df, column_types)

        valid = screening["p_value"].notna()
        assert (screening.loc[valid, "p_adj"] >= screening.loc[valid, "p_value"] - 1e-12).all()

    def test_significant_requires_both_fdr_and_effect_size(self):
        df = make_screening_df(n=200)
        column_types = base_column_types()

        screening = screen_associations(df, column_types)
        flagged = screening[screening["significant"]]

        thresholds = {
            "correlation": 0.2, "cramers_v": 0.1, "hedges_g": 0.2, "rank_biserial": 0.2,
        }
        for _, row in flagged.iterrows():
            assert row["p_adj"] < 0.05
            assert row["effect_size"] >= thresholds[row["effect_size_metric"]]

        # the deliberately-noise pairs should not be flagged
        noise_pairs = {("num_a", "num_noise"), ("num_a", "grp_weak"), ("cat_p", "cat_q")}
        flagged_pairs = set(zip(flagged["var1"], flagged["var2"])) | set(zip(flagged["var2"], flagged["var1"]))
        assert noise_pairs.isdisjoint(flagged_pairs)

    def test_handles_nan_p_values_without_corrupting_fdr_correction(self):
        """
        Regression test for the FDR/NaN bug found during end-to-end testing:
        multipletests propagates a single NaN p-value to every adjusted value,
        which used to make `significant` False for the entire screen. Build a
        screening table by hand (bypassing the test-running machinery) to
        force a NaN p-value alongside valid ones, and check the correction
        still produces sensible p_adj values for the valid rows.
        """
        df = make_screening_df(n=200)
        column_types = base_column_types()

        # Monkeypatch _auto_correlation to inject one NaN p-value into the mix.
        import data_report.statistical_analysis.local.inferential_analysis as ia
        original = ia._auto_correlation
        calls = {"n": 0}

        def flaky_correlation(df, var1, var2):
            calls["n"] += 1
            result = original(df, var1, var2)
            if calls["n"] == 1:
                result = dict(result, p_value=np.nan)
            return result

        ia._auto_correlation = flaky_correlation
        try:
            screening = screen_associations(df, column_types)
        finally:
            ia._auto_correlation = original

        assert screening["p_value"].isna().any()
        nan_rows = screening[screening["p_value"].isna()]
        valid_rows = screening[screening["p_value"].notna()]

        # NaN p-value rows must not be (mis)classified as significant, and
        # must not corrupt the adjustment of the valid rows.
        assert (nan_rows["p_adj"].isna()).all()
        assert not nan_rows["significant"].any()
        assert valid_rows["p_adj"].notna().all()
        assert (valid_rows["p_adj"] >= valid_rows["p_value"] - 1e-12).all()
