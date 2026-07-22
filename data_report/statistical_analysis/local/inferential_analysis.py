"""Statistical testing and association screening for tabular datasets.

Provides automatic test selection (parametric vs. non-parametric based on
distribution diagnostics), pairwise group comparisons, post-hoc procedures,
time-series and longitudinal analysis helpers, and a BH-FDR-corrected
association screening pipeline across column pairs.
"""
import itertools
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pingouin as pg
from scipy.stats import linregress, chi2_contingency, fisher_exact
from scipy.signal import find_peaks, detrend as _linear_detrend
from scipy.signal.windows import blackman
from scipy.fft import rfft, rfftfreq
from statsmodels.stats.multitest import multipletests
from data_report.statistical_analysis.local.compute_statistics import (detect_id_column, detect_column_types)


logger = logging.getLogger(__name__)

# compare_two_groups' variance/effect-size math (ddof=1) is undefined for a
# group of size 1 (division by zero), so screen_associations skips any
# num-cat pair where a group falls below this size rather than surfacing a
# NaN effect size.
_MIN_GROUP_SIZE_FOR_COMPARISON = 2

# helper functions
def _group_values(df, value_col, group_col):
    """Extract non-null value arrays and their labels for each level of group_col."""
    groups = []
    labels = []
    for name, g in df.groupby(group_col):
        vals = g[value_col].dropna().values
        if len(vals) > 0:
            groups.append(vals)
            labels.append(name)
    return groups, labels

def _distribution_diagnostics(groups):
    """Return per-group normality diagnostics (Shapiro-Wilk, skewness, kurtosis, outliers)."""
    diagnostics = []

    for g in groups:

        g = np.asarray(g)
        n = len(g)

        if n < 3:
            diagnostics.append({
                "n": n,
                "shapiro_p": np.nan,
                "skewness": np.nan,
                "kurtosis": np.nan,
                "approx_normal": False,
                "has_outliers": False
            })
            continue

        # Avoid Shapiro on very large samples
        shapiro_p = stats.shapiro(g)[1] if n <= 5000 else np.nan

        skewness = stats.skew(g, bias=False)
        kurt = stats.kurtosis(g, fisher=True, bias=False)

        # IQR outlier detection
        q1, q3 = np.percentile(g, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = np.sum((g < lower) | (g > upper))

        has_outliers = outliers > 0

        if n <= 50 and not np.isnan(shapiro_p):
            approx_normal = shapiro_p > 0.05
        else:
            # Fallback for n > 50 (Shapiro-Wilk over-rejects at large n): rule-of-thumb
            # bounds on excess kurtosis (Fisher, 0 = normal) and skewness, common in
            # applied normality screening (e.g. George & Mallery's |skew|, |kurt| < 2).
            approx_normal = (
                    abs(skewness) < 1 and
                    abs(kurt) < 2
            )

        diagnostics.append({
            "n": n,
            "shapiro_p": shapiro_p,
            "skewness": skewness,
            "kurtosis": kurt,
            "approx_normal": approx_normal,
            "has_outliers": has_outliers
        })

    overall_normal = all(d["approx_normal"] for d in diagnostics)

    return {
        "overall_normal": overall_normal,
        "groups": diagnostics
    }

def _cohens_d(g1, g2):
    """Compute Cohen's d (pooled-SD effect size) for two independent samples."""
    n1, n2 = len(g1), len(g2)

    s1 = np.var(g1, ddof=1)
    s2 = np.var(g2, ddof=1)

    pooled_sd = np.sqrt(
        ((n1 - 1) * s1 + (n2 - 1) * s2) /
        (n1 + n2 - 2)
    )

    if pooled_sd == 0:
        return 0.0 if np.mean(g1) == np.mean(g2) else np.inf

    return (np.mean(g1) - np.mean(g2)) / pooled_sd


def _hedges_g(g1, g2):
    """Compute Hedges' g (bias-corrected Cohen's d) for two independent samples."""
    d = _cohens_d(g1, g2)

    n = len(g1) + len(g2)

    denom = 4 * n - 9
    correction = 1 - (3 / denom) if denom > 0 else 1.0

    return d * correction

def _rank_biserial(u, n1, n2):
    """Compute the rank-biserial correlation effect size from a Mann-Whitney U statistic."""
    # u: Mann-Whitney U statistic for g1 relative to g2 (as returned by
    # stats.mannwhitneyu(g1, g2, ...)) -- passed in so the test isn't run twice
    return 1 - (2 * u) / (n1 * n2)

def _should_use_nonparametric(diagnostics):
    """Return True when data characteristics favour a non-parametric test over a parametric one."""
    min_n = min(d["n"] for d in diagnostics["groups"])

    severe_skew = any(
        abs(d["skewness"]) > 2
        for d in diagnostics["groups"]
        if not np.isnan(d["skewness"])
    )

    severe_kurtosis = any(
        abs(d["kurtosis"]) > 4
        for d in diagnostics["groups"]
        if not np.isnan(d["kurtosis"])
    )

    outliers = any(
        d["has_outliers"]
        for d in diagnostics["groups"]
    )

    overall_normal = diagnostics["overall_normal"]

    return (
        (severe_skew or severe_kurtosis or outliers) and
        not (overall_normal and min_n >= 30)
    )

def _check_variance(groups):
    """Test equality of variances across groups using Brown-Forsythe (median-centred Levene)."""
    #  Brown-Forsythe (median-centered Levene)
    stat, p = stats.levene(*groups, center="median")
    return {
        "equal_variance": p > 0.05,
        "p_value": p
    }

def _diagnose_groups(groups):
    """Run distribution and variance diagnostics for a list of groups."""
    # Shared by compare_two_groups and one_way_group_comparison so the
    # distribution/variance checks that drive "auto" method selection are
    # computed in exactly one place.
    diagnostics = _distribution_diagnostics(groups)
    variance = _check_variance(groups)
    return diagnostics, variance
#----------------------------------------------------------------
def compare_two_groups(df, value_col, group_col):
    """Compare two independent groups on a continuous variable using an auto-selected test.

    The test is chosen automatically from distribution diagnostics:

    * Mann-Whitney U when data are non-normal or have severe outliers or skew.
    * Student t-test when data are approximately normal with equal variances.
    * Welch t-test when data are approximately normal but variances differ.

    Args:
        df (pd.DataFrame): The dataset containing both columns.
        value_col (str): Name of the continuous outcome column.
        group_col (str): Name of the grouping column, which must have exactly
            two distinct non-null levels.

    Returns:
        dict: A result dictionary with keys ``method``, ``statistic``,
            ``p_value``, ``effect_size`` (Cohen's d / Hedges' g for t-tests,
            rank-biserial for Mann-Whitney), and ``assumptions`` (distribution
            diagnostics and variance-test output).

    Raises:
        ValueError: If ``group_col`` does not yield exactly two non-empty groups.
    """
    groups, labels = _group_values(df, value_col, group_col)
    if len(groups) != 2:
        raise ValueError("Exactly 2 groups required")

    g1, g2 = groups

    diagnostics, variance = _diagnose_groups(groups)

    if _should_use_nonparametric(diagnostics):
        method = "mannwhitney"
    elif variance["equal_variance"]:
        method = "student_ttest"
    else:
        method = "welch_ttest"

    if method == "student_ttest":
        stat, p = stats.ttest_ind(g1, g2, equal_var=True)
        effect_size = {
            "cohens_d": _cohens_d(g1, g2),
            "hedges_g": _hedges_g(g1, g2)
        }

    elif method == "welch_ttest":
        stat, p = stats.ttest_ind(g1, g2, equal_var=False)
        effect_size = {
            "cohens_d": _cohens_d(g1, g2),
            "hedges_g": _hedges_g(g1, g2)
        }

    elif method == "mannwhitney":
        stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        effect_size = {
            "rank_biserial": _rank_biserial(stat, len(g1), len(g2))
        }
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "method": method,
        "statistic": stat,
        "p_value": p,
        "effect_size": effect_size,
        "assumptions": {
            "distribution_diagnostics": diagnostics,
            "equal_variance": variance["equal_variance"],
            "variance_test_pvalue": variance["p_value"]  # Brown-Forsythe (median-centered Levene) test p-value for equal variances
        }
    }

#--------------------------------------------------------
# ANOVA/Welch:compare means
# Kruskal: compare distributions/ranks, often interpreted as medians
# one-way comparison of 2 or more independent groups
def one_way_group_comparison(df, value_col, group_col):
    """Compare two or more independent groups on a continuous variable using an auto-selected test.

    Automatically selects the test based on distribution diagnostics:

    * Kruskal-Wallis when data are non-normal or have severe outliers or skew.
    * Welch ANOVA otherwise (robust to unequal group variances).

    Args:
        df (pd.DataFrame): The dataset containing both columns.
        value_col (str): Name of the continuous outcome column.
        group_col (str): Name of the grouping column with two or more distinct
            non-null levels.

    Returns:
        dict: A result dictionary with keys ``method``, ``statistic``,
            ``p_value``, and ``assumptions`` (distribution and variance
            diagnostics).

    Raises:
        ValueError: If fewer than two non-empty groups are found.
    """
    groups, labels = _group_values(df, value_col, group_col)
    if len(groups) < 2:
        raise ValueError("Need at least 2 groups")

    diagnostics, variance = _diagnose_groups(groups)
    if _should_use_nonparametric(diagnostics):
        method = "kruskal"
    else:
        method = "welch"

    result = {
        "method": method,
        "statistic": None,
        "p_value": None,
        "assumptions": {
            "distribution_diagnostics": diagnostics,
            "equal_variance": variance["equal_variance"],
            "variance_test_pvalue": variance["p_value"]  # Brown-Forsythe (median-centered Levene) test p-value for equal variances
        }
    }

    if method == "welch":
        welch = pg.welch_anova( data=df, dv=value_col, between=group_col)
        stat = welch["F"].iloc[0]
        # pingouin names this column "p-unc" in some versions and "p_unc" in
        # others (e.g. 0.6.x) -- support both so the call doesn't KeyError.
        p_col = "p-unc" if "p-unc" in welch.columns else "p_unc"
        p = welch[p_col].iloc[0]

    elif method == "kruskal":
        stat, p = stats.kruskal(*groups)

    else:
        raise ValueError(f"Unknown method: {method}")

    result["statistic"] = stat
    result["p_value"] = p
    return result

#--------------------------
# Post-hoc tests
def posthoc_test(df, value_col, group_col, method):
    """Run the appropriate post-hoc pairwise test after a significant group comparison.

    The ``method`` argument identifies the primary test returned by
    ``one_way_group_comparison``, not the post-hoc procedure itself. Each
    primary test maps to a standard companion:

    * ``"welch"``   -> Games-Howell (via pingouin).
    * ``"kruskal"`` -> Pairwise Mann-Whitney U with Holm-Bonferroni correction.

    Args:
        df (pd.DataFrame): The dataset containing both columns.
        value_col (str): Name of the continuous outcome column.
        group_col (str): Name of the grouping column.
        method (str): The primary comparison method; one of ``"welch"`` or
            ``"kruskal"``.

    Returns:
        pd.DataFrame: For ``"welch"``, a pingouin Games-Howell result table. For
            ``"kruskal"``, a square DataFrame of Holm-adjusted p-values indexed
            by group label.

    Raises:
        ValueError: If ``method`` is not one of the supported values.
    """
    # NOTE: `method` names the *primary* group-comparison test that was run
    # (matches the "method" returned by one_way_group_comparison), not the
    # post-hoc test itself. Each primary test maps to its standard companion:
    #   "welch"   -> Games-Howell       (pg.pairwise_gameshowell)
    #   "kruskal" -> pairwise MWU + Holm correction
    if method == "welch":
        return pg.pairwise_gameshowell(
            data=df,
            dv=value_col,
            between=group_col
        )
    elif method == "kruskal":
        # Pairwise Mann-Whitney U with Holm-Bonferroni correction —
        # equivalent to Dunn's test without the scikit-posthocs dependency.
        # Returns a square p-value matrix indexed by group label.
        groups = sorted(df[group_col].dropna().unique())
        pairs = [(a, b) for i, a in enumerate(groups) for b in groups[i + 1:]]
        raw_p = []
        for a, b in pairs:
            g1 = df.loc[df[group_col] == a, value_col].dropna().values
            g2 = df.loc[df[group_col] == b, value_col].dropna().values
            _, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            raw_p.append(p)
        _, adj_p, _, _ = multipletests(raw_p, method="holm")
        mat = pd.DataFrame(np.nan, index=groups, columns=groups)
        for (a, b), p in zip(pairs, adj_p):
            mat.loc[a, b] = p
            mat.loc[b, a] = p
        return mat
    else:
        raise ValueError(f"Unsupported method: {method}")

def detect_event_columns(df, patient_col, date_cols=None):
    """Score and rank columns that are likely to represent recurring clinical events.

    A column is a candidate when it shows moderate overall diversity (unique
    ratio between 1 % and 50 %), multiple distinct values per patient on average,
    and many patients with varying values. Each property contributes to a score;
    columns that reach a minimum threshold are returned.

    Args:
        df (pd.DataFrame): The dataset to search.
        patient_col (str): Name of the patient identifier column.
        date_cols: Unused; reserved for future date-column filtering.

    Returns:
        list[dict]: Candidate columns sorted by descending score. Each entry
            contains ``column``, ``score``, ``avg_unique_per_patient``, and
            ``patient_diversity``.
    """
    candidates = []

    object_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in object_cols:
        if col == patient_col:
            continue

        total_unique = df[col].nunique(dropna=True)
        unique_ratio = total_unique / len(df)

        per_patient_unique = df.groupby(patient_col)[col].nunique()
        avg_unique_patient = per_patient_unique.mean()

        patient_diversity = (per_patient_unique > 1).mean()

        score = 0

        # moderate diversity
        if 0.01 < unique_ratio < 0.5:
            score += 1

        # multiple values within patient
        if avg_unique_patient > 1.2:
            score += 2

        # many patients with varying values
        if patient_diversity > 0.3:
            score += 2

        if score >= 3:
            candidates.append({
                "column": col,
                "score": score,
                "avg_unique_per_patient": avg_unique_patient,
                "patient_diversity": patient_diversity
            })

    return sorted(candidates, key=lambda x: x["score"], reverse=True)

def event_dataset_score(df, patient_col):
    """Score how event-like a dataset is based on per-patient row variability.

    Args:
        df (pd.DataFrame): The dataset to score.
        patient_col (str): Name of the patient identifier column.

    Returns:
        dict: A dictionary with ``row_variability`` (coefficient of variation of
            per-patient row counts) and ``repeat_ratio`` (fraction of patients
            with more than one row).
    """
    counts = df.groupby(patient_col).size()

    row_variability = counts.std() / counts.mean()
    repeated_rows = (counts > 1).mean()

    return {
        "row_variability": row_variability,
        "repeat_ratio": repeated_rows
    }

# trend analysis
# use FFT: Fast Fourier Transformation to detect seasonality
def detect_seasonality_fft(values, time=None, sampling_rate=1.0, detrend=True, window=True, min_periods=8,
                           top_k=3, remove_zero_frequency=True,):
    """Detect periodic patterns in a signal using Fast Fourier Transform analysis.

    Missing values are removed before processing. When timestamps are provided
    the sampling rate is estimated from the median inter-sample interval;
    signals with highly irregular sampling (coefficient of variation > 0.5) are
    rejected. A Blackman window reduces spectral leakage; linear detrending
    removes slope and offset before windowing.

    Args:
        values (array-like): The signal values to analyse.
        time (array-like, optional): Timestamps corresponding to ``values``.
            Accepts numeric arrays or numpy datetime64 arrays. When provided the
            sampling rate is estimated from the data. Defaults to None.
        sampling_rate (float): Fallback sampling rate used when ``time`` is not
            provided. Defaults to 1.0.
        detrend (bool): Remove linear trend before FFT to reduce spectral
            leakage. Defaults to True.
        window (bool): Apply a Blackman window before FFT. Defaults to True.
        min_periods (int): Minimum number of non-null observations required to
            run the analysis. Defaults to 8.
        top_k (int): Number of dominant frequencies to report. Defaults to 3.
        remove_zero_frequency (bool): Discard the DC (zero-frequency) component.
            Defaults to True.

    Returns:
        dict: A dictionary containing ``dominant_frequency``,
            ``dominant_period``, ``seasonality_strength`` (amplitude of the
            dominant frequency), ``frequencies``, ``power_spectrum``,
            ``top_frequencies``, ``top_periods``, and ``top_powers``.
            All numeric fields are NaN and array fields are empty when the
            minimum observation count is not met or sampling is too irregular.

    Note:
        ``seasonality_strength`` is the raw FFT amplitude, not a normalised
        index; compare values only within the same signal.
    """
    values = np.asarray(values, dtype=float)
    # remove missing values
    mask = ~np.isnan(values)
    values = values[mask]

    # time alignment + datetime handling
    if time is not None:
        time = np.asarray(time)[mask]
        # safely convert datetime if needed
        if np.issubdtype(time.dtype, np.datetime64):
            time = time.astype("datetime64[s]").astype(float)


    # minimum observations
    if len(values) < min_periods:
        return {
            "dominant_frequency": np.nan,
            "dominant_period": np.nan,
            "seasonality_strength": np.nan,
            "frequencies": np.array([]),
            "power_spectrum": np.array([]),
            "top_frequencies": [],
            "top_periods": [],
            "top_powers": [],
        }

    # estimate sampling rate from irregular timestamps
    if time is not None:
        time = np.asarray(time, dtype=float)
        dt = np.diff(time)
        dt = dt[dt > 0]
        if len(dt) > 0:
            median_dt = np.median(dt)

            irregularity = np.std(dt) / (median_dt + 1e-8)
            # reject unreliable time series
            if irregularity > 0.5:
                return {
                    "dominant_frequency": np.nan,
                    "dominant_period": np.nan,
                    "seasonality_strength": np.nan,
                    "frequencies": np.array([]),
                    "power_spectrum": np.array([]),
                    "top_frequencies": [],
                    "top_periods": [],
                    "top_powers": [],
                    "note": "irregular_sampling_rejected"
                }

            sampling_rate = 1.0 / median_dt

    # signal processing
    signal = values.copy()
    # baseline removal (linear detrend removes slope as well as offset,
    # avoiding the spectral leakage that median-subtraction would leave behind
    # for data with a linear trend)
    if detrend:
        signal = _linear_detrend(signal, type="linear")
    # windowing
    if window:
        signal = signal * blackman(len(signal))

    # raw fourier coefficients
    fft_vals = rfft(signal)
    # amplitude spectrum
    power = np.abs(fft_vals)
    freqs = rfftfreq(
        len(signal),
        d=1.0 / sampling_rate
    )

    # remove DC component
    if remove_zero_frequency and len(freqs) > 1:
        freqs = freqs[1:]
        power = power[1:]

    if len(power) == 0:
        return {
            "dominant_frequency": np.nan,
            "dominant_period": np.nan,
            "seasonality_strength": np.nan,
            "frequencies": np.array([]),
            "power_spectrum": np.array([]),
            "top_frequencies": [],
            "top_periods": [],
            "top_powers": [],
        }
    # dominant frequency
    dominant_idx = np.argmax(power)
    dominant_frequency = freqs[dominant_idx]
    dominant_power = power[dominant_idx]
    if dominant_frequency > 0:
        dominant_period = 1.0 / dominant_frequency
    else:
        dominant_period = np.nan
    # top k frequencies
    sorted_idx = np.argsort(power)[::-1]
    top_idx = sorted_idx[:top_k]
    top_frequencies = freqs[top_idx]
    top_powers = power[top_idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        top_periods = np.where(
            top_frequencies > 0,
            1.0 / top_frequencies,
            np.nan
        )
    return {
        "dominant_frequency": dominant_frequency,
        "dominant_period": dominant_period,
        "seasonality_strength": dominant_power,
        "frequencies": freqs,
        "power_spectrum": power,
        "top_frequencies": top_frequencies,
        "top_periods": top_periods,
        "top_powers": top_powers,
    }

# function for annotation of the peaks
# annotation of the highest peak + added: annotation for the k highest peaks for example k=5 -> top 5
def peak_annotation(x, y, k, ax=None, min_height=None, fft_labels=True):
    """Annotate the top-k peaks on a matplotlib axes object.

    Args:
        x (array-like): X-axis values (e.g. frequencies or time points).
        y (array-like): Y-axis values (e.g. power or amplitude).
        k (int): Number of peaks to annotate. Pass ``1`` to annotate only the
            global maximum.
        ax (matplotlib.axes.Axes, optional): Target axes. Defaults to the
            current axes (``plt.gca()``).
        min_height (float, optional): Minimum height threshold for peak
            detection. Defaults to None.
        fft_labels (bool): When True, annotations show ``freq`` and ``power``;
            when False, annotations show generic ``x`` and ``y`` labels.
            Defaults to True.
    """
    if ax is None:
        ax = plt.gca()
    x = np.asarray(x)
    y = np.asarray(y)
    # find peaks
    peaks, properties = find_peaks(y, height=min_height)
    # if no peaks detected
    if len(peaks) == 0:
        peaks = [np.argmax(y)]
    # only the highest peak
    if k == 1:
        peaks = [np.argmax(y)]
    # top k peaks
    else:
        peaks, properties = find_peaks(y, height=min_height)
        peak_heights = y[peaks]
        # find the order of the indices if the values were sorted
        # smallest value is first
        sorted_idx = np.argsort(peak_heights)
        # reverse the order
        # now biggest value first
        sorted_idx = sorted_idx[::-1]
        # get the top k peaks
        # avoid getting out of range
        peaks = peaks[sorted_idx[:min(k, len(sorted_idx))]]

    for peak in peaks:
        xmax = x[peak]
        ymax = y[peak]
        if fft_labels:
            text = ("freq={:.3f}\npower={:.3f}".format(xmax, ymax))
        # generic labels
        else:
            text = ("x={:.3f}\ny={:.3f}".format(xmax, ymax))

        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        arrowprops = dict(arrowstyle="->")
        # single peak placement
        if k == 1:
            ax.annotate(text, xy=(xmax, ymax),  xytext=(0.94, 0.96), textcoords="axes fraction", bbox=bbox_props,
                    arrowprops=arrowprops, ha="right", va="top")
        # multiple peak placement
        else:
            ax.annotate(text, xy=(xmax, ymax), xytext=(20, 20), textcoords='offset points', bbox=bbox_props,
                        arrowprops=arrowprops)

# time series
# one variable changing over time
# example: monthly admissions
def time_series(df, date_col, value_col):
    """Compute trend, seasonality, and summary statistics for a univariate time series.

    The DataFrame is sorted by ``date_col`` before processing. A linear trend
    is fitted via ordinary least-squares regression on a zero-based time index;
    seasonality is detected via FFT (see ``detect_seasonality_fft``).

    Args:
        df (pd.DataFrame): The dataset containing the date and value columns.
        date_col (str): Name of the column containing timestamps.
        value_col (str): Name of the numeric value column.

    Returns:
        dict: A result dictionary containing ``min``, ``max``, ``range``,
            ``mean``, ``rolling_mean`` (3-period), ``growth_rate``
            (period-over-period percentage change), ``trend_slope``, and
            ``trend_r2``.
    """
    # sort rows by date
    # important for correct trend analysis!
    # time series must be ordered correctly!
    df = df.sort_values(date_col)
    # take only the numeric series
    # kept as a pandas Series so rolling()/pct_change() remain available;
    # convert to numpy only where explicitly required (e.g. detect_seasonality_fft)
    series = df[value_col].astype(float)
    # create a simple time index: 0, 1, 2, 3...
    # used for regression and trend detection
    time_index = np.arange(len(series))
    # result dictionary
    result = {}

    result["min"] = np.nanmin(series)
    result["max"] = np.nanmax(series)
    # amplitude of the signal
    result["range"] = result["max"] - result["min"]
    # overall average
    result["mean"] = series.mean()
    # rolling average: smooths fluctuations
    # 3 = previous 3 observations
    result["rolling_mean"] = (series.rolling(window=3).mean().tolist())
    # shows how it developed
    result["growth_rate"] = (series.pct_change().tolist())
    # seasonality
    # detect repeating patterns (weekly, yearly, etc.)
    # use frequency decomposition
    seasonality = detect_seasonality_fft(values=series.values,time=df[date_col].values)
    # seasonality = detect_seasonality_fft(values=series.values, sampling_rate=12)  # monthly data example
    # trend detection
    # we fit a straight line: y = a*x + b -> affine function
    # slope = direction of trend (up/down)
    # r² = how strong the trend is (0 = weak, 1 = strong)
    valid_mask = ~np.isnan(series)
    if np.sum(valid_mask) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(time_index[valid_mask], series[valid_mask])
        # direction + speed of change
        result["trend_slope"] = slope
        # strength
        result["trend_r2"] = r_value ** 2
    else:
        result["trend_slope"] = np.nan
        result["trend_r2"] = np.nan

    return result



# longitudinal: many observations per patient (several rows)
# same subjects followed over time
# example: same patient across multiple visits
def analyze_longitudinal(df, subject_col, time_col, value_col):
    """Summarise the longitudinal trajectory of each subject over repeated measurements.

    For each subject the function computes: baseline and last recorded value,
    absolute and relative change, individual trend slope, and within-subject
    variance. Subjects with fewer than two measurements are skipped.

    Args:
        df (pd.DataFrame): Long-format dataset with one row per observation.
        subject_col (str): Name of the subject identifier column.
        time_col (str): Name of the column containing timestamps or numeric
            time values.
        value_col (str): Name of the numeric outcome column.

    Returns:
        dict: Mapping of subject identifier to a summary dictionary containing
            ``n_observations``, ``baseline``, ``last_value``,
            ``absolute_change``, ``relative_change`` (percentage), ``slope``,
            and ``within_var``.

    Note:
        A slope derived from exactly two observations is a degenerate perfect
        line (R squared = 1) and should be interpreted with caution.
    """
    # store results per subject
    results = {}
    # loop through each patient
    for subject, group in df.groupby(subject_col):
        # sort records of one subject by time
        group = group.sort_values(time_col)
        # extract values
        values = group[value_col].values
        # skip if only one measurement
        if len(values) < 2:
            continue

        # individual slope (trend over time)
        # use the actual time values (not visit order) so the slope reflects a
        # real rate of change even when visits are unevenly spaced
        time_values = group[time_col]
        if pd.api.types.is_numeric_dtype(time_values):
            x = time_values.values.astype(float)
        else:
            if not pd.api.types.is_datetime64_any_dtype(time_values):
                time_values = pd.to_datetime(time_values, errors="coerce")
            # ensure a Series (with a usable .iloc[0]) even if a plain
            # array/list is ever passed in for time_values
            time_values = pd.Series(time_values).reset_index(drop=True)
            x = (time_values - time_values.iloc[0]).dt.total_seconds().values / 86400.0
        # calculate subject trend
        slope, _, _, _, _ = linregress(x, values)

        # absolute change
        absolute_change = values[-1] - values[0]
        # relative change
        relative_change = np.nan
        # Make sure we are not dividing by zero
        if values[0] != 0:
            # divide by the first value to get a ratio
            # multiply by 100 to convert to percentage
            relative_change = (absolute_change / values[0]) * 100
        # within-subject variance (stability)
        # measures how stable or variable a single subject is over time
        # describes individual behavior changes rather than just averages
        within_var = np.var(values, ddof=1)
        # values = measurements from one subject over time
        # ddof = 1 = uses sample variance (for a subset not a whole population so more correct for real-world data)


        # save subject summary
        results[subject] = {
            # number of measurements behind this summary -- a slope/within_var
            # fitted from exactly 2 points is a degenerate (perfect, R^2=1)
            # line, not evidence of a real trend; consumers should treat
            # n_observations == 2 with caution
            "n_observations": len(values),
            # first recorded value
            "baseline": values[0],
            # last recorded value
            "last_value": values[-1],
            # total change
            "absolute_change": absolute_change,
            # relative change
            "relative_change": relative_change,
            # direction of trend
            "slope": slope,
            # stability (within-subject variance)
            "within_var": within_var,
        }
    return results

# event-based
# discrete events like surgeries ...
def analyze_event_based(df, time_col):
    """Compute event frequency and timing statistics for a discrete event log.

    Args:
        df (pd.DataFrame): Dataset with one row per event; must contain a
            datetime column named ``time_col``.
        time_col (str): Name of the datetime column.

    Returns:
        dict: A result dictionary containing ``event_count``, ``daily_counts``,
            ``event_density_mean``, ``event_density_std``, ``peak_day``,
            ``peak_day_count``, ``unusual_high_activity_days``,
            ``avg_time_between_events`` (seconds), and
            ``time_between_events_spread`` (seconds).
    """
    # sort by time
    df = df.sort_values(time_col)
    result = {}
    # total number of events
    result["event_count"] = len(df)
    # event density per day
    # group events by calendar day
    daily_counts = df.groupby(df[time_col].dt.date).size()
    # save event counts per day
    result["daily_counts"] = daily_counts.to_dict()
    result["event_density_mean"] = daily_counts.mean()
    result["event_density_std"] = daily_counts.std()
    # peak periods
    # day with highest number of events
    result["peak_day"] = daily_counts.idxmax()
    result["peak_day_count"] = int(daily_counts.max())
    # unusual high activity

    # burst = unusually high daily activity
    if len(daily_counts) > 1:
        threshold = daily_counts.mean() + 2 * daily_counts.std()
        unusual_days = daily_counts[daily_counts > threshold]
        result["unusual_high_activity_days"] = len(unusual_days)
    else:
        result["unusual_high_activity_days"] = 0
    # time gaps
    if len(df) > 1:
        # difference between consecutive event timestamps
        time_gaps = df[time_col].diff().dropna()
        # convert time differences into seconds (numeric form)
        time_gaps_seconds = time_gaps.dt.total_seconds()
        # average waiting time between events
        result["avg_time_between_events"] = time_gaps_seconds.mean()
        # how consistent or irregular the timing is
        result["time_between_events_spread"] = time_gaps_seconds.std()
    else:
        result["avg_time_between_events"] = np.nan
        result["time_between_events_spread"] = np.nan
    return result


# panel data
# multiple entities over time
def analyze_panel(df, entity_col, time_col, value_col):
    """Summarise panel data by computing per-entity trend and growth statistics.

    Entities with fewer than two observations are skipped. The trend slope is
    fitted on actual time values rather than row order so the slope reflects a
    real rate of change even when observations are unevenly spaced.

    Args:
        df (pd.DataFrame): Long-format panel dataset with one row per
            entity-time observation.
        entity_col (str): Name of the entity identifier column.
        time_col (str): Name of the column containing timestamps or numeric
            time values.
        value_col (str): Name of the numeric outcome column.

    Returns:
        dict: Mapping of entity identifier to a summary dictionary containing
            ``n_observations``, ``mean``, ``growth`` (last minus first value),
            and ``slope``.
    """
    # store results per entity
    results = {}
    # loop through entities (hospitals)
    for entity, group in df.groupby(entity_col):
        # sort each entity over time
        group = group.sort_values(time_col)
        # extract numeric values
        values = group[value_col].values
        # skip short series
        if len(values) < 2:
            continue
        # use actual time values (not row order) so the slope reflects a real
        # rate of change even when observations are unevenly spaced
        time_values = group[time_col]
        if pd.api.types.is_numeric_dtype(time_values):
            x = time_values.values.astype(float)
        else:
            if not pd.api.types.is_datetime64_any_dtype(time_values):
                time_values = pd.to_datetime(time_values, errors="coerce")
            # ensure a Series (with a usable .iloc[0]) even if a plain
            # array/list is ever passed in for time_values
            time_values = pd.Series(time_values).reset_index(drop=True)
            x = (time_values - time_values.iloc[0]).dt.total_seconds().values / 86400.0
        # compute trend line slope
        slope, _, _, _, _ = linregress(x, values)
        # save results
        results[entity] = {
            # number of observations behind this summary -- a slope fitted
            # from exactly 2 points is a degenerate (perfect, R^2=1) line
            "n_observations": len(values),
            # average level
            "mean": values.mean(),
            # first vs last change
            "growth": values[-1] - values[0],
            # direction
            "slope": slope
        }
    return results


# ---------------------------------------------------------------------------
# Outcome-column detection
#
# The report runs unattended on hub nodes -- nobody can point it at "the"
# outcome variable. Hospital datasets vary in column naming, so candidates are
# matched by keyword in priority order (mortality/survival ranks above generic
# status/diagnosis labels) and only accepted if they are actually usable as a
# group/target variable: not constant, low cardinality, and every level has
# enough observations to support a stable comparison. Mirrors the keyword +
# validation pattern already used for sex-column detection in analyze.py.
# ---------------------------------------------------------------------------
def detect_outcome_column(df, column_types, keyword_groups, min_class_size=20, max_levels=5):
    """Find the most suitable outcome column in a dataset using keyword matching.

    Candidate columns are drawn from the categorical column type list. Keywords
    are evaluated in priority order (earlier groups take precedence); within a
    group the first matching column that passes all validity checks is returned.

    A column passes validity when it has between 2 and ``max_levels`` distinct
    non-null values and every class has at least ``min_class_size`` observations.

    Args:
        df (pd.DataFrame): The dataset to search.
        column_types (dict): Column-type dictionary as returned by
            ``detect_column_types``, used to restrict the search to categorical
            columns.
        keyword_groups (list[list[str]]): Ordered list of keyword lists. Each
            inner list is a priority tier; keywords within a tier are ORed.
        min_class_size (int): Minimum number of observations required per class
            level. Defaults to 20.
        max_levels (int): Maximum number of distinct values allowed for a column
            to be considered a valid grouping variable. Defaults to 5.

    Returns:
        Optional[str]: The name of the detected outcome column, or ``None`` if
            no suitable column is found.
    """
    candidates = column_types.get("categorical", [])

    for group in keyword_groups:
        for col in candidates:
            name = str(col).lower()
            if not any(keyword in name for keyword in group):
                continue

            counts = df[col].value_counts(dropna=True)
            # not constant, not too many levels for a group comparison / logistic target
            if len(counts) < 2 or len(counts) > max_levels:
                continue
            # every class needs enough observations for a stable comparison
            if counts.min() < min_class_size:
                continue

            return col

    return None


# ---------------------------------------------------------------------------
# Association screening
#
# Tests every numeric-numeric pair and every (cardinality-limited)
# categorical-categorical pair without a human picking them, rather than
# restricting to a data-driven subset -- with ~200 columns the full pairwise
# set is tens of thousands of tests, but each test is cheap, and
# Benjamini-Hochberg FDR correction (below) is specifically designed to stay
# valid regardless of how many tests are run, so there is no statistical need
# to pre-filter which pairs get tested. Every p-value is corrected for
# running many tests at once (BH-FDR -- standard for exploratory screening,
# less conservative than Bonferroni), and a pair is only flagged
# "significant" when BOTH the corrected p-value clears alpha AND the effect
# size clears a conventional small-effect threshold (Cohen's conventions):
# with large n, trivial differences become "significant" by p-value alone.
# ---------------------------------------------------------------------------
def _chi2_association(df, col1, col2):
    """Chi-square (or Fisher exact for 2×2) test + Cramér's V."""
    tbl = pd.crosstab(df[col1], df[col2])
    rows, cols = tbl.shape
    chi2, p, dof, expected = chi2_contingency(tbl)
    fisher_p = None
    if rows == 2 and cols == 2:
        _, fisher_p = fisher_exact(tbl)
    n = tbl.to_numpy().sum()
    min_dim = min(rows, cols) - 1
    if min_dim > 0:
        chi2_unc, _, _, _ = chi2_contingency(tbl, correction=False)
        cramers_v = np.sqrt(chi2_unc / (n * min_dim))
    else:
        cramers_v = np.nan
    assumptions_ok = (expected >= 1).all() and ((expected < 5).sum() / expected.size) <= 0.20
    if rows == 2 and cols == 2 and not assumptions_ok:
        return {"test_used": "Fisher Exact Test", "chi2_statistic": chi2, "p_value": fisher_p, "cramers_v": cramers_v}
    return {"test_used": "Chi-Square Test", "chi2_statistic": chi2, "p_value": p, "cramers_v": cramers_v}


def _auto_correlation(df, var1, var2):
    """Pearson or Spearman based on skewness/outlier heuristics."""
    data = df[[var1, var2]].dropna()
    if len(data) < 3:
        raise ValueError("Need at least 3 observations")
    x, y = data[var1], data[var2]
    q1x, q3x = np.percentile(x, [25, 75])
    iqrx = q3x - q1x
    q1y, q3y = np.percentile(y, [25, 75])
    iqry = q3y - q1y
    outlier = np.any((x < q1x - 1.5 * iqrx) | (x > q3x + 1.5 * iqrx)) or \
              np.any((y < q1y - 1.5 * iqry) | (y > q3y + 1.5 * iqry))
    use_pearson = (len(data) >= 30 and abs(stats.skew(x)) < 2
                   and abs(stats.skew(y)) < 2 and not outlier)
    if use_pearson:
        r, p = stats.pearsonr(x, y)
        method = "pearson"
    else:
        r, p = stats.spearmanr(x, y)
        method = "spearman"
    return {"method": method, "correlation": r, "p_value": p}


def screen_associations(df, column_types, alpha=0.05,
                        effect_size_thresholds=None, max_group_levels=6):
    """Screen column pairs for statistically significant associations with BH-FDR correction.

    Three pair types are tested:

    * **numeric-numeric**: every pair of numeric columns, using Pearson or
      Spearman correlation selected automatically by ``_auto_correlation``.
    * **categorical-categorical**: every pair of low-cardinality categorical
      columns, using chi-square or Fisher exact via ``_chi2_association``.
    * **numeric-categorical**: every numeric column against binary categorical
      columns (exactly two levels), using ``compare_two_groups``.

    All p-values are corrected together using Benjamini-Hochberg FDR. A pair is
    marked ``significant`` only when both the adjusted p-value is below
    ``alpha`` and the effect size clears the conventional small-effect threshold
    for its metric.

    Args:
        df (pd.DataFrame): The dataset to screen.
        column_types (dict): Column-type dictionary as returned by
            ``detect_column_types``.
        alpha (float): Significance level for the FDR correction. Defaults to
            0.05.
        effect_size_thresholds (dict, optional): Mapping of effect-size metric
            name to minimum threshold. Defaults to Cohen's small-effect
            conventions (correlation >= 0.2, Cramers V >= 0.1, Hedges g >= 0.2,
            rank-biserial >= 0.2).
        max_group_levels (int): Maximum number of distinct values a categorical
            column may have to be included in the categorical-categorical screen.
            Defaults to 6.

    Returns:
        pd.DataFrame: A DataFrame with one row per tested pair and columns
            ``var1``, ``var2``, ``pair_type``, ``test``, ``statistic``,
            ``p_value``, ``effect_size``, ``effect_size_metric``, ``p_adj``, and
            ``significant``. An empty DataFrame with those columns is returned
            when no pairs could be tested.
    """
    if effect_size_thresholds is None:
        # "small effect" cutoffs by Cohen's conventions
        effect_size_thresholds = {
            "correlation": 0.2,
            "cramers_v": 0.1,
            "hedges_g": 0.2,
            "rank_biserial": 0.2,
        }

    numeric_cols = column_types.get("numeric", [])
    categorical_cols = column_types.get("categorical", [])
    rows = []

    # numeric ~ numeric: every pair of numeric columns. BH-FDR correction
    # below is what keeps this valid at scale, not a pre-filter on which
    # pairs look promising.
    for var1, var2 in itertools.combinations(numeric_cols, 2):
        try:
            corr = _auto_correlation(df, var1, var2)
        except Exception:
            logger.warning("Correlation screening failed for (%s, %s)", var1, var2, exc_info=True)
            continue
        rows.append({
            "var1": var1, "var2": var2, "pair_type": "num-num",
            "test": corr["method"], "statistic": corr["correlation"],
            "p_value": corr["p_value"],
            "effect_size": abs(corr["correlation"]), "effect_size_metric": "correlation",
        })

    # categorical ~ categorical: every pair among low-cardinality categorical
    # columns. A chi-square test between two high-cardinality columns is both
    # expensive and statistically meaningless (most expected counts fall below
    # 5), so the cardinality shortlist stays -- but every pair within it is
    # tested, with BH-FDR correcting for the resulting number of tests.
    shortlist = [c for c in categorical_cols if df[c].nunique(dropna=True) <= max_group_levels]
    for col1, col2 in itertools.combinations(shortlist, 2):
        try:
            assoc = _chi2_association(df, col1, col2)
        except Exception:
            logger.warning("Categorical association screening failed for (%s, %s)", col1, col2, exc_info=True)
            continue
        rows.append({
            "var1": col1, "var2": col2, "pair_type": "cat-cat",
            "test": assoc["test_used"], "statistic": assoc["chi2_statistic"],
            "p_value": assoc["p_value"],
            "effect_size": assoc["cramers_v"], "effect_size_metric": "cramers_v",
        })

    # numeric ~ categorical: every numeric column against binary categorical
    # columns (group_col has exactly 2 levels). Restricted to the binary case
    # because compare_two_groups is the only group-comparison function that
    # already returns a standardized effect size (Cohen's d / Hedges' g /
    # rank-biserial) regardless of which test it auto-selects -- reusing it
    # keeps the screen built entirely on already-tested outputs instead of
    # inventing new effect-size formulas for the 3+-group case.
    for value_col in numeric_cols:
        for group_col in categorical_cols:
            if df[group_col].nunique(dropna=True) != 2:
                continue
            group_sizes = df.groupby(group_col)[value_col].apply(
                lambda s: s.dropna().shape[0]
            )
            if (group_sizes < _MIN_GROUP_SIZE_FOR_COMPARISON).any():
                logger.warning(
                    "Skipping group comparison for (%s, %s): a group has "
                    "fewer than %d observations",
                    value_col, group_col, _MIN_GROUP_SIZE_FOR_COMPARISON,
                )
                continue
            try:
                cmp = compare_two_groups(df, value_col, group_col)
            except Exception:
                logger.warning("Group comparison screening failed for (%s, %s)", value_col, group_col, exc_info=True)
                continue
            if "hedges_g" in cmp["effect_size"]:
                metric, effect = "hedges_g", cmp["effect_size"]["hedges_g"]
            else:
                metric, effect = "rank_biserial", cmp["effect_size"]["rank_biserial"]
            rows.append({
                "var1": value_col, "var2": group_col, "pair_type": "num-cat",
                "test": cmp["method"], "statistic": cmp["statistic"],
                "p_value": cmp["p_value"],
                "effect_size": abs(effect), "effect_size_metric": metric,
            })

    columns = ["var1", "var2", "pair_type", "test", "statistic", "p_value",
               "effect_size", "effect_size_metric", "p_adj", "significant"]
    if not rows:
        return pd.DataFrame(columns=columns)

    screening = pd.DataFrame(rows)

    # Benjamini-Hochberg FDR correction across every test run in this screen --
    # without it, running hundreds of tests at alpha=0.05 would flag ~5% of
    # them as "significant" by chance alone.
    #
    # A handful of tests can come back with a NaN p-value (e.g. Welch's ANOVA
    # is undefined for degenerate group variances) -- multipletests propagates
    # a single NaN to *every* adjusted value, silently breaking the correction
    # for the whole screen. Correct only over the valid p-values and leave the
    # NaN rows as NaN/not-significant.
    from statsmodels.stats.multitest import multipletests
    screening["p_adj"] = np.nan
    valid = screening["p_value"].notna()
    if valid.any():
        _, p_adj, _, _ = multipletests(screening.loc[valid, "p_value"], alpha=alpha, method="fdr_bh")
        screening.loc[valid, "p_adj"] = p_adj

    thresholds = screening["effect_size_metric"].map(effect_size_thresholds)
    screening["significant"] = (
        (screening["p_adj"] < alpha)
        & screening["effect_size"].notna()
        & (screening["effect_size"] >= thresholds)
    )

    return screening[columns]