"""Descriptive statistics computation for typed DataFrame columns.

Provides column-type detection (numeric, categorical, binary, temporal) and
per-type summary statistic functions designed to be called once per local
dataset; results can be aggregated across nodes by a federated hub.
"""
import re
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List

# detect numeric, categorical and temporal columns
def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Classify every column of a DataFrame into one of four typed buckets.

    Columns that contain no non-missing values are excluded from all buckets so
    that empty columns do not pollute downstream similarity matrices.

    Args:
        df (pd.DataFrame): The input dataset whose columns are to be classified.

    Returns:
        Dict[str, List[str]]: A dictionary with keys ``"numeric"``,
            ``"categorical"``, ``"temporal"``, and ``"binary"``, each mapping to
            a list of column names belonging to that type.
    """
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    temporal_columns = df.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns.tolist()
    candidate_columns = [
        col for col in df.columns
        if col not in temporal_columns
    ]
    binary_columns = [col for col in candidate_columns if is_binary(df[col], column_name=col)]
    categorical_columns = [
        col for col in df.columns
        if col not in numeric_columns and col not in temporal_columns
    ]
    numeric_columns = [col for col in numeric_columns if col not in binary_columns]
    categorical_columns = categorical_columns + [
        col for col in binary_columns if col not in categorical_columns
    ]

    # Columns with no non-missing values carry no information for any
    # statistic (compute_*_statistics already skip them) -- drop them here
    # too so they don't pollute downstream statistics with all-NaN values
    # (e.g. showing up as a nonsensical "numeric" variable).
    non_empty = {col for col in df.columns if df[col].notna().any()}
    numeric_columns = [c for c in numeric_columns if c in non_empty]
    categorical_columns = [c for c in categorical_columns if c in non_empty]
    temporal_columns = [c for c in temporal_columns if c in non_empty]
    binary_columns = [c for c in binary_columns if c in non_empty]

    return {
        "numeric": numeric_columns,
        "categorical": categorical_columns,
        "temporal": temporal_columns,
        "binary": binary_columns
    }


def detect_quasi_numeric_categorical_columns(
    df: pd.DataFrame, categorical_columns: List[str], threshold: float = 0.5,
) -> List[str]:
    """Identify categorical columns where most (but not all) values look numeric.

    A column like lab results recorded as ``["12.5", "8.1", "<5", "300.2"]`` is
    routed to the categorical bucket by ``detect_column_types`` because the
    censored value ``"<5"`` keeps the whole column at object dtype -- with no
    indication anywhere that this happened. This function flags such columns
    so a report notice can explain the gap, without silently coercing the
    unparseable values to NaN (which would discard the below/above-detection-
    limit signal without a human deciding that's the right tradeoff for that
    specific column).

    Args:
        df (pd.DataFrame): The dataset containing the columns to check.
        categorical_columns (List[str]): Categorical column names to check.
        threshold (float): Minimum fraction of non-null values that must be
            numeric-parseable for a column to be flagged. Defaults to 0.5.

    Returns:
        List[str]: Column names where ``threshold <= fraction numeric < 1.0``.
    """
    flagged = []
    for col in categorical_columns:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        fraction_numeric = pd.to_numeric(s, errors="coerce").notna().mean()
        if threshold <= fraction_numeric < 1.0:
            flagged.append(col)
    return flagged

# Medical/clinical terms that strongly imply a column is a binary flag
# (diagnosis present/absent, symptom present/absent, treatment given/not given).
# Used by is_binary to rescue legitimately binary columns that are very sparse
# (fewer than _BINARY_MIN_COUNT non-null values).
_MEDICAL_BINARY_KEYWORDS = {
    # Comorbidities / chronic conditions
    "hypertension", "diabetes", "cardiovasc", "cardiac", "cardio", "coronary", "chd",
    "asthma", "copd", "crd", "ckd", "cld", "renal", "hepatic", "pulmonary",
    "neuro", "psych", "psychiatric", "neurological",
    "tumor", "cancer", "oncol", "malignant", "lesion",
    "immuno", "transplant", "comorb",
    "metabol", "obesity", "obese",
    "hiv", "tb", "tuberculosis", "infectious",
    # Acute symptoms
    "fever", "cough", "dyspnea", "fatigue", "malaise", "anosmia",
    "pain", "confusion", "diarrhea", "nausea", "rash", "fainting",
    "headache", "myalgia", "arthralgia", "ageusia", "anorexia",
    "tinnitus", "dizziness", "seizures", "tremors", "constipation",
    "dismobility", "depressive", "parasthesia",
    "memory", "acute",
    # Hospital events / outcomes
    "hospitalization", "icu", "admission", "discharge",
    "mortality", "death",
    # Treatments / interventions
    "treated", "treatment", "therapy", "procedure", "surgery",
    "oxygen", "ventil",
    "viral", "antiviral", "anti", "antibiotic",
    "glucocorticoid", "steroid", "corticoid",
    "monoclonal", "antibody", "antagonist", "inhibitor",
    "lopinavir", "remdesivir", "molnupiravir", "paxlovid", "ribavirin",
    "il1", "il6",
    "vaccination", "vaccin", "vac",
    # Generic binary-indicator suffixes / prefixes
    "flag", "indicator",
}

# Minimum non-null observations required to classify a numeric {0, 1} column as
# binary on value evidence alone.  Below this threshold the column name must
# contain a medical keyword (see _MEDICAL_BINARY_KEYWORDS) to be classified as
# binary — this prevents a very sparse continuous column that happens to show
# only {0, 1} from being misclassified.
_BINARY_MIN_COUNT = 3


def _name_suggests_binary(column_name: str) -> bool:
    """Return True if the column name contains a medical or binary-indicator keyword."""
    tokens = set(re.split(r"[^a-z0-9]+", column_name.lower())) - {""}
    return bool(tokens & _MEDICAL_BINARY_KEYWORDS)


# checks if the column is binary
def is_binary(series: pd.Series, allow_bool: bool = True,
              column_name: str = "") -> bool:
    """Determine whether a pandas Series represents a binary variable.

    A series is considered binary if it has at most two distinct non-null values
    and those values belong to one of the recognised binary encodings: numeric
    {0, 1}, boolean dtype, or a semantic pair such as yes/no, true/false, or
    ja/nein.

    For sparse numeric {0, 1} columns (fewer than ``_BINARY_MIN_COUNT`` non-null
    rows) the function requires the column name to contain a medical keyword from
    ``_MEDICAL_BINARY_KEYWORDS`` before returning True, guarding against
    continuous columns that happen to show only {0, 1} by coincidence.

    Args:
        series (pd.Series): The column to test.
        allow_bool (bool): If True, accept boolean-dtype series unconditionally.
            Defaults to True.
        column_name (str): The name of the column, used for keyword-based
            disambiguation of sparse numeric columns. Defaults to ``""``.

    Returns:
        bool: True when the series is recognised as a binary variable.
    """
    s = series.dropna()
    if s.empty:
        return False
    if s.nunique() > 2:
        return False
    # numeric binary: 0/1 or 0.0/1.0
    numeric = pd.to_numeric(series, errors="coerce")
    valid_values = numeric.dropna()
    if len(valid_values) > 0:
        unique_vals = set(valid_values.unique())
        if unique_vals.issubset({0, 1}):
            # Require sufficient observations OR a medical keyword in the name.
            # A very sparse column (< _BINARY_MIN_COUNT non-null values) that
            # shows only {0, 1} could be a continuous measurement that happens
            # to have those values by coincidence; the keyword acts as a
            # positive signal that it is genuinely a binary clinical flag.
            if len(valid_values) >= _BINARY_MIN_COUNT or _name_suggests_binary(column_name):
                return True
    # boolean: True or False
    if allow_bool and pd.api.types.is_bool_dtype(series):
        return True
    # semantic binary: yes/no, y/n, true/false, ja/nein
    str_series = series.dropna().astype(str).str.strip().str.lower()
    unique_vals = set(str_series.dropna().unique())
    valid_sets = [
        {"yes", "no"},
        {"y", "n"},
        {"true", "false"},
        {"t", "f"},
        {"ja", "nein"},
    ]
    for valid_set in valid_sets:
        if unique_vals.issubset(valid_set):
            return True

    return False

def detect_id_column(series: pd.Series, column_name: str) -> bool:
    """Detect whether a column is likely a patient or record identifier.

    Uses token-split keyword matching only: columns whose normalised name
    contains a strong identifier keyword are accepted unconditionally;
    columns matching weaker keywords (patient, subject, …) are accepted
    only when the uniqueness ratio exceeds 0.9. Datetime columns are never
    classified as identifiers.

    The bare uniqueness heuristic (ratio ≥ 0.95 for any column) has been
    intentionally removed because it produces false positives on date
    columns and other high-cardinality clinical features.

    Args:
        series (pd.Series): The column to evaluate.
        column_name (str): The name of the column.

    Returns:
        bool: True when the column is likely an identifier.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return False

    name = column_name.lower()
    tokens = set(re.split(r'[^a-z0-9]+', name)) - {""}

    strong_keywords = {
        "id", "identifier", "identifikator", "pid",
        "name", "surname", "firstname", "lastname", "rownames",
        "first_name", "last_name", "family_name",
        "vorname", "nachname", "familienname",
        "telefon", "phone", "email", "adresse", "address",
        "postcode", "zip", "ssn", "dob", "birthdate", "geburtsdatum",
    }
    weak_keywords = {"patient", "person", "subject", "record", "case"}

    if bool(tokens & strong_keywords):
        return True

    valid_values = series.dropna()
    n_valid = len(valid_values)
    if n_valid == 0:
        return False

    # Value-based check: string values matching "patient_001", "subject_02", etc.
    # Sample up to 50 rows to keep this fast.
    if pd.api.types.is_object_dtype(valid_values):
        _id_value_re = re.compile(
            r'^(patient|pat|subject|sub|person|record|case|participant)[_\-]?\d+$',
            re.IGNORECASE,
        )
        sample = valid_values.iloc[:50]
        if sample.apply(lambda v: bool(_id_value_re.match(str(v)))).all():
            return True

    uniqueness_ratio = valid_values.nunique() / n_valid

    if bool(tokens & weak_keywords) and uniqueness_ratio > 0.9:
        return True
    return False

def compute_numeric_statistics(numeric_df: pd.DataFrame) -> dict:
    """Compute descriptive statistics for every numeric column in a DataFrame.

    Non-numeric values within a column are coerced to NaN and excluded from all
    calculations. Columns that are entirely non-numeric after coercion are
    silently skipped.

    Args:
        numeric_df (pd.DataFrame): DataFrame containing only numeric (or
            mixed-but-coercible) columns.

    Returns:
        dict: Mapping of column name to a statistics dictionary containing
            ``mean``, ``median``, ``mode``, ``min``, ``max``, ``variance``,
            ``std_dev``, ``iqr``, ``count``, ``frequency``,
            ``relative_frequency``, ``outliers`` (IQR-fence count), ``skewness``,
            ``kurtosis``, and ``missing_values``.
    """
    statistics = {}
    for col in numeric_df.columns:

        # number of missing values in the column
        missing_values = int(numeric_df[col].isna().sum())

        # s = pandas Series
        s = pd.to_numeric(numeric_df[col], errors="coerce").dropna()
        if len(s) == 0:
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = int(((s < lower_bound) | (s > upper_bound)).sum())
        vc = s.value_counts()
        statistics[col] = {
            "mean": round(float(s.mean()), 3),
            "median": round(float(s.median()), 3),
            "mode": float(s.mode().iloc[0]) if not s.mode().empty else None,
            "min": float(s.min()),
            "max": float(s.max()),
            "variance": round(float(s.var()), 3),
            "std_dev": round(float(s.std()), 3),
            # "q1": float(q1),
            # "q3": float(q3),
            "iqr": round(float(iqr), 3),
            "count": int(s.count()),
            'frequency': int(vc.iloc[0]),
            'relative_frequency': round(vc.iloc[0] / len(s) * 100, 2),
            'outliers': outlier_count,
            'skewness': round(s.skew(), 3),
            'kurtosis': round(s.kurtosis(), 3),
            'missing_values': missing_values
        }
    return statistics


def compute_categorical_statistics(categorical_df: pd.DataFrame) -> Dict:
    """Compute frequency and imbalance statistics for every categorical column.

    NaN values are excluded before computing category counts so they do not
    appear as a separate ``"nan"`` category. Columns that are entirely missing
    are silently skipped.

    Args:
        categorical_df (pd.DataFrame): DataFrame containing only categorical
            columns.

    Returns:
        Dict: Mapping of column name to a statistics dictionary containing
            ``count``, ``number_of_categories``, ``most_frequent_category``,
            ``least_frequent_category``, ``category_counts``,
            ``relative_frequencies``, ``class_imbalance_ratio``, and
            ``missing_values``.
    """
    statistics = {}

    for col in categorical_df.columns:
        # number of missing values
        missing_values = int(categorical_df[col].isna().sum())

        # drop missing values *before* converting to string -- otherwise
        # NaN becomes the literal string "nan" and is counted as its own
        # category below.
        s = categorical_df[col].dropna().astype(str)
        if s.empty:
            continue
        # count occurrences of each category
        # vc = the number of observations belonging to each category(frequency of each category)
        # for example: if we have 3 possible categories: A, B and C for a column:
        # we want to know how many times do we have A in the rows?
        # how many rows habe B? how many rows have C?
        vc = s.value_counts()
        if vc.empty:
            continue
        # relative frequency (%) of each category
        rel_freq = (vc / len(s) * 100).round(2)
        # most frequent category
        most_frequent = vc.index[0]
        # least frequent category
        least_frequent = vc.index[-1]
        # simple class imbalance indicator
        # ratio between most frequent and least frequent
        # (vc.iloc[-1] is the smallest non-zero count, so this is always > 0)
        imbalance_ratio = float(vc.iloc[0] / vc.iloc[-1])

        statistics[col] = {
            # total number of valid observations
            "count": int(s.count()),
            # number of distinct categories
            "number_of_categories": int(s.nunique()),
            "most_frequent_category": most_frequent,
            "least_frequent_category": least_frequent,
            # counts of each category
            "category_counts": vc.to_dict(),
            # relative frequency (%) of each category
            "relative_frequencies": {k: round(v, 3) for k, v in rel_freq.items()},
            # imbalance ratio (large value = strong imbalance)
            "class_imbalance_ratio": round(imbalance_ratio, 4),
            "missing_values": missing_values
        }
    return statistics

def compute_temporal_statistics(temporal_df: pd.DataFrame, patient_series: Optional[pd.Series] = None,
    freq: str = "M") -> Dict:
    """Compute temporal distribution statistics for every datetime column.

    For each column the function records the observed time range, counts
    observations per period (daily/weekly/monthly/yearly), identifies the most
    active period, and lists periods within the range that have no observations.

    Args:
        temporal_df (pd.DataFrame): DataFrame whose columns are or can be
            coerced to datetime.
        patient_series (Optional[pd.Series]): Series of patient identifiers with
            the same index as ``temporal_df``. When provided, a per-patient
            observation count is computed. Defaults to None.
        freq (str): Period frequency for bucketing. Accepted values are ``"D"``
            (daily), ``"W"`` (weekly), ``"M"`` (monthly), and ``"Y"`` (yearly).
            Defaults to ``"M"``.

    Returns:
        Dict: Mapping of column name to a statistics dictionary containing
            ``count``, ``time_range``, ``range_days``,
            ``observations_per_period``, ``most_active_period``,
            ``missing_periods``, and ``missing_values``.

    Raises:
        ValueError: If ``patient_series`` is provided but its index does not
            match the index of ``temporal_df``.
    """
    statistics = {}
    # validate alignment if patient_series is provided
    if patient_series is not None:
        if not patient_series.index.equals(temporal_df.index):
            raise ValueError("patient_series must have the same index as temporal_df")

    for col in temporal_df.columns:

        # count missing values
        missing_values = int(temporal_df[col].isna().sum())

        # convert to datetime
        s = pd.to_datetime(temporal_df[col], errors="coerce").dropna()

        if len(s) == 0:
            continue

        start_date = s.min()
        end_date = s.max()
        # observations per time unit
        obs_per_period = s.dt.to_period(freq).value_counts().sort_index()
        # most active period
        most_active_period = obs_per_period.idxmax()
        # detect missing periods
        full_range = pd.period_range(
            start=start_date.to_period(freq),
            end=end_date.to_period(freq),
            freq=freq
        )
        missing_periods = list(set(full_range) - set(obs_per_period.index))

        # observations per patient over time

        # initialize the variable
        # if the dataset does not contain a patient ID column, we leave it None
        # ->  this avoids errors later and clearly signals no patient-level analysis available
        obs_per_patient = None
        if patient_series is not None:
            # create a temporary dataframe with only the columns needed
            tmp = pd.DataFrame({
                "patient": patient_series,
                "time": pd.to_datetime(temporal_df[col], errors="coerce") # convert the column to datetime
                # if invalid values exist,  they become NaT = Not a Time
            }).dropna() # remove rows with missing values (rows where patient is missing or date is missing)

            if not tmp.empty:
                obs_per_patient = (
                # group all rows belonging to the same patient
                    tmp.groupby("patient")["time"]
                    .count()
                    .to_dict()
                )

        statistics[col] = {
            # number of valid timestamps
            "count": int(s.count()),
            # start and end of the dataset timeline
            "time_range": {
                "start": str(start_date),
                "end": str(end_date)
            },
            # number of days between start and end
            "range_days": int((end_date - start_date).days),
            # observations per time unit (month by default)
            "observations_per_period": obs_per_period.to_dict(),
            "most_active_period": str(most_active_period),
            "missing_periods": [str(p) for p in missing_periods],
            # "observations_per_patient": obs_per_patient,
            "missing_values": missing_values
        }
    return statistics


def compute_age_histogram(
    df: pd.DataFrame,
    age_col: str = "age",
    bin_size: int = 5,
    max_age: int = 100,
) -> Tuple[Optional[list], Optional[list]]:
    """Compute a fixed-width age histogram from a DataFrame.

    Returns (counts_list, edges_list) suitable for federated aggregation:
    counts are summed across nodes and edges are identical so bins stay aligned.

    Args:
        df (pd.DataFrame): The input dataset.
        age_col (str): Name of the column containing age values. Defaults to
            ``"age"``.
        bin_size (int): Width of each histogram bin in years. Defaults to 5.
        max_age (int): Upper bound for the last bin edge in years. Defaults to
            100.

    Returns:
        Tuple[Optional[list], Optional[list]]: A pair ``(counts, edges)`` where
            ``counts`` is a list of integer observation counts per bin and
            ``edges`` is a list of float bin boundaries, or ``(None, None)`` if
            the column is absent or empty after numeric coercion.
    """
    if age_col not in df.columns:
        return None, None
    ages = pd.to_numeric(df[age_col], errors="coerce").dropna()
    if ages.empty:
        return None, None
    bins = np.arange(0, max_age + bin_size, bin_size)
    hist, edges = np.histogram(ages, bins=bins)
    return hist.astype(int).tolist(), edges.astype(float).tolist()


def count_out_of_range_ages(
    df: pd.DataFrame, age_col: str = "age", max_age: int = 100,
) -> int:
    """Count parseable age values that fall outside the plausible [0, max_age] range.

    compute_age_histogram's fixed-width bins silently drop values below 0 or
    above ``max_age`` (e.g. a negative age or a data-entry typo like 999) --
    they simply don't appear in any bin. This count lets the report surface
    those values instead of letting them vanish from the age distribution
    with no explanation.

    Args:
        df (pd.DataFrame): The input dataset.
        age_col (str): Name of the column containing age values. Defaults to
            ``"age"``.
        max_age (int): Upper bound of the plausible age range in years.
            Defaults to 100.

    Returns:
        int: Number of non-null, numeric-parseable values in ``age_col``
            that are negative or greater than ``max_age``. 0 if the column
            is absent or empty after numeric coercion.
    """
    if age_col not in df.columns:
        return 0
    ages = pd.to_numeric(df[age_col], errors="coerce").dropna()
    if ages.empty:
        return 0
    return int(((ages < 0) | (ages > max_age)).sum())
