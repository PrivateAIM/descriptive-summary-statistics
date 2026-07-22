"""Categorical column analysis helpers for tabular datasets.

Provides lightweight frequency and distribution utilities that are consumed
by the main reporting pipeline to summarise categorical and label columns.
"""
from typing import Dict
import pandas as pd


def compute_value_counts(df: pd.DataFrame, column: str) -> Dict[str, int]:
    """Count occurrences of each value in a single DataFrame column.

    Args:
        df (pd.DataFrame): The input dataset.
        column (str): Name of the column to count.

    Returns:
        Dict[str, int]: Mapping of string-cast category label to integer count,
            including a ``"nan"`` entry for missing values when present. Returns
            an empty dict if ``column`` is not in ``df``.
    """
    if column not in df.columns:
        return {}

    return {
        str(k): int(v)
        for k, v in df[column].value_counts(dropna=False).items()
    }


def compute_sex_distribution(df: pd.DataFrame) -> Dict[str, int]:
    """Return the value-count distribution for the sex or gender column.

    The function looks for a column named ``"sex"`` first, then ``"gender"``.
    Missing values are counted under the string ``"nan"``.

    Args:
        df (pd.DataFrame): The input dataset.

    Returns:
        Dict[str, int]: Mapping of sex/gender label to count, or an empty dict
            when neither ``"sex"`` nor ``"gender"`` is present.
    """
    # Sex distribution
    sex_col = "sex" if "sex" in df.columns else ("gender" if "gender" in df.columns else None)
    sex_counts = (
        df[sex_col].astype(str).value_counts(dropna=False).to_dict()
        if sex_col else {}
    )
    return sex_counts


def compute_label_distribution(
    df: pd.DataFrame,
    label_col: str
) -> Dict[str, int]:
    """Return the value-count distribution for a target label column.

    Args:
        df (pd.DataFrame): The input dataset.
        label_col (str): Name of the target label column.

    Returns:
        Dict[str, int]: Mapping of label value (as string) to count, or an
            empty dict if ``label_col`` is not in ``df``.
    """
    if label_col not in df.columns:
        return {}

    return compute_value_counts(df, label_col)