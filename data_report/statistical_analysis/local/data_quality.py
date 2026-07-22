"""Data quality checks for pandas DataFrames.

Provides column-level and dataset-level missing-value counts used by the
reporting pipeline to quantify data completeness.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


def compute_missing_by_column(df: pd.DataFrame) -> Dict[str, int]:
    """Count the number of missing values in each column of a DataFrame.

    Args:
        df (pd.DataFrame): The input dataset.

    Returns:
        Dict[str, int]: Mapping of column name to the count of NaN/None values
            in that column.
    """
    return df.isna().sum().astype(int).to_dict()


def compute_total_missing(df: pd.DataFrame) -> int:
    """Count the total number of missing values across an entire DataFrame.

    Args:
        df (pd.DataFrame): The input dataset.

    Returns:
        int: Total count of NaN/None cells across all rows and columns.
    """
    return int(df.isna().sum().sum())