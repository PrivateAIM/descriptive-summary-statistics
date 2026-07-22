import numpy as np
import pandas as pd
import pytest

from data_report.statistical_analysis.local.compute_statistics import detect_column_types


def test_numeric_columns():
    df = pd.DataFrame({"age": [10, 20, 30], "score": [1.5, 2.5, 3.5]})
    result = detect_column_types(df)
    assert "age" in result["numeric"]
    assert "score" in result["numeric"]
    assert result["categorical"] == []
    assert result["temporal"] == []
    assert result["binary"] == []


def test_categorical_columns():
    df = pd.DataFrame({"city": ["Paris", "London", "Berlin"], "color": ["red", "blue", "green"]})
    result = detect_column_types(df)
    assert "city" in result["categorical"]
    assert "color" in result["categorical"]
    assert result["numeric"] == []
    assert result["temporal"] == []
    assert result["binary"] == []


def test_temporal_columns():
    df = pd.DataFrame({"event_date": pd.to_datetime(["2021-01-01", "2022-06-15", "2023-03-10"])})
    result = detect_column_types(df)
    assert "event_date" in result["temporal"]
    assert result["numeric"] == []
    assert result["categorical"] == []
    assert result["binary"] == []


def test_binary_numeric_01():
    df = pd.DataFrame({"flag": [0, 1, 0, 1]})
    result = detect_column_types(df)
    assert "flag" in result["binary"]
    assert "flag" not in result["numeric"]
    assert "flag" in result["categorical"]


def test_binary_bool_dtype():
    df = pd.DataFrame({"active": [True, False, True]})
    result = detect_column_types(df)
    assert "active" in result["binary"]
    assert "active" in result["categorical"]


def test_binary_yes_no():
    df = pd.DataFrame({"enrolled": ["yes", "no", "yes"]})
    result = detect_column_types(df)
    assert "enrolled" in result["binary"]
    assert "enrolled" in result["categorical"]


def test_binary_y_n():
    df = pd.DataFrame({"consent": ["y", "n", "y"]})
    result = detect_column_types(df)
    assert "consent" in result["binary"]


def test_binary_true_false_strings():
    df = pd.DataFrame({"valid": ["true", "false", "true"]})
    result = detect_column_types(df)
    assert "valid" in result["binary"]


def test_binary_case_insensitive():
    df = pd.DataFrame({"flag": ["YES", "No", "YES"]})
    result = detect_column_types(df)
    assert "flag" in result["binary"]


def test_non_binary_categorical_not_in_binary():
    df = pd.DataFrame({"status": ["low", "medium", "high"]})
    result = detect_column_types(df)
    assert "status" not in result["binary"]
    assert "status" in result["categorical"]


def test_mixed_column_types():
    df = pd.DataFrame({
        "age": [25, 40, 35],
        "city": ["NY", "LA", "SF"],
        "dob": pd.to_datetime(["1999-01-01", "1984-06-01", "1989-03-15"]),
        "active": [0, 1, 1],
    })
    result = detect_column_types(df)
    assert "age" in result["numeric"]
    assert "city" in result["categorical"]
    assert "dob" in result["temporal"]
    assert "active" in result["binary"]
    assert "active" not in result["numeric"]


def test_empty_dataframe():
    df = pd.DataFrame()
    result = detect_column_types(df)
    assert result["numeric"] == []
    assert result["categorical"] == []
    assert result["temporal"] == []
    assert result["binary"] == []


def test_column_with_nulls_still_detected():
    df = pd.DataFrame({"score": [1.0, None, 3.0], "flag": [0, None, 1]})
    result = detect_column_types(df)
    assert "score" in result["numeric"]
    assert "flag" in result["binary"]


def test_binary_column_not_in_numeric():
    df = pd.DataFrame({"x": [0, 1, 0]})
    result = detect_column_types(df)
    assert "x" not in result["numeric"]
    assert "x" in result["binary"]
