"""Synthetic PASC label generation for HALTA node CSVs.

``add_synthetic_pasc_label`` generates a binary post-acute sequelae of
SARS-CoV-2 (PASC) label for a DataFrame that does not already contain one.
The synthetic label is correlated with symptom burden (count of symptom-flag
columns that are positive for the patient) and severity proxies
(hospitalisation and ICU flags) via a logistic model with added Gaussian
noise, so it is statistically plausible without being derived from real
outcomes.

``label_and_write_csv`` is a convenience wrapper that reads a CSV file, calls
``add_synthetic_pasc_label``, and writes the labelled result back to disk.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from data_report.config import PASC_SYMPTOM_KEYWORDS


def _find_symptom_columns(df: pd.DataFrame) -> List[str]:
    """Return de-duplicated list of columns whose names match any PASC symptom keyword."""
    cols: List[str] = []
    for c in df.columns:
        cl = str(c).lower()
        for kw in PASC_SYMPTOM_KEYWORDS:
            if kw in cl:
                cols.append(c)
                break
    seen: set = set()
    out: List[str] = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _to_numeric_flag_series(s: pd.Series) -> pd.Series:
    """Convert a boolean-coded string or numeric series to a float series of 0 / 1 flags."""
    if s.dtype == object:
        mapped = (
            s.astype(str)
             .str.strip()
             .str.lower()
             .map({
                 "1": 1, "true": 1, "yes": 1, "y": 1, "t": 1,
                 "0": 0, "false": 0, "no": 0, "n": 0, "f": 0,
             })
        )
        return mapped.fillna(pd.to_numeric(s, errors="coerce"))
    return pd.to_numeric(s, errors="coerce")


def add_synthetic_pasc_label(
    df: pd.DataFrame,
    label_col: str = "pasc",
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Add a synthetic binary PASC label column to a DataFrame.

    If the target column already exists it is left untouched and the function
    returns immediately with ``status="exists"``.  Otherwise a label is
    generated using a logistic model whose score is::

        score = -1.2 + 2.6 * symptom_burden + 0.6 * hosp + 0.8 * icu

    where ``symptom_burden`` is the fraction of symptom-flag columns that are
    positive for each patient, and ``hosp`` / ``icu`` are binary severity
    proxies read from columns named ``"hospitalization"`` and ``"icu"`` if
    present.  Gaussian noise (σ = 0.05) is added to the probability before
    Bernoulli sampling so the label is not perfectly deterministic.

    Args:

        df (pd.DataFrame): Input DataFrame.  All columns are inspected for
            symptom keywords; ``"hospitalization"`` and ``"icu"`` columns are
            used as severity proxies if present.
        label_col (str): Name of the column to create or check.  Defaults to
            ``"pasc"``.
        seed (int): Random seed for reproducibility.  Defaults to ``42``.

    Returns:

        tuple:
            - pd.DataFrame: Copy of ``df`` with the label column appended (or
              the original ``df`` unchanged if the column already existed).
            - dict: Metadata dictionary with keys ``"label_col"``,
              ``"status"`` (``"exists"`` or ``"created"``), ``"n_rows"``,
              ``"pasc_positive"``, and — when created — ``"symptom_cols_used"``,
              ``"num_symptom_cols_used"``, and ``"pasc_prevalence"``.
    """
    if label_col in df.columns:
        meta = {
            "label_col": label_col,
            "status": "exists",
            "n_rows": int(df.shape[0]),
            "pasc_positive": int(pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int).sum()),
        }
        return df, meta

    rng = np.random.default_rng(seed)

    symptom_cols = _find_symptom_columns(df)

    if symptom_cols:
        sym = df[symptom_cols].copy()
        for c in symptom_cols:
            sym[c] = _to_numeric_flag_series(sym[c])
        sym = sym.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        symptom_burden = sym.clip(lower=0).astype(float).sum(axis=1).to_numpy(dtype=float)
        symptom_burden_norm = symptom_burden / max(1.0, float(len(symptom_cols)))
    else:
        flag_like = [c for c in df.columns if any(k in str(c).lower() for k in ["symptom", "complaint", "score"])][:25]
        if flag_like:
            tmp = df[flag_like].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            symptom_burden_norm = (tmp.clip(lower=0).sum(axis=1) / max(1.0, float(len(flag_like)))).to_numpy(dtype=float)
        else:
            symptom_burden_norm = np.zeros(int(df.shape[0]), dtype=float)

    hosp = (
        pd.to_numeric(df["hospitalization"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if "hospitalization" in df.columns
        else 0.0
    )
    icu = (
        pd.to_numeric(df["icu"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if "icu" in df.columns
        else 0.0
    )

    score = -1.2 + 2.6 * symptom_burden_norm + 0.6 * hosp + 0.8 * icu
    prob = 1.0 / (1.0 + np.exp(-score))
    prob = np.clip(prob + rng.normal(0.0, 0.05, size=prob.shape[0]), 0.0, 1.0)

    y = rng.binomial(1, prob).astype(int)

    out_df = df.copy()
    out_df[label_col] = y

    meta = {
        "label_col": label_col,
        "status": "created",
        "n_rows": int(df.shape[0]),
        "symptom_cols_used": symptom_cols,
        "num_symptom_cols_used": int(len(symptom_cols)),
        "pasc_positive": int(y.sum()),
        "pasc_prevalence": float(y.mean()) if len(y) else 0.0,
    }
    return out_df, meta


def label_and_write_csv(
    input_path: str,
    output_path: str,
    label_col: str = "pasc",
    seed: int = 42,
) -> Dict[str, Any]:
    """Read a CSV file, add a synthetic PASC label, and write the result to disk.

    Args:

        input_path (str): Path to the source CSV file.
        output_path (str): Path where the labelled CSV will be written.
        label_col (str): Name of the label column to add.  Defaults to
            ``"pasc"``.
        seed (int): Random seed forwarded to ``add_synthetic_pasc_label``.
            Defaults to ``42``.

    Returns:

        dict: Metadata dictionary returned by ``add_synthetic_pasc_label``.
    """
    df = pd.read_csv(input_path)
    df_labeled, meta = add_synthetic_pasc_label(df, label_col=label_col, seed=seed)
    df_labeled.to_csv(output_path, index=False)
    return meta
