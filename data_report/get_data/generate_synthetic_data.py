"""Synthetic medical-data generator for exercising inferential statistics on a
realistic, multi-hospital dataset.

Produces four kinds of data, all with a deliberate correlation structure so
that ``screen_associations`` / ``compare_two_groups`` /
``one_way_group_comparison`` / ``correlation_between_two_variables`` have
real signal to detect:

1. Cross-sectional, multi-node hospital dataset (``data/dataset4/node{1,2,3}``)
   -- one row per patient, mixed numeric/categorical/binary/temporal columns,
   ready to run through ``dr-analyze`` / the StarModel pipeline. node2 and
   node3 each drop one numeric column (``ldl_cholesterol`` /
   ``crp_level``) and node3 also drops the temporal ``followup_date``
   column, while node1 has an extra ``referral_source`` categorical column
   that the other nodes don't collect. This gives the federated aggregator
   a mix of "common_all" / "common_partial" / "unique" columns across all
   three column types to classify.

2. Longitudinal dataset (``data/dataset4_longitudinal/node{1,2}``) -- repeated
   visits per patient with trending measurements, for
   ``analyze_longitudinal`` and ``detect_dataset_type`` (-> "longitudinal").

3. Event-based dataset (``data/dataset4_event_based/node{1,2}``) -- irregular
   event log per patient, for ``analyze_event_based`` and
   ``detect_dataset_type`` (-> "event_based").

4. Panel dataset (``data/dataset4_panel/node{1,2}``) -- hospital x month
   metrics (different hospitals per node), for ``analyze_panel``.

Run directly to (re)generate everything under ``data/``:

    python -m data_report.get_data.generate_synthetic_data
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

REGIONS = ["North", "South", "East", "West"]
BLOOD_TYPES = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
INSURANCE_TYPES = ["Public", "Private", "Uninsured"]
DIAGNOSIS_CATEGORIES = ["Cardiovascular", "Respiratory", "Endocrine", "Renal", "Other"]
EVENT_TYPES = ["ED_visit", "medication_change", "lab_test", "specialist_referral", "hospitalization"]
REFERRAL_SOURCES = ["Self", "GP_referral", "ED_transfer", "Specialist_referral"]

# Region -> insurance-type probabilities (creates a categorical/categorical
# association so region and insurance_type land in the same cluster).
REGION_INSURANCE_PROBS = {
    "North": [0.55, 0.40, 0.05],
    "South": [0.35, 0.30, 0.35],
    "East":  [0.50, 0.45, 0.05],
    "West":  [0.40, 0.50, 0.10],
}


# ---------------------------------------------------------------------------
# 1. Cross-sectional, multi-node hospital dataset
# ---------------------------------------------------------------------------
def make_hospital_dataset(node_label: str, n_patients: int, age_mean: float,
                           severity_shift: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    patient_id = [f"{node_label}_P{idx:04d}" for idx in range(1, n_patients + 1)]
    age = np.clip(rng.normal(age_mean, 14, n_patients), 18, 95)
    sex = rng.choice(["M", "F"], n_patients)

    # Latent severity / risk factors drive several observed columns each,
    # which is what gives the clustering algorithms real structure to find.
    metabolic_severity = rng.normal(0, 1, n_patients) + severity_shift + 0.02 * (age - age_mean)
    acute_severity = rng.normal(0, 1, n_patients) + 0.5 * severity_shift
    respiratory_risk = rng.normal(0, 1, n_patients)
    body_size = rng.normal(0, 1, n_patients)

    # --- Anthropometric cluster ---
    height_cm = np.clip(rng.normal(165, 9, n_patients), 140, 210)
    weight_kg = np.clip(60 + body_size * 12 + 0.15 * (height_cm - 165) + rng.normal(0, 5, n_patients), 40, 180)
    bmi = weight_kg / (height_cm / 100) ** 2
    waist_circumference_cm = 70 + body_size * 10 + 0.3 * (bmi - 25) + rng.normal(0, 4, n_patients)

    # --- Cardiometabolic cluster ---
    # weight/noise ratios tuned so pairwise correlations land around 0.85,
    # which is high enough for the TOM-based clustering (power=3,
    # distance_cut=0.25) to actually group these columns together.
    systolic_bp = 115 + metabolic_severity * 15 + rng.normal(0, 6, n_patients)
    diastolic_bp = 75 + metabolic_severity * 9 + rng.normal(0, 4, n_patients)
    glucose_mg_dl = 95 + metabolic_severity * 25 + rng.normal(0, 10, n_patients)
    hba1c_percent = 5.4 + metabolic_severity * 1.0 + rng.normal(0, 0.42, n_patients)
    ldl_cholesterol = 100 + metabolic_severity * 18 + rng.normal(0, 7.5, n_patients)

    # --- Acute / inflammation cluster ---
    heart_rate = 75 + acute_severity * 10 + rng.normal(0, 4, n_patients)
    temperature_c = 36.8 + acute_severity * 0.5 + rng.normal(0, 0.2, n_patients)
    wbc_count = np.clip(7 + acute_severity * 2.5 + rng.normal(0, 1.0, n_patients), 1.5, None)
    crp_level = np.clip(5 + acute_severity * 8 + rng.normal(0, 3.5, n_patients), 0, None)

    # --- Binary flags (form their own binary clusters) ---
    diabetes = (metabolic_severity + rng.normal(0, 0.5, n_patients) > 0.8).astype(int)
    hypertension = (metabolic_severity + rng.normal(0, 0.5, n_patients) > 0.5).astype(int)
    obesity = (bmi > 30).astype(int)
    current_smoker = (respiratory_risk + rng.normal(0, 0.5, n_patients) > 0.7).astype(int)

    # --- Categorical columns with deliberate associations ---
    smoking_status = np.where(
        current_smoker == 1, "Current",
        np.where(respiratory_risk > -0.2, "Former", "Never"),
    )

    # respiratory_diagnosis is sampled conditional on smoking_status with a
    # strongly (but not deterministically) skewed distribution per category.
    # For a 2-variable TOM cluster (power=3, distance_cut=0.25), Cramer's V
    # needs to exceed 0.5**(1/3) =~ 0.794 -- below that, even with TOM's
    # shared-neighbour boost, the pair stays in separate clusters.
    respiratory_diagnosis_probs = {
        "Never": [0.99, 0.01, 0.00],    # None, Asthma, COPD
        "Former": [0.05, 0.85, 0.10],
        "Current": [0.01, 0.03, 0.96],
    }
    respiratory_diagnosis = np.array([
        rng.choice(["None", "Asthma", "COPD"], p=respiratory_diagnosis_probs[s])
        for s in smoking_status
    ])
    copd = (respiratory_diagnosis == "COPD").astype(int)

    region = rng.choice(REGIONS, n_patients)
    insurance_type = np.array([
        rng.choice(INSURANCE_TYPES, p=REGION_INSURANCE_PROBS[r]) for r in region
    ])
    blood_type = rng.choice(BLOOD_TYPES, n_patients)
    referral_source = rng.choice(REFERRAL_SOURCES, n_patients, p=[0.4, 0.3, 0.2, 0.1])

    # diagnosis_category leans on the same severity factors that drive
    # mortality below, so diagnosis_category <-> mortality is associated.
    combined_severity = metabolic_severity + acute_severity
    diag_idx = np.clip(
        np.digitize(combined_severity, bins=[-1.0, -0.3, 0.3, 1.0]), 0, len(DIAGNOSIS_CATEGORIES) - 1
    )
    # nudge respiratory cases toward "Respiratory" using respiratory_risk
    diagnosis_category = np.array(DIAGNOSIS_CATEGORIES)[diag_idx]
    diagnosis_category = np.where(
        (respiratory_risk > 0.8) & (diagnosis_category == "Other"), "Respiratory", diagnosis_category
    )

    # --- Outcomes ---
    severity_score = acute_severity + 0.3 * metabolic_severity + 0.03 * (age - 60)
    mortality_p = 1 / (1 + np.exp(-(severity_score - 1.5)))
    icu_p = 1 / (1 + np.exp(-(severity_score - 0.8)))
    mortality = (rng.random(n_patients) < mortality_p).astype(int)
    icu_admission = (rng.random(n_patients) < icu_p).astype(int)

    # --- Length of stay + temporal cluster (admission/discharge/follow-up) ---
    length_of_stay_days = np.clip(
        3 + acute_severity * 2 + 0.05 * (age - 60) + rng.normal(0, 1.5, n_patients), 1, 60
    ).round().astype(int)

    admission_offsets = rng.integers(0, 730, n_patients)  # spread over ~2 years
    admission_date = pd.Timestamp("2022-01-01") + pd.to_timedelta(admission_offsets, unit="D")
    discharge_date = admission_date + pd.to_timedelta(length_of_stay_days, unit="D")
    followup_days = rng.integers(10, 60, n_patients)
    followup_date = discharge_date + pd.to_timedelta(followup_days, unit="D")

    return pd.DataFrame({
        "patient_id": patient_id,
        "age": age.round(1),
        "sex": sex,
        "region": region,
        "insurance_type": insurance_type,
        "blood_type": blood_type,
        "referral_source": referral_source,
        "height_cm": height_cm.round(1),
        "weight_kg": weight_kg.round(1),
        "bmi": bmi.round(2),
        "waist_circumference_cm": waist_circumference_cm.round(1),
        "systolic_bp": systolic_bp.round(1),
        "diastolic_bp": diastolic_bp.round(1),
        "glucose_mg_dl": glucose_mg_dl.round(1),
        "hba1c_percent": hba1c_percent.round(2),
        "ldl_cholesterol": ldl_cholesterol.round(1),
        "heart_rate": heart_rate.round(1),
        "temperature_c": temperature_c.round(2),
        "wbc_count": wbc_count.round(2),
        "crp_level": crp_level.round(2),
        "diabetes": diabetes,
        "hypertension": hypertension,
        "obesity": obesity,
        "current_smoker": current_smoker,
        "copd": copd,
        "smoking_status": smoking_status,
        "respiratory_diagnosis": respiratory_diagnosis,
        "diagnosis_category": diagnosis_category,
        "length_of_stay_days": length_of_stay_days,
        "admission_date": admission_date,
        "discharge_date": discharge_date,
        "followup_date": followup_date,
        "icu_admission": icu_admission,
        "mortality": mortality,
    })


# ---------------------------------------------------------------------------
# 2. Longitudinal dataset (repeated visits per patient, trending measurements)
# ---------------------------------------------------------------------------
def make_longitudinal_dataset(node_label: str = "L", n_patients: int = 150, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for i in range(1, n_patients + 1):
        patient_id = f"{node_label}_P{i:04d}"
        age = round(float(np.clip(rng.normal(60, 12), 18, 95)), 1)
        sex = rng.choice(["M", "F"])

        n_visits = int(rng.integers(3, 7))  # 3-6 visits
        baseline_weight = rng.normal(78, 12)
        baseline_glucose = rng.normal(100, 15)
        baseline_sbp = rng.normal(125, 12)

        # Per-patient trends (progression over time) -- this is what makes
        # measurement_change high enough for detect_dataset_type ->
        # "longitudinal".
        trend_weight = rng.normal(0.4, 0.6)
        trend_glucose = rng.normal(1.8, 1.2)
        trend_sbp = rng.normal(1.0, 1.2)

        visit_date = pd.Timestamp("2021-01-01") + pd.to_timedelta(int(rng.integers(0, 365)), unit="D")

        for v in range(n_visits):
            visit_date = visit_date + pd.to_timedelta(int(rng.integers(60, 121)), unit="D")
            rows.append({
                "patient_id": patient_id,
                "age": age,
                "sex": sex,
                "visit_number": v + 1,
                "visit_date": visit_date,
                "weight_kg": round(float(baseline_weight + trend_weight * v + rng.normal(0, 1.5)), 1),
                "glucose_mg_dl": round(float(baseline_glucose + trend_glucose * v + rng.normal(0, 5)), 1),
                "systolic_bp": round(float(baseline_sbp + trend_sbp * v + rng.normal(0, 5)), 1),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Event-based dataset (irregular event log, no trending measurements)
# ---------------------------------------------------------------------------
def make_event_based_dataset(node_label: str = "E", n_patients: int = 150, seed: int = 456) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for i in range(1, n_patients + 1):
        patient_id = f"{node_label}_P{i:04d}"
        age = round(float(np.clip(rng.normal(55, 15), 18, 95)), 1)
        sex = rng.choice(["M", "F"])
        # Constant per patient -> measurement_change stays low, so
        # detect_dataset_type doesn't mistake this for "longitudinal".
        baseline_severity = int(rng.integers(1, 5))

        n_events = int(rng.integers(1, 9))
        event_date = pd.Timestamp("2022-01-01") + pd.to_timedelta(int(rng.integers(0, 200)), unit="D")

        for _ in range(n_events):
            event_date = event_date + pd.to_timedelta(int(rng.integers(1, 90)), unit="D")
            rows.append({
                "patient_id": patient_id,
                "age": age,
                "sex": sex,
                "event_date": event_date,
                "event_type": rng.choice(EVENT_TYPES),
                "severity_score": baseline_severity,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Panel dataset (hospital x month metrics)
# ---------------------------------------------------------------------------
def make_panel_dataset(hospital_ids: list[int], n_months: int = 36, seed: int = 789) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for h in hospital_ids:
        hospital_id = f"Hospital_{h}"
        base_admissions = rng.integers(80, 200)
        admissions_trend = rng.normal(0.5, 1.0)
        base_los = rng.normal(5, 1)
        los_trend = rng.normal(0.02, 0.05)
        base_occupancy = rng.uniform(0.6, 0.85)
        base_mortality_rate = rng.uniform(0.02, 0.06)

        month_start = pd.Timestamp("2021-01-01")
        for m in range(n_months):
            month = month_start + pd.DateOffset(months=m)
            admissions = max(0, int(base_admissions + admissions_trend * m + rng.normal(0, 10)))
            avg_los_days = max(1.0, base_los + los_trend * m + rng.normal(0, 0.4))
            bed_occupancy_rate = float(np.clip(base_occupancy + 0.002 * m + rng.normal(0, 0.03), 0, 1))
            mortality_rate = float(np.clip(base_mortality_rate + rng.normal(0, 0.01), 0, 1))

            rows.append({
                "hospital_id": hospital_id,
                "month": month,
                "admissions": admissions,
                "avg_los_days": round(avg_los_days, 2),
                "bed_occupancy_rate": round(bed_occupancy_rate, 3),
                "mortality_rate": round(mortality_rate, 4),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Cross-sectional, multi-node hospital dataset — multi-group outcome
#    Identical structure to dataset4 except `mortality` (binary) is replaced
#    by `mortality_status` (4 ordered severity groups). The keyword "mortality"
#    still triggers the highest-priority outcome detection so the one-way
#    comparison + post-hoc pipeline fires for every significant numeric variable.
# ---------------------------------------------------------------------------
MORTALITY_STATUS_GROUPS = [
    "Mild_Recovery",      # low severity
    "Standard_Recovery",  # moderate severity
    "Complex_Recovery",   # high severity, survived
    "Deceased",           # very high severity
]
# Severity thresholds that partition severity_score into the four groups above.
# Chosen so that for ~N(0,1) severity each group has >20% of patients even
# under node-level severity shifts of ±0.4, keeping min_class_size=20 satisfied
# for node sizes down to ~150.
MORTALITY_STATUS_THRESHOLDS = [-0.5, 0.5, 1.5]


def make_hospital_dataset_5(node_label: str, n_patients: int, age_mean: float,
                            severity_shift: float, seed: int) -> pd.DataFrame:
    """Multi-node hospital dataset with 4-group mortality_status and rich cluster structure.

    Three tightly-correlated numeric clusters (cardiometabolic, anthropometric,
    acute-inflammation), two categorical cluster pairs (respiratory_diagnosis /
    respiratory_severity and mortality_status / clinical_outcome_group), and
    seven temporal variables all derived from admission_date ensure TOM clustering,
    PCA, MCA, correlation graphics, and missing-value heatmaps all render in
    every federated report.
    """
    rng = np.random.default_rng(seed)

    patient_id = [f"{node_label}_P{idx:04d}" for idx in range(1, n_patients + 1)]
    age = np.clip(rng.normal(age_mean, 14, n_patients), 18, 95)
    sex = rng.choice(["M", "F"], n_patients)

    metabolic_severity = rng.normal(0, 1, n_patients) + severity_shift + 0.02 * (age - age_mean)
    acute_severity = rng.normal(0, 1, n_patients) + 0.5 * severity_shift
    respiratory_risk = rng.normal(0, 1, n_patients)
    body_size = rng.normal(0, 1, n_patients)

    # --- Anthropometric cluster (body_size latent factor; tight noise → r ≥ 0.97) ---
    height_cm = np.clip(rng.normal(165, 9, n_patients), 140, 210)
    weight_kg = np.clip(60 + body_size * 12 + 0.15 * (height_cm - 165) + rng.normal(0, 2, n_patients), 40, 180)
    bmi = weight_kg / (height_cm / 100) ** 2
    waist_circumference_cm = np.clip(70 + body_size * 10 + rng.normal(0, 1.5, n_patients), 50, 150)
    hip_circumference_cm = np.clip(95 + body_size * 10 + rng.normal(0, 1.5, n_patients), 70, 160)
    body_fat_percent = np.clip(18 + body_size * 5 + rng.normal(0, 1.0, n_patients), 5, 60)

    # --- Cardiometabolic cluster (metabolic_severity latent factor; tight noise → r ≥ 0.97) ---
    systolic_bp = 115 + metabolic_severity * 15 + rng.normal(0, 2, n_patients)
    diastolic_bp = 75 + metabolic_severity * 9 + rng.normal(0, 1.5, n_patients)
    glucose_mg_dl = 95 + metabolic_severity * 25 + rng.normal(0, 3, n_patients)
    hba1c_percent = 5.4 + metabolic_severity * 1.0 + rng.normal(0, 0.12, n_patients)
    ldl_cholesterol = 100 + metabolic_severity * 18 + rng.normal(0, 2, n_patients)
    triglycerides = np.clip(100 + metabolic_severity * 30 + rng.normal(0, 2.5, n_patients), 30, None)
    insulin_resistance_index = np.clip(3.5 + metabolic_severity * 1.5 + rng.normal(0, 0.12, n_patients), 0.5, None)

    # --- Acute / inflammation cluster (acute_severity latent factor; tight noise → r ≥ 0.96) ---
    heart_rate = 75 + acute_severity * 10 + rng.normal(0, 1.5, n_patients)
    temperature_c = 36.8 + acute_severity * 0.5 + rng.normal(0, 0.08, n_patients)
    wbc_count = np.clip(7 + acute_severity * 2.5 + rng.normal(0, 0.3, n_patients), 1.5, None)
    crp_level = np.clip(5 + acute_severity * 8 + rng.normal(0, 1.0, n_patients), 0, None)
    ferritin_level = np.clip(50 + acute_severity * 100 + rng.normal(0, 5, n_patients), 10, None)
    procalcitonin = np.clip(
        0.05 + np.clip(acute_severity, 0, None) * 2.5 + rng.normal(0, 0.08, n_patients), 0.01, None
    )
    length_of_stay_days = np.clip(
        5 + acute_severity * 5 + 0.05 * (age - 60) + rng.normal(0, 1.0, n_patients), 1, 60
    ).round().astype(int)

    # --- Binary flags ---
    diabetes = (metabolic_severity + rng.normal(0, 0.5, n_patients) > 0.8).astype(int)
    hypertension = (metabolic_severity + rng.normal(0, 0.5, n_patients) > 0.5).astype(int)
    obesity = (bmi > 30).astype(int)
    current_smoker = (respiratory_risk + rng.normal(0, 0.5, n_patients) > 0.7).astype(int)

    # --- Categorical: smoking ↔ respiratory_diagnosis ↔ respiratory_severity (V ≈ 0.97) ---
    smoking_status = np.where(
        current_smoker == 1, "Current",
        np.where(respiratory_risk > -0.2, "Former", "Never"),
    )
    respiratory_diagnosis_probs = {
        "Never": [0.99, 0.01, 0.00],
        "Former": [0.05, 0.85, 0.10],
        "Current": [0.01, 0.03, 0.96],
    }
    respiratory_diagnosis = np.array([
        rng.choice(["None", "Asthma", "COPD"], p=respiratory_diagnosis_probs[s])
        for s in smoking_status
    ])
    # Deterministic mapping → V ≈ 1.0 so respiratory_diagnosis and respiratory_severity
    # cluster together via TOM (needed because shared-neighbour noise from smoking_status
    # requires adj > 0.94 to keep TOM distance below the 0.25 cut-off)
    respiratory_severity = np.where(
        respiratory_diagnosis == "None", "Mild",
        np.where(respiratory_diagnosis == "Asthma", "Moderate", "Severe")
    )
    copd = (respiratory_diagnosis == "COPD").astype(int)

    region = rng.choice(REGIONS, n_patients)
    insurance_type = np.array([
        rng.choice(INSURANCE_TYPES, p=REGION_INSURANCE_PROBS[r]) for r in region
    ])
    blood_type = rng.choice(BLOOD_TYPES, n_patients)
    referral_source = rng.choice(REFERRAL_SOURCES, n_patients, p=[0.4, 0.3, 0.2, 0.1])

    combined_severity = metabolic_severity + acute_severity
    diag_idx = np.clip(
        np.digitize(combined_severity, bins=[-1.0, -0.3, 0.3, 1.0]), 0, len(DIAGNOSIS_CATEGORIES) - 1
    )
    diagnosis_category = np.array(DIAGNOSIS_CATEGORIES)[diag_idx]
    diagnosis_category = np.where(
        (respiratory_risk > 0.8) & (diagnosis_category == "Other"), "Respiratory", diagnosis_category
    )

    # --- 4-group mortality_status (threshold on severity_score) ---
    severity_score = acute_severity + 0.3 * metabolic_severity + 0.03 * (age - age_mean)
    lo, mid, hi = MORTALITY_STATUS_THRESHOLDS
    mortality_status = np.select(
        [severity_score <= lo, severity_score <= mid, severity_score <= hi],
        MORTALITY_STATUS_GROUPS[:3],
        default=MORTALITY_STATUS_GROUPS[3],
    )
    # clinical_outcome_group: near-deterministic mapping from mortality_status.
    # Complex_Recovery and Deceased both map to Unfavorable so the 4-group →
    # 3-group V approaches 1.0, making adj > 0.94 and TOM distance < 0.25.
    _outcome_probs = {
        "Mild_Recovery":    (0.99, 0.01, 0.00),
        "Standard_Recovery":(0.01, 0.99, 0.00),
        "Complex_Recovery": (0.00, 0.02, 0.98),
        "Deceased":         (0.00, 0.01, 0.99),
    }
    clinical_outcome_group = np.array([
        rng.choice(["Favorable", "Moderate", "Unfavorable"], p=_outcome_probs[m])
        for m in mortality_status
    ])

    icu_p = 1 / (1 + np.exp(-(severity_score - 0.8)))
    icu_admission = (rng.random(n_patients) < icu_p).astype(int)

    # --- Temporal cluster: all derived from admission_date (r ≈ 1.0) ---
    admission_offsets = rng.integers(0, 730, n_patients)
    admission_date = pd.Timestamp("2022-01-01") + pd.to_timedelta(admission_offsets, unit="D")

    lab_delay = np.clip(rng.normal(1.5, 0.5, n_patients).round().astype(int), 0, 3)
    imaging_delay = np.clip(rng.normal(3.0, 1.0, n_patients).round().astype(int), 1, 7)
    medication_delay = np.clip(rng.normal(1.0, 0.3, n_patients).round().astype(int), 0, 2)
    specialist_delay = np.clip(rng.normal(5.0, 1.5, n_patients).round().astype(int), 2, 10)

    lab_date = admission_date + pd.to_timedelta(lab_delay, unit="D")
    imaging_date = admission_date + pd.to_timedelta(imaging_delay, unit="D")
    medication_start_date = admission_date + pd.to_timedelta(medication_delay, unit="D")
    specialist_consult_date = admission_date + pd.to_timedelta(specialist_delay, unit="D")
    discharge_date = admission_date + pd.to_timedelta(length_of_stay_days, unit="D")
    followup_days = rng.integers(10, 60, n_patients)
    followup_date = discharge_date + pd.to_timedelta(followup_days, unit="D")

    return pd.DataFrame({
        "patient_id": patient_id,
        "age": age.round(1),
        "sex": sex,
        "region": region,
        "insurance_type": insurance_type,
        "blood_type": blood_type,
        "referral_source": referral_source,
        "height_cm": height_cm.round(1),
        "weight_kg": weight_kg.round(1),
        "bmi": bmi.round(2),
        "waist_circumference_cm": waist_circumference_cm.round(1),
        "hip_circumference_cm": hip_circumference_cm.round(1),
        "body_fat_percent": body_fat_percent.round(1),
        "systolic_bp": systolic_bp.round(1),
        "diastolic_bp": diastolic_bp.round(1),
        "glucose_mg_dl": glucose_mg_dl.round(1),
        "hba1c_percent": hba1c_percent.round(2),
        "ldl_cholesterol": ldl_cholesterol.round(1),
        "triglycerides": triglycerides.round(1),
        "insulin_resistance_index": insulin_resistance_index.round(2),
        "heart_rate": heart_rate.round(1),
        "temperature_c": temperature_c.round(2),
        "wbc_count": wbc_count.round(2),
        "crp_level": crp_level.round(2),
        "ferritin_level": ferritin_level.round(1),
        "procalcitonin": procalcitonin.round(3),
        "length_of_stay_days": length_of_stay_days,
        "diabetes": diabetes,
        "hypertension": hypertension,
        "obesity": obesity,
        "current_smoker": current_smoker,
        "copd": copd,
        "icu_admission": icu_admission,
        "smoking_status": smoking_status,
        "respiratory_diagnosis": respiratory_diagnosis,
        "respiratory_severity": respiratory_severity,
        "diagnosis_category": diagnosis_category,
        "mortality_status": mortality_status,
        "clinical_outcome_group": clinical_outcome_group,
        "admission_date": admission_date,
        "lab_date": lab_date,
        "imaging_date": imaging_date,
        "medication_start_date": medication_start_date,
        "specialist_consult_date": specialist_consult_date,
        "discharge_date": discharge_date,
        "followup_date": followup_date,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    # 1. Cross-sectional, 3-node hospital dataset
    node_configs = [
        ("node1", 450, 55, 0.0, 1001),
        ("node2", 380, 65, 0.4, 2002),
        ("node3", 520, 50, -0.3, 3003),
    ]
    # Per-node column drops, layered on top of the full 34-column schema
    # produced by make_hospital_dataset, so the federated aggregator sees a
    # mix of common_all / common_partial / unique columns:
    #   - referral_source: only node1 collects it (unique).
    #   - ldl_cholesterol: missing from node2 (common_partial, numeric).
    #   - crp_level / followup_date: missing from node3 (common_partial,
    #     numeric and temporal respectively).
    node_column_drops = {
        "node1": [],
        "node2": ["referral_source", "ldl_cholesterol"],
        "node3": ["referral_source", "crp_level", "followup_date"],
    }
    for node_label, n_patients, age_mean, severity_shift, seed in node_configs:
        df = make_hospital_dataset(node_label, n_patients, age_mean, severity_shift, seed)
        df = df.drop(columns=node_column_drops[node_label])
        out_dir = DATA_DIR / "dataset4" / node_label
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"hospital_{node_label}.csv"
        df.to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({df.shape[0]} rows, {df.shape[1]} columns)")

    # 2. Longitudinal dataset, 2 nodes
    longitudinal_configs = [
        ("node1", "L1", 100, 123),
        ("node2", "L2", 80, 124),
    ]
    for node_label, patient_prefix, n_patients, seed in longitudinal_configs:
        df = make_longitudinal_dataset(patient_prefix, n_patients, seed)
        out_dir = DATA_DIR / "dataset4_longitudinal" / node_label
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"longitudinal_{node_label}.csv"
        df.to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({df.shape[0]} rows, {df.shape[1]} columns)")

    # 3. Event-based dataset, 2 nodes
    event_configs = [
        ("node1", "E1", 100, 456),
        ("node2", "E2", 80, 457),
    ]
    for node_label, patient_prefix, n_patients, seed in event_configs:
        df = make_event_based_dataset(patient_prefix, n_patients, seed)
        out_dir = DATA_DIR / "dataset4_event_based" / node_label
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"event_based_{node_label}.csv"
        df.to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({df.shape[0]} rows, {df.shape[1]} columns)")

    # 4. Panel dataset, 2 nodes (disjoint hospital sets)
    panel_configs = [
        ("node1", [1, 2, 3], 789),
        ("node2", [4, 5], 790),
    ]
    for node_label, hospital_ids, seed in panel_configs:
        df = make_panel_dataset(hospital_ids, seed=seed)
        out_dir = DATA_DIR / "dataset4_panel" / node_label
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"hospital_panel_{node_label}.csv"
        df.to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({df.shape[0]} rows, {df.shape[1]} columns)")

    # 5. Cross-sectional dataset with 4-group mortality_status, rich numeric /
    #    categorical / temporal clusters, and per-node correlated missing values.
    node_configs_5 = [
        ("node1", 450, 55, 0.0, 5001),
        ("node2", 380, 65, 0.4, 5002),
        ("node3", 520, 50, -0.3, 5003),
    ]
    # node2: drops referral_source and two cardiometabolic labs (incomplete panel)
    # node3: drops referral_source, three acute-inflammation labs, and two temporal
    #         columns (specialist consult and follow-up not collected at this site)
    node_column_drops_5 = {
        "node1": [],
        "node2": ["referral_source", "ldl_cholesterol", "triglycerides"],
        "node3": ["referral_source", "crp_level", "ferritin_level", "procalcitonin",
                  "specialist_consult_date", "followup_date"],
    }
    # Separate RNGs so missing-value injection is reproducible and independent
    # from the data-generation seeds.
    _rng5 = {
        "node1": np.random.default_rng(9001),
        "node2": np.random.default_rng(9002),
        "node3": np.random.default_rng(9003),
    }
    for node_label, n_patients, age_mean, severity_shift, seed in node_configs_5:
        df = make_hospital_dataset_5(node_label, n_patients, age_mean, severity_shift, seed)
        df = df.drop(columns=node_column_drops_5[node_label])
        rng_m = _rng5[node_label]

        # Inject correlated missing values so nullity patterns are correlated and
        # the missingno heatmap renders for every node.
        if node_label == "node1":
            # ~8 % of patients miss the full lab panel (correlated block)
            missed_lab = rng_m.random(n_patients) < 0.08
            for _col in ["glucose_mg_dl", "hba1c_percent", "ferritin_level", "insulin_resistance_index"]:
                if _col in df.columns:
                    df.loc[missed_lab, _col] = np.nan
            df.loc[rng_m.random(n_patients) < 0.03, "crp_level"] = np.nan
            df.loc[rng_m.random(n_patients) < 0.04, "waist_circumference_cm"] = np.nan

        elif node_label == "node2":
            # ~15 % of patients miss the vitals panel (correlated block)
            missed_vitals = rng_m.random(n_patients) < 0.15
            for _col in ["heart_rate", "temperature_c", "wbc_count"]:
                if _col in df.columns:
                    df.loc[missed_vitals, _col] = np.nan
            df.loc[rng_m.random(n_patients) < 0.06, "procalcitonin"] = np.nan
            df.loc[rng_m.random(n_patients) < 0.05, "insurance_type"] = np.nan

        elif node_label == "node3":
            # ~3 % of patients miss anthropometric measurements (correlated block)
            missed_anthro = rng_m.random(n_patients) < 0.03
            for _col in ["bmi", "waist_circumference_cm", "hip_circumference_cm", "body_fat_percent"]:
                if _col in df.columns:
                    df.loc[missed_anthro, _col] = np.nan
            df.loc[rng_m.random(n_patients) < 0.02, "weight_kg"] = np.nan
            df.loc[rng_m.random(n_patients) < 0.02, "height_cm"] = np.nan

        out_dir = DATA_DIR / "dataset5" / node_label
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"hospital_{node_label}.csv"
        df.to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({df.shape[0]} rows, {df.shape[1]} columns)")
        counts = df["mortality_status"].value_counts()
        print(f"  mortality_status groups: {counts.to_dict()}")


if __name__ == "__main__":
    main()
