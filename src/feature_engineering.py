"""
feature_engineering.py
-----------------------
Feature preparation module for the TB treatment dropout prediction model.
Simulates the output of the SQL queries defined in 02_analytical_queries.sql.

ENROLLMENT-ONLY MODEL — LEAKAGE-FREE DESIGN:
All ML features are restricted to information available at patient registration
(day 0 of treatment). Visit-derived behavioral features (adherence scores,
attendance rates, side-effect counts, weight change) are EXCLUDED because:

1. Even when windowed to the first 2 months, these features leak the outcome:
   patients who default during that window already show lower adherence,
   making avg_adherence trivially predictive of the very event it encodes.
2. A real-world early-warning system must generate risk scores at enrollment,
   before any behavioral data is collected.

FEATURES USED: demographics, distance, household size, clinical baseline
(HIV, diabetes, alcohol, smoking, TB type, drug resistance), treatment
assignment (regimen, DOT method, supporter), and accessibility_score
(a composite structural barrier index; Horter et al., 2020).

Methodological reference: Alene et al. (2024), Scientific Reports.
"""

PREDICTION_WINDOW_MONTHS = 2  # Months used to compute behavioral visit features

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════
# 1. LOAD AND MERGE (equivalent to the patient_analytics SQL VIEW)
# ══════════════════════════════════════════════════════════════

def load_and_merge(data_dir: str = "../data/raw") -> pd.DataFrame:
    import os
    df_pat  = pd.read_csv(os.path.join(data_dir, "patients.csv"),
                           parse_dates=["registration_date"])
    df_trt  = pd.read_csv(os.path.join(data_dir, "treatments.csv"),
                           parse_dates=["start_date", "end_date"])
    df_vis  = pd.read_csv(os.path.join(data_dir, "visits.csv"),
                           parse_dates=["visit_date"])
    df_res  = pd.read_csv(os.path.join(data_dir, "results.csv"),
                           parse_dates=["result_date"])

    print(f"  Patients:    {len(df_pat):>5,}")
    print(f"  Treatments:  {len(df_trt):>5,}")
    print(f"  Visits:      {len(df_vis):>5,}")
    print(f"  Lab results: {len(df_res):>5,}")

    # ANTI-LEAKAGE: use only the first N months of visits
    df_vis_window = df_vis[df_vis["visit_month"] <= PREDICTION_WINDOW_MONTHS].copy()

    visit_agg = df_vis_window.groupby("patient_id").agg(
        visits_in_window      = ("visit_id", "count"),
        visits_attended       = ("attendance", "sum"),
        avg_adherence         = ("adherence_score", "mean"),
        n_severe_side_effects = ("side_effects",
                                  lambda x: (x.isin(["Moderate", "Severe"])).sum()),
        n_social_worker       = ("social_worker_visit", "sum"),
        weight_start          = ("weight_kg", "first"),
        weight_end            = ("weight_kg", "last"),
    ).reset_index()

    visit_agg["visits_missed"] = visit_agg["visits_in_window"] - visit_agg["visits_attended"]
    visit_agg["attendance_rate_pct"] = (
        visit_agg["visits_attended"] / visit_agg["visits_in_window"].replace(0, 1) * 100
    ).round(2)
    visit_agg["weight_gain_kg"] = (
        visit_agg["weight_end"] - visit_agg["weight_start"]
    ).round(2)

    res_baseline = df_res[df_res["test_month"] == 0][
        ["patient_id", "smear_result", "xray_result", "cd4_count", "viral_load"]
    ].rename(columns={"smear_result": "smear_baseline",
                      "xray_result":  "xray_baseline"}
    ).drop_duplicates("patient_id")

    res_month2 = df_res[df_res["test_month"] == 2][
        ["patient_id", "smear_result"]
    ].rename(columns={"smear_result": "smear_month2"}).drop_duplicates("patient_id")

    df = (df_pat
          .merge(df_trt, on="patient_id", how="left")
          .merge(visit_agg, on="patient_id", how="left")
          .merge(res_baseline, on="patient_id", how="left")
          .merge(res_month2, on="patient_id", how="left"))

    df = df[~df["outcome"].isin(["Still on Treatment", "Transferred Out", None])].copy()
    df = df[df["outcome"].notna()].copy()

    print(f"\n  Merged dataset: {df.shape[0]:,} patients x {df.shape[1]} columns")
    return df


# ══════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TARGET
    df["dropout_label"] = (df["outcome"] == "Defaulted").astype(int)

    # Age groups
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 25, 35, 50, 65, 120],
        labels=["<25", "25-35", "35-50", "50-65", ">65"]
    ).astype(str)

    # Distance flags
    df["far_from_clinic"] = (df["distance_km"] > 20).astype(int)
    df["very_far"]        = (df["distance_km"] > 40).astype(int)

    # HIV interactions
    df["hiv_positive"] = (df["hiv_status"] == "Positive").astype(int)
    df["hiv_and_far"]  = df["hiv_positive"] * df["far_from_clinic"]

    # Expected treatment duration
    df["expected_months"] = df["regimen"].map({
        "2HRZE/4HR": 6, "2HRZE/4HR3": 6, "HRZE/HRE": 8,
        "BPaL": 18, "BDQ/DLM/LZD": 24, "Other": 9
    }).fillna(6)

    # Social support composite score (enrollment features only)
    df["support_score"] = (
        (df["supporter"] == "CHW").astype(int) * 3 +
        (df["supporter"] == "Family").astype(int) * 2 +
        (df["supporter"] == "Clinic").astype(int) * 1 +
        (df["dot_method"] == "In-person").astype(int) * 2
    )

    # Binary risk factor flags
    df["heavy_alcohol"]  = (df["alcohol_use"] == "Heavy").astype(int)
    df["current_smoker"] = (df["smoking"] == "Current").astype(int)
    df["unemployed"]     = (df["employment_status"] == "Unemployed").astype(int)
    df["no_education"]   = (df["education_level"] == "No Formal").astype(int)
    df["mdr_xdr"]        = (df["drug_resistance"].isin(["MDR", "XDR"])).astype(int)

    # Days on treatment — excluded from features (proxy of outcome)
    df["days_on_treatment"] = (
        pd.to_datetime(df["end_date"]) - pd.to_datetime(df["start_date"])
    ).dt.days.fillna(0)

    # Seasonality
    df["start_year"]    = pd.to_datetime(df["start_date"]).dt.year
    df["start_quarter"] = pd.to_datetime(df["start_date"]).dt.quarter

    # ── Accessibility score (Horter et al., 2020; Datiko et al., 2022) ──
    # Composite feature: distance × DOT method penalty × support factor.
    # Captures structural access barriers beyond linear distance alone.
    dot_penalty = df["dot_method"].map({
        "In-person": 1.0,
        "Video DOT": 1.3,
        "Self-administered": 1.7,
    }).fillna(1.5)
    support_factor = df["supporter"].map({
        "CHW":    0.7,
        "Family": 0.85,
        "Clinic": 1.0,
        "None":   1.3,
    }).fillna(1.0)
    df["accessibility_score"] = (
        df["distance_km"] * dot_penalty * support_factor
    ).round(2)

    return df


# ══════════════════════════════════════════════════════════════
# 3. FINAL ML DATASET PREPARATION
# ══════════════════════════════════════════════════════════════

# ENROLLMENT-ONLY features: all known at patient registration (day 0).
# EXCLUDED (post-enrollment / visit-derived):
#   visits_in_window, visits_attended, visits_missed, attendance_rate_pct,
#   avg_adherence, n_severe_side_effects, n_social_worker, weight_gain_kg,
#   low_attendance, smear_positive_m2, high_toxicity.
# days_on_treatment and dropout_month also excluded: proxies of the outcome.
# accessibility_score included: structural composite (Horter et al., 2020).
NUMERIC_FEATURES = [
    "age", "distance_km", "household_contacts",
    "expected_months", "accessibility_score",
]

BINARY_FEATURES = [
    "diabetes", "previous_tb",
    "hiv_positive", "far_from_clinic", "very_far", "hiv_and_far",
    "heavy_alcohol", "current_smoker", "unemployed",
    "no_education", "mdr_xdr",
]

CATEGORICAL_FEATURES = [
    "sex", "region", "education_level", "employment_status",
    "hiv_status", "alcohol_use", "smoking", "tb_type",
    "drug_resistance", "regimen", "supporter", "dot_method",
    "age_group",
]


def prepare_ml_dataset(df: pd.DataFrame,
                       encode_categoricals: bool = True) -> tuple:
    df = df.copy()

    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    for col in BINARY_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    encoders = {}
    if encode_categoricals:
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                le = LabelEncoder()
                df[col + "_enc"] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
        cat_encoded = [c + "_enc" for c in CATEGORICAL_FEATURES if c in df.columns]
    else:
        cat_encoded = []

    all_features = (
        [f for f in NUMERIC_FEATURES if f in df.columns] +
        [f for f in BINARY_FEATURES if f in df.columns] +
        cat_encoded
    )

    X = df[all_features].values
    y = df["dropout_label"].values

    print(f"  Feature matrix: {X.shape[0]:,} patients x {X.shape[1]} features")
    print(f"    Numeric:      {len([f for f in NUMERIC_FEATURES if f in df.columns])}")
    print(f"    Binary:       {len([f for f in BINARY_FEATURES if f in df.columns])}")
    print(f"    Categorical:  {len(cat_encoded)} (encoded)")
    print(f"\n  Target (dropout_label):")
    print(f"    Defaulted:  {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"    Completed:  {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")

    return X, y, all_features, encoders, df


if __name__ == "__main__":
    import sys, os
    sys.path.append("..")
    print("Feature engineering module — self-test:")
    print("=" * 50)
    df = load_and_merge("../data/raw")
    df = engineer_features(df)
    X, y, features, encoders, df_ml = prepare_ml_dataset(df)
    print("\n✅ Feature pipeline OK")
