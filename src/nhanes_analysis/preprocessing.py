from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

PHQ_ITEM_PREFIX = "DPQ"
PHQ_SCORE_COLUMNS = [f"{PHQ_ITEM_PREFIX}{code:03d}" for code in range(10, 100, 10)]

SYSTOLIC_COLS = ["BPXSY1", "BPXSY2", "BPXSY3"]
DIASTOLIC_COLS = ["BPXDI1", "BPXDI2", "BPXDI3"]

CATEGORICAL_MISSING_CODES = {7, 9, 77, 99}

SEX_MAP = {1.0: "Male", 2.0: "Female"}
RACE_MAP = {
    1.0: "Mexican American",
    2.0: "Other Hispanic",
    3.0: "Non-Hispanic White",
    4.0: "Non-Hispanic Black",
    6.0: "Non-Hispanic Asian",
    7.0: "Other Race",
}
EDUCATION_MAP = {
    1.0: "Less than 9th grade",
    2.0: "9-11th grade",
    3.0: "High school/GED",
    4.0: "Some college/AA",
    5.0: "College graduate",
}
MARITAL_MAP = {
    1.0: "Married",
    2.0: "Widowed",
    3.0: "Divorced",
    4.0: "Separated",
    5.0: "Never married",
    6.0: "Living with partner",
}
BOOLEAN_MAP = {1.0: "Yes", 2.0: "No"}


def compute_mean_bp(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean systolic and diastolic blood pressure across replicates."""

    def _row_mean(row: pd.Series, cols: Iterable[str]) -> float:
        available = [col for col in cols if col in row.index]
        if not available:
            return np.nan
        values = row[available]
        valid = values.dropna()
        return float(valid.mean()) if not valid.empty else np.nan

    df = df.copy()
    df["systolic_bp_mean"] = df.apply(lambda row: _row_mean(row, SYSTOLIC_COLS), axis=1)
    df["diastolic_bp_mean"] = df.apply(lambda row: _row_mean(row, DIASTOLIC_COLS), axis=1)
    cols_to_drop = [col for col in SYSTOLIC_COLS + DIASTOLIC_COLS if col in df.columns]
    return df.drop(columns=cols_to_drop)


def compute_phq_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Create PHQ-9 total score and binary depression flag."""

    df = df.copy()
    phq_columns = [col for col in df.columns if col.startswith(PHQ_ITEM_PREFIX)]
    df["phq9_total"] = df[phq_columns].sum(axis=1, min_count=len(phq_columns))
    df["depression_flag"] = np.where(df["phq9_total"] >= 10, 1, 0).astype(float)
    df.loc[df["phq9_total"].isna(), "depression_flag"] = np.nan
    return df


def normalise_categorical_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NHANES special codes with NaN before mapping to strings."""

    df = df.copy()
    categorical_cols = [
        "sex",
        "race_ethnicity",
        "education_level",
        "marital_status",
        "congestive_heart_failure",
        "coronary_heart_disease",
        "angina",
        "heart_attack",
        "stroke",
        "weak_kidneys",
    ]
    for col in categorical_cols:
        if col in df.columns:
            df.loc[df[col].isin(CATEGORICAL_MISSING_CODES), col] = np.nan
    return df


def apply_value_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map categorical codes to descriptive labels."""

    df = df.copy()
    if "sex" in df.columns:
        df["sex"] = df["sex"].map(SEX_MAP)
    if "race_ethnicity" in df.columns:
        df["race_ethnicity"] = df["race_ethnicity"].map(RACE_MAP)
    if "education_level" in df.columns:
        df["education_level"] = df["education_level"].map(EDUCATION_MAP)
    if "marital_status" in df.columns:
        df["marital_status"] = df["marital_status"].map(MARITAL_MAP)
    for col in [
        "congestive_heart_failure",
        "coronary_heart_disease",
        "angina",
        "heart_attack",
        "stroke",
        "weak_kidneys",
    ]:
        if col in df.columns:
            df[col] = df[col].map(BOOLEAN_MAP)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional engineered features for better prediction."""
    
    df = df.copy()
    
    # Age groups (cardiovascular risk increases with age)
    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 40, 50, 60, 70, 100],
            labels=["<40", "40-50", "50-60", "60-70", "70+"],
            include_lowest=True
        )
    
    # BMI categories (obesity is a major CV risk factor)
    if "bmi" in df.columns:
        df["bmi_category"] = pd.cut(
            df["bmi"],
            bins=[0, 18.5, 25, 30, 100],
            labels=["Underweight", "Normal", "Overweight", "Obese"],
            include_lowest=True
        )
    
    # Blood pressure categories (hypertension is a major CV risk factor)
    if "systolic_bp_mean" in df.columns and "diastolic_bp_mean" in df.columns:
        # Normal: <120/<80, Elevated: 120-129/<80, Stage 1: 130-139/80-89, Stage 2: >=140/>=90
        bp_category = pd.Series(np.nan, index=df.index, dtype=object)
        
        # Check for missing values first
        has_bp_data = df["systolic_bp_mean"].notna() & df["diastolic_bp_mean"].notna()
        
        if has_bp_data.any():
            sbp = df["systolic_bp_mean"]
            dbp = df["diastolic_bp_mean"]
            
            # Stage 2 (highest priority): SBP >= 140 OR DBP >= 90
            stage2_mask = has_bp_data & ((sbp >= 140) | (dbp >= 90))
            bp_category[stage2_mask] = "Stage2_Hypertension"
            
            # Stage 1: SBP 130-139 OR DBP 80-89 (and not Stage 2)
            stage1_mask = has_bp_data & ~stage2_mask & (((sbp >= 130) & (sbp < 140)) | ((dbp >= 80) & (dbp < 90)))
            bp_category[stage1_mask] = "Stage1_Hypertension"
            
            # Elevated: SBP 120-129 AND DBP < 80 (and not Stage 1 or 2)
            elevated_mask = has_bp_data & ~stage2_mask & ~stage1_mask & (sbp >= 120) & (sbp < 130) & (dbp < 80)
            bp_category[elevated_mask] = "Elevated"
            
            # Normal: SBP < 120 AND DBP < 80 (and not any of the above)
            normal_mask = has_bp_data & ~stage2_mask & ~stage1_mask & ~elevated_mask & (sbp < 120) & (dbp < 80)
            bp_category[normal_mask] = "Normal"
        
        df["bp_category"] = bp_category
    
    # Depression severity categories (more granular than binary)
    if "phq9_total" in df.columns:
        df["depression_severity"] = pd.cut(
            df["phq9_total"],
            bins=[-1, 4, 9, 14, 19, 27],
            labels=["None", "Mild", "Moderate", "Moderately_Severe", "Severe"],
            include_lowest=True
        )
    
    # Interaction: Age * Depression (older depressed individuals at higher CV risk)
    if "age" in df.columns and "depression_flag" in df.columns:
        df["age_depression_interaction"] = df["age"] * df["depression_flag"]
    
    # Interaction: BMI * Depression (obese depressed individuals at higher risk)
    if "bmi" in df.columns and "depression_flag" in df.columns:
        df["bmi_depression_interaction"] = df["bmi"] * df["depression_flag"]
    
    return df


def preprocess_merged(df: pd.DataFrame) -> pd.DataFrame:
    """Run all preprocessing steps on the merged dataset."""

    df = compute_mean_bp(df)
    df = compute_phq_scores(df)
    df = normalise_categorical_codes(df)
    df = apply_value_labels(df)
    df = engineer_features(df)
    return df


def build_analysis_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return final dataset filtered for complete target information."""

    df = preprocess_merged(df)
    analysis_df = df.dropna(subset=["phq9_total", "depression_flag"])
    return analysis_df

