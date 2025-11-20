from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from .config import get_project_paths


DATA_FILES = {
    "DEMO": "P_DEMO.xpt",
    "DPQ": "P_DPQ.xpt",
    "MCQ": "P_MCQ.xpt",
    "BPX": "P_BPXO.xpt",
    "BMX": "P_BMX.xpt",
}

DEMO_COLS = [
    "SEQN",
    "RIDAGEYR",
    "RIAGENDR",
    "RIDRETH3",
    "DMDEDUC2",
    "DMDMARTZ",
    "INDFMPIR",
]

DPQ_COLS = ["SEQN"] + [f"DPQ0{i:02d}" for i in range(10, 100, 10)]

MCQ_COLS = [
    "SEQN",
    "MCQ160B",
    "MCQ160C",
    "MCQ160D",
    "MCQ160E",
    "MCQ160F",
    "MCQ160K",
]

BPX_COLS = [
    "SEQN",
    "BPXPLS",
    "BPXSY1",
    "BPXSY2",
    "BPXSY3",
    "BPXDI1",
    "BPXDI2",
    "BPXDI3",
]

BMX_COLS = [
    "SEQN",
    "BMXWT",
    "BMXBMI",
    "BMXWAIST",
]


RENAME_COLUMNS = {
    "RIDAGEYR": "age",
    "RIAGENDR": "sex",
    "RIDRETH3": "race_ethnicity",
    "DMDEDUC2": "education_level",
    "DMDMARTZ": "marital_status",
    "INDFMPIR": "income_poverty_ratio",
    "MCQ160B": "congestive_heart_failure",
    "MCQ160C": "coronary_heart_disease",
    "MCQ160D": "angina",
    "MCQ160E": "heart_attack",
    "MCQ160F": "stroke",
    "MCQ160K": "weak_kidneys",
    "BPXPLS": "pulse",
    "BMXWT": "weight_kg",
    "BMXBMI": "bmi",
    "BMXWAIST": "waist_circumference",
}


def load_table(
    path: Path,
    columns: Iterable[str] | None = None,
    *,
    strict: bool = True,
) -> pd.DataFrame:
    """Load a SAS transport file into a DataFrame."""

    df = pd.read_sas(path, format="xport", encoding="utf-8")
    if columns is not None:
        missing_cols = sorted(set(columns) - set(df.columns))
        if missing_cols and strict:
            raise KeyError(f"Columns not found in {path.name}: {missing_cols}")
        if missing_cols and not strict:
            print(f"[WARN] {path.name}: missing expected columns {missing_cols} — continuing with available columns.")
        available_cols = [col for col in columns if col in df.columns]
        df = df[available_cols]
    return df


def load_tables(data_dir: Path | None = None) -> Dict[str, pd.DataFrame]:
    """Load NHANES tables required for the analysis."""

    paths = get_project_paths()
    base_dir = data_dir or paths.data
    tables = {}
    for name, filename in DATA_FILES.items():
        file_path = base_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Missing expected NHANES file: {file_path}")
        if name == "DEMO":
            tables[name] = load_table(file_path, DEMO_COLS)
        elif name == "DPQ":
            tables[name] = load_table(file_path, DPQ_COLS)
        elif name == "MCQ":
            tables[name] = load_table(file_path, MCQ_COLS, strict=False)
        elif name == "BPX":
            tables[name] = load_table(file_path, BPX_COLS, strict=False)
        elif name == "BMX":
            tables[name] = load_table(file_path, BMX_COLS)
        else:
            tables[name] = load_table(file_path)
    return tables


def merge_tables(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge tables on SEQN and apply standard column renaming."""

    required = ["DEMO", "DPQ", "MCQ", "BPX", "BMX"]
    missing = [name for name in required if name not in tables]
    if missing:
        raise ValueError(f"Missing tables for merge: {missing}")

    merged = tables["DEMO"]
    for name in ["DPQ", "MCQ", "BPX", "BMX"]:
        merged = merged.merge(tables[name], on="SEQN", how="inner")

    merged = merged.rename(columns=RENAME_COLUMNS)
    return merged

