from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid tkinter errors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def save_plot(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_phq_distribution(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots()
    sns.histplot(df["phq9_total"], kde=True, ax=ax, bins=15, color="#4e79a7")
    ax.set_title("Distribution of PHQ-9 Total Scores")
    ax.set_xlabel("PHQ-9 total score")
    ax.set_ylabel("Count")
    save_plot(fig, output_path)


def plot_depression_by_condition(
    df: pd.DataFrame,
    conditions: Iterable[str],
    output_path: Path,
) -> None:
    melt_df = (
        df.melt(
            id_vars=["SEQN", "depression_flag"],
            value_vars=list(conditions),
            var_name="condition",
            value_name="diagnosis",
        )
        .dropna(subset=["diagnosis"])
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=melt_df,
        x="condition",
        y="depression_flag",
        hue="diagnosis",
        estimator=np.mean,
        ax=ax,
    )
    ax.set_ylabel("Proportion with depression (PHQ-9 >= 10)")
    ax.set_xlabel("Cardiovascular condition")
    ax.set_title("Depression prevalence by heart condition status")
    ax.legend(title="Diagnosed")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20)
    save_plot(fig, output_path)


def plot_numeric_correlation(df: pd.DataFrame, output_path: Path) -> None:
    numeric_cols = df.select_dtypes(include=np.number).columns
    corr_matrix = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation heatmap (numeric features)")
    save_plot(fig, output_path)

