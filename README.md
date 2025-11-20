# NHANES Heart Stability & Depression Analysis

This project analyses the relationship between cardiovascular stability and depressive symptoms using the NHANES 2017–2020 public-use datasets. It provides a reproducible Python pipeline that:

- loads and merges demographic (`DEMO`), depression (`DPQ`), medical history (`MCQ`), blood pressure (`BPX`), and body measurement (`BMX`) data,
- cleans and engineers features (including PHQ-9 scores and blood pressure summaries),
- runs descriptive/exploratory analysis with ready-made visualisations,
- trains modern machine learning models (histogram gradient boosting and random forest),
- exports metrics, trained models, and interpretability artefacts.

All stages are orchestrated from the command line so the workflow can be scripted or containerised easily.

## Prerequisites

- Python 3.10 or later
- NHANES 2017–2020 XPT files for the modules listed above placed under `data/`

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

All commands are run from the project root (`run.py` handles path setup for you).

### 1. Process the Raw Data

```bash
python run.py load-data
```

Creates `outputs/nhanes_processed.parquet` with merged, feature-engineered data.

### 2. Generate EDA Artefacts

```bash
python run.py run-eda
```

Produces figures in `outputs/figures/` (PHQ-9 distribution, depression prevalence by heart condition, numeric correlation heatmap) and a descriptive statistics table at `outputs/eda_summary.csv`.

### 3. Train & Evaluate Models

```bash
python run.py train-models
```

Trains the configured models, writes metrics to `outputs/model_results.json`, and stores each model (with confusion matrix, ROC curve, permutation importances, and raw importance CSV) under `outputs/models/<model_name>/`.

### 4. Evaluate a Saved Model

```bash
# Evaluate on the test split from training
python run.py evaluate randomforest

# Evaluate on a custom test dataset
python run.py evaluate histgradientboosting --test-data path/to/test_data.parquet

# Save evaluation results to a file
python run.py evaluate xgboost --output evaluation_results.json
```

Evaluates a saved model and prints metrics. Available models: `randomforest`, `histgradientboosting`, `xgboost`.

### 5. Run the Full Pipeline

```bash
python run.py run-all
```

Executes stages 1-3 sequentially.

## Outputs

- `outputs/nhanes_processed.parquet` — cleaned, feature-rich dataset ready for reuse.
- `outputs/figures/` — visualisations produced during EDA and evaluation.
- `outputs/eda_summary.csv` — tabular descriptive statistics for quick review.
- `outputs/model_results.json` — summary metrics (accuracy, precision, recall, F1) for each trained model.
- `outputs/models/<model_name>/` — persisted scikit-learn pipeline, metrics, confusion matrix, ROC curve, permutation importance plot, and feature importance CSV.
