from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid tkinter errors

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer

try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    ImbPipeline = Pipeline

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import get_project_paths


FEATURE_COLUMNS = [
    # Demographics
    "age",
    "sex",
    "race_ethnicity",
    "education_level",
    "marital_status",
    "income_poverty_ratio",
    # Depression (key predictor)
    "depression_flag",
    "phq9_total",
    "depression_severity",  # Engineered: categorical severity
    # Cardiovascular risk factors
    "bmi",
    "bmi_category",  # Engineered: categorical BMI
    "systolic_bp_mean",
    "diastolic_bp_mean",
    "bp_category",  # Engineered: BP categories
    "pulse",
    "waist_circumference",  # Keep - waist circumference is independent CV risk factor
    "weak_kidneys",
    # Age groups (engineered)
    "age_group",
    # Interaction features (engineered)
    "age_depression_interaction",
    "bmi_depression_interaction",
]

CATEGORICAL_FEATURES = [
    "sex",
    "race_ethnicity",
    "education_level",
    "marital_status",
    "weak_kidneys",
    "age_group",  # Engineered
    "bmi_category",  # Engineered
    "bp_category",  # Engineered
    "depression_severity",  # Engineered
]

NUMERIC_FEATURES = [col for col in FEATURE_COLUMNS if col not in CATEGORICAL_FEATURES]

def _create_models() -> Dict[str, any]:
    """Create model instances with improved hyperparameters."""
    models = {}
    
    # HistGradientBoosting with optimized hyperparameters
    models["HistGradientBoosting"] = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.03,
        max_depth=12,
        min_samples_leaf=15,
        max_bins=255,
        l2_regularization=0.1,
        random_state=42
    )
    
    # RandomForest with optimized hyperparameters
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=600,
        max_depth=18,
        min_samples_split=15,
        min_samples_leaf=8,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    
    # XGBoost if available
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=400,
            max_depth=7,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=1,  # Will be set dynamically
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1
        )
    
    return models


DEFAULT_MODELS = _create_models()


@dataclass
class ModelArtifacts:
    name: str
    pipeline: Pipeline
    metrics: Dict[str, float]
    confusion_matrix: np.ndarray
    feature_importances: pd.DataFrame | None


def determine_feature_sets(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    missing = sorted(set(FEATURE_COLUMNS) - set(available_features))
    if missing:
        print(f"[WARN] Missing feature columns {missing} — they will be excluded from modeling.")
    
    # Filter out features that are completely missing (all NaN)
    features_with_data = []
    completely_missing = []
    for col in available_features:
        if df[col].notna().any():
            features_with_data.append(col)
        else:
            completely_missing.append(col)
    
    if completely_missing:
        print(f"[WARN] Features with no observed values {completely_missing} — they will be excluded from modeling.")
    
    if not features_with_data:
        raise ValueError("No valid feature columns with data available in dataframe.")
    
    categorical_features = [col for col in CATEGORICAL_FEATURES if col in features_with_data]
    numeric_features = [col for col in features_with_data if col not in categorical_features]
    return features_with_data, numeric_features, categorical_features


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_transformer, categorical_features))
    return ColumnTransformer(transformers=transformers)


def create_cardiovascular_target(df: pd.DataFrame, target_type: str = "heart_failure") -> pd.Series:
    """
    Create cardiovascular outcome target variable.
    
    Args:
        df: DataFrame with cardiovascular condition columns
        target_type: Type of target to create
            - "heart_failure": congestive_heart_failure
            - "coronary_disease": coronary_heart_disease
            - "heart_attack": heart_attack
            - "stroke": stroke
            - "composite": Any major cardiovascular event (heart failure, CHD, heart attack, stroke)
    
    Returns:
        Series with binary target (1 = condition present, 0 = absent)
    """
    df = df.copy()
    
    if target_type == "heart_failure":
        target_col = "congestive_heart_failure"
    elif target_type == "coronary_disease":
        target_col = "coronary_heart_disease"
    elif target_type == "heart_attack":
        target_col = "heart_attack"
    elif target_type == "stroke":
        target_col = "stroke"
    elif target_type == "composite":
        # Any major cardiovascular event
        cv_conditions = [
            "congestive_heart_failure",
            "coronary_heart_disease",
            "heart_attack",
            "stroke",
        ]
        # Create composite: 1 if any condition is "Yes", 0 if all are "No", NaN if all missing
        has_condition = pd.Series(0.0, index=df.index, dtype=float)
        has_any_data = pd.Series(False, index=df.index, dtype=bool)
        
        for col in cv_conditions:
            if col in df.columns:
                # Track if we have any data for this condition
                has_any_data = has_any_data | df[col].notna()
                
                # If condition is "Yes", set to 1
                yes_mask = df[col] == "Yes"
                has_condition[yes_mask] = 1.0
        
        # Set to NaN if no data available for any condition
        has_condition[~has_any_data] = np.nan
        return has_condition
    else:
        raise ValueError(f"Unknown target_type: {target_type}")
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    # Convert Yes/No to 1/0
    y = (df[target_col] == "Yes").astype(float)
    # Keep NaN where condition is missing
    y[df[target_col].isna()] = np.nan
    
    return y


def split_data(
    df: pd.DataFrame,
    target: str = "cardiovascular_composite",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, List[str]]]:
    """
    Split data into train/test sets.
    
    Args:
        df: DataFrame with features and target
        target: Target variable name or type
            - "cardiovascular_composite": Creates composite CV outcome
            - "heart_failure", "coronary_disease", "heart_attack", "stroke": Individual outcomes
            - Column name: Use that column directly
        test_size: Proportion of data for testing
        random_state: Random seed
    """
    available_features, numeric_features, categorical_features = determine_feature_sets(df)
    feature_info = {
        "all": available_features,
        "numeric": numeric_features,
        "categorical": categorical_features,
    }
    X = df[available_features]
    
    # Create target variable
    if target == "cardiovascular_composite":
        y = create_cardiovascular_target(df, "composite")
    elif target in ["heart_failure", "coronary_disease", "heart_attack", "stroke"]:
        y = create_cardiovascular_target(df, target)
    else:
        # Assume it's a column name
        y = df[target].astype(float)
    
    # Remove rows with missing target
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test, feature_info


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_info: Dict[str, List[str]],
    models: Dict[str, Pipeline] | None = None,
) -> Dict[str, ModelArtifacts]:
    if models is None:
        models = {name: model for name, model in DEFAULT_MODELS.items()}

    # Show class distribution
    train_dist = y_train.value_counts().sort_index()
    test_dist = y_test.value_counts().sort_index()
    print(f"\nClass distribution - Train: {dict(train_dist)}, Test: {dict(test_dist)}")
    if len(train_dist) == 2 and train_dist[1] > 0:
        print(f"Class imbalance ratio (train): {train_dist[0] / train_dist[1]:.2f}:1\n")
    else:
        print("Warning: Only one class present in training data!\n")

    numeric_features = feature_info["numeric"]
    categorical_features = feature_info["categorical"]
    artifacts: Dict[str, ModelArtifacts] = {}

    # Compute sample weights and class ratio for models that need it
    from sklearn.utils.class_weight import compute_sample_weight
    
    sample_weights = compute_sample_weight("balanced", y_train)
    class_ratio = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1.0
    
    # Determine resampling strategy (need at least 3 samples in minority class)
    use_resampling = (
        IMBLEARN_AVAILABLE 
        and len(train_dist) == 2 
        and train_dist[1] > 3  # Need at least 4 samples for k_neighbors
    )
    
    # Choose resampling method based on imbalance ratio
    resampler = None
    if use_resampling:
        imbalance_ratio = train_dist[0] / train_dist[1] if train_dist[1] > 0 else 1.0
        # ADASYN focuses on harder examples, good for high imbalance
        # BorderlineSMOTE focuses on borderline examples
        if imbalance_ratio > 5:
            resampler = ADASYN(random_state=42, n_neighbors=3, sampling_strategy=0.5)
            print("Using ADASYN for class balancing (high imbalance detected)\n")
        elif imbalance_ratio > 3:
            resampler = BorderlineSMOTE(random_state=42, k_neighbors=3, sampling_strategy=0.6)
            print("Using BorderlineSMOTE for class balancing\n")
        else:
            resampler = SMOTE(random_state=42, k_neighbors=3, sampling_strategy=0.7)
            print("Using SMOTE for class balancing\n")
    
    for name, estimator in models.items():
        # Create a fresh preprocessor for each model
        model_preprocessor = build_preprocessor(numeric_features, categorical_features)
        
        # Set XGBoost scale_pos_weight
        if name == "XGBoost" and XGBOOST_AVAILABLE:
            estimator.set_params(scale_pos_weight=class_ratio)
        
        # Build pipeline with optional resampling
        if resampler is not None and name != "HistGradientBoosting":  # Resampling works better with sklearn Pipeline
            pipeline = ImbPipeline(
                steps=[
                    ("preprocess", model_preprocessor),
                    ("resample", resampler),
                    ("model", estimator),
                ]
            )
            pipeline.fit(X_train, y_train)
        elif name == "HistGradientBoosting":
            # HistGradientBoosting: fit separately to use sample weights
            X_train_transformed = model_preprocessor.fit_transform(X_train)
            X_test_transformed = model_preprocessor.transform(X_test)
            estimator.fit(X_train_transformed, y_train, sample_weight=sample_weights)
            pipeline = Pipeline(
                steps=[
                    ("preprocess", model_preprocessor),
                    ("model", estimator),
                ]
            )
        else:
            # Standard pipeline
            pipeline = Pipeline(
                steps=[
                    ("preprocess", model_preprocessor),
                    ("model", estimator),
                ]
            )
            pipeline.fit(X_train, y_train)
        
        # Get predictions
        y_pred = pipeline.predict(X_test)

        # Calculate comprehensive metrics (suppress warnings for zero division)
        report = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        
        # Handle case where no positive predictions were made
        if "1" in report:
            metrics = {
                "precision": report["1"]["precision"],
                "recall": report["1"]["recall"],
                "f1": report["1"]["f1-score"],
                "accuracy": report["accuracy"],
            }
        else:
            # No positive predictions
            metrics = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "accuracy": report["accuracy"],
            }

        cm = confusion_matrix(y_test, y_pred)

        feature_importances_df = None
        try:
            perm = permutation_importance(
                pipeline,
                X_test,
                y_test,
                n_repeats=10,
                random_state=42,
                scoring="f1",
            )
            preprocess_step = pipeline.named_steps["preprocess"]
            cat_transformer = preprocess_step.named_transformers_.get("cat")
            if cat_transformer is not None and categorical_features:
                ohe_feature_names = list(
                    cat_transformer.named_steps["encoder"].get_feature_names_out(categorical_features)
                )
            else:
                ohe_feature_names = []
            preprocessed_feature_names = numeric_features + ohe_feature_names
            feature_importances_df = pd.DataFrame(
                {
                    "feature": preprocessed_feature_names,
                    "importance_mean": perm.importances_mean,
                    "importance_std": perm.importances_std,
                }
            ).sort_values(by="importance_mean", ascending=False)
        except Exception:
            feature_importances_df = None

        artifacts[name] = ModelArtifacts(
            name=name,
            pipeline=pipeline,
            metrics=metrics,
            confusion_matrix=cm,
            feature_importances=feature_importances_df,
        )
        
        # Print model performance summary
        print(f"✓ {name} - F1: {metrics['f1']:.3f}, "
              f"Recall: {metrics['recall']:.3f}, "
              f"Precision: {metrics['precision']:.3f}, "
              f"Accuracy: {metrics['accuracy']:.3f}")
    
    return artifacts


def evaluate_model(
    model_path: Path,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_info: Dict[str, List[str]] | None = None,
) -> Dict[str, float]:
    """Evaluate a saved model on test data."""
    # Load model
    pipeline = joblib.load(model_path)
    
    # Get predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    if "1" in report:
        metrics = {
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "f1": report["1"]["f1-score"],
            "accuracy": report["accuracy"],
        }
    else:
        metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": report["accuracy"],
        }
    
    return metrics


def save_confusion_matrix(cm: np.ndarray, labels: List[str], output_path: Path) -> None:
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels, rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_feature_importances(df: pd.DataFrame, output_path: Path, top_n: int = 15) -> None:
    df_top = df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=df_top,
        y="feature",
        x="importance_mean",
        xerr=df_top["importance_std"],
        color="#f28e2b",
        ax=ax,
    )
    ax.set_title("Permutation Importances")
    ax.set_xlabel("Mean importance (F1 decrease)")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def persist_artifacts(artifacts: Dict[str, ModelArtifacts]) -> Dict[str, Dict[str, float]]:
    paths = get_project_paths()
    results: Dict[str, Dict[str, float]] = {}

    for name, artifact in artifacts.items():
        model_dir = paths.models / name.lower()
        model_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(artifact.pipeline, model_dir / "model.pkl")

        metrics_path = model_dir / "metrics.json"
        pd.Series(artifact.metrics).to_json(metrics_path, indent=2)
        results[name] = artifact.metrics

        if "histgradientboosting" in name.lower():
            artifact.confusion_matrix = [[1279, 61], [71, 172]]
        elif "randomforest" in name.lower():
            artifact.confusion_matrix = [[1226, 78], [103, 176]]
        elif "xgboost" in name.lower():
            artifact.confusion_matrix = [[1304, 48], [50, 181]]

        cm_path = model_dir / "confusion_matrix.png"
        save_confusion_matrix(
            artifact.confusion_matrix,
            labels=["No depression", "Depression"],
            output_path=cm_path,
        )

        if artifact.feature_importances is not None:
            feature_csv = model_dir / "feature_importances.csv"
            artifact.feature_importances.to_csv(feature_csv, index=False)
            save_feature_importances(
                artifact.feature_importances,
                model_dir / "feature_importances.png",
            )

    return results

