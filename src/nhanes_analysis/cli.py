from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import data as data_module
from . import eda, modeling, preprocessing
from .config import get_project_paths


def stage_load(args: argparse.Namespace) -> None:
    paths = get_project_paths()
    tables = data_module.load_tables(paths.data)
    merged = data_module.merge_tables(tables)
    processed = preprocessing.preprocess_merged(merged)

    processed_path = paths.outputs / "nhanes_processed.parquet"
    processed.to_parquet(processed_path, index=False)
    print(f"Saved processed dataset to {processed_path}")


def stage_eda(args: argparse.Namespace) -> None:
    paths = get_project_paths()
    processed_path = paths.outputs / "nhanes_processed.parquet"
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {processed_path}. Run 'load-data' first."
        )
    df = pd.read_parquet(processed_path)

    essentials = df.dropna(subset=["phq9_total", "depression_flag"])
    print(f"Dataset rows used for EDA: {essentials.shape[0]}")

    fig_dir = paths.figures
    eda.plot_phq_distribution(essentials, fig_dir / "phq9_distribution.png")
    eda.plot_depression_by_condition(
        essentials,
        conditions=[
            "congestive_heart_failure",
            "coronary_heart_disease",
            "angina",
            "heart_attack",
        ],
        output_path=fig_dir / "depression_by_heart_condition.png",
    )
    eda.plot_numeric_correlation(essentials, fig_dir / "numeric_correlation_heatmap.png")
    essentials.describe(include="all").transpose().to_csv(paths.outputs / "eda_summary.csv")
    print(f"EDA artifacts written to {fig_dir} and {paths.outputs / 'eda_summary.csv'}")


def stage_train(args: argparse.Namespace) -> None:
    paths = get_project_paths()
    processed_path = paths.outputs / "nhanes_processed.parquet"
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {processed_path}. Run 'load-data' first."
        )
    df = pd.read_parquet(processed_path)
    
    # Filter for rows with depression data (needed as a feature)
    # The target (cardiovascular outcome) will be created in split_data
    analysis_df = df.dropna(subset=["phq9_total", "depression_flag"])

    # Get target type from args (default to composite)
    target_type = getattr(args, "target", "cardiovascular_composite")
    
    print(f"Training models to predict: {target_type}")
    X_train, X_test, y_train, y_test, feature_info = modeling.split_data(
        analysis_df, target=target_type
    )
    artifacts = modeling.train_models(X_train, y_train, X_test, y_test, feature_info)
    results = modeling.persist_artifacts(artifacts)

    results_path = paths.outputs / "model_results.json"
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print("Training complete. Metrics:")
    print(json.dumps(results, indent=2))
    print(f"Saved models and diagnostics under {paths.models}")


def stage_evaluate(args: argparse.Namespace) -> None:
    """Evaluate saved models on test data."""
    paths = get_project_paths()
    
    # Get model name
    model_name = args.model_name.lower()
    model_path = paths.models / model_name / "model.pkl"
    
    if not model_path.exists():
        available = [d.name for d in paths.models.iterdir() if d.is_dir()]
        raise FileNotFoundError(
            f"Model '{model_name}' not found at {model_path}. "
            f"Available models: {available}"
        )
    
    # Get target type from args or default to composite
    target_type = getattr(args, "target", "cardiovascular_composite")
    
    # Load test data
    if args.test_data:
        test_df = pd.read_parquet(args.test_data)
        # Ensure it has depression data (needed as feature)
        if "depression_flag" not in test_df.columns or "phq9_total" not in test_df.columns:
            raise ValueError(
                "Test data must include 'depression_flag' and 'phq9_total' columns"
            )
        # Create target variable
        y_test = modeling.create_cardiovascular_target(test_df, target_type.replace("cardiovascular_", ""))
        # Get available features (handles missing columns)
        available_features, _, _ = modeling.determine_feature_sets(test_df)
        X_test = test_df[available_features]
        # Remove rows with missing target
        valid_mask = y_test.notna()
        X_test = X_test[valid_mask]
        y_test = y_test[valid_mask].astype(int)
    else:
        # Use the same processed data and split (with same random seed)
        processed_path = paths.outputs / "nhanes_processed.parquet"
        if not processed_path.exists():
            raise FileNotFoundError(
                f"Processed dataset not found at {processed_path}. Run 'load-data' first."
            )
        df = pd.read_parquet(processed_path)
        # Filter for rows with depression data (needed as feature)
        analysis_df = df.dropna(subset=["phq9_total", "depression_flag"])
        # Create test split with same target type
        _, X_test, _, y_test, _ = modeling.split_data(analysis_df, target=target_type)
    
    # Evaluate
    print(f"Evaluating {model_name} on {len(X_test)} samples...")
    metrics = modeling.evaluate_model(model_path, X_test, y_test)
    
    # Print results
    print(f"\n=== Evaluation Results for {model_name} ===")
    print(f"F1 Score: {metrics['f1']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to {output_path}")


def stage_all(args: argparse.Namespace) -> None:
    stage_load(args)
    stage_eda(args)
    stage_train(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NHANES heart stability and depression analysis pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    load_parser = subparsers.add_parser("load-data", help="Load raw data and produce processed dataset")
    load_parser.set_defaults(func=stage_load)

    eda_parser = subparsers.add_parser("run-eda", help="Generate descriptive analyses and plots")
    eda_parser.set_defaults(func=stage_eda)

    train_parser = subparsers.add_parser("train-models", help="Train ML models and save metrics")
    train_parser.add_argument(
        "--target",
        type=str,
        default="cardiovascular_composite",
        choices=["cardiovascular_composite", "heart_failure", "coronary_disease", "heart_attack", "stroke"],
        help="Target cardiovascular outcome to predict (default: cardiovascular_composite)"
    )
    train_parser.set_defaults(func=stage_train)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a saved model on test data")
    eval_parser.add_argument(
        "model_name",
        help="Name of the model to evaluate (e.g., 'randomforest', 'histgradientboosting', 'xgboost')"
    )
    eval_parser.add_argument(
        "--test-data",
        type=str,
        help="Path to test dataset (parquet file). If not provided, uses the test split from training."
    )
    eval_parser.add_argument(
        "--target",
        type=str,
        default="cardiovascular_composite",
        choices=["cardiovascular_composite", "heart_failure", "coronary_disease", "heart_attack", "stroke"],
        help="Target cardiovascular outcome that the model was trained to predict (default: cardiovascular_composite)"
    )
    eval_parser.add_argument(
        "--output",
        type=str,
        help="Path to save evaluation results (JSON file)"
    )
    eval_parser.set_defaults(func=stage_evaluate)

    all_parser = subparsers.add_parser("run-all", help="Execute the full pipeline sequentially")
    all_parser.set_defaults(func=stage_all)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

