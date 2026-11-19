"""Retrain the final attrition model of project 4 and save it with its metadata."""

import argparse
import json
import platform
from pathlib import Path

import catboost
import joblib
import numpy as np
import pandas as pd
import sklearn
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from oc_p05_attrition_ml_deployment.features import TARGET, build_inputs
from oc_p05_attrition_ml_deployment.model import (
    BINARY_COLUMNS,
    NOMINAL_COLUMNS,
    NUMERIC_COLUMNS,
    PIPELINE_COLUMNS,
    TRAVEL_COLUMNS,
    build_pipeline,
)

# Values from the final model of the project 4 notebook (02_modeling.ipynb)
MODEL_VERSION = "1.0.0"
RANDOM_STATE = 42
TEST_SIZE = 0.2
THRESHOLD = 0.2524600136446524
CATBOOST_PARAMS = {
    "auto_class_weights": "SqrtBalanced",
    "depth": 3,
    "iterations": 978,
    "l2_leaf_reg": 3.1900416724175513,
    "learning_rate": 0.01471005003927122,
}

EXPECTED_SPLIT = {"train": (1176, 190), "test": (294, 47)}


def parse_args() -> argparse.Namespace:
    """Read the command line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/employees_cleaned.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    return parser.parse_args()


def check_columns(X: pd.DataFrame) -> None:
    """Check that the four column groups cover exactly the columns of X."""
    groups = [NUMERIC_COLUMNS, TRAVEL_COLUMNS, BINARY_COLUMNS, NOMINAL_COLUMNS]
    grouped = [column for group in groups for column in group]
    assert len(grouped) == len(set(grouped)), "A column belongs to more than one group"
    missing = sorted(set(X.columns) - set(grouped))
    unknown = sorted(set(grouped) - set(X.columns))
    assert not missing and not unknown, (
        f"Groups and data differ. Missing from the groups: {missing}. Not in the data: {unknown}"
    )
    assert len(grouped) == len(PIPELINE_COLUMNS) == 27, f"Expected 27 columns, got {len(grouped)}"


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Build the inputs and the target, then split them as in the notebook."""
    X = build_inputs(df)
    check_columns(X)
    y = (df[TARGET] == "Oui").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    sizes = {
        "train": (len(X_train), int(y_train.sum())),
        "test": (len(X_test), int(y_test.sum())),
    }
    assert sizes == EXPECTED_SPLIT, (
        f"The split differs from project 4 (rows, leavers): got {sizes}, expected {EXPECTED_SPLIT}"
    )
    return X_train, X_test, y_train, y_test


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float | int]:
    """Return the test metrics at the alert threshold."""
    proba = pipeline.predict_proba(X_test)[:, 1]
    predicted = (proba >= THRESHOLD).astype(int)
    return {
        "pr_auc": round(float(average_precision_score(y_test, proba)), 3),
        "precision": round(float(precision_score(y_test, predicted)), 3),
        "recall": round(float(recall_score(y_test, predicted)), 3),
        "leavers_found": int(((predicted == 1) & (y_test == 1)).sum()),
        "leavers_total": int(y_test.sum()),
    }


def build_metadata(metrics: dict[str, float | int]) -> dict:
    """Return the metadata saved next to the pipeline, with a fixed key order."""
    return {
        "model_version": MODEL_VERSION,
        "threshold": THRESHOLD,
        "catboost_params": CATBOOST_PARAMS,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "pipeline_columns": PIPELINE_COLUMNS,
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "catboost": catboost.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "test_metrics": metrics,
    }


def main() -> None:
    """Train the pipeline, print the test metrics and save the files."""
    args = parse_args()
    df = pd.read_parquet(args.data)
    X_train, X_test, y_train, y_test = split_data(df)

    model = CatBoostClassifier(
        **CATBOOST_PARAMS, verbose=0, random_state=RANDOM_STATE, allow_writing_files=False
    )
    pipeline = build_pipeline(model).fit(X_train, y_train)
    metrics = evaluate(pipeline, X_test, y_test)

    print(f"PR-AUC: {metrics['pr_auc']}")
    print(f"Precision at {THRESHOLD:.4f}: {metrics['precision']}")
    print(f"Recall at {THRESHOLD:.4f}: {metrics['recall']}")
    print(f"Leavers found: {metrics['leavers_found']} of {metrics['leavers_total']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.output_dir / "attrition_pipeline.joblib")
    metadata_path = args.output_dir / "attrition_metadata.json"
    metadata_path.write_text(
        json.dumps(build_metadata(metrics), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
