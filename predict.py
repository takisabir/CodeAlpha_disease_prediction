"""
predict.py
----------------
Load a saved model bundle and predict disease risk for a new patient.

Example usage:
    python predict.py --dataset breast_cancer
    python predict.py --dataset heart
    python predict.py --dataset diabetes

This demo version predicts on the first row of the test-shaped synthetic
sample data just to prove the saved model works end-to-end. In a real
deployment, replace `sample_input` with actual patient values in the
same order as `feature_names` (printed below).
"""

import argparse
import os
import joblib
import numpy as np

from src.data_loader import load_dataset

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def predict_for_dataset(dataset_name):
    model_path = os.path.join(MODELS_DIR, f"{dataset_name}_best_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No saved model found at {model_path}. Run `python main.py` first."
        )

    bundle = joblib.load(model_path)
    model = bundle["model"]
    scaler = bundle["scaler"]
    imputer = bundle["imputer"]
    feature_names = bundle["feature_names"]

    print(f"Loaded '{bundle['model_name']}' model for '{dataset_name}'")
    print(f"Expected feature order: {feature_names}")

    # Demo: grab one real row from the dataset (excluding the target) to show it works
    df, target_col = load_dataset(dataset_name)
    sample_row = df.drop(columns=[target_col]).iloc[[0]]
    actual_label = df[target_col].iloc[0]

    X_imputed = imputer.transform(sample_row)
    X_scaled = scaler.transform(X_imputed)

    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0][1] if hasattr(model, "predict_proba") else None

    print(f"\nSample patient features:\n{sample_row.to_dict(orient='records')[0]}")
    print(f"\nPredicted class: {pred} (1 = disease present, 0 = no disease)")
    if proba is not None:
        print(f"Predicted probability of disease: {proba:.3f}")
    print(f"Actual label (ground truth): {actual_label}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="breast_cancer",
        choices=["breast_cancer", "heart", "diabetes"],
        help="Which trained model to use for prediction"
    )
    args = parser.parse_args()
    predict_for_dataset(args.dataset)
