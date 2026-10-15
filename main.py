"""
main.py
==================================================
Disease Prediction from Medical Data - End-to-End Pipeline
==================================================
Runs, for EACH of the 3 datasets (Breast Cancer, Heart Disease, Diabetes):
    1. Load data
    2. Preprocess (clean, impute, split, scale)
    3. Train 4 classifiers: Logistic Regression, Random Forest, SVM, XGBoost
    4. Evaluate (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, ROC curve)
    5. Save the best model (highest ROC-AUC) to models/

Run with:
    python main.py
"""

import os
import joblib
import pandas as pd

from src.data_loader import load_dataset
from src.preprocess import split_and_scale
from src.train import train_models
from src.evaluate import evaluate_models

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DATASETS_TO_RUN = ["breast_cancer", "heart", "diabetes"]


def run_pipeline(dataset_name):
    print("\n" + "=" * 60)
    print(f" DATASET: {dataset_name.upper()}")
    print("=" * 60)

    df, target_col = load_dataset(dataset_name)
    print(f"Data shape: {df.shape} | Target column: '{target_col}'")

    data = split_and_scale(df, target_col)
    print(f"Train size: {len(data['y_train'])} | Test size: {len(data['y_test'])}")

    fitted_models = train_models(data)
    print(f"Trained models: {list(fitted_models.keys())}")

    results_df = evaluate_models(fitted_models, data, dataset_name)
    print(f"\nResults for {dataset_name}:")
    print(results_df.to_string(index=False))

    # Save the best model (by ROC-AUC) + its scaler/imputer for later inference
    os.makedirs(MODELS_DIR, exist_ok=True)
    best_model_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_model_name]

    bundle = {
        "model": best_model,
        "model_name": best_model_name,
        "scaler": data["scaler"],
        "imputer": data["imputer"],
        "feature_names": data["feature_names"],
        "target_col": target_col,
    }
    model_path = os.path.join(MODELS_DIR, f"{dataset_name}_best_model.joblib")
    joblib.dump(bundle, model_path)
    print(f"Saved best model ('{best_model_name}') -> {model_path}")

    return results_df


def main():
    all_results = {}
    for name in DATASETS_TO_RUN:
        all_results[name] = run_pipeline(name)

    print("\n" + "#" * 60)
    print(" FINAL SUMMARY (best model per dataset)")
    print("#" * 60)
    for name, df in all_results.items():
        best = df.iloc[0]
        print(f"{name:15s} -> Best: {best['Model']:20s} "
              f"| Accuracy={best['Accuracy']:.3f} | ROC-AUC={best['ROC-AUC']:.3f}")

    print("\nAll plots & CSV comparisons saved in the 'outputs/' folder.")
    print("All trained models saved in the 'models/' folder.")


if __name__ == "__main__":
    main()
