"""
evaluate.py
----------------
Evaluates trained models: accuracy, precision, recall, F1, ROC-AUC,
confusion matrix, and saves comparison plots.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def evaluate_models(fitted_models, data, dataset_name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X_test, y_test = data["X_test"], data["y_test"]

    results = []
    plt.figure(figsize=(7, 6))

    for name, model in fitted_models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)

        results.append({
            "Model": name, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1-Score": f1, "ROC-AUC": auc
        })

        print(f"\n--- {name} ({dataset_name}) ---")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Confusion matrix plot (one per model)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"{name} - Confusion Matrix\n({dataset_name})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        cm_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_{name.replace(' ', '_')}_confusion_matrix.png")
        plt.savefig(cm_path, dpi=120)
        plt.close()

        # Add to combined ROC plot
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(1, figsize=(7, 6))
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    # Finish combined ROC curve plot
    plt.figure(1)
    plt.plot([0, 1], [0, 1], "k--", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve Comparison - {dataset_name}")
    plt.legend()
    plt.tight_layout()
    roc_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_roc_comparison.png")
    plt.savefig(roc_path, dpi=120)
    plt.close("all")

    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    results_df.to_csv(os.path.join(OUTPUT_DIR, f"{dataset_name}_model_comparison.csv"), index=False)

    # Bar chart comparing all models across metrics
    metrics_melted = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    plt.figure(figsize=(9, 5))
    sns.barplot(data=metrics_melted, x="Metric", y="Score", hue="Model")
    plt.title(f"Model Comparison - {dataset_name}")
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{dataset_name}_metrics_comparison.png"), dpi=120)
    plt.close()

    return results_df
