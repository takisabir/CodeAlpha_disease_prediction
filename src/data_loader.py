"""
data_loader.py
----------------
Loads the three disease datasets used in this project:
    1. Breast Cancer  -> loaded directly from scikit-learn (real UCI data, built-in)
    2. Heart Disease   -> loaded from data/heart.csv if present, else a realistic
                          synthetic dataset (same feature schema as the UCI
                          Cleveland Heart Disease dataset) is generated so the
                          project runs out-of-the-box.
    3. Diabetes         -> loaded from data/diabetes.csv if present, else a
                          realistic synthetic dataset (same feature schema as the
                          UCI Pima Indians Diabetes dataset) is generated.

Why synthetic fallback?
    This sandbox has no internet access to download the real Kaggle/UCI CSV
    files. The code is written so that the moment you place the REAL
    'heart.csv' / 'diabetes.csv' (downloaded from Kaggle/UCI) inside the
    data/ folder, this loader will automatically pick up the real data
    instead of the synthetic one. See README.md for download links.
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

RANDOM_STATE = 42
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_breast_cancer_data():
    """Real UCI Breast Cancer Wisconsin dataset, shipped with scikit-learn."""
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    # sklearn encodes target as 1=benign, 0=malignant -> flip so 1 = disease present (malignant)
    df["target"] = df["target"].map({0: 1, 1: 0})
    return df, "target"


def _generate_synthetic_heart(n=1000, seed=RANDOM_STATE):
    """Synthetic data mimicking the UCI Cleveland Heart Disease schema."""
    rng = np.random.default_rng(seed)
    age = rng.integers(29, 77, n)
    sex = rng.integers(0, 2, n)                       # 1 = male, 0 = female
    cp = rng.integers(0, 4, n)                        # chest pain type
    trestbps = rng.normal(131, 17, n).clip(94, 200)    # resting blood pressure
    chol = rng.normal(246, 51, n).clip(126, 564)       # cholesterol
    fbs = rng.binomial(1, 0.15, n)                     # fasting blood sugar > 120
    restecg = rng.integers(0, 3, n)
    thalach = rng.normal(150, 22, n).clip(71, 202)     # max heart rate achieved
    exang = rng.binomial(1, 0.33, n)                   # exercise induced angina
    oldpeak = rng.exponential(1.0, n).clip(0, 6.2)
    slope = rng.integers(0, 3, n)
    ca = rng.integers(0, 4, n)                         # number of major vessels
    thal = rng.integers(0, 3, n)

    # risk score drives the probability of disease -> keeps relationships realistic
    risk = (
        0.03 * (age - 50)
        + 0.8 * sex
        + 0.5 * cp
        + 0.02 * (trestbps - 130)
        + 0.01 * (chol - 240)
        + 0.4 * fbs
        - 0.02 * (thalach - 150)
        + 0.9 * exang
        + 0.35 * oldpeak
        + 0.4 * ca
        + 0.3 * thal
    )
    risk_std = risk.std()
    prob = 1 / (1 + np.exp(-1.6 * (risk - risk.mean()) / risk_std))
    target = rng.binomial(1, prob)

    df = pd.DataFrame({
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps.round(1),
        "chol": chol.round(1), "fbs": fbs, "restecg": restecg,
        "thalach": thalach.round(1), "exang": exang, "oldpeak": oldpeak.round(2),
        "slope": slope, "ca": ca, "thal": thal, "target": target
    })
    return df


def load_heart_disease_data():
    """
    Looks for data/heart.csv (real UCI/Kaggle data). If not found, generates
    a synthetic dataset with the same 13-feature schema so the pipeline runs.
    Download real data: https://archive.ics.uci.edu/dataset/45/heart+disease
    or https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
    """
    path = os.path.join(DATA_DIR, "heart.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        target_col = "target" if "target" in df.columns else df.columns[-1]
        print(f"[data_loader] Loaded REAL heart disease data from {path}")
    else:
        df = _generate_synthetic_heart()
        target_col = "target"
        df.to_csv(path, index=False)
        print(f"[data_loader] heart.csv not found -> generated SYNTHETIC data and saved to {path}")
    return df, target_col


def _generate_synthetic_diabetes(n=1000, seed=RANDOM_STATE):
    """Synthetic data mimicking the UCI Pima Indians Diabetes schema."""
    rng = np.random.default_rng(seed)
    pregnancies = rng.poisson(3, n).clip(0, 17)
    glucose = rng.normal(120, 32, n).clip(0, 200)
    blood_pressure = rng.normal(69, 19, n).clip(0, 122)
    skin_thickness = rng.normal(20, 16, n).clip(0, 99)
    insulin = rng.normal(80, 115, n).clip(0, 846)
    bmi = rng.normal(32, 7.9, n).clip(0, 67)
    dpf = rng.exponential(0.47, n).clip(0.08, 2.42)     # diabetes pedigree function
    age = rng.integers(21, 81, n)

    risk = (
        0.02 * glucose
        + 0.015 * bmi
        + 0.3 * dpf
        + 0.02 * age
        + 0.1 * pregnancies
        - 0.01 * blood_pressure
    )
    risk_std = risk.std()
    prob = 1 / (1 + np.exp(-1.5 * (risk - risk.mean()) / risk_std))
    target = rng.binomial(1, prob)

    df = pd.DataFrame({
        "Pregnancies": pregnancies, "Glucose": glucose.round(1),
        "BloodPressure": blood_pressure.round(1), "SkinThickness": skin_thickness.round(1),
        "Insulin": insulin.round(1), "BMI": bmi.round(1),
        "DiabetesPedigreeFunction": dpf.round(3), "Age": age, "Outcome": target
    })
    return df


def load_diabetes_data():
    """
    Looks for data/diabetes.csv (real Pima Indians Diabetes data from Kaggle).
    If not found, generates a synthetic dataset with the same 8-feature schema.
    Download real data: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
    """
    path = os.path.join(DATA_DIR, "diabetes.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        target_col = "Outcome" if "Outcome" in df.columns else df.columns[-1]
        print(f"[data_loader] Loaded REAL diabetes data from {path}")
    else:
        df = _generate_synthetic_diabetes()
        target_col = "Outcome"
        df.to_csv(path, index=False)
        print(f"[data_loader] diabetes.csv not found -> generated SYNTHETIC data and saved to {path}")
    return df, target_col


DATASETS = {
    "breast_cancer": load_breast_cancer_data,
    "heart": load_heart_disease_data,
    "diabetes": load_diabetes_data,
}


def load_dataset(name):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(DATASETS.keys())}")
    return DATASETS[name]()


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    for name in DATASETS:
        df, target = load_dataset(name)
        print(f"{name}: shape={df.shape}, target='{target}', positive_rate={df[target].mean():.2f}")
