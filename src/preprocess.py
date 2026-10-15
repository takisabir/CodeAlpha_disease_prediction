"""
preprocess.py
----------------
Cleans a dataset and prepares it for training:
    - handles missing / zero-as-missing values
    - splits into train/test
    - scales features (fit on train only, to avoid data leakage)
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

RANDOM_STATE = 42

# In the Pima Diabetes dataset, 0 is not a physically valid value for these
# columns and actually represents a missing measurement.
ZERO_AS_MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def clean_data(df, target_col):
    df = df.copy()

    # Replace biologically-impossible zeros with NaN so they can be imputed properly
    for col in ZERO_AS_MISSING_COLS:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    return df


def split_and_scale(df, target_col, test_size=0.2, random_state=RANDOM_STATE):
    df = clean_data(df, target_col)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Impute missing values (median, fit on train only)
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # Scale features (fit on train only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    feature_names = X.columns.tolist()

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "feature_names": feature_names,
        "scaler": scaler,
        "imputer": imputer,
    }
