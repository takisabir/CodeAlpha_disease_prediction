"""
train.py
----------------
Defines the 4 classification algorithms required by the task
(Logistic Regression, Random Forest, SVM, XGBoost) and trains them.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

RANDOM_STATE = 42


def get_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            eval_metric="logloss", random_state=RANDOM_STATE
        )
    else:
        print("[train] WARNING: xgboost is not installed, skipping it. "
              "Run: pip install xgboost --break-system-packages")
    return models


def train_models(data):
    """Trains every model on data['X_train']/data['y_train'] and returns fitted models."""
    models = get_models()
    fitted = {}
    for name, model in models.items():
        model.fit(data["X_train"], data["y_train"])
        fitted[name] = model
    return fitted
