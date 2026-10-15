# Disease Prediction from Medical Data — Internship Task 4

A machine learning project that predicts the possibility of disease based on patient data
(symptoms, age, blood test results, etc.). The project works on three datasets — **Heart
Disease**, **Diabetes**, and **Breast Cancer** — and trains/compares 4 classification
algorithms on each: **Logistic Regression, Random Forest, SVM, XGBoost**.

## 📁 Project Structure

```
disease_prediction/
├── data/                          # datasets are stored here (auto-generated or real CSV)
├── models/                        # trained models are saved here (.joblib files)
├── outputs/                       # confusion matrices, ROC curves, comparison charts, CSV results
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # loads / generates the data
│   ├── preprocess.py              # cleaning, missing value handling, scaling, train/test split
│   ├── train.py                   # defines and trains the 4 models
│   └── evaluate.py                # generates metrics and plots
├── main.py                        # MAIN SCRIPT — runs the entire pipeline
├── predict.py                     # runs a prediction on a sample patient using a saved model
├── requirements.txt
└── README.md
```

## ⚠️ A note about the data

- The **Breast Cancer** dataset ships directly with scikit-learn (it's real UCI data, no
  download required).
- For **Heart Disease** and **Diabetes**, this code auto-generates realistic **synthetic
  data** (with the same feature columns as the real UCI/Kaggle datasets) so the project runs
  immediately without needing internet access.
- If you want to use **real data** (recommended for your final submission), just download
  these 2 files and place them in the `data/` folder — the code will automatically switch to
  using the real data:
  - Heart Disease: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset →
    save the file as `heart.csv`, with the target column named `target`.
  - Diabetes (Pima Indians): https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database →
    save the file as `diabetes.csv` (already comes in this format).

## 🚀 Step-by-Step: Setting Up and Running in VSCode

### Step 1 — Open the project folder in VSCode
1. Open VSCode.
2. `File > Open Folder` → select this `disease_prediction` folder.

### Step 2 — Check that Python is installed
Open the terminal (in VSCode: `Terminal > New Terminal`, or shortcut `` Ctrl+` ``):
```bash
python --version
```
You need Python 3.9+. If it's not installed, get it from
https://www.python.org/downloads/ (make sure to tick "Add Python to PATH" during install).

### Step 3 — Create a virtual environment (recommended, optional but best practice)
```bash
python -m venv venv
```
Activate it:
- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

VSCode will usually prompt you to select the interpreter in the bottom-right corner — pick the
`venv` Python (or `Ctrl+Shift+P` → "Python: Select Interpreter" → choose venv).

### Step 4 — Install the required libraries
```bash
pip install -r requirements.txt
```
This installs: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, joblib.

> If you get an error installing `xgboost`, everything else will still work — the code
> automatically skips XGBoost and shows a warning. You can try installing it separately with:
> `pip install xgboost`

### Step 5 — Run the full pipeline
```bash
python main.py
```
This single command will:
1. Load all three datasets
2. Clean/preprocess the data
3. Train 4 models on each dataset (12 models total)
4. Print accuracy, precision, recall, F1-score, and ROC-AUC for each in the terminal
5. Save confusion matrices, ROC curves, and comparison charts to the `outputs/` folder
6. Save the best model for each dataset to the `models/` folder

You'll see something like this at the end of the terminal output:
```
############################################################
 FINAL SUMMARY (best model per dataset)
############################################################
breast_cancer   -> Best: Logistic Regression  | Accuracy=0.965 | ROC-AUC=0.996
heart           -> Best: Logistic Regression  | Accuracy=0.745 | ROC-AUC=0.829
diabetes        -> Best: Random Forest        | Accuracy=0.740 | ROC-AUC=0.779
```

### Step 6 — Check the results
- Open the `outputs/` folder — you'll find three types of files for each dataset:
  - `<dataset>_model_comparison.csv` — a metrics table for all models
  - `<dataset>_metrics_comparison.png` — bar chart comparing all 4 models
  - `<dataset>_roc_comparison.png` — ROC curve of all models
  - `<dataset>_<model>_confusion_matrix.png` — confusion matrix for each model

### Step 7 — Test a prediction from a trained model (optional demo)
```bash
python predict.py --dataset breast_cancer
python predict.py --dataset heart
python predict.py --dataset diabetes
```
This loads the saved model and shows a prediction (disease present or not + probability) for
a sample patient row.

## 🧠 How the project works (for your report/viva)

1. **Data Loading** (`data_loader.py`): Each dataset is loaded with the proper feature columns
   (symptoms, age, blood test values like Glucose, Cholesterol, BMI, etc.) and a target label
   (disease present = 1, absent = 0).
2. **Preprocessing** (`preprocess.py`): Missing/invalid values (e.g., a BMI of 0 in the
   diabetes data) are imputed with the median, then the data is split 80/20 into train/test,
   and finally scaled with `StandardScaler` so all features are on the same scale (this helps
   Logistic Regression and SVM perform better).
3. **Model Training** (`train.py`): 4 different classification algorithms are trained:
   - **Logistic Regression** — a simple, interpretable baseline
   - **Random Forest** — an ensemble of decision trees, good at capturing non-linear patterns
   - **SVM (RBF kernel)** — for complex decision boundaries
   - **XGBoost** — gradient boosting, generally the strongest performer
4. **Evaluation** (`evaluate.py`): Accuracy, Precision, Recall, F1-score, and ROC-AUC are all
   calculated, since accuracy alone isn't enough for medical prediction (a False Negative — i.e.
   missing a disease — is far more dangerous, which is why Recall matters too).
5. **Model Selection**: For each dataset, the model with the highest ROC-AUC is treated as the
   best model and saved in the `models/` folder for future predictions.

## 🔧 Customization ideas (if you want to improve it further)

- Tune hyperparameters in `src/train.py` (e.g. `RandomForestClassifier(n_estimators=500)`)
- Add cross-validation (`sklearn.model_selection.cross_val_score`) for more robust results
- Extract feature importance from Random Forest/XGBoost for explainability
- Once you add the real datasets, the results will reflect real-world performance

## 📦 requirements.txt
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
matplotlib>=3.6.0
seaborn>=0.12.0
joblib>=1.2.0
