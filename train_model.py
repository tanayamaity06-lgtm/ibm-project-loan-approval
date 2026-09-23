"""
train_model.py
--------------
Trains a Random Forest classifier on the loan_approval dataset,
saves the model and scaler to models/, and prints evaluation metrics.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, accuracy_score
)
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "loan_approval.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
print(df.head(3))

# ── Feature engineering ────────────────────────────────────────────────────
df["loan_approved"] = df["loan_approved"].map({"True": 1, "False": 0, True: 1, False: 0})
df["debt_to_income"] = df["loan_amount"] / (df["income"] + 1)

FEATURES = ["income", "credit_score", "loan_amount", "years_employed", "points", "debt_to_income"]
TARGET   = "loan_approved"

X = df[FEATURES]
y = df[TARGET]

# ── Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Scale features ─────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── Train models and pick best ─────────────────────────────────────────────
candidates = {
    "RandomForest":        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "GradientBoosting":    GradientBoostingClassifier(n_estimators=200, random_state=42),
    "LogisticRegression":  LogisticRegression(max_iter=1000, random_state=42),
}

best_name, best_model, best_score = None, None, 0.0
results = {}

for name, clf in candidates.items():
    cv_scores = cross_val_score(clf, X_train_s, y_train, cv=5, scoring="roc_auc")
    mean_auc  = cv_scores.mean()
    results[name] = mean_auc
    print(f"{name:25s} CV AUC = {mean_auc:.4f} ± {cv_scores.std():.4f}")
    if mean_auc > best_score:
        best_score = mean_auc
        best_name  = name
        best_model = clf

print(f"\nBest model: {best_name}  (CV AUC = {best_score:.4f})")

# ── Fit best model on full training set ───────────────────────────────────
best_model.fit(X_train_s, y_train)
y_pred      = best_model.predict(X_test_s)
y_pred_prob = best_model.predict_proba(X_test_s)[:, 1]

print("\n-- Test-set evaluation --")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_pred_prob):.4f}")
print(classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# ── Feature importances (for tree-based models) ────────────────────────────
if hasattr(best_model, "feature_importances_"):
    fi = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nFeature importances:\n", fi.to_string())
    fi.to_csv(os.path.join(MODEL_DIR, "feature_importances.csv"))

# ── Save artefacts ─────────────────────────────────────────────────────────
joblib.dump(best_model, os.path.join(MODEL_DIR, "model.pkl"))
joblib.dump(scaler,     os.path.join(MODEL_DIR, "scaler.pkl"))

# Save metadata (feature names, model name, accuracy, AUC)
metadata = {
    "model_name": best_name,
    "features":   FEATURES,
    "accuracy":   float(accuracy_score(y_test, y_pred)),
    "roc_auc":    float(roc_auc_score(y_test, y_pred_prob)),
    "cv_results": {k: float(v) for k, v in results.items()},
}
import json
with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nSaved model + scaler + metadata to {MODEL_DIR}")
