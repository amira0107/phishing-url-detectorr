"""
train_model.py
---------------
Trains a Random Forest classifier on the URL feature dataset and
saves the trained model + evaluation report.

Usage:
    python3 train_model.py
"""

import os
import sys
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "app"))
from features import extract_features, FEATURE_NAMES  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "urls_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "phishing_model.joblib")
REPORT_PATH = os.path.join(BASE_DIR, "models", "training_report.json")


def load_dataset(path):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    return df


def featurize(df):
    feature_rows = [extract_features(u) for u in df["url"]]
    X = pd.DataFrame(feature_rows)[FEATURE_NAMES]
    y = df["label"]
    return X, y


def train():
    df = load_dataset(DATA_PATH)
    X, y = featurize(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    print("\n=== Evaluation on held-out test set ===")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"{k:>10}: {v:.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(metrics["confusion_matrix"])
    print("\nFull classification report:")
    print(classification_report(y_test, y_pred, target_names=["legit", "phishing"]))

    # feature importance
    importances = sorted(
        zip(FEATURE_NAMES, clf.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    print("Top 8 most important features:")
    for name, score in importances[:8]:
        print(f"  {name:<28} {score:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": clf, "feature_names": FEATURE_NAMES}, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    with open(REPORT_PATH, "w") as f:
        json.dump({
            **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
            "confusion_matrix": metrics["confusion_matrix"],
            "top_features": [[n, float(s)] for n, s in importances],
            "n_train": len(X_train),
            "n_test": len(X_test),
        }, f, indent=2)
    print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    train()
