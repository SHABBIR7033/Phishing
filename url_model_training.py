"""
URL Phishing Classifier — Feature Extraction + Model Training
Phase 1 of Phishing Detection System

Expected input: a CSV with two columns -> url,label
    label = 1 for phishing, 0 for legitimate

Build this CSV by combining:
  - PhishTank verified feed (label=1)   https://phishtank.org/developer_info.php
  - Tranco top-1M list, sampled (label=0)  https://tranco-list.eu/
Save the merged, deduplicated file as data/urls_raw.csv before running this script.
"""

import re
import joblib
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import time

RAW_CSV = "data/urls_raw.csv"
MODEL_OUT = "models/url_random_forest.joblib"
SCALER_OUT = "models/url_scaler.joblib"

SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly",
    "is.gd", "buff.ly", "shorte.st", "adf.ly", "cutt.ly"
}
SUSPICIOUS_TLDS = {"tk", "ml", "ga", "cf", "gq", "xyz", "top", "work", "click"}
IP_PATTERN = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")


def extract_features(url: str) -> dict:
    """Extract lexical/structural features from a single URL."""
    url = str(url).strip()
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
    except ValueError:
        parsed = None

    hostname = (parsed.hostname or "") if parsed else ""
    scheme = (parsed.scheme or "") if parsed else ""
    path = (parsed.path or "") if parsed else ""
    full = url

    length = len(full)
    special_chars = sum(full.count(c) for c in ["%", "=", "&", "_", "//"])
    digits = sum(c.isdigit() for c in full)
    tld = hostname.split(".")[-1].lower() if "." in hostname else ""

    return {
        "url_length": length,
        "hostname_length": len(hostname),
        "dot_count": full.count("."),
        "hyphen_count": full.count("-"),
        "at_symbol_present": int("@" in full),
        "ip_address_present": int(bool(IP_PATTERN.match(hostname))),
        "https_present": int(scheme == "https"),
        "subdomain_count": max(hostname.count(".") - 1, 0),
        "special_char_ratio": special_chars / length if length else 0,
        "digit_ratio": digits / length if length else 0,
        "url_shortener_flag": int(hostname in SHORTENERS),
        "suspicious_tld_flag": int(tld in SUSPICIOUS_TLDS),
        "path_depth": path.count("/"),
    }


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    feats = df["url"].apply(extract_features).apply(pd.Series)
    return feats


def train():
    df = pd.read_csv(RAW_CSV)
    df = df.dropna(subset=["url", "label"]).drop_duplicates(subset=["url"])
    print(f"Loaded {len(df)} rows | phishing={df['label'].sum()} legit={(df['label']==0).sum()}")

    X = build_feature_matrix(df)
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1765, stratify=y_train, random_state=42
    )  # ~15% of original for val, remaining ~70% train

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    candidates = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=20, n_jobs=-1, random_state=42
        ),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        # LinearSVC scales to large datasets far better than kernel SVC (RBF SVC
        # is O(n^2)-O(n^3) and becomes impractically slow above ~20k rows).
        # CalibratedClassifierCV wraps it so we still get predict_proba if needed.
        "LinearSVM": CalibratedClassifierCV(LinearSVC(max_iter=5000), cv=3),
    }

    results = []
    best_model, best_name, best_f1 = None, None, -1

    for name, model in candidates.items():
        # Random Forest trains fine on unscaled features; linear models need scaling
        Xtr = X_train if name == "RandomForest" else X_train_scaled
        Xva = X_val if name == "RandomForest" else X_val_scaled

        t0 = time.time()
        model.fit(Xtr, y_train)
        train_time = time.time() - t0

        t0 = time.time()
        preds = model.predict(Xva)
        infer_ms_per_sample = (time.time() - t0) / len(Xva) * 1000

        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds)
        rec = recall_score(y_val, preds)
        f1 = f1_score(y_val, preds)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        fpr = fp / (fp + tn)

        results.append({
            "model": name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "fpr": fpr,
            "train_time_s": train_time, "infer_ms_per_sample": infer_ms_per_sample,
        })

        if f1 > best_f1:
            best_model, best_name, best_f1 = model, name, f1

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    print("\n=== Validation Results ===")
    print(results_df.to_string(index=False))

    # Final test-set evaluation with the best model
    Xte = X_test if best_name == "RandomForest" else X_test_scaled
    test_preds = best_model.predict(Xte)
    print(f"\n=== Best model: {best_name} — Test Set Report ===")
    print(classification_report(y_test, test_preds, target_names=["legit", "phishing"]))

    joblib.dump(best_model, MODEL_OUT)
    joblib.dump(scaler, SCALER_OUT)
    print(f"\nSaved best model to {MODEL_OUT}")
    print(f"Saved scaler to {SCALER_OUT}")
    print("Feature order (must match at inference time):", list(X.columns))


if __name__ == "__main__":
    train()
