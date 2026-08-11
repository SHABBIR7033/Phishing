"""
Email Phishing / Scam Classifier — Feature Extraction + Model Training
Phase 1 of Phishing Detection System

Expected input: a CSV with columns -> subject,body,from_addr,reply_to,display_name,
                                        spf_pass,dkim_pass,label
    label = 1 for phishing, 0 for legitimate

Build this CSV by combining:
  - Enron Email Dataset (label=0, legitimate)   https://www.cs.cmu.edu/~enron/
  - Nazario Phishing Corpus (label=1)           https://monkey.org/~jose/phishing/
  - Optional extra volume: SpamAssassin public corpus, CEAS 2008 dataset

For SPF/DKIM/header fields not present in a public dataset, either parse them
from raw .eml headers with Python's `email` module, or set them to a neutral
default (0) if unavailable — just be consistent between train and inference.
"""

import re
import time
import joblib
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

RAW_CSV = "data/emails_raw.csv"
VECTORIZER_OUT = "models/email_tfidf_vectorizer.joblib"
MODEL_OUT = "models/email_classifier.joblib"

URGENCY_TERMS = [
    "urgent", "verify", "account suspended", "act now", "immediately",
    "limited time", "confirm your identity", "unusual activity",
    "click here", "final notice",
]
REWARD_TERMS = [
    "congratulations", "you've won", "you have won", "claim now",
    "free gift", "selected winner", "reward",
]
CREDENTIAL_TERMS = [
    "password", "ssn", "social security", "bank account",
    "otp", "one time password", "credit card number", "pin number",
]

LINK_PATTERN = re.compile(r"href=[\"'](https?://[^\"']+)[\"']", re.IGNORECASE)


def clean_body(raw_html_or_text: str) -> str:
    text = str(raw_html_or_text)
    try:
        text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    except Exception:
        pass
    text = re.sub(r"\s+", " ", text).strip()
    return text


def count_terms(text: str, terms: list) -> int:
    text_lower = text.lower()
    return sum(text_lower.count(t) for t in terms)


def extract_link_mismatch(raw_html: str) -> int:
    """Rough heuristic: anchor display text contains a domain different from href domain."""
    if not isinstance(raw_html, str):
        return 0
    anchors = re.findall(r"<a[^>]+href=[\"'](https?://[^\"'/]+)[^>]*>(.*?)</a>", raw_html, re.IGNORECASE)
    for href_domain, anchor_text in anchors:
        domain_in_text = re.findall(r"[\w.-]+\.\w{2,}", anchor_text)
        if domain_in_text and href_domain.split("//")[-1] not in domain_in_text[0]:
            return 1
    return 0


def build_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)
    combined_text = (df["subject"].fillna("") + " " + df["clean_body"].fillna(""))

    feats["urgency_keyword_count"] = combined_text.apply(lambda t: count_terms(t, URGENCY_TERMS))
    feats["reward_lure_count"] = combined_text.apply(lambda t: count_terms(t, REWARD_TERMS))
    feats["credential_request_flag"] = combined_text.apply(
        lambda t: int(count_terms(t, CREDENTIAL_TERMS) > 0)
    )
    feats["link_count"] = df["body"].fillna("").apply(lambda b: len(LINK_PATTERN.findall(b)))
    feats["link_domain_mismatch"] = df["body"].fillna("").apply(extract_link_mismatch)

    feats["spf_pass"] = df.get("spf_pass", 0).fillna(0).astype(int)
    feats["dkim_pass"] = df.get("dkim_pass", 0).fillna(0).astype(int)

    from_domain = df.get("from_addr", "").fillna("").str.extract(r"@([\w.-]+)")[0].fillna("")
    display_name = df.get("display_name", "").fillna("")
    feats["display_name_mismatch"] = [
        int(bool(dn) and dom not in dn.lower()) for dn, dom in zip(display_name, from_domain)
    ]

    reply_to_domain = df.get("reply_to", "").fillna("").str.extract(r"@([\w.-]+)")[0].fillna("")
    feats["reply_to_mismatch"] = [
        int(bool(rt) and rt != frm) for rt, frm in zip(reply_to_domain, from_domain)
    ]

    return feats


def train():
    df = pd.read_csv(RAW_CSV)
    df = df.dropna(subset=["body", "label"]).drop_duplicates(subset=["subject", "body"])
    df["clean_body"] = df["body"].apply(clean_body)
    print(f"Loaded {len(df)} rows | phishing={df['label'].sum()} legit={(df['label']==0).sum()}")

    y = df["label"].astype(int)

    X_train_df, X_temp_df, y_train, y_temp = train_test_split(
        df, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val_df, X_test_df, y_val, y_test = train_test_split(
        X_temp_df, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    # TF-IDF on subject + cleaned body
    text_train = X_train_df["subject"].fillna("") + " " + X_train_df["clean_body"]
    text_val = X_val_df["subject"].fillna("") + " " + X_val_df["clean_body"]
    text_test = X_test_df["subject"].fillna("") + " " + X_test_df["clean_body"]

    vectorizer = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), stop_words="english"
    )
    tfidf_train = vectorizer.fit_transform(text_train)
    tfidf_val = vectorizer.transform(text_val)
    tfidf_test = vectorizer.transform(text_test)

    # Engineered features
    eng_train = build_engineered_features(X_train_df)
    eng_val = build_engineered_features(X_val_df)
    eng_test = build_engineered_features(X_test_df)

    X_train = hstack([tfidf_train, csr_matrix(eng_train.values)])
    X_val = hstack([tfidf_val, csr_matrix(eng_val.values)])
    X_test = hstack([tfidf_test, csr_matrix(eng_test.values)])

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42),
        "LinearSVM": CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3),
    }

    results = []
    best_model, best_name, best_f1 = None, None, -1

    for name, model in candidates.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        t0 = time.time()
        preds = model.predict(X_val)
        infer_ms = (time.time() - t0) / X_val.shape[0] * 1000

        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds)
        rec = recall_score(y_val, preds)
        f1 = f1_score(y_val, preds)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        fpr = fp / (fp + tn)

        results.append({
            "model": name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "fpr": fpr,
            "train_time_s": train_time, "infer_ms_per_sample": infer_ms,
        })

        if f1 > best_f1:
            best_model, best_name, best_f1 = model, name, f1

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    print("\n=== Validation Results ===")
    print(results_df.to_string(index=False))

    test_preds = best_model.predict(X_test)
    print(f"\n=== Best model: {best_name} — Test Set Report ===")
    print(classification_report(y_test, test_preds, target_names=["legit", "phishing"]))

    joblib.dump(vectorizer, VECTORIZER_OUT)
    joblib.dump(best_model, MODEL_OUT)
    print(f"\nSaved vectorizer to {VECTORIZER_OUT}")
    print(f"Saved model to {MODEL_OUT}")
    print("Engineered feature order (must match at inference time):", list(eng_train.columns))


if __name__ == "__main__":
    train()
