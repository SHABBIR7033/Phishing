"""
Phishing Detection System — Streamlit App

Loads the trained URL and email classifiers (produced by url_model_training.py
and email_model_training.py) and exposes a simple web UI to test a URL or an
email for phishing.

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    Push this repo to GitHub, then point share.streamlit.io at app.py.
    Model files in models/ must be committed to the repo (see README) since
    there's no training step in the deployed app -- it only loads and predicts.
"""

import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from urllib.parse import urlparse
from scipy.sparse import hstack, csr_matrix

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Phishing Detection System", page_icon="🛡️", layout="centered")

# ----------------------------------------------------------------------------
# Feature extraction (must mirror url_model_training.py / email_model_training.py exactly)
# ----------------------------------------------------------------------------
SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly",
    "is.gd", "buff.ly", "shorte.st", "adf.ly", "cutt.ly"
}
SUSPICIOUS_TLDS = {"tk", "ml", "ga", "cf", "gq", "xyz", "top", "work", "click"}
IP_PATTERN = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")

URL_FEATURE_ORDER = [
    "url_length", "hostname_length", "dot_count", "hyphen_count",
    "at_symbol_present", "ip_address_present", "https_present",
    "subdomain_count", "special_char_ratio", "digit_ratio",
    "url_shortener_flag", "suspicious_tld_flag", "path_depth",
]

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


def extract_url_features(url: str) -> dict:
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


def count_terms(text: str, terms: list) -> int:
    text_lower = text.lower()
    return sum(text_lower.count(t) for t in terms)


def extract_link_mismatch(raw_html: str) -> int:
    if not isinstance(raw_html, str):
        return 0
    anchors = re.findall(r"<a[^>]+href=[\"'](https?://[^\"'/]+)[^>]*>(.*?)</a>", raw_html, re.IGNORECASE)
    for href_domain, anchor_text in anchors:
        domain_in_text = re.findall(r"[\w.-]+\.\w{2,}", anchor_text)
        if domain_in_text and href_domain.split("//")[-1] not in domain_in_text[0]:
            return 1
    return 0


def extract_email_engineered_features(subject: str, body: str, from_addr: str, reply_to: str) -> dict:
    combined_text = f"{subject} {body}"
    from_domain = from_addr.split("@")[-1] if "@" in from_addr else ""
    reply_domain = reply_to.split("@")[-1] if "@" in reply_to else ""

    return {
        "urgency_keyword_count": count_terms(combined_text, URGENCY_TERMS),
        "reward_lure_count": count_terms(combined_text, REWARD_TERMS),
        "credential_request_flag": int(count_terms(combined_text, CREDENTIAL_TERMS) > 0),
        "link_count": len(LINK_PATTERN.findall(body)),
        "link_domain_mismatch": extract_link_mismatch(body),
        "spf_pass": 0,   # unknown at inference time unless headers are supplied
        "dkim_pass": 0,  # unknown at inference time unless headers are supplied
        "display_name_mismatch": 0,
        "reply_to_mismatch": int(bool(reply_domain) and reply_domain != from_domain),
    }


# ----------------------------------------------------------------------------
# Model loading (cached so it only happens once per session)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_url_model():
    model = joblib.load("models/url_random_forest.joblib")
    return model


@st.cache_resource
def load_email_model():
    vectorizer = joblib.load("models/email_tfidf_vectorizer.joblib")
    model = joblib.load("models/email_classifier.joblib")
    return vectorizer, model


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("🛡️ Phishing Detection System")
st.caption("Lightweight ML-based phishing detection for URLs and emails — Random Forest / Logistic Regression, trained on PhiUSIIL and email corpora.")

tab1, tab2 = st.tabs(["🔗 URL Scanner", "✉️ Email Scanner"])

# --- URL tab ---
with tab1:
    st.subheader("Check a URL")
    url_input = st.text_input("Enter a URL", placeholder="https://example.com/login")

    if st.button("Scan URL", type="primary"):
        if not url_input.strip():
            st.warning("Please enter a URL.")
        else:
            try:
                model = load_url_model()
                feats = extract_url_features(url_input)
                X = pd.DataFrame([feats])[URL_FEATURE_ORDER]
                pred = model.predict(X)[0]
                proba = model.predict_proba(X)[0]

                if pred == 1:
                    st.error(f"⚠️ Likely **PHISHING** — confidence {proba[1]*100:.1f}%")
                else:
                    st.success(f"✅ Likely **legitimate** — confidence {proba[0]*100:.1f}%")

                with st.expander("See extracted features"):
                    st.json(feats)
            except FileNotFoundError:
                st.error(
                    "URL model not found. Run `python url_model_training.py` first, "
                    "or make sure `models/url_random_forest.joblib` is committed to the repo."
                )

# --- Email tab ---
with tab2:
    st.subheader("Check an email")
    subject_input = st.text_input("Subject", placeholder="Urgent: verify your account")
    body_input = st.text_area("Body (HTML or plain text)", height=180,
                               placeholder="Paste the email body here...")
    col1, col2 = st.columns(2)
    with col1:
        from_input = st.text_input("From address (optional)", placeholder="sender@example.com")
    with col2:
        reply_input = st.text_input("Reply-To address (optional)", placeholder="reply@example.com")

    if st.button("Scan Email", type="primary"):
        if not body_input.strip():
            st.warning("Please enter an email body.")
        else:
            try:
                vectorizer, model = load_email_model()
                text = f"{subject_input} {body_input}"
                tfidf_vec = vectorizer.transform([text])
                eng_feats = extract_email_engineered_features(
                    subject_input, body_input, from_input, reply_input
                )
                eng_vec = csr_matrix(pd.DataFrame([eng_feats]).values)
                X = hstack([tfidf_vec, eng_vec])

                pred = model.predict(X)[0]
                proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None

                if pred == 1:
                    conf = f" — confidence {proba[1]*100:.1f}%" if proba is not None else ""
                    st.error(f"⚠️ Likely **PHISHING**{conf}")
                else:
                    conf = f" — confidence {proba[0]*100:.1f}%" if proba is not None else ""
                    st.success(f"✅ Likely **legitimate**{conf}")

                with st.expander("See extracted features"):
                    st.json(eng_feats)

                st.caption(
                    "Note: SPF/DKIM pass flags default to 0 here since raw email "
                    "headers aren't available from pasted text alone."
                )
            except FileNotFoundError:
                st.error(
                    "Email model not found. Run `python email_model_training.py` first, "
                    "or make sure `models/email_classifier.joblib` and "
                    "`models/email_tfidf_vectorizer.joblib` are committed to the repo."
                )

st.divider()
st.caption("Built with scikit-learn + Streamlit. This tool assists in identifying suspicious content — it does not replace organizational email security controls.")
