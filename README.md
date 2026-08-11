# 🛡️ Phishing Detection System

A lightweight ML-based phishing detector for URLs and emails, with a
Streamlit web app for interactive testing and standalone training scripts
for both classifiers.

**Live features:**
- URL classifier (Random Forest, 13 lexical/structural features)
- Email classifier (TF-IDF + engineered features, Logistic Regression / Linear SVM)
- Streamlit UI to test either in your browser

## Project structure

```
.
├── app.py                     # Streamlit web app (loads trained models, no training here)
├── url_model_training.py      # Trains the URL classifier
├── email_model_training.py    # Trains the email classifier
├── convert_phiusiil.py        # Converts PhiUSIIL dataset -> urls_raw.csv format
├── convert_emails.py          # Converts a raw email CSV -> emails_raw.csv format
├── requirements.txt
├── LICENSE
├── data/                      # Raw datasets (gitignored — not committed)
└── models/                    # Trained model files (.joblib)
```

## Setup

```bash
git clone <your-repo-url>
cd <your-repo-folder>
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.9–3.11 recommended (see library compatibility notes below).

## 1. Train the URL classifier

**Option A — PhiUSIIL dataset (recommended, tested):**

Download the [PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)
(235k rows, `label`: 1=legitimate, 0=phishing — opposite of what this
pipeline expects), then convert it:

```bash
python convert_phiusiil.py /path/to/PhiUSIIL_Phishing_URL_Dataset.csv data/urls_raw.csv
```

Verified results on the real dataset: **99.6% accuracy, 99.5% F1, 0.13%
false-positive rate** with Random Forest.

**Option B — PhishTank + Tranco:**

Build `data/urls_raw.csv` with columns `url,label` (1=phishing, 0=legitimate) using
[PhishTank](https://phishtank.org/developer_info.php) (phishing) and
[Tranco](https://tranco-list.eu/) top-1M (legitimate).

Then train:

```bash
python url_model_training.py
```

Trains Random Forest / Logistic Regression / Linear SVM, prints a comparison
table (accuracy, precision, recall, F1, false-positive rate, inference
latency), and saves the best model to `models/url_random_forest.joblib`.

## 2. Train the email classifier

Build `data/emails_raw.csv` with columns:
`subject,body,from_addr,reply_to,display_name,spf_pass,dkim_pass,label`

Use `convert_emails.py` to reshape a raw email CSV (e.g. from Kaggle) into
this format, or combine [Enron](https://www.cs.cmu.edu/~enron/) (legitimate)
+ [Nazario Phishing Corpus](https://monkey.org/~jose/phishing/) (phishing)
manually. If `spf_pass`/`dkim_pass`/`reply_to`/`display_name` aren't
available, default them to 0 — just stay consistent between training and
inference.

```bash
python email_model_training.py
```

Builds a TF-IDF matrix (unigrams + bigrams, top 5000 terms) combined with
engineered features (urgency/reward keyword counts, credential-request flag,
link-domain mismatch, SPF/DKIM pass, reply-to mismatch), trains Logistic
Regression / Random Forest / Linear SVM, and saves the best model +
vectorizer to `models/`.

> ⚠️ **If your dataset is synthetic/templated**, expect suspiciously perfect
> scores (100% accuracy) — that reflects overly clean class separation in
> the data, not real-world performance. Report this honestly if it happens,
> or blend in real corpora (Enron/Nazario) for more realistic results.

## 3. Run the Streamlit app locally

Once both models are trained and saved in `models/`:

```bash
streamlit run app.py
```

Opens a browser tab with two tabs — a URL scanner and an email scanner —
that load the trained `.joblib` models and return a phishing/legitimate
prediction with confidence score.

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (see steps below) — **make sure `models/*.joblib`
   files are committed**, since the deployed app only loads models, it
   doesn't train them.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, click "New app".
3. Select your repo, branch, and set the main file path to `app.py`.
4. Click Deploy. Streamlit Cloud will install from `requirements.txt`
   automatically.

## Pushing this project to GitHub

```bash
git init
git add .
git commit -m "Initial commit: phishing detection system"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```

Note: `data/*.csv` is gitignored (raw datasets are large and easily
regenerated from the public sources above) — only `data/.gitkeep` is
tracked so the folder structure survives cloning. Trained models in
`models/` **are** tracked since the Streamlit app needs them to run without
retraining.

If GitHub rejects a push for a large file (>100MB), check `models/*.joblib`
sizes — if any model file is too large, consider
[Git LFS](https://git-lfs.com/) or retraining with fewer trees /
lower `max_features` to shrink it.

## Notes

- Feature order is printed at the end of each training run and must match
  exactly what `app.py` and any backend inference endpoint use — `app.py`
  already mirrors the training scripts' feature extraction logic exactly,
  so no changes are needed there if you retrain with the same scripts.
- Class balance matters for the URL model (PhishTank alone is much smaller
  than a Tranco sample) — undersample or apply SMOTE if your ratio is
  skewed beyond ~70/30.
- The SVM baseline uses `LinearSVC` (not kernel SVM) since RBF-kernel SVC
  doesn't scale past ~20k rows and this pipeline is built for full-size
  datasets (PhiUSIIL's 235k rows trains in under two minutes).
