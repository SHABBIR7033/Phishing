"""
Convert email_raw.csv (raw_text, subject, body_plain, body_html, from_address,
from_domain, reply_to, spf_result, dkim_result, ..., label) into the
emails_raw.csv format expected by email_model_training.py:

    subject,body,from_addr,reply_to,display_name,spf_pass,dkim_pass,label

Notes on the mapping:
- body: prefer body_html (script's HTML-stripping + link-mismatch logic needs
  real <a href> tags); fall back to body_plain when body_html is missing.
- spf_pass / dkim_pass: spf_result/dkim_result are 'pass'/'fail'/'neutral'
  strings here -> converted to 1 if 'pass' else 0.
- display_name: not present in this source file, so it's left blank. The
  email_model_training.py display-name-mismatch feature will simply read as
  0 (no mismatch detected) for every row, which is fine -- from_addr,
  reply_to, and the SPF/DKIM/text features still carry plenty of signal.
- label convention already matches: 1 = phishing, 0 = legitimate. No flip needed.

Usage:
    python convert_emails.py /path/to/email_raw.csv data/emails_raw.csv
"""

import sys
import pandas as pd


def convert(input_path: str, output_path: str):
    df = pd.read_csv(input_path)

    required = {"subject", "body_plain", "from_address", "spf_result", "dkim_result", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing expected columns: {missing}")

    body = df["body_html"].fillna(df["body_plain"])

    out = pd.DataFrame({
        "subject": df["subject"].fillna(""),
        "body": body.fillna(""),
        "from_addr": df["from_address"].fillna(""),
        "reply_to": df["reply_to"].fillna(""),
        "display_name": "",  # not present in source data
        "spf_pass": (df["spf_result"].str.lower() == "pass").astype(int),
        "dkim_pass": (df["dkim_result"].str.lower() == "pass").astype(int),
        "label": df["label"].astype(int),
    })

    out = out.dropna(subset=["body"]).drop_duplicates(subset=["subject", "body"])

    out.to_csv(output_path, index=False)
    print(f"Converted {len(out)} rows -> {output_path}")
    print(out["label"].value_counts().rename({1: "phishing", 0: "legitimate"}))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_emails.py <input_csv> <output_csv>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
