"""
Convert PhiUSIIL_Phishing_URL_Dataset.csv into the urls_raw.csv format
expected by url_model_training.py (columns: url,label).

PhiUSIIL label convention: 1 = legitimate, 0 = phishing
url_model_training.py convention: 1 = phishing, 0 = legitimate
--> this script flips the label.

Usage:
    python convert_phiusiil.py /path/to/PhiUSIIL_Phishing_URL_Dataset.csv data/urls_raw.csv
"""

import sys
import pandas as pd


def convert(input_path: str, output_path: str):
    df = pd.read_csv(input_path)

    required = {"URL", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing expected columns: {missing}")

    out = pd.DataFrame({
        "url": df["URL"],
        "label": (1 - df["label"]).astype(int),  # flip: PhiUSIIL 1=legit -> our 0=legit
    })

    out = out.dropna(subset=["url"]).drop_duplicates(subset=["url"])

    out.to_csv(output_path, index=False)
    print(f"Converted {len(out)} rows -> {output_path}")
    print(out["label"].value_counts().rename({1: "phishing", 0: "legitimate"}))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_phiusiil.py <input_csv> <output_csv>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
