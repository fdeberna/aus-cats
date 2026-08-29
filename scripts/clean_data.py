# clean_data.py

import pandas as pd
import numpy as np
import re

INPUT_FILE = "data/Cases_08222026.csv"
OUTPUT_FILE = "data/cases_cleaned_08222026.csv"


def parse_numeric(x):
    if pd.isna(x):
        return np.nan
    x = str(x)
    x = re.sub(r"[^\d\.]", "", x)
    try:
        return float(x)
    except:
        return np.nan


def main():
    df = pd.read_csv(INPUT_FILE)

    if "Unnamed: 0" in df.columns and "Name" not in df.columns:
        df = df.rename(columns={"Unnamed: 0": "Name"})

    # Keep only IBD and low grade lymphoma
    dx = df["shorthand dx"].str.lower().str.strip()
    df = df[dx.isin(["ibd", "low grade lymphoma"])].copy()
    dx = df["shorthand dx"].str.lower().str.strip()

    df["shorthand dx"] = np.where(
        dx == "ibd",
        "IBD",
        "low grade lymphoma"
    )

    # Binary target
    df["target"] = (df["shorthand dx"] == "low grade lymphoma").astype(int)

    # Clean labs
    lab_cols = [
        "cobalamin (290-1500)",
        "folate (9.7-21.6)",
        "TLI (12-82)",
        "PLI (<= 4.4)"
    ]

    for col in lab_cols:
        df[col] = df[col].apply(parse_numeric)

    # Gender standardization
    df["gender"] = df["gender"].str.lower().str.strip()

    # Procedure standardization
    df["procedure"] = df["procedure"].str.lower().str.strip()

    df.to_csv(OUTPUT_FILE, index=False)
    print("Saved cleaned data to", OUTPUT_FILE)


if __name__ == "__main__":
    main()
