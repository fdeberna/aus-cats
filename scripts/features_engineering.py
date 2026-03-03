# feature_engineering.py

import pandas as pd
import numpy as np
import re

INPUT_FILE = "data/cases_cleaned.csv"
OUTPUT_FILE = "data/cases_features.csv"

MIN_TOKEN_COUNT = 5

REMOVE_WORDS = {"mild", "moderate", "severe", "marked"}

# --------------------------------------------------
# Allowed AUS Structures
# --------------------------------------------------

AUS_STRUCTURES = [
    "duodenum",
    "jejunum",
    "ileum",
    "colon",
    "mesenteric lymphadenopathy",
    "kidneys",
    "pancreas",
    "splenomegaly",
    "ileus",
    "gall bladder",
    "wnl"
]


# --------------------------------------------------
# Clinical Sign Normalization
# --------------------------------------------------

def normalize_clinical_token(token: str):

    token = token.strip()

    # Ignore elevated liver value
    if "liver" in token and "elevated" in token:
        return None

    # Normalize vomiting
    if "vomiting" in token:
        return "vomiting"

    # Normalize anorexia/inappetence
    if "inappetence" in token or "anorexia" in token:
        return "anorexia"

    return token


# --------------------------------------------------
# AUS Standardization
# --------------------------------------------------

def extract_aus_structures(text):
    """
    Extract only allowed anatomical structures from AUS.
    Ignore descriptors like thickened, chronic, enlarged, etc.
    """

    if pd.isna(text):
        return []

    text = text.lower()

    found = []

    # Special rule: if WNL appears → treat as normal only
    if "wnl" in text:
        return ["wnl"]

    for structure in AUS_STRUCTURES:
        if structure == "wnl":
            continue

        if structure in text:
            found.append(structure)

    return list(set(found))


# --------------------------------------------------
# Generic Multi-Label Tokenizer (for CBC, chem, clinical signs)
# --------------------------------------------------

def tokenize_multilabel(text, is_clinical=False):

    if pd.isna(text):
        return []

    text = text.lower()
    parts = re.split(r",|\n|;", text)

    cleaned = []

    for p in parts:
        p = p.strip()
        p = re.sub(r"\s+", " ", p)

        words = [w for w in p.split() if w not in REMOVE_WORDS]
        token = " ".join(words)

        if not token:
            continue

        if is_clinical:
            token = normalize_clinical_token(token)

        if token:
            cleaned.append(token)

    return list(set(cleaned))


# --------------------------------------------------
# Expand Tokens into Binary Columns
# --------------------------------------------------

def expand_tokens(df, column, min_count=5, is_clinical=False):

    token_col = column + "_tokens"

    df[token_col] = df[column].apply(
        lambda x: tokenize_multilabel(x, is_clinical=is_clinical)
    )

    all_tokens = df[token_col].explode()
    counts = all_tokens.value_counts()
    valid_tokens = counts[counts >= min_count].index

    for token in valid_tokens:
        df[f"{column}_{token}"] = df[token_col].apply(
            lambda x: int(token in x)
        )

    return df


# --------------------------------------------------
# Main Pipeline
# --------------------------------------------------

def main():

    df = pd.read_csv(INPUT_FILE)

    # -------------------------------
    # Lab Abnormal Flags
    # -------------------------------

    df["cobalamin_low"] = (df["cobalamin (290-1500)"] < 800).astype(float)

    df["folate_abn"] = (
        (df["folate (9.7-21.6)"] < 9.7) |
        (df["folate (9.7-21.6)"] > 21.6)
    ).astype(float)

    df["tli_abn"] = (
        (df["TLI (12-82)"] < 12) |
        (df["TLI (12-82)"] > 82)
    ).astype(float)

    df["pli_abn"] = (
        df["PLI (<= 4.4)"] > 4.4
    ).astype(float)

    # -------------------------------
    # Gender Binary
    # -------------------------------

    df["male"] = (df["gender"] == "m").astype(int)

    # -------------------------------
    # Collapse Rare Breeds
    # -------------------------------

    breed_counts = df["breed"].value_counts()
    common_breeds = breed_counts[breed_counts >= 5].index

    df["breed_group"] = df["breed"].where(
        df["breed"].isin(common_breeds),
        "other"
    )

    # -------------------------------
    # Collapse Procedure
    # -------------------------------

    df["procedure_clean"] = df["procedure"].replace({
        "upper - fb": "other",
        "upper, lower": "other"
    })

    # -------------------------------
    # AUS Standardized Expansion
    # -------------------------------

    df["AUS_tokens"] = df["AUS"].apply(extract_aus_structures)

    for structure in AUS_STRUCTURES:
        df[f"AUS_{structure}"] = df["AUS_tokens"].apply(
            lambda x: int(structure in x)
        )

    # -------------------------------
    # CBC + chem (standard expansion)
    # -------------------------------

    for col in ["CBC", "chem"]:
        df = expand_tokens(df, col, min_count=MIN_TOKEN_COUNT)

    # -------------------------------
    # Clinical signs (with normalization)
    # -------------------------------

    df = expand_tokens(
        df,
        "clinical signs",
        min_count=MIN_TOKEN_COUNT,
        is_clinical=True
    )

    # -------------------------------
    # Save Output
    # -------------------------------

    df.to_csv(OUTPUT_FILE, index=False)

    print("Feature engineering complete.")
    print("Saved to:", OUTPUT_FILE)
    print("Final shape:", df.shape)


if __name__ == "__main__":
    main()