# feature_engineering.py

import pandas as pd
import re

INPUT_FILE = "data/cases_cleaned_08222026.csv"
OUTPUT_FILE = "data/cases_features_08222026.csv"


# --------------------------------------------------
# Requested Text Feature Patterns
# --------------------------------------------------

REQUESTED_FEATURE_PATTERNS = {
    "clinical signs": [
        ("none", [r"\bnone\b"]),
        ("vomiting", [r"\bvomiting\b"]),
        ("diarrhea", [r"\bdiarrhea\b"]),
        ("hyporexia", [r"\bhyporexia\b"]),
        ("weight loss", [r"\bweight loss\b"]),
        ("hairball obstruction", [r"\bhairball obstruction\b"]),
        ("constipation", [r"\bconstipation\b"]),
    ],
    "AUS": [
        ("no abnormalities", [r"\bwnl\b", r"\bno abnormalities\b"]),
        ("duodenum", [r"\bduodenum\b", r"\bduodenun\b", r"\bdudenum\b"]),
        ("jejunum", [r"\bjejunum\b", r"\bjeunum\b"]),
        ("ileum", [r"\bileum\b"]),
        ("colon", [r"\bcolon\b"]),
        (
            "mesenteric lymphadenopathy",
            [r"\bmesenteric lymphadenopathy\b", r"\bmeseneteric lymphadenopathy\b"],
        ),
        (
            "chronic degenerative renal changes",
            [
                r"\bchronic degenerative renal changes\b",
                r"\bchronic degenerative changes to kidneys\b",
            ],
        ),
        ("enlarged pancreas", [r"\benlarged pancreas\b"]),
        ("splenomegaly", [r"\bsplenomegaly\b"]),
        ("ileus", [r"\bileus\b"]),
        ("gall bladder sludge", [r"\bgall bladder sludge\b"]),
    ],
    "CBC": [
        ("no abnormalities", [r"\bwnl\b", r"\bno abnormalities\b"]),
        ("neutrophilia", [r"\bneutrophilia\b"]),
        ("neutropenia", [r"\bneutropenia\b"]),
        ("eosinophilia", [r"\beosinophilia\b"]),
        ("eosinopenia", [r"\beosinopenia\b"]),
        ("basophilia", [r"\bbasophilia\b"]),
        ("basocytosis", [r"\bbasocytosis\b"]),
        ("lymphocytosis", [r"\blymphocytosis\b"]),
        ("lymphopenia", [r"\blymphopenia\b"]),
        ("monocytosis", [r"\bmonocytosis\b"]),
        ("anemia", [r"\banemia\b"]),
    ],
    "chem": [
        ("no abnormalities", [r"\bwnl\b", r"\bno abnormalities\b"]),
        ("hypoproteinemia", [r"\bhypoproteinemia\b"]),
        ("hyperproteinemia", [r"\bhyperproteinemia\b"]),
        ("decreased alt", [r"\bdecreased alt\b"]),
        ("elevated alt", [r"\belevated alt\b", r"\balt\b.*\belevated\b"]),
        ("elevated ast", [r"\belevated ast\b", r"\bast\b.*\belevated\b"]),
        ("elevated alp", [r"\belevated alp\b", r"\balp\b.*\belevated\b"]),
        (
            "elevated total bilirubin",
            [
                r"\belevated total bilirubin\b",
                r"\btbili\b",
                r"\btibil\b.*\belevated\b",
                r"\bbilirubin\b.*\belevated\b",
            ],
        ),
        ("azotemia", [r"\bazotemia\b"]),
        ("discordantly elevated bun", [r"\bdiscordantly elevated bun\b", r"\belevated bun\b"]),
        ("hyperglycemia", [r"\bhyperglycemia\b"]),
        ("hypercholesterolemia", [r"\bhypercholesterolemia\b"]),
        ("hypercalcemia", [r"\bhypercalcemia\b"]),
        ("hypokalemia", [r"\bhypokalemia\b"]),
        ("hypophosphatemia", [r"\bhypophosphatemia\b"]),
    ],
}


def text_has_pattern(text, patterns):
    if pd.isna(text):
        return False

    normalized = re.sub(r"\s+", " ", str(text).lower())
    return any(re.search(pattern, normalized) for pattern in patterns)


def extract_requested_terms(text, term_patterns):
    return [
        label
        for label, patterns in term_patterns
        if text_has_pattern(text, patterns)
    ]


def expand_requested_terms(df, column):
    term_patterns = REQUESTED_FEATURE_PATTERNS[column]
    token_col = column + "_tokens"

    df[token_col] = df[column].apply(
        lambda x: extract_requested_terms(x, term_patterns)
    )

    for label, _ in term_patterns:
        df[f"{column}_{label}"] = df[token_col].apply(
            lambda x: int(label in x)
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

    df["cobalamin_low"] = (df["cobalamin (290-1500)"] < 600).astype(float)

    df["folate_high"] = (
        (df["folate (9.7-21.6)"] > 21.6)
    ).astype(float)

    df["folate_low"] = (
        (df["folate (9.7-21.6)"] < 9.7)
    ).astype(float)

    df["tli_high"] = (
        (df["TLI (12-82)"] > 82)
    ).astype(float)

    df["tli_low"] = (
        (df["TLI (12-82)"] < 12)
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
    # Requested Text Feature Expansion
    # -------------------------------

    for col in ["clinical signs", "AUS", "CBC", "chem"]:
        df = expand_requested_terms(df, col)

    # -------------------------------
    # Save Output
    # -------------------------------

    df.to_csv(OUTPUT_FILE, index=False)

    print("Feature engineering complete.")
    print("Saved to:", OUTPUT_FILE)
    print("Final shape:", df.shape)


if __name__ == "__main__":
    main()
