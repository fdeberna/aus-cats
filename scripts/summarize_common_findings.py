import pandas as pd
from itertools import combinations
from pathlib import Path


INPUT_FILE = Path("data/cases_features_08222026.csv")
OUTPUT_FILE = Path("results/common_findings_08222026.csv")


CLINICAL_SIGNS = [
    "none",
    "vomiting",
    "diarrhea",
    "hyporexia",
    "weight loss",
    "hairball obstruction",
    "constipation",
]

CBC_FINDINGS = [
    "no abnormalities",
    "neutrophilia",
    "neutropenia",
    "eosinophilia",
    "eosinopenia",
    "basophilia",
    "basocytosis",
    "lymphocytosis",
    "lymphopenia",
    "monocytosis",
    "anemia",
]

CHEM_FINDINGS = [
    "no abnormalities",
    "hypoproteinemia",
    "hyperproteinemia",
    "decreased alt",
    "elevated alt",
    "elevated ast",
    "elevated alp",
    "elevated total bilirubin",
    "azotemia",
    "discordantly elevated bun",
    "hyperglycemia",
    "hypercholesterolemia",
    "hypercalcemia",
    "hypokalemia",
    "hypophosphatemia",
]

AUS_FINDINGS = [
    "no abnormalities",
    "duodenum",
    "jejunum",
    "ileum",
    "colon",
    "mesenteric lymphadenopathy",
    "chronic degenerative renal changes",
    "enlarged pancreas",
    "splenomegaly",
    "ileus",
    "gall bladder sludge",
]

LAB_RANGES = {
    "cobalamin": ("cobalamin (290-1500)", 290, 1500),
    "folate": ("folate (9.7-21.6)", 9.7, 21.6),
    "TLI": ("TLI (12-82)", 12, 82),
    "PLI": ("PLI (<= 4.4)", None, 4.4),
}


def most_common_feature(df, prefix, labels, exclude_no_abnormalities=False):
    counts = []
    for label in labels:
        if exclude_no_abnormalities and label == "no abnormalities":
            continue

        col = f"{prefix}_{label}"
        counts.append((label, int(df[col].fillna(0).astype(int).sum())))

    counts.sort(key=lambda x: (-x[1], x[0]))
    top_count = counts[0][1]
    top_labels = [label for label, count in counts if count == top_count]

    return "; ".join(top_labels), top_count, len(df)


def most_common_clinical_pair(df):
    pair_counts = {}

    for _, row in df.iterrows():
        present = [
            label
            for label in CLINICAL_SIGNS
            if label != "none"
            and int(row.get(f"clinical signs_{label}", 0) or 0) == 1
        ]

        for first, second in combinations(sorted(present), 2):
            pair_counts[(first, second)] = pair_counts.get((first, second), 0) + 1

    if not pair_counts:
        return "none", 0, len(df)

    top_count = max(pair_counts.values())
    top_pairs = [
        " + ".join(pair)
        for pair, count in sorted(pair_counts.items())
        if count == top_count
    ]

    return "; ".join(top_pairs), top_count, len(df)


def most_common_lab_status(df, lab_name):
    col, low, high = LAB_RANGES[lab_name]
    values = pd.to_numeric(df[col], errors="coerce").dropna()
    statuses = []

    for value in values:
        if low is not None and value < low:
            statuses.append("decreased")
        elif high is not None and value > high:
            statuses.append("elevated")
        else:
            statuses.append("normal")

    counts = pd.Series(statuses).value_counts()
    if counts.empty:
        return "not reported", 0, 0

    top_count = int(counts.iloc[0])
    top_statuses = sorted(counts[counts == top_count].index.tolist())

    return "; ".join(top_statuses), top_count, int(len(values))


def add_row(rows, group_name, group_df, measure, result):
    finding, count, denominator = result
    rows.append(
        {
            "group": group_name,
            "group_n": len(group_df),
            "measure": measure,
            "most_common_finding_or_status": finding,
            "count": count,
            "denominator": denominator,
            "percent": round(count / denominator * 100, 1) if denominator else "",
        }
    )


def main():
    df = pd.read_csv(INPUT_FILE)
    groups = [
        ("Overall", df),
        ("IBD", df[df["shorthand dx"] == "IBD"]),
        ("Lymphoma", df[df["shorthand dx"] == "low grade lymphoma"]),
    ]

    rows = []
    for group_name, group_df in groups:
        add_row(
            rows,
            group_name,
            group_df,
            "Most common clinical sign",
            most_common_feature(group_df, "clinical signs", CLINICAL_SIGNS),
        )
        add_row(
            rows,
            group_name,
            group_df,
            "Most common clinical sign pair",
            most_common_clinical_pair(group_df),
        )
        add_row(
            rows,
            group_name,
            group_df,
            "Most common CBC finding",
            most_common_feature(group_df, "CBC", CBC_FINDINGS),
        )
        add_row(
            rows,
            group_name,
            group_df,
            "Most common biochemistry abnormality",
            most_common_feature(
                group_df,
                "chem",
                CHEM_FINDINGS,
                exclude_no_abnormalities=True,
            ),
        )
        add_row(
            rows,
            group_name,
            group_df,
            "Most common ultrasonographic abnormality",
            most_common_feature(
                group_df,
                "AUS",
                AUS_FINDINGS,
                exclude_no_abnormalities=True,
            ),
        )

        for lab_name in LAB_RANGES:
            add_row(
                rows,
                group_name,
                group_df,
                f"Most common {lab_name} status",
                most_common_lab_status(group_df, lab_name),
            )

    output = pd.DataFrame(rows)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(output)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
