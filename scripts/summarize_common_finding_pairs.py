import pandas as pd
from itertools import combinations
from pathlib import Path


INPUT_FILE = Path("data/cases_features_08222026.csv")
OUTPUT_FILE = Path("results/common_finding_pairs_08222026.csv")


CATEGORIES = [
    (
        "Clinical signs",
        "clinical signs",
        [
            "vomiting",
            "diarrhea",
            "hyporexia",
            "weight loss",
            "hairball obstruction",
            "constipation",
        ],
    ),
    (
        "CBC abnormalities",
        "CBC",
        [
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
        ],
    ),
    (
        "Biochemistry abnormalities",
        "chem",
        [
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
        ],
    ),
    (
        "Ultrasonographic abnormalities",
        "AUS",
        [
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
        ],
    ),
]


def most_common_pair(df, prefix, labels):
    pair_counts = {}

    for _, row in df.iterrows():
        present = [
            label
            for label in labels
            if int(row.get(f"{prefix}_{label}", 0) or 0) == 1
        ]

        for first, second in combinations(sorted(present), 2):
            pair_counts[(first, second)] = pair_counts.get((first, second), 0) + 1

    if not pair_counts:
        return "none", 0

    top_count = max(pair_counts.values())
    top_pairs = [
        " + ".join(pair)
        for pair, count in sorted(pair_counts.items())
        if count == top_count
    ]

    return "; ".join(top_pairs), top_count


def main():
    df = pd.read_csv(INPUT_FILE)
    groups = [
        ("Overall", df),
        ("IBD", df[df["shorthand dx"] == "IBD"]),
        ("Lymphoma", df[df["shorthand dx"] == "low grade lymphoma"]),
    ]

    rows = []
    for group_name, group_df in groups:
        for category, prefix, labels in CATEGORIES:
            pair, count = most_common_pair(group_df, prefix, labels)
            denominator = len(group_df)
            rows.append(
                {
                    "group": group_name,
                    "group_n": denominator,
                    "category": category,
                    "most_common_pair": pair,
                    "count": count,
                    "denominator": denominator,
                    "percent": round(count / denominator * 100, 1)
                    if denominator
                    else "",
                }
            )

    output = pd.DataFrame(rows)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(output)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
