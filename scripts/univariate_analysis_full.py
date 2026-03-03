# univariate_analysis.py

import pandas as pd
import numpy as np
from scipy.stats import fisher_exact, chi2_contingency, mannwhitneyu
from statsmodels.stats.multitest import multipletests

INPUT_FILE = "data/cases_features.csv"
OUTPUT_FILE = "data/univariate_results_full.csv"


def mann_whitney_test(df, column):
    group1 = df[df.target == 1][column].dropna()
    group0 = df[df.target == 0][column].dropna()

    if len(group1) > 0 and len(group0) > 0:
        _, p = mannwhitneyu(group1, group0, alternative="two-sided")
        return p
    return np.nan


def main():

    df = pd.read_csv(INPUT_FILE)

    results = []

    # -----------------------------
    # Continuous variables
    # -----------------------------

    continuous_cols = [
        "age",
        "cobalamin (290-1500)",
        "folate (9.7-21.6)",
        "TLI (12-82)",
        "PLI (<= 4.4)"
    ]

    for col in continuous_cols:
        if col in df.columns:
            p = mann_whitney_test(df, col)
            results.append((col, "mannwhitney", p))

    # -----------------------------
    # Binary variables (Fisher)
    # -----------------------------

    for col in df.columns:
        if col == "target":
            continue

        if df[col].dropna().nunique() == 2:
            table = pd.crosstab(df[col], df["target"])
            if table.shape == (2, 2):
                _, p = fisher_exact(table)
                results.append((col, "fisher", p))

    # -----------------------------
    # Multi-category variables
    # -----------------------------

    for col in ["procedure_clean", "breed_group"]:
        if col in df.columns:
            table = pd.crosstab(df[col], df["target"])
            if table.shape[0] > 1:
                chi2, p, _, _ = chi2_contingency(table)
                results.append((col, "chi2", p))

    # -----------------------------
    # Compile results
    # -----------------------------

    results_df = pd.DataFrame(results, columns=["feature", "test", "p_value"])

    # Remove NaNs
    results_df = results_df.dropna()

    # FDR correction
    results_df["q_value"] = multipletests(
        results_df["p_value"],
        method="fdr_bh"
    )[1]

    results_df = results_df.sort_values("p_value")

    results_df.to_csv(OUTPUT_FILE, index=False)

    print("Saved univariate results to:", OUTPUT_FILE)
    print(results_df.head(15))


if __name__ == "__main__":
    main()
