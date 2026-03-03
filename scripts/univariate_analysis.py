# univariate_analysis.py

import pandas as pd
from scipy.stats import fisher_exact, chi2_contingency, mannwhitneyu
from statsmodels.stats.multitest import multipletests

INPUT_FILE = "data/cases_features.csv"
OUTPUT_FILE = "data/univariate_results.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    results = []

    # Age (continuous)
    # check if values are generally higher or lower
    u, p = mannwhitneyu(
        df[df.target == 1]["age"],
        df[df.target == 0]["age"],
        alternative="two-sided"
    )
    results.append(("age", "mannwhitney", p))

    # Binary features
    # Fisher exact: check if the proportion of certain binary variables is different
    for col in df.columns:
        if df[col].dropna().nunique() == 2 and col != "target":
            table = pd.crosstab(df[col], df["target"])
            if table.shape == (2, 2):
                _, p = fisher_exact(table)
                results.append((col, "fisher", p))

    # Procedure chi-square
    table = pd.crosstab(df["procedure_clean"], df["target"])
    chi2, p, _, _ = chi2_contingency(table)
    results.append(("procedure_clean", "chi2", p))

    results_df = pd.DataFrame(results, columns=["feature", "test", "p_value"])

    results_df["q_value"] = multipletests(
        results_df["p_value"],
        method="fdr_bh"
    )[1]

    results_df = results_df.sort_values("p_value")
    results_df.to_csv(OUTPUT_FILE, index=False)

    print("Saved univariate results to", OUTPUT_FILE)


if __name__ == "__main__":
    main()
