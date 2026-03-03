# multivariate_analysis.py

import pandas as pd
import statsmodels.api as sm
import numpy as np

INPUT_FILE = "data/cases_features.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    # Select predictors (keep model small!)
    predictors = [
        "age",
        "gender","breed"
    ]

    X = df[predictors]
    X = sm.add_constant(X)

    y = df["target"]

    model = sm.Logit(y, X, missing="drop")
    result = model.fit()

    print(result.summary())

    # Odds ratios
    OR = np.exp(result.params)
    print("\nOdds Ratios:")
    print(OR)


if __name__ == "__main__":
    main()
