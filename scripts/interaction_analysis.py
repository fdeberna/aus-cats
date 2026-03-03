# multivariate_interactions.py

import argparse
import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

INPUT_FILE = "data/cases_features.csv"


def to_formula_term(column_name):
    """Return a patsy-safe term for any column name."""
    if column_name.isidentifier():
        return column_name

    escaped = column_name.replace("\\", "\\\\").replace('"', '\\"')
    return f'Q("{escaped}")'


def normalize_formula_columns(formula, columns):
    """Wrap non-identifier column names with patsy's Q("...")."""
    normalized = formula

    # Longest first avoids partial replacements when names overlap.
    problematic = sorted(
        [c for c in columns if not c.isidentifier()],
        key=len,
        reverse=True,
    )

    for col in problematic:
        escaped = re.escape(col)
        quoted_col = col.replace('"', '\\"')
        replacement = f'Q("{quoted_col}")'

        # Support users writing 'col name' or "col name" in formulas.
        normalized = re.sub(rf"(?<!Q\()'{escaped}'", replacement, normalized)
        normalized = re.sub(rf'(?<!Q\()"{escaped}"', replacement, normalized)

        # Support bare usage like: target ~ clinical signs_vomiting + age
        normalized = re.sub(
            rf'(?<![\w\"])({escaped})(?![\w\"])',
            replacement,
            normalized,
        )

    return normalized


def run_model(df, formula):
    formula = normalize_formula_columns(formula, df.columns)

    print("\n=========================================")
    print("Running model:")
    print(formula)
    print("=========================================\n")

    model = smf.logit(formula=formula, data=df)
    result = model.fit()

    print(result.summary())

    odds = np.exp(result.params)
    conf = result.conf_int()
    conf_odds = np.exp(conf)

    or_table = pd.DataFrame(
        {
            "Odds Ratio": odds,
            "CI_lower": conf_odds[0],
            "CI_upper": conf_odds[1],
            "p_value": result.pvalues,
        }
    )

    print("\nOdds ratios:\n")
    print(or_table)


def build_formulas(target_col, col_a, col_b):
    """Build additive and interaction formulas for two predictors."""
    target = to_formula_term(target_col)
    a = to_formula_term(col_a)
    b = to_formula_term(col_b)

    additive = f"{target} ~ {a} + {b}"
    interaction = f"{target} ~ {a} * {b}"
    return additive, interaction


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run logistic models for two predictors: additive (a + b) "
            "and interaction (a * b)."
        )
    )
    parser.add_argument(
        "--input-file",
        default=INPUT_FILE,
        help=f"CSV input file path (default: {INPUT_FILE})",
    )
    parser.add_argument(
        "--target",
        default="target",
        help="Target column name (default: target)",
    )
    parser.add_argument(
        "--a",
        "--col-a",
        dest="col_a",
        required=True,
        help="First predictor column name",
    )
    parser.add_argument(
        "--b",
        "--col-b",
        dest="col_b",
        required=True,
        help="Second predictor column name",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_file)

    required_cols = [args.target, args.col_a, args.col_b]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    additive_formula, interaction_formula = build_formulas(
        args.target, args.col_a, args.col_b
    )

    run_model(df, additive_formula)
    run_model(df, interaction_formula)


if __name__ == "__main__":
    main()
