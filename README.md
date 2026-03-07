# AUS Cats: IBD vs Low-Grade Lymphoma Analysis

This project analyzes feline clinical data to identify variables associated with **IBD** versus **low-grade lymphoma** diagnosis.

The workflow includes:
- data cleaning
- feature engineering from labs/clinical text/imaging text
- univariate association testing
- logistic regression with interaction terms

## Clinical Question

Given historical case data (clinical signs, labs, CBC/chem findings, abdominal ultrasound notes, and procedure information), which variables are associated with diagnosis outcome:
- `target = 0`: IBD
- `target = 1`: low-grade lymphoma

## Repository Layout

- `data/cases.csv`: raw input data
- `data/cases_cleaned.csv`: cleaned subset (IBD + low-grade lymphoma only)
- `data/cases_features.csv`: engineered features for modeling
- `data/univariate_results.csv`: compact univariate results
- `data/univariate_results_full.csv`: expanded univariate results
- `scripts/clean_data.py`: cleaning and target creation
- `scripts/features_engineering.py`: feature construction
- `scripts/univariate_analysis.py`: univariate screen (compact)
- `scripts/univariate_analysis_full.py`: univariate screen (full)
- `scripts/interaction_analysis.py`: additive + interaction logistic models for two predictors
- `scripts/multivariate_analysis.py`: simple multivariate logistic prototype

## Environment Setup

Use Python 3.10+.

Install dependencies:

```bash
pip install pandas numpy scipy statsmodels
```

## End-to-End Pipeline

Run from repository root.

1. Clean data

```bash
python scripts/clean_data.py
```

What it does:
- keeps only `IBD` and `low grade lymphoma`
- creates binary `target`
- parses numeric lab values
- standardizes selected text fields (`gender`, `procedure`)

2. Engineer features

```bash
python scripts/features_engineering.py
```

What it does:
- creates lab abnormality flags (`cobalamin_low`, `folate_abn`, `tli_abn`, `pli_abn`)
- creates binary demographic features (for example `male`)
- groups low-frequency breeds into `breed_group = other`
- normalizes/expands text-derived features from:
  - `CBC`
  - `chem`
  - `clinical signs`
- extracts and binarizes allowed AUS structures (imaging findings)

3. Run univariate analysis

Compact version:

```bash
python scripts/univariate_analysis.py
```

Full version:

```bash
python scripts/univariate_analysis_full.py
```

Tests used:
- Mann-Whitney U for continuous variables
- Fisher exact test for binary variables
- chi-square for multicategory variables
- Benjamini-Hochberg FDR correction (`q_value`)

### How FDR Is Applied in `univariate_analysis_full.py`

In `scripts/univariate_analysis_full.py`, FDR correction is applied after all univariate tests are collected:
- all test p-values are stored in `results_df["p_value"]`
- rows with missing p-values are removed (`dropna()`)
- adjusted p-values are computed with `statsmodels.stats.multitest.multipletests(..., method="fdr_bh")`
- the adjusted values are saved to `results_df["q_value"]`

This means FDR is controlled across the full set of tested features in that run (not separately by test type). In practice, use `q_value < 0.05` as a common threshold for post-correction significance.

4. Run interaction models (two predictors)

```bash
python scripts/interaction_analysis.py --a age --b gender
```

This script automatically runs:
- additive model: `target ~ a + b`
- interaction model: `target ~ a * b`

It prints:
- logistic regression summary
- odds ratios
- confidence intervals
- p-values

## Interaction Script: Any Column Names

`interaction_analysis.py` supports predictor names with spaces or special characters.

Example using a feature with spaces:

```bash
python scripts/interaction_analysis.py --a "clinical signs_vomiting" --b "breed_group"
```

The script safely converts non-identifier column names to Patsy-safe terms internally.

You can also override defaults:

```bash
python scripts/interaction_analysis.py \
  --input-file data/cases_features.csv \
  --target target \
  --a "clinical signs_vomiting" \
  --b "AUS_duodenum"
```

## Notes on Interpretation

- Univariate significance does not imply independent predictive value.
- Interaction terms (`a*b`) test whether the effect of one predictor depends on the other.
- Small sample sizes or sparse binary features can produce unstable estimates.
- Review confidence intervals alongside p-values and odds ratios.

## Common Troubleshooting

- `Missing column(s)` error in interaction script:
  - verify exact column names in `data/cases_features.csv`
- model convergence warnings:
  - check class imbalance and sparse predictors
  - consider simplifying predictors or collapsing rare levels

## Typical Analysis Flow

```bash
python scripts/clean_data.py
python scripts/features_engineering.py
python scripts/univariate_analysis_full.py
python scripts/interaction_analysis.py --a age --b gender
python scripts/interaction_analysis.py --a "clinical signs_vomiting" --b "AUS_duodenum"
```
