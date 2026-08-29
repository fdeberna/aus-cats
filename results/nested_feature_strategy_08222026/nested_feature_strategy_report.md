# Nested CV Feature Strategy Analysis

## Dataset

- Input file: `data/cases_features_08222026.csv`
- Rows: 108
- Target column: `target`; target=1 is low grade lymphoma.
- Class balance: [{'target': 0, 'shorthand dx': 'IBD', 'count': 34, 'fraction': 0.3148148148148148}, {'target': 1, 'shorthand dx': 'low grade lymphoma', 'count': 74, 'fraction': 0.6851851851851852}]
- Candidate predictors used after documented exclusions: 52
- Singles evaluated per outer fold: 52
- Pairs evaluated per outer fold: 1326
- Triples evaluated: no

The candidate set uses `age`, lab abnormality flags, the requested binary clinical/AUS/CBC/chem features, and `breed_group` plus `procedure_clean`. Raw text, diagnosis labels, outcomes after diagnosis, dates, IDs, raw lab values, token helper columns, and no-variation fields were excluded.

## Validation Method

Repeated nested stratified CV was used with 5 repeats of 5-fold outer CV and 3-fold inner CV. Within each outer training fold, all single features and all pairs were evaluated by inner-CV log loss, including ridge penalty selection over C values [0.1, 0.3, 1.0]. The selected model was then refit on the full outer training fold and evaluated once on the untouched outer test fold.

Binary-by-binary interactions were included only when all four joint cells in the outer training fold had at least 5 observations. Other pair interactions were omitted to avoid adding unstable parameters for age or multi-level categorical variables.

Threshold-dependent metrics used a fixed probability threshold of 0.50; no threshold was optimized on test data.

## Out-of-Sample Performance

| Strategy | Log loss mean | AUC mean | Brier mean | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Single | 0.632 | 0.542 | 0.217 | 0.968 | 0.100 |
| Pair | 0.650 | 0.533 | 0.224 | 0.919 | 0.112 |

Positive delta log loss and Brier values mean the pair strategy improved over the single-feature strategy. Positive delta AUC means the pair strategy had higher AUC.

| Difference | Mean | 2.5% | 97.5% | Fraction positive |
| --- | ---: | ---: | ---: | ---: |
| Log loss: single - pair | -0.0175 | -0.0354 | 0.0044 | 0.20 |
| AUC: pair - single | -0.0093 | -0.0709 | 0.0209 | 0.60 |
| Brier: single - pair | -0.0069 | -0.0134 | 0.0014 | 0.20 |

The uncertainty intervals above are empirical quantiles across repeated outer-CV runs. These repeats are not fully independent because the same patients appear across repeats, so the intervals should be read as a stability/sensitivity summary rather than formal independent-sample confidence intervals.

## Selection Stability

- Most frequently selected single feature: `age` selected 13 times.
- Most frequently selected pair: `age` + `AUS_enlarged pancreas` selected 4 times.
- Top feature appearing in selected pairs: `age`.

If selection frequencies are spread across many features or pairs, the apparent winner should be treated cautiously because small perturbations of the data change the selected model.

## Sparse Diagnostics

- Rare binary predictors flagged: 18.
- See `sparse_cell_diagnostics.csv`, `feature_diagnostic_summary.csv`, and `categorical_level_counts.csv` for details.

## Exploratory Full-Data Ranking

The full-data rankings are descriptive and optimistically biased because the same dataset is used to search many features/pairs and rank the winners. Use the nested-CV comparison above for the primary out-of-sample inference.

Top exploratory single features by full-data CV log loss:

| features | cv_log_loss | best_c |
| --- | --- | --- |
| age | 0.6020 | 0.3000 |
| AUS_enlarged pancreas | 0.6083 | 1.0000 |
| chem_hyperproteinemia | 0.6106 | 1.0000 |
| tli_high | 0.6112 | 1.0000 |
| chem_discordantly elevated bun | 0.6166 | 1.0000 |
| CBC_neutrophilia | 0.6168 | 1.0000 |
| chem_decreased alt | 0.6168 | 1.0000 |
| clinical signs_hyporexia | 0.6181 | 0.3000 |
| AUS_ileus | 0.6194 | 1.0000 |
| AUS_splenomegaly | 0.6195 | 1.0000 |

Top exploratory pairs by full-data CV log loss:

| features | cv_log_loss | best_c | interaction_reason |
| --- | --- | --- | --- |
| age \| AUS_enlarged pancreas | 0.5780 | 1.0000 | omitted_non_binary_or_categorical_pair |
| age \| chem_hyperproteinemia | 0.5830 | 1.0000 | omitted_non_binary_or_categorical_pair |
| age \| CBC_neutrophilia | 0.5857 | 1.0000 | omitted_non_binary_or_categorical_pair |
| age \| clinical signs_hyporexia | 0.5859 | 1.0000 | omitted_non_binary_or_categorical_pair |
| age \| chem_discordantly elevated bun | 0.5869 | 1.0000 | omitted_non_binary_or_categorical_pair |
| age \| CBC_eosinophilia | 0.5871 | 1.0000 | omitted_non_binary_or_categorical_pair |
| age \| chem_decreased alt | 0.5881 | 1.0000 | omitted_non_binary_or_categorical_pair |
| age \| chem_no abnormalities | 0.5885 | 0.3000 | omitted_non_binary_or_categorical_pair |
| age \| tli_high | 0.5893 | 0.3000 | omitted_non_binary_or_categorical_pair |
| age \| CBC_lymphocytosis | 0.5897 | 1.0000 | omitted_non_binary_or_categorical_pair |

## Files

- `feature_diagnostic_summary.csv`
- `class_balance.csv`
- `binary_feature_prevalence.csv`
- `categorical_level_counts.csv`
- `sparse_cell_diagnostics.csv`
- `nested_cv_overall_performance.csv`
- `nested_cv_delta_summary.csv`
- `nested_cv_repeat_level_performance.csv`
- `nested_cv_fold_level_results.csv`
- `nested_cv_out_of_fold_predictions.csv`
- `single_feature_selection_frequency.csv`
- `pair_selection_frequency.csv`
- `feature_frequency_within_selected_pairs.csv`
- `exploratory_single_feature_ranking.csv`
- `exploratory_pair_ranking.csv`
- `plots/`