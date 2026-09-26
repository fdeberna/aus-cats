# Exploratory Association Analysis

This is a descriptive association search within the 108-cat cohort, not an out-of-sample diagnostic or predictive modeling analysis.

## Dataset and Feature Eligibility

- Input file: `data/cases_features_08222026.csv`
- Rows: 108
- Target: `0 = IBD`, `1 = low-grade lymphoma`
- Class balance: 34 IBD and 74 low-grade lymphoma
- Eligible predictors: 51
- Excluded columns: 27
- Procedure fields were excluded because procedure choice may encode clinician suspicion rather than biological information.
- Sex was excluded because raw `gender` contains `n` and `s`; the engineered `male` variable is constant at 0 and cannot be reliably decoded as male/female.

Raw free text, IDs, dates, DOB, diagnosis-defining histopathology, diagnosis labels, treatment, response, current status, token helper columns, target-derived information, raw GI-panel values, and procedure variables were excluded from the main search.

## Missing GI-Panel Handling

GI-panel abnormality flags were corrected before analysis: when the raw measurement was missing, the corresponding engineered abnormality flag was set to missing and excluded only from candidate combinations that used that analyte. This avoids treating an unmeasured analyte as normal while preserving all 108 cats for combinations that do not use that analyte.

| feature | raw_measurement | n_raw_missing | n_positive_before | n_positive_after |
| --- | --- | --- | --- | --- |
| cobalamin_low | cobalamin (290-1500) | 18 | 34 | 34 |
| folate_high | folate (9.7-21.6) | 18 | 31 | 31 |
| folate_low | folate (9.7-21.6) | 18 | 5 | 5 |
| tli_high | TLI (12-82) | 21 | 27 | 27 |
| tli_low | TLI (12-82) | 21 | 0 | 0 |
| pli_abn | PLI (<= 4.4) | 22 | 15 | 15 |

## Association Statistics

- All-binary and categorical-only combinations used contingency-table likelihood-ratio G statistics and mutual information directly from the joint state table.
- Combinations containing age used logistic deviance improvement over an intercept-only model, with age kept continuous and linear.
- For age plus binary/categorical variables, the exhaustive model used one common linear age slope plus intercepts for the joint non-age state. Age-by-state interactions were not used in the exhaustive search because many joint states are sparse or separated in this 108-cat dataset.
- Empirical p-values used diagnosis-label permutations with fixed features.
- Permutations: 1000; random seed: 20260829.
- Search-adjusted p-values are max-statistic family-wise p-values computed separately for singles, pairs, and triples.

## Validation Checks

- Validation subset passed: True
- Checked that binary contingency G matched the equivalent grouped likelihood-ratio calculation on sampled binary pairs.
- Checked that GI-panel raw missingness remained missing in corrected flags.
- Checked that redundant no-abnormality combinations were excluded from the candidate search.

## Top Single Variables

| rank_by_association_statistic | display_variables | association_statistic | mutual_information_nats | raw_empirical_p_value | search_adjusted_empirical_p_value | usable_n | sparse_warnings |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | age | 7.383 |  | 0.006993 | 0.2567 | 108 |  |
| 2 | chem_hyperproteinemia | 5.513 | 0.02552 | 0.06094 | 0.6284 | 108 |  |
| 3 | AUS_enlarged_pancreas | 5.111 | 0.02366 | 0.02098 | 0.7133 | 108 |  |
| 4 | CBC_neutrophilia | 4.706 | 0.02179 | 0.09291 | 0.8851 | 108 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 5 | chem_decreased_ALT | 4.706 | 0.02179 | 0.09291 | 0.8851 | 108 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 6 | chem_discordantly_elevated_BUN | 4.706 | 0.02179 | 0.1069 | 0.8851 | 108 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 7 | chem_hyperglycemia | 4.706 | 0.02179 | 0.09191 | 0.8851 | 108 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 8 | tli_high | 4.609 | 0.02649 | 0.03696 | 0.9101 | 87 |  |
| 9 | clinical_signs_hyporexia | 3.036 | 0.01406 | 0.1489 | 0.995 | 108 |  |
| 10 | breed_group | 2.624 | 0.01215 | 0.4785 | 0.998 | 108 |  |
| 11 | CBC_eosinopenia | 2.332 | 0.0108 | 0.3067 | 1 | 108 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 12 | chem_elevated_ALP | 2.332 | 0.0108 | 0.3257 | 1 | 108 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 13 | AUS_ileus | 2.104 | 0.009739 | 0.2018 | 1 | 108 |  |
| 14 | CBC_eosinophilia | 2.104 | 0.009739 | 0.2368 | 1 | 108 |  |
| 15 | CBC_basophilia | 1.809 | 0.008375 | 0.3467 | 1 | 108 |  |

The strongest individual associations should be read as cohort-level descriptive contrasts. Sparse warnings mark variables where at least one state is based on very few cats.

## Top Pairs

| rank_by_association_statistic | display_variables | association_statistic | mutual_information_nats | raw_empirical_p_value | search_adjusted_empirical_p_value | usable_n | sparse_warnings |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | age<br>CBC_neutrophilia | 13.66 |  | 0.001998 | 0.7742 | 108 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state;age_model_newton_not_fully_converged |
| 2 | breed_group<br>AUS_duodenal_thickening | 13.45 | 0.06225 | 0.08791 | 0.8012 | 108 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 3 | folate_high<br>breed_group | 12.18 | 0.06765 | 0.1419 | 0.9191 | 90 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 4 | age<br>chem_hyperproteinemia | 12.12 |  | 0.002997 | 0.9221 | 108 | age_model_newton_not_fully_converged |
| 5 | breed_group<br>clinical_signs_hyporexia | 11.81 | 0.05468 | 0.1678 | 0.9451 | 108 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 6 | AUS_splenomegaly<br>CBC_basophilia | 11.75 | 0.05441 | 0.00999 | 0.9481 | 108 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 7 | age<br>chem_discordantly_elevated_BUN | 11.71 |  | 0.001998 | 0.953 | 108 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state;age_model_newton_not_fully_converged |
| 8 | age<br>breed_group | 11.3 |  | 0.03497 | 0.97 | 108 |  |
| 9 | tli_high<br>CBC_basophilia | 11.23 | 0.06456 | 0.005994 | 0.976 | 87 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 10 | AUS_enlarged_pancreas<br>CBC_neutrophilia | 11.17 | 0.05173 | 0.001998 | 0.978 | 108 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 11 | age<br>AUS_enlarged_pancreas | 11.03 |  | 0.002997 | 0.982 | 108 |  |
| 12 | breed_group<br>AUS_enlarged_pancreas | 10.77 | 0.04984 | 0.2158 | 0.987 | 108 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 13 | tli_high<br>CBC_neutrophilia | 10.7 | 0.0615 | 0.006993 | 0.987 | 87 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 14 | tli_high<br>breed_group | 10.69 | 0.06146 | 0.2288 | 0.987 | 87 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 15 | AUS_gall_bladder_sludge<br>CBC_lymphopenia | 10.68 | 0.04943 | 0.01598 | 0.988 | 108 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |

## Top Triples

| rank_by_association_statistic | display_variables | association_statistic | mutual_information_nats | raw_empirical_p_value | search_adjusted_empirical_p_value | usable_n | sparse_warnings |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | tli_high<br>breed_group<br>AUS_enlarged_pancreas | 27.86 | 0.1601 | 0.02597 | 0.6583 | 87 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 2 | folate_high<br>tli_high<br>breed_group | 26.24 | 0.1508 | 0.02697 | 0.8062 | 87 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 3 | breed_group<br>AUS_duodenal_thickening<br>AUS_ileal_thickening | 25.84 | 0.1196 | 0.02597 | 0.8362 | 108 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 4 | cobalamin_low<br>folate_high<br>breed_group | 25.22 | 0.1401 | 0.07592 | 0.8801 | 90 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 5 | folate_high<br>pli_abn<br>breed_group | 25.14 | 0.1461 | 0.01898 | 0.8861 | 86 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 6 | breed_group<br>AUS_enlarged_pancreas<br>chem_no_abnormalities | 24.21 | 0.1121 | 0.1469 | 0.9331 | 108 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 7 | breed_group<br>AUS_duodenal_thickening<br>AUS_enlarged_pancreas | 23.31 | 0.1079 | 0.06593 | 0.965 | 108 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 8 | tli_high<br>breed_group<br>clinical_signs_weight_loss | 22.73 | 0.1307 | 0.08192 | 0.976 | 87 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 9 | breed_group<br>clinical_signs_diarrhea<br>AUS_enlarged_pancreas | 22.21 | 0.1028 | 0.2068 | 0.988 | 108 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 10 | breed_group<br>clinical_signs_diarrhea<br>clinical_signs_weight_loss | 22.15 | 0.1025 | 0.1558 | 0.989 | 108 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 11 | tli_high<br>breed_group<br>chem_azotemia | 21.91 | 0.1259 | 0.04795 | 0.992 | 87 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 12 | breed_group<br>clinical_signs_hyporexia<br>AUS_duodenal_thickening | 21.81 | 0.101 | 0.1099 | 0.994 | 108 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 13 | breed_group<br>clinical_signs_hyporexia<br>CBC_no_abnormalities | 21.75 | 0.1007 | 0.1349 | 0.994 | 108 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 14 | breed_group<br>AUS_jejunal_thickening<br>chem_no_abnormalities | 21.43 | 0.09919 | 0.1868 | 0.998 | 108 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 15 | breed_group<br>clinical_signs_weight_loss<br>chem_no_abnormalities | 21.24 | 0.09836 | 0.2498 | 0.999 | 108 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |

## Multiple-Search Correction

- Best search-adjusted empirical p-value among singles: 0.2567
- Best search-adjusted empirical p-value among pairs: 0.7742
- Best search-adjusted empirical p-value among triples: 0.6583

These adjusted values answer whether a candidate exceeded the best candidate found anywhere in a label-shuffled dataset of the same search size. They are the primary correction for the exhaustive search.

## Age in Combinations

| rank_by_association_statistic | display_variables | association_statistic | mutual_information_nats | raw_empirical_p_value | search_adjusted_empirical_p_value | usable_n | sparse_warnings | combination_size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 22 | age<br>breed_group<br>clinical_signs_hyporexia | 20.15 |  | 0.01798 | 1 | 108 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state;age_model_newton_not_fully_converged | 3 |
| 35 | age<br>folate_high<br>breed_group | 19.22 |  | 0.02797 | 1 | 90 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state;age_model_newton_not_fully_converged | 3 |
| 45 | age<br>AUS_enlarged_pancreas<br>CBC_neutrophilia | 18.74 |  | 0.000999 | 1 | 108 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state;age_model_newton_not_fully_converged | 3 |
| 67 | age<br>CBC_neutrophilia<br>chem_discordantly_elevated_BUN | 18.15 |  | 0.000999 | 1 | 108 | empty_joint_state;joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state;age_model_newton_not_fully_converged | 3 |
| 69 | age<br>CBC_neutrophilia<br>chem_hyperproteinemia | 18.13 |  | 0.000999 | 1 | 108 | empty_joint_state;joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state;age_model_newton_not_fully_converged | 3 |
| 73 | age<br>breed_group<br>AUS_duodenal_thickening | 17.95 |  | 0.03796 | 1 | 108 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state;age_model_newton_not_fully_converged | 3 |
| 85 | age<br>breed_group<br>AUS_enlarged_pancreas | 17.69 |  | 0.03696 | 1 | 108 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state;age_model_newton_not_fully_converged | 3 |
| 95 | age<br>breed_group<br>chem_azotemia | 17.41 |  | 0.03796 | 1 | 108 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state;age_model_newton_not_fully_converged | 3 |
| 109 | age<br>breed_group<br>AUS_splenomegaly | 17.01 |  | 0.01898 | 1 | 108 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state;age_model_newton_not_fully_converged | 3 |
| 111 | age<br>breed_group<br>AUS_ileal_thickening | 16.97 |  | 0.04196 | 1 | 108 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state;age_model_newton_not_fully_converged | 3 |
| 119 | age<br>AUS_splenomegaly<br>CBC_basophilia | 16.81 |  | 0.002997 | 1 | 108 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state;age_model_newton_not_fully_converged | 3 |
| 130 | age<br>AUS_gall_bladder_sludge<br>CBC_lymphopenia | 16.43 |  | 0.004995 | 1 | 108 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state;age_model_newton_not_fully_converged | 3 |
| 131 | age<br>breed_group<br>AUS_mesenteric_lymphadenopathy | 16.42 |  | 0.07193 | 1 | 108 | joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state;age_model_newton_not_fully_converged | 3 |
| 135 | age<br>pli_abn<br>breed_group | 16.39 |  | 0.03896 | 1 | 86 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state;age_model_newton_not_fully_converged | 3 |
| 139 | age<br>CBC_neutrophilia<br>chem_elevated_ALP | 16.35 |  | 0.001998 | 1 | 108 | empty_joint_state;joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state;age_model_newton_not_fully_converged | 3 |

Age appears repeatedly in this section only if its continuous deviance contribution plus the joint non-age state ranks highly. This is still descriptive and should not be converted into an age threshold.

## Modality-Specific Patterns

| combination_size | modality_group | rank_within_modality_group | display_variables | association_statistic | search_adjusted_empirical_p_value | sparse_warnings |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | cross_CBC_plus_demographic | 1 | age<br>CBC_neutrophilia | 13.66 | 0.7742 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state;age_model_newton_not_fully_converged |
| 2 | cross_AUS_plus_demographic | 1 | breed_group<br>AUS_duodenal_thickening | 13.45 | 0.8012 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 2 | cross_GI_panel_plus_demographic | 1 | folate_high<br>breed_group | 12.18 | 0.9191 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 2 | cross_chemistry_plus_demographic | 1 | age<br>chem_hyperproteinemia | 12.12 | 0.9221 | age_model_newton_not_fully_converged |
| 2 | cross_clinical_signs_plus_demographic | 1 | breed_group<br>clinical_signs_hyporexia | 11.81 | 0.9451 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 2 | cross_AUS_plus_CBC | 1 | AUS_splenomegaly<br>CBC_basophilia | 11.75 | 0.9481 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 2 | cross_chemistry_plus_demographic | 2 | age<br>chem_discordantly_elevated_BUN | 11.71 | 0.953 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state;age_model_newton_not_fully_converged |
| 2 | within_demographic | 1 | age<br>breed_group | 11.3 | 0.97 |  |
| 2 | cross_CBC_plus_GI_panel | 1 | tli_high<br>CBC_basophilia | 11.23 | 0.976 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 2 | cross_AUS_plus_CBC | 2 | AUS_enlarged_pancreas<br>CBC_neutrophilia | 11.17 | 0.978 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 2 | cross_AUS_plus_demographic | 2 | age<br>AUS_enlarged_pancreas | 11.03 | 0.982 |  |
| 2 | cross_AUS_plus_demographic | 3 | breed_group<br>AUS_enlarged_pancreas | 10.77 | 0.987 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 2 | cross_CBC_plus_GI_panel | 2 | tli_high<br>CBC_neutrophilia | 10.7 | 0.987 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 2 | cross_GI_panel_plus_demographic | 2 | tli_high<br>breed_group | 10.69 | 0.987 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 2 | cross_AUS_plus_CBC | 3 | AUS_gall_bladder_sludge<br>CBC_lymphopenia | 10.68 | 0.988 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 2 | cross_AUS_plus_chemistry | 1 | AUS_enlarged_pancreas<br>chem_hyperproteinemia | 10.6 | 0.991 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 2 | cross_chemistry_plus_demographic | 3 | age<br>chem_decreased_ALT | 10.56 | 0.992 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state;age_model_newton_not_fully_converged |
| 2 | cross_clinical_signs_plus_demographic | 2 | age<br>clinical_signs_hyporexia | 9.99 | 0.997 |  |
| 2 | cross_CBC_plus_chemistry | 1 | CBC_neutrophilia<br>chem_hyperproteinemia | 9.948 | 0.997 | empty_joint_state;joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 2 | within_chemistry | 1 | chem_hyperproteinemia<br>chem_decreased_ALT | 9.948 | 0.997 | empty_joint_state;joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 2 | within_chemistry | 2 | chem_hyperproteinemia<br>chem_discordantly_elevated_BUN | 9.948 | 0.997 | empty_joint_state;joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 2 | within_chemistry | 3 | chem_hyperproteinemia<br>chem_hyperglycemia | 9.948 | 0.997 | empty_joint_state;joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 2 | cross_chemistry_plus_demographic | 4 | age<br>chem_elevated_ALP | 9.927 | 0.997 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state;age_model_newton_not_fully_converged |
| 2 | cross_AUS_plus_CBC | 4 | AUS_enlarged_pancreas<br>CBC_eosinophilia | 9.911 | 0.998 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state |
| 2 | cross_CBC_plus_demographic | 2 | age<br>CBC_lymphocytosis | 9.902 | 0.998 | joint_state_n_lt_3;joint_state_n_lt_5;apparent_extreme_fraction_in_tiny_joint_state;age_model_newton_not_fully_converged |

## Feature Frequency Among Top Combinations

Pair feature frequency:

| top_n | display_name | count | fraction_of_top_n |
| --- | --- | --- | --- |
| 20 | age | 7 | 0.35 |
| 20 | breed_group | 6 | 0.3 |
| 20 | CBC_neutrophilia | 4 | 0.2 |
| 20 | chem_hyperproteinemia | 4 | 0.2 |
| 20 | AUS_enlarged_pancreas | 4 | 0.2 |
| 20 | tli_high | 3 | 0.15 |
| 20 | clinical_signs_hyporexia | 2 | 0.1 |
| 20 | CBC_basophilia | 2 | 0.1 |
| 20 | chem_decreased_ALT | 2 | 0.1 |
| 20 | AUS_duodenal_thickening | 1 | 0.05 |
| 20 | folate_high | 1 | 0.05 |
| 20 | AUS_splenomegaly | 1 | 0.05 |
| 20 | chem_discordantly_elevated_BUN | 1 | 0.05 |
| 20 | AUS_gall_bladder_sludge | 1 | 0.05 |
| 20 | CBC_lymphopenia | 1 | 0.05 |
| 100 | age | 45 | 0.45 |
| 100 | AUS_enlarged_pancreas | 15 | 0.15 |
| 100 | breed_group | 14 | 0.14 |
| 100 | chem_hyperproteinemia | 12 | 0.12 |
| 100 | tli_high | 10 | 0.1 |
| 100 | CBC_neutrophilia | 7 | 0.07 |
| 100 | chem_discordantly_elevated_BUN | 7 | 0.07 |
| 100 | chem_decreased_ALT | 7 | 0.07 |
| 100 | chem_hyperglycemia | 7 | 0.07 |
| 100 | clinical_signs_hyporexia | 6 | 0.06 |
| 100 | AUS_splenomegaly | 5 | 0.05 |
| 100 | CBC_basophilia | 4 | 0.04 |
| 100 | AUS_ileus | 4 | 0.04 |
| 100 | CBC_eosinophilia | 3 | 0.03 |
| 100 | CBC_anemia | 3 | 0.03 |

Triple feature frequency:

| top_n | display_name | count | fraction_of_top_n |
| --- | --- | --- | --- |
| 20 | breed_group | 20 | 1 |
| 20 | AUS_enlarged_pancreas | 7 | 0.35 |
| 20 | tli_high | 5 | 0.25 |
| 20 | chem_no_abnormalities | 5 | 0.25 |
| 20 | folate_high | 4 | 0.2 |
| 20 | AUS_duodenal_thickening | 3 | 0.15 |
| 20 | clinical_signs_weight_loss | 3 | 0.15 |
| 20 | cobalamin_low | 2 | 0.1 |
| 20 | pli_abn | 2 | 0.1 |
| 20 | clinical_signs_diarrhea | 2 | 0.1 |
| 20 | clinical_signs_hyporexia | 2 | 0.1 |
| 20 | AUS_ileal_thickening | 1 | 0.05 |
| 20 | chem_azotemia | 1 | 0.05 |
| 20 | CBC_no_abnormalities | 1 | 0.05 |
| 20 | AUS_jejunal_thickening | 1 | 0.05 |
| 20 | AUS_mesenteric_lymphadenopathy | 1 | 0.05 |
| 100 | breed_group | 89 | 0.89 |
| 100 | AUS_duodenal_thickening | 22 | 0.22 |
| 100 | AUS_enlarged_pancreas | 20 | 0.2 |
| 100 | tli_high | 15 | 0.15 |
| 100 | folate_high | 15 | 0.15 |
| 100 | pli_abn | 10 | 0.1 |
| 100 | chem_no_abnormalities | 10 | 0.1 |
| 100 | clinical_signs_weight_loss | 10 | 0.1 |
| 100 | chem_azotemia | 10 | 0.1 |
| 100 | clinical_signs_hyporexia | 10 | 0.1 |
| 100 | AUS_splenomegaly | 10 | 0.1 |
| 100 | AUS_ileal_thickening | 8 | 0.08 |
| 100 | cobalamin_low | 8 | 0.08 |
| 100 | age | 8 | 0.08 |

## Clinically Interpretable State Examples

Pair states with the largest distance from the cohort lymphoma fraction among top-ranked pairs:

| display_variables | state | n | IBD | lymphoma | lymphoma_fraction | sparse_warning |
| --- | --- | --- | --- | --- | --- | --- |
| AUS_splenomegaly<br>CBC_basophilia | 01 | 3 | 3 | 0 | 0 | n_lt_5 |
| tli_high<br>CBC_basophilia | 01 | 3 | 3 | 0 | 0 | n_lt_5 |
| age<br>CBC_neutrophilia | 1 | 2 | 2 | 0 | 0 | n_lt_3 |
| breed_group<br>AUS_duodenal_thickening | DMH<br>0 | 2 | 2 | 0 | 0 | n_lt_3 |
| age<br>chem_discordantly_elevated_BUN | 1 | 2 | 2 | 0 | 0 | n_lt_3 |
| AUS_gall_bladder_sludge<br>CBC_lymphopenia | 11 | 2 | 2 | 0 | 0 | n_lt_3 |
| age<br>chem_decreased_ALT | 1 | 2 | 2 | 0 | 0 | n_lt_3 |
| CBC_neutrophilia<br>chem_hyperproteinemia | 10 | 2 | 2 | 0 | 0 | n_lt_3 |
| chem_hyperproteinemia<br>chem_decreased_ALT | 01 | 2 | 2 | 0 | 0 | n_lt_3 |
| chem_hyperproteinemia<br>chem_discordantly_elevated_BUN | 01 | 2 | 2 | 0 | 0 | n_lt_3 |
| chem_hyperproteinemia<br>chem_hyperglycemia | 01 | 2 | 2 | 0 | 0 | n_lt_3 |
| folate_high<br>breed_group | 1.0<br>DMH | 1 | 1 | 0 | 0 | n_lt_3 |

Triple states with the largest distance from the cohort lymphoma fraction among top-ranked triples:

| display_variables | state | n | IBD | lymphoma | lymphoma_fraction | sparse_warning |
| --- | --- | --- | --- | --- | --- | --- |
| tli_high<br>breed_group<br>AUS_enlarged_pancreas | 0.0<br>DMH<br>0 | 3 | 3 | 0 | 0 | n_lt_5 |
| breed_group<br>AUS_duodenal_thickening<br>AUS_ileal_thickening | DLH<br>0<br>0 | 3 | 3 | 0 | 0 | n_lt_5 |
| breed_group<br>AUS_enlarged_pancreas<br>chem_no_abnormalities | other<br>0<br>0 | 3 | 3 | 0 | 0 | n_lt_5 |
| breed_group<br>clinical_signs_diarrhea<br>clinical_signs_weight_loss | DMH<br>0<br>1 | 3 | 3 | 0 | 0 | n_lt_5 |
| tli_high<br>breed_group<br>chem_no_abnormalities | 0.0<br>other<br>0 | 3 | 3 | 0 | 0 | n_lt_5 |
| breed_group<br>AUS_duodenal_thickening<br>AUS_ileal_thickening | DMH<br>0<br>0 | 2 | 2 | 0 | 0 | n_lt_3 |
| cobalamin_low<br>folate_high<br>breed_group | 1.0<br>0.0<br>other | 2 | 2 | 0 | 0 | n_lt_3 |
| folate_high<br>pli_abn<br>breed_group | 0.0<br>1.0<br>other | 2 | 2 | 0 | 0 | n_lt_3 |
| breed_group<br>AUS_duodenal_thickening<br>AUS_enlarged_pancreas | DMH<br>0<br>0 | 2 | 2 | 0 | 0 | n_lt_3 |
| tli_high<br>breed_group<br>clinical_signs_weight_loss | 0.0<br>DMH<br>0 | 2 | 2 | 0 | 0 | n_lt_3 |
| breed_group<br>clinical_signs_diarrhea<br>clinical_signs_weight_loss | DMH<br>1<br>0 | 2 | 2 | 0 | 0 | n_lt_3 |
| breed_group<br>clinical_signs_hyporexia<br>CBC_no_abnormalities | DMH<br>0<br>0 | 2 | 2 | 0 | 0 | n_lt_3 |

## Interpretation

The top-ranked findings and combinations identify where diagnosis proportions differ most inside this cohort. Results with low search-adjusted empirical p-values are less compatible with random label assignment after accounting for the full search, while results with sparse-cell warnings are fragile and hypothesis-generating.

Do not interpret any pair or triple here as a diagnostic rule, predictive signature, or sufficient test. That requires a separate out-of-sample predictive analysis.

## Output Files

- `association_feature_inventory.csv`
- `single_associations.csv`
- `pair_associations.csv`
- `triple_associations.csv`
- `top_single_state_tables.csv`
- `top_pair_state_tables.csv`
- `top_triple_state_tables.csv`
- `permutation_max_statistics.csv`
- `age_partner_rankings.csv`
- `modality_specific_rankings.csv`
- `feature_frequency_top_combinations.csv`
- `validation_checks.csv`
- `gi_panel_missingness_corrections.csv`
- `plots/`