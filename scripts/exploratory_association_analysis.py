import argparse
import itertools
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit


DEFAULT_INPUT = "data/cases_features_08222026.csv"
DEFAULT_OUTPUT_DIR = "results/exploratory_association_analysis_08292026"
TARGET_COL = "target"
RANDOM_SEED = 20260829

BINARY_PREFIXES = ("clinical signs_", "AUS_", "CBC_", "chem_")
RAW_LAB_TO_FLAGS = {
    "cobalamin": {
        "raw": "cobalamin (290-1500)",
        "flags": {"cobalamin_low": lambda s: s < 600},
        "source": "GI panel",
    },
    "folate": {
        "raw": "folate (9.7-21.6)",
        "flags": {
            "folate_high": lambda s: s > 21.6,
            "folate_low": lambda s: s < 9.7,
        },
        "source": "GI panel",
    },
    "TLI": {
        "raw": "TLI (12-82)",
        "flags": {
            "tli_high": lambda s: s > 82,
            "tli_low": lambda s: s < 12,
        },
        "source": "GI panel",
    },
    "PLI": {
        "raw": "PLI (<= 4.4)",
        "flags": {"pli_abn": lambda s: s > 4.4},
        "source": "GI panel",
    },
}

DISPLAY_NAME_OVERRIDES = {
    "clinical signs_none": "clinical_signs_none_reported",
    "clinical signs_vomiting": "clinical_signs_vomiting",
    "clinical signs_diarrhea": "clinical_signs_diarrhea",
    "clinical signs_hyporexia": "clinical_signs_hyporexia",
    "clinical signs_weight loss": "clinical_signs_weight_loss",
    "clinical signs_hairball obstruction": "clinical_signs_hairball_obstruction",
    "clinical signs_constipation": "clinical_signs_constipation",
    "AUS_no abnormalities": "AUS_no_abnormalities",
    "AUS_duodenum": "AUS_duodenal_thickening",
    "AUS_jejunum": "AUS_jejunal_thickening",
    "AUS_ileum": "AUS_ileal_thickening",
    "AUS_colon": "AUS_colonic_thickening",
    "AUS_mesenteric lymphadenopathy": "AUS_mesenteric_lymphadenopathy",
    "AUS_chronic degenerative renal changes": "AUS_chronic_degenerative_renal_changes",
    "AUS_enlarged pancreas": "AUS_enlarged_pancreas",
    "AUS_gall bladder sludge": "AUS_gall_bladder_sludge",
    "CBC_no abnormalities": "CBC_no_abnormalities",
    "chem_no abnormalities": "chem_no_abnormalities",
    "chem_decreased alt": "chem_decreased_ALT",
    "chem_elevated alt": "chem_elevated_ALT",
    "chem_elevated ast": "chem_elevated_AST",
    "chem_elevated alp": "chem_elevated_ALP",
    "chem_elevated total bilirubin": "chem_elevated_total_bilirubin",
    "chem_discordantly elevated bun": "chem_discordantly_elevated_BUN",
}

EXCLUDED_ALWAYS = {
    "Name": "excluded_id",
    "procedure date": "excluded_date",
    "DOB": "excluded_date_birth_age_used_instead",
    "histopathology": "excluded_diagnosis_defining_post_procedure_information",
    "shorthand dx": "excluded_diagnosis_label",
    "Treatment": "excluded_post_diagnostic_treatment",
    "Response": "excluded_post_diagnostic_response",
    "Current Status": "excluded_post_diagnostic_status",
    TARGET_COL: "excluded_target",
    "procedure": "excluded_procedure_choice_may_encode_clinician_suspicion",
    "procedure_clean": "excluded_procedure_choice_may_encode_clinician_suspicion",
    "clinical signs": "excluded_raw_text_engineered_findings_used",
    "AUS": "excluded_raw_text_engineered_findings_used",
    "CBC": "excluded_raw_text_engineered_findings_used",
    "chem": "excluded_raw_text_engineered_findings_used",
    "breed": "excluded_raw_breed_breed_group_used",
    "gender": "excluded_ambiguous_codes_n_s_do_not_identify_male_female",
    "male": "excluded_invalid_constant_from_ambiguous_gender_codes",
}

NO_ABNORMALITY_GROUPS = {
    "clinical signs": {
        "none": "clinical signs_none",
        "members_prefix": "clinical signs_",
    },
    "AUS": {
        "none": "AUS_no abnormalities",
        "members_prefix": "AUS_",
    },
    "CBC": {
        "none": "CBC_no abnormalities",
        "members_prefix": "CBC_",
    },
    "chem": {
        "none": "chem_no abnormalities",
        "members_prefix": "chem_",
    },
}


def display_name(feature):
    if feature in DISPLAY_NAME_OVERRIDES:
        return DISPLAY_NAME_OVERRIDES[feature]
    return re.sub(r"[^A-Za-z0-9]+", "_", feature).strip("_")


def source_for_feature(feature):
    if feature == "age":
        return "demographic"
    if feature == "breed_group":
        return "demographic"
    if feature.startswith("clinical signs_"):
        return "clinical signs"
    if feature.startswith("AUS_"):
        return "AUS"
    if feature.startswith("CBC_"):
        return "CBC"
    if feature.startswith("chem_"):
        return "chemistry"
    if feature in {flag for spec in RAW_LAB_TO_FLAGS.values() for flag in spec["flags"]}:
        return "GI panel"
    return "other"


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if pd.isna(value):
        return None
    return str(value)


def finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def bh_fdr(p_values):
    p = np.asarray(p_values, dtype=float)
    q = np.ones_like(p, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return q
    valid_idx = np.where(valid)[0]
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    q_valid = np.empty_like(adjusted)
    q_valid[order] = adjusted
    q[valid_idx] = q_valid
    return q


def prepare_analysis_frame(df):
    out = df.copy()
    corrections = []
    for analyte, spec in RAW_LAB_TO_FLAGS.items():
        raw = spec["raw"]
        if raw not in out.columns:
            continue
        measured = out[raw].notna()
        for flag, rule in spec["flags"].items():
            if flag not in out.columns:
                continue
            original_missing = int(out[flag].isna().sum())
            original_positive = int((out[flag] == 1).sum())
            corrected = pd.Series(np.nan, index=out.index, dtype=float)
            corrected.loc[measured] = rule(out.loc[measured, raw]).astype(float)
            out[flag] = corrected
            corrections.append(
                {
                    "feature": flag,
                    "raw_measurement": raw,
                    "n_raw_missing": int((~measured).sum()),
                    "n_flag_missing_before": original_missing,
                    "n_positive_before": original_positive,
                    "n_flag_missing_after": int(out[flag].isna().sum()),
                    "n_positive_after": int((out[flag] == 1).sum()),
                    "correction": "raw_missing_set_to_feature_missing_not_normal",
                }
            )
    return out, pd.DataFrame(corrections)


def infer_feature_types(df):
    rows = []
    candidates = []
    feature_types = {}
    excluded_map = dict(EXCLUDED_ALWAYS)

    for col in df.columns:
        if col.endswith("_tokens"):
            excluded_map[col] = "excluded_token_helper_column"
        if col in {spec["raw"] for spec in RAW_LAB_TO_FLAGS.values()}:
            excluded_map[col] = "excluded_raw_gi_panel_value_engineered_abnormality_state_used"

    planned = []
    if "age" in df.columns:
        planned.append("age")
    if "breed_group" in df.columns:
        planned.append("breed_group")
    planned.extend(
        c
        for c in df.columns
        if c.startswith(BINARY_PREFIXES) and not c.endswith("_tokens")
    )
    planned.extend(flag for spec in RAW_LAB_TO_FLAGS.values() for flag in spec["flags"])
    planned = list(dict.fromkeys(c for c in planned if c in df.columns))

    for col in df.columns:
        eligible = col in planned and col not in excluded_map
        reason = "included"
        data_type = ""

        if not eligible:
            reason = excluded_map.get(col, "excluded_not_preplanned_predictor")
        else:
            if col == "age":
                data_type = "continuous"
            elif col == "breed_group":
                data_type = "categorical"
            else:
                values = set(df[col].dropna().unique().tolist())
                if values <= {0, 1, 0.0, 1.0, False, True}:
                    data_type = "binary"
                else:
                    data_type = "categorical"

            n_unique = df[col].nunique(dropna=True)
            if n_unique < 2:
                eligible = False
                reason = "excluded_no_variation_after_missingness_correction"

        if eligible:
            candidates.append(col)
            feature_types[col] = data_type

        rows.append(feature_inventory_row(df, col, data_type, eligible, reason))

    return candidates, feature_types, pd.DataFrame(rows)


def feature_inventory_row(df, col, data_type, eligible, reason):
    row = {
        "feature": col,
        "display_name": display_name(col),
        "source_modality": source_for_feature(col),
        "data_type": data_type if data_type else "excluded",
        "eligible": bool(eligible),
        "inclusion_exclusion_reason": reason,
        "n_rows": int(len(df)),
        "n_missing": int(df[col].isna().sum()),
        "missing_fraction": float(df[col].isna().mean()),
        "n_unique_nonmissing": int(df[col].nunique(dropna=True)),
        "prevalence_or_counts": "",
    }
    if data_type == "binary" or (col in df.columns and set(df[col].dropna().unique()) <= {0, 1, 0.0, 1.0, False, True}):
        nonmissing = df[col].notna()
        denom = int(nonmissing.sum())
        positives = int((df.loc[nonmissing, col] == 1).sum())
        row["positive_count_nonmissing"] = positives
        row["positive_fraction_nonmissing"] = float(positives / denom) if denom else np.nan
        row["prevalence_or_counts"] = json.dumps(
            df[col].value_counts(dropna=False).to_dict(), default=json_default
        )
    elif col in df.columns and data_type == "continuous":
        s = pd.to_numeric(df[col], errors="coerce")
        row["mean"] = finite_float(s.mean())
        row["median"] = finite_float(s.median())
        row["min"] = finite_float(s.min())
        row["max"] = finite_float(s.max())
    elif col in df.columns:
        row["prevalence_or_counts"] = json.dumps(
            df[col].value_counts(dropna=False).to_dict(), default=json_default
        )
    return row


def invalid_combination(features):
    feature_set = set(features)
    for _, spec in NO_ABNORMALITY_GROUPS.items():
        none_col = spec["none"]
        if none_col in feature_set:
            others = [
                f
                for f in feature_set
                if f != none_col and f.startswith(spec["members_prefix"])
            ]
            if others:
                return True, f"redundant_no_abnormality_with_{source_for_feature(none_col)}_abnormality"

    if "folate_high" in feature_set and "folate_low" in feature_set:
        return True, "mutually_exclusive_same_analyte_flags"
    if "tli_high" in feature_set and "tli_low" in feature_set:
        return True, "mutually_exclusive_same_analyte_flags"
    return False, ""


def build_combinations(candidates, feature_types, size):
    rows = []
    for combo in itertools.combinations(candidates, size):
        invalid, reason = invalid_combination(combo)
        if invalid:
            continue
        has_age = any(feature_types[f] == "continuous" for f in combo)
        rows.append(
            {
                "features": combo,
                "has_continuous": has_age,
                "has_age": "age" in combo,
                "feature_types": tuple(feature_types[f] for f in combo),
            }
        )
    return rows


def encode_joint_states(df, features, feature_types):
    valid = np.ones(len(df), dtype=bool)
    arrays = []
    levels = []
    all_binary = all(feature_types[f] == "binary" for f in features)

    for feature in features:
        valid &= df[feature].notna().to_numpy()

    if not valid.any():
        return np.full(len(df), -1, dtype=int), [], valid

    if all_binary:
        code = np.zeros(len(df), dtype=int)
        for i, feature in enumerate(features):
            values = df[feature].fillna(0).astype(int).to_numpy()
            code += values * (2 ** (len(features) - i - 1))
        code[~valid] = -1
        levels = [
            "".join(bits)
            for bits in itertools.product("01", repeat=len(features))
        ]
        return code, levels, valid

    tuples = []
    for feature in features:
        values = df[feature].astype(object).where(df[feature].notna(), None)
        arrays.append(values.to_numpy())
    for idx in range(len(df)):
        if not valid[idx]:
            tuples.append(None)
        else:
            tuples.append(tuple(arr[idx] for arr in arrays))
    observed_levels = sorted({t for t in tuples if t is not None}, key=lambda x: tuple(map(str, x)))
    mapping = {level: i for i, level in enumerate(observed_levels)}
    code = np.full(len(df), -1, dtype=int)
    for idx, value in enumerate(tuples):
        if value is not None:
            code[idx] = mapping[value]
    levels = ["|".join(str(v) for v in level) for level in observed_levels]
    return code, levels, valid


def safe_loglik_intercept(y):
    n = len(y)
    if n == 0:
        return np.nan
    p = np.clip(np.mean(y), 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def contingency_stat_from_codes(codes, row_totals, y, n_states):
    valid = codes >= 0
    n = int(valid.sum())
    if n == 0:
        return np.nan, np.nan, 0, 0
    y_valid = y[valid].astype(float)
    codes_valid = codes[valid]
    counts1 = np.bincount(codes_valid, weights=y_valid, minlength=n_states).astype(float)
    row = row_totals.astype(float)
    counts0 = row - counts1
    total1 = float(counts1.sum())
    total0 = float(counts0.sum())
    if total1 == 0 or total0 == 0:
        return 0.0, 0.0, max(int((row > 0).sum()) - 1, 0), n
    observed = np.vstack([counts0, counts1]).T
    col_totals = np.array([total0, total1], dtype=float)
    expected = np.outer(row, col_totals) / n
    mask = observed > 0
    g = 2.0 * float(np.sum(observed[mask] * np.log(observed[mask] / expected[mask])))
    mi = g / (2.0 * n)
    df = max(int((row > 0).sum()) - 1, 0)
    return g, mi, df, n


def group_age_loglik(y, age_z, group_codes, n_groups, max_iter=40, tol=1e-8):
    y = y.astype(float)
    if len(y) == 0:
        return np.nan, False
    if y.sum() == 0 or y.sum() == len(y):
        return 0.0, False

    theta = np.zeros(n_groups + 1, dtype=float)
    converged = False
    for _ in range(max_iter):
        alpha = theta[:n_groups]
        beta = theta[-1]
        eta = np.clip(alpha[group_codes] + beta * age_z, -35, 35)
        mu = expit(eta)
        w = np.clip(mu * (1 - mu), 1e-10, None)
        resid = y - mu

        grad = np.zeros(n_groups + 1, dtype=float)
        grad[:n_groups] = np.bincount(group_codes, weights=resid, minlength=n_groups)
        grad[-1] = float(np.sum(age_z * resid))

        hess = np.zeros((n_groups + 1, n_groups + 1), dtype=float)
        hess[np.arange(n_groups), np.arange(n_groups)] = np.bincount(
            group_codes, weights=w, minlength=n_groups
        )
        cross = np.bincount(group_codes, weights=w * age_z, minlength=n_groups)
        hess[:n_groups, -1] = cross
        hess[-1, :n_groups] = cross
        hess[-1, -1] = float(np.sum(w * age_z * age_z))
        hess.flat[:: n_groups + 2] += 1e-8

        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hess).dot(grad)
        theta += np.clip(step, -5, 5)
        theta = np.clip(theta, -40, 40)
        if np.max(np.abs(step)) < tol:
            converged = True
            break

    eta = np.clip(theta[:n_groups][group_codes] + theta[-1] * age_z, -35, 35)
    mu = np.clip(expit(eta), 1e-12, 1 - 1e-12)
    ll = float(np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu)))
    return ll, converged


def age_model_stat(df, y, features, feature_types):
    other_features = [f for f in features if feature_types[f] != "continuous"]
    valid = df["age"].notna().to_numpy().copy()
    if other_features:
        joint_codes, levels, joint_valid = encode_joint_states(df, other_features, feature_types)
        valid &= joint_valid
        codes = joint_codes[valid]
        n_groups = len(levels)
    else:
        codes = np.zeros(int(valid.sum()), dtype=int)
        n_groups = 1
        levels = ["all"]

    y_valid = y[valid]
    age = pd.to_numeric(df.loc[valid, "age"], errors="coerce").to_numpy(dtype=float)
    if len(y_valid) < 3 or np.nanstd(age) == 0 or n_groups < 1:
        return np.nan, np.nan, 0, int(len(y_valid)), "unidentifiable_age_model"
    age_z = (age - np.nanmean(age)) / np.nanstd(age)
    ll_null = safe_loglik_intercept(y_valid)
    ll_model, converged = group_age_loglik(y_valid, age_z, codes.astype(int), n_groups)
    stat = max(0.0, 2.0 * (ll_model - ll_null))
    # Model is one common age slope plus one intercept per joint state.
    df_model = n_groups
    warning = "" if converged else "age_model_newton_not_fully_converged"
    return stat, np.nan, df_model, int(len(y_valid)), warning


def age_stat_from_precomputed(item, y):
    valid = item["valid"]
    y_valid = y[valid]
    if len(y_valid) < 3:
        return np.nan, np.nan, 0, int(len(y_valid)), "unidentifiable_age_model"
    ll_null = safe_loglik_intercept(y_valid)
    ll_model, converged = group_age_loglik(
        y_valid,
        item["age_z"],
        item["group_codes"],
        item["n_groups"],
    )
    stat = max(0.0, 2.0 * (ll_model - ll_null))
    warning = "" if converged else "age_model_newton_not_fully_converged"
    return stat, np.nan, item["degrees_of_freedom"], int(len(y_valid)), warning


def precompute_candidate(df, combo, feature_types):
    if any(feature_types[f] == "continuous" for f in combo):
        other_features = [f for f in combo if feature_types[f] != "continuous"]
        valid = df["age"].notna().to_numpy().copy()
        if other_features:
            joint_codes, levels, joint_valid = encode_joint_states(df, other_features, feature_types)
            valid &= joint_valid
            codes = joint_codes[valid].astype(int)
            n_groups = len(levels)
        else:
            levels = ["all"]
            codes = np.zeros(int(valid.sum()), dtype=int)
            n_groups = 1
        age = pd.to_numeric(df.loc[valid, "age"], errors="coerce").to_numpy(dtype=float)
        if len(age) and np.nanstd(age) > 0:
            age_z = (age - np.nanmean(age)) / np.nanstd(age)
        else:
            age_z = np.zeros(len(age), dtype=float)
        return {
            "features": combo,
            "kind": "age",
            "valid": valid,
            "group_codes": codes,
            "levels": levels,
            "n_groups": n_groups,
            "age_z": age_z,
            "degrees_of_freedom": n_groups,
        }
    codes, levels, valid = encode_joint_states(df, combo, feature_types)
    n_states = len(levels)
    row_totals = np.bincount(codes[valid], minlength=n_states).astype(float)
    return {
        "features": combo,
        "kind": "contingency",
        "codes": codes,
        "levels": levels,
        "valid": valid,
        "row_totals": row_totals,
        "n_states": n_states,
    }


def sparse_warnings_from_table(table):
    n_values = table["n"].to_numpy()
    warnings = []
    if np.any(n_values == 0):
        warnings.append("empty_joint_state")
    if np.any((n_values > 0) & (n_values < 3)):
        warnings.append("joint_state_n_lt_3")
    if np.any((n_values > 0) & (n_values < 5)):
        warnings.append("joint_state_n_lt_5")
    nonempty = table[table["n"] > 0]
    if len(nonempty):
        lymphoma_fracs = nonempty["lymphoma_fraction"].dropna()
        tiny = nonempty[nonempty["n"] < 5]
        if len(tiny) and len(lymphoma_fracs):
            global_min = lymphoma_fracs.min()
            global_max = lymphoma_fracs.max()
            tiny_extreme = tiny["lymphoma_fraction"].isin([global_min, global_max]).any()
            if tiny_extreme:
                warnings.append("apparent_extreme_fraction_in_tiny_joint_state")
    return ";".join(warnings)


def state_table(df, y, combo, feature_types, label_map):
    other_features = [f for f in combo if feature_types[f] != "continuous"]
    has_age = len(other_features) < len(combo)
    if other_features:
        codes, levels, valid = encode_joint_states(df, other_features, feature_types)
    else:
        valid = df["age"].notna().to_numpy().copy()
        codes = np.zeros(len(df), dtype=int)
        levels = ["all"]
    if has_age:
        valid &= df["age"].notna().to_numpy()

    rows = []
    y_valid = y[valid]
    total_0 = int((y_valid == 0).sum())
    total_1 = int((y_valid == 1).sum())
    total_n = int(valid.sum())
    for code, level in enumerate(levels):
        mask = valid & (codes == code)
        n = int(mask.sum())
        ibd = int((y[mask] == 0).sum())
        lymphoma = int((y[mask] == 1).sum())
        expected_ibd = n * total_0 / total_n if total_n else np.nan
        expected_lymphoma = n * total_1 / total_n if total_n else np.nan
        row = {
            "state": level,
            "n": n,
            "IBD": ibd,
            "lymphoma": lymphoma,
            "lymphoma_fraction": lymphoma / n if n else np.nan,
            "expected_IBD_under_independence": expected_ibd,
            "expected_lymphoma_under_independence": expected_lymphoma,
        }
        if other_features and all(feature_types[f] == "binary" for f in other_features):
            for i, feature in enumerate(other_features):
                row[label_map[feature]] = int(level[i])
        elif other_features:
            values = level.split("|")
            for feature, value in zip(other_features, values):
                row[label_map[feature]] = value
        if has_age and n:
            ages = pd.to_numeric(df.loc[mask, "age"], errors="coerce")
            row["age_mean"] = float(ages.mean())
            row["age_median"] = float(ages.median())
            row["age_min"] = float(ages.min())
            row["age_max"] = float(ages.max())
        rows.append(row)
    table = pd.DataFrame(rows)
    table["sparse_warning"] = ""
    if len(table):
        table["sparse_warning"] = table.apply(
            lambda r: "empty" if r["n"] == 0 else ("n_lt_3" if r["n"] < 3 else ("n_lt_5" if r["n"] < 5 else "")),
            axis=1,
        )
    return table


def evaluate_observed(df, y, precomputed, size, feature_types, label_map):
    rows = []
    for idx, item in enumerate(precomputed):
        combo = item["features"]
        if item["kind"] == "contingency":
            stat, mi, df_stat, n = contingency_stat_from_codes(
                item["codes"], item["row_totals"], y, item["n_states"]
            )
            table = state_table(df, y, combo, feature_types, label_map)
            warning = sparse_warnings_from_table(table)
            statistic_type = "contingency_likelihood_ratio_G"
        else:
            stat, mi, df_stat, n, age_warning = age_stat_from_precomputed(item, y)
            table = state_table(df, y, combo, feature_types, label_map)
            warning = sparse_warnings_from_table(table)
            if age_warning:
                warning = ";".join([w for w in [warning, age_warning] if w])
            statistic_type = "logistic_deviance_improvement_age_linear_additive_to_joint_state"

        rows.append(
            {
                "candidate_id": f"{size}_{idx:05d}",
                "combination_size": size,
                "variables": "|".join(combo),
                "display_variables": "|".join(label_map[f] for f in combo),
                "association_statistic": stat,
                "mutual_information_nats": mi,
                "degrees_of_freedom": df_stat,
                "usable_n": n,
                "statistic_type": statistic_type,
                "sparse_warnings": warning,
                "sources": "|".join(source_for_feature(f) for f in combo),
                "contains_age": "age" in combo,
            }
        )
    return pd.DataFrame(rows)


def compute_permutation_stats(df, y_perm, precomputed, feature_types):
    stats = np.empty(len(precomputed), dtype=float)
    for idx, item in enumerate(precomputed):
        if item["kind"] == "contingency":
            stats[idx] = contingency_stat_from_codes(
                item["codes"], item["row_totals"], y_perm, item["n_states"]
            )[0]
        else:
            stats[idx] = age_stat_from_precomputed(item, y_perm)[0]
    return stats


def permutation_test(df, y, candidate_sets, observed_sets, feature_types, n_permutations, seed):
    rng = np.random.default_rng(seed)
    exceed = {
        size: np.zeros(len(candidate_sets[size]), dtype=int)
        for size in candidate_sets
    }
    max_rows = []
    observed_values = {
        size: observed_sets[size]["association_statistic"].to_numpy()
        for size in observed_sets
    }

    start = time.time()
    for perm_idx in range(n_permutations):
        y_perm = rng.permutation(y)
        row = {"permutation": perm_idx + 1}
        for size, precomputed in candidate_sets.items():
            stats = compute_permutation_stats(df, y_perm, precomputed, feature_types)
            exceed[size] += stats >= observed_values[size] - 1e-12
            row[f"G_max_{size}"] = float(np.nanmax(stats)) if len(stats) else np.nan
        max_rows.append(row)
        if (perm_idx + 1) % 100 == 0:
            elapsed = time.time() - start
            print(f"Completed {perm_idx + 1}/{n_permutations} permutations in {elapsed:.1f}s", flush=True)

    max_df = pd.DataFrame(max_rows)
    return exceed, max_df


def add_permutation_columns(results, exceed, max_df, size, n_permutations):
    out = results.copy()
    out["raw_empirical_p_value"] = (exceed + 1) / (n_permutations + 1)
    max_col = f"G_max_{size}"
    out["search_adjusted_empirical_p_value"] = [
        (int((max_df[max_col] >= stat - 1e-12).sum()) + 1) / (n_permutations + 1)
        for stat in out["association_statistic"]
    ]
    pvals = out["raw_empirical_p_value"].fillna(1.0).to_numpy()
    out["bh_fdr_q_value"] = bh_fdr(pvals) if len(out) else []
    out = out.sort_values(
        ["association_statistic", "mutual_information_nats"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)
    out.insert(0, "rank_by_association_statistic", np.arange(1, len(out) + 1))
    return out


def validation_checks(df, y, candidates, feature_types):
    rows = []
    binary = [f for f in candidates if feature_types[f] == "binary"]
    checked = 0
    for combo in itertools.combinations(binary, 2):
        invalid, _ = invalid_combination(combo)
        if invalid:
            continue
        item = precompute_candidate(df, combo, feature_types)
        stat, _, _, n = contingency_stat_from_codes(item["codes"], item["row_totals"], y, item["n_states"])
        group_codes = item["codes"][item["valid"]]
        y_valid = y[item["valid"]]
        ll_null = safe_loglik_intercept(y_valid)
        row_totals = item["row_totals"]
        counts1 = np.bincount(group_codes, weights=y_valid, minlength=item["n_states"])
        counts0 = row_totals - counts1
        ll_group = 0.0
        for n0, n1 in zip(counts0, counts1):
            total = n0 + n1
            if total == 0:
                continue
            p = np.clip(n1 / total, 1e-12, 1 - 1e-12)
            ll_group += n1 * np.log(p) + n0 * np.log(1 - p)
        lr_stat = 2.0 * (ll_group - ll_null)
        rows.append(
            {
                "check": "binary_pair_contingency_G_matches_group_logistic_LR",
                "features": "|".join(combo),
                "contingency_G": stat,
                "group_logistic_LR": lr_stat,
                "absolute_difference": abs(stat - lr_stat),
                "passed": bool(abs(stat - lr_stat) < 1e-8),
            }
        )
        checked += 1
        if checked >= 10:
            break

    for analyte, spec in RAW_LAB_TO_FLAGS.items():
        raw = spec["raw"]
        if raw not in df.columns:
            continue
        for flag in spec["flags"]:
            if flag not in df.columns:
                continue
            missing_raw_flag_missing = bool(df.loc[df[raw].isna(), flag].isna().all())
            rows.append(
                {
                    "check": "gi_panel_raw_missing_preserved_in_flag",
                    "features": flag,
                    "contingency_G": np.nan,
                    "group_logistic_LR": np.nan,
                    "absolute_difference": np.nan,
                    "passed": missing_raw_flag_missing,
                }
            )

    for combo in itertools.combinations(candidates, 2):
        invalid, reason = invalid_combination(combo)
        if invalid:
            rows.append(
                {
                    "check": "redundant_or_impossible_combination_excluded",
                    "features": "|".join(combo),
                    "contingency_G": np.nan,
                    "group_logistic_LR": np.nan,
                    "absolute_difference": np.nan,
                    "passed": True,
                    "reason": reason,
                }
            )
            if sum(r["check"] == "redundant_or_impossible_combination_excluded" for r in rows) >= 10:
                break
    return pd.DataFrame(rows)


def top_state_tables(df, y, results, feature_types, label_map, top_n):
    tables = []
    for _, result in results.head(top_n).iterrows():
        combo = tuple(result["variables"].split("|"))
        table = state_table(df, y, combo, feature_types, label_map)
        table.insert(0, "candidate_id", result["candidate_id"])
        table.insert(1, "rank_by_association_statistic", int(result["rank_by_association_statistic"]))
        table.insert(2, "display_variables", result["display_variables"])
        tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def classify_modality(variables):
    sources = [source_for_feature(v) for v in variables]
    if len(set(sources)) == 1:
        return f"within_{sources[0].replace(' ', '_')}"
    return "cross_" + "_plus_".join(sorted(set(s.replace(" ", "_") for s in sources)))


def modality_rankings(pair_df, triple_df):
    pieces = []
    for df, size in [(pair_df, 2), (triple_df, 3)]:
        tmp = df.copy()
        tmp["modality_group"] = tmp["variables"].apply(lambda s: classify_modality(s.split("|")))
        tmp["within_or_cross_modality"] = np.where(
            tmp["modality_group"].str.startswith("within_"), "within", "cross"
        )
        tmp["rank_within_modality_group"] = (
            tmp.groupby("modality_group")["association_statistic"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
        tmp = tmp[tmp["rank_within_modality_group"] <= 25].copy()
        tmp["combination_size"] = size
        pieces.append(tmp)
    return pd.concat(pieces, ignore_index=True)


def feature_frequency(results, top_ns=(20, 100)):
    rows = []
    for top_n in top_ns:
        top = results.head(top_n)
        counts = Counter()
        for variables in top["variables"]:
            counts.update(variables.split("|"))
        for feature, count in counts.most_common():
            rows.append(
                {
                    "combination_size": int(top["combination_size"].iloc[0]) if len(top) else np.nan,
                    "top_n": top_n,
                    "feature": feature,
                    "display_name": display_name(feature),
                    "count": count,
                    "fraction_of_top_n": count / max(len(top), 1),
                }
            )
    return pd.DataFrame(rows)


def plot_bar(df, value_col, label_col, title, path):
    if df.empty:
        return
    plot_df = df.head(20).iloc[::-1]
    height = max(5, 0.32 * len(plot_df) + 1.5)
    plt.figure(figsize=(11, height))
    plt.barh(plot_df[label_col], plot_df[value_col], color="#3b6ea8")
    plt.xlabel(value_col.replace("_", " "))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def make_plots(outdir, single_df, pair_df, triple_df, max_df, pair_freq, triple_freq):
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_bar(single_df, "association_statistic", "display_variables", "Top 20 Single Associations", plot_dir / "top_20_singles.png")
    plot_bar(pair_df, "association_statistic", "display_variables", "Top 20 Pair Associations", plot_dir / "top_20_pairs.png")
    plot_bar(triple_df, "association_statistic", "display_variables", "Top 20 Triple Associations", plot_dir / "top_20_triples.png")
    plot_bar(single_df, "mutual_information_nats", "display_variables", "Top 20 Single Mutual Information", plot_dir / "top_20_single_mi.png")
    plot_bar(pair_df, "mutual_information_nats", "display_variables", "Top 20 Pair Mutual Information", plot_dir / "top_20_pair_mi.png")
    plot_bar(triple_df, "mutual_information_nats", "display_variables", "Top 20 Triple Mutual Information", plot_dir / "top_20_triple_mi.png")

    for size, df in [(1, single_df), (2, pair_df), (3, triple_df)]:
        col = f"G_max_{size}"
        if col not in max_df:
            continue
        plt.figure(figsize=(8, 5))
        plt.hist(max_df[col].dropna(), bins=35, color="#8fa7bf", edgecolor="white")
        observed = df["association_statistic"].max()
        plt.axvline(observed, color="#b7410e", linewidth=2, label="Observed best")
        plt.xlabel(f"Permutation maximum statistic, size {size}")
        plt.ylabel("Permutation count")
        plt.title(f"Observed Best vs Permutation Null: Size {size}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"permutation_max_size_{size}.png", dpi=200)
        plt.close()

    for freq, name in [(pair_freq, "pair"), (triple_freq, "triple")]:
        if freq.empty:
            continue
        for top_n in sorted(freq["top_n"].unique()):
            plot_df = freq[freq["top_n"] == top_n].sort_values("count", ascending=False).head(20).iloc[::-1]
            plt.figure(figsize=(9, max(5, 0.3 * len(plot_df) + 1.5)))
            plt.barh(plot_df["display_name"], plot_df["count"], color="#5c8f68")
            plt.xlabel("Occurrences")
            plt.title(f"Feature Frequency Among Top {top_n} {name.title()}s")
            plt.tight_layout()
            plt.savefig(plot_dir / f"feature_frequency_top_{top_n}_{name}s.png", dpi=200)
            plt.close()

    heat = pair_df.head(40).copy()
    if not heat.empty:
        features = sorted({f for s in heat["variables"] for f in s.split("|")})
        mat = pd.DataFrame(np.nan, index=[display_name(f) for f in features], columns=[display_name(f) for f in features])
        for _, row in heat.iterrows():
            f1, f2 = row["variables"].split("|")
            mat.loc[display_name(f1), display_name(f2)] = row["association_statistic"]
            mat.loc[display_name(f2), display_name(f1)] = row["association_statistic"]
        plt.figure(figsize=(12, 10))
        plt.imshow(mat, cmap="viridis")
        plt.xticks(range(len(mat.columns)), mat.columns, rotation=90, fontsize=7)
        plt.yticks(range(len(mat.index)), mat.index, fontsize=7)
        plt.colorbar(label="Association statistic")
        plt.title("Strongest Pair Associations Heatmap")
        plt.tight_layout()
        plt.savefig(plot_dir / "strongest_pair_association_heatmap.png", dpi=220)
        plt.close()


def markdown_table(df, cols, max_rows=10):
    if df.empty:
        return "_No rows._"
    view = df[cols].head(max_rows).copy()
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in view.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append("" if pd.isna(value) else f"{value:.4g}")
            else:
                values.append(str(value).replace("|", "<br>"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def high_low_state_examples(state_df, top_n=12):
    if state_df.empty:
        return pd.DataFrame()
    nonempty = state_df[state_df["n"] > 0].copy()
    nonempty["distance_from_overall"] = (nonempty["lymphoma_fraction"] - 74 / 108).abs()
    return nonempty.sort_values(["distance_from_overall", "n"], ascending=[False, False]).head(top_n)


def write_report(
    outdir,
    args,
    df,
    inventory,
    corrections,
    validation,
    single_df,
    pair_df,
    triple_df,
    single_tables,
    pair_tables,
    triple_tables,
    age_rankings,
    modality_df,
    pair_freq,
    triple_freq,
):
    included = inventory[inventory["eligible"]]
    excluded = inventory[~inventory["eligible"]]
    class_counts = df[TARGET_COL].value_counts().sort_index()
    best_single_adj = single_df["search_adjusted_empirical_p_value"].min()
    best_pair_adj = pair_df["search_adjusted_empirical_p_value"].min()
    best_triple_adj = triple_df["search_adjusted_empirical_p_value"].min()
    validation_passed = bool(validation["passed"].all()) if "passed" in validation else False

    top_cols = [
        "rank_by_association_statistic",
        "display_variables",
        "association_statistic",
        "mutual_information_nats",
        "raw_empirical_p_value",
        "search_adjusted_empirical_p_value",
        "usable_n",
        "sparse_warnings",
    ]
    lines = [
        "# Exploratory Association Analysis",
        "",
        "This is a descriptive association search within the 108-cat cohort, not an out-of-sample diagnostic or predictive modeling analysis.",
        "",
        "## Dataset and Feature Eligibility",
        "",
        f"- Input file: `{args.input_file}`",
        f"- Rows: {len(df)}",
        f"- Target: `0 = IBD`, `1 = low-grade lymphoma`",
        f"- Class balance: {int(class_counts.get(0, 0))} IBD and {int(class_counts.get(1, 0))} low-grade lymphoma",
        f"- Eligible predictors: {len(included)}",
        f"- Excluded columns: {len(excluded)}",
        f"- Procedure fields were excluded because procedure choice may encode clinician suspicion rather than biological information.",
        f"- Sex was excluded because raw `gender` contains `n` and `s`; the engineered `male` variable is constant at 0 and cannot be reliably decoded as male/female.",
        "",
        "Raw free text, IDs, dates, DOB, diagnosis-defining histopathology, diagnosis labels, treatment, response, current status, token helper columns, target-derived information, raw GI-panel values, and procedure variables were excluded from the main search.",
        "",
        "## Missing GI-Panel Handling",
        "",
        "GI-panel abnormality flags were corrected before analysis: when the raw measurement was missing, the corresponding engineered abnormality flag was set to missing and excluded only from candidate combinations that used that analyte. This avoids treating an unmeasured analyte as normal while preserving all 108 cats for combinations that do not use that analyte.",
        "",
        markdown_table(corrections, ["feature", "raw_measurement", "n_raw_missing", "n_positive_before", "n_positive_after"], max_rows=20),
        "",
        "## Association Statistics",
        "",
        "- All-binary and categorical-only combinations used contingency-table likelihood-ratio G statistics and mutual information directly from the joint state table.",
        "- Combinations containing age used logistic deviance improvement over an intercept-only model, with age kept continuous and linear.",
        "- For age plus binary/categorical variables, the exhaustive model used one common linear age slope plus intercepts for the joint non-age state. Age-by-state interactions were not used in the exhaustive search because many joint states are sparse or separated in this 108-cat dataset.",
        "- Empirical p-values used diagnosis-label permutations with fixed features.",
        f"- Permutations: {args.permutations}; random seed: {args.seed}.",
        "- Search-adjusted p-values are max-statistic family-wise p-values computed separately for singles, pairs, and triples.",
        "",
        "## Validation Checks",
        "",
        f"- Validation subset passed: {validation_passed}",
        "- Checked that binary contingency G matched the equivalent grouped likelihood-ratio calculation on sampled binary pairs.",
        "- Checked that GI-panel raw missingness remained missing in corrected flags.",
        "- Checked that redundant no-abnormality combinations were excluded from the candidate search.",
        "",
        "## Top Single Variables",
        "",
        markdown_table(single_df, top_cols, max_rows=15),
        "",
        "The strongest individual associations should be read as cohort-level descriptive contrasts. Sparse warnings mark variables where at least one state is based on very few cats.",
        "",
        "## Top Pairs",
        "",
        markdown_table(pair_df, top_cols, max_rows=15),
        "",
        "## Top Triples",
        "",
        markdown_table(triple_df, top_cols, max_rows=15),
        "",
        "## Multiple-Search Correction",
        "",
        f"- Best search-adjusted empirical p-value among singles: {best_single_adj:.4g}",
        f"- Best search-adjusted empirical p-value among pairs: {best_pair_adj:.4g}",
        f"- Best search-adjusted empirical p-value among triples: {best_triple_adj:.4g}",
        "",
        "These adjusted values answer whether a candidate exceeded the best candidate found anywhere in a label-shuffled dataset of the same search size. They are the primary correction for the exhaustive search.",
        "",
        "## Age in Combinations",
        "",
        markdown_table(age_rankings, top_cols + ["combination_size"], max_rows=15),
        "",
        "Age appears repeatedly in this section only if its continuous deviance contribution plus the joint non-age state ranks highly. This is still descriptive and should not be converted into an age threshold.",
        "",
        "## Modality-Specific Patterns",
        "",
        markdown_table(
            modality_df,
            [
                "combination_size",
                "modality_group",
                "rank_within_modality_group",
                "display_variables",
                "association_statistic",
                "search_adjusted_empirical_p_value",
                "sparse_warnings",
            ],
            max_rows=25,
        ),
        "",
        "## Feature Frequency Among Top Combinations",
        "",
        "Pair feature frequency:",
        "",
        markdown_table(pair_freq, ["top_n", "display_name", "count", "fraction_of_top_n"], max_rows=30),
        "",
        "Triple feature frequency:",
        "",
        markdown_table(triple_freq, ["top_n", "display_name", "count", "fraction_of_top_n"], max_rows=30),
        "",
        "## Clinically Interpretable State Examples",
        "",
        "Pair states with the largest distance from the cohort lymphoma fraction among top-ranked pairs:",
        "",
        markdown_table(
            high_low_state_examples(pair_tables),
            ["display_variables", "state", "n", "IBD", "lymphoma", "lymphoma_fraction", "sparse_warning"],
            max_rows=12,
        ),
        "",
        "Triple states with the largest distance from the cohort lymphoma fraction among top-ranked triples:",
        "",
        markdown_table(
            high_low_state_examples(triple_tables),
            ["display_variables", "state", "n", "IBD", "lymphoma", "lymphoma_fraction", "sparse_warning"],
            max_rows=12,
        ),
        "",
        "## Interpretation",
        "",
        "The top-ranked findings and combinations identify where diagnosis proportions differ most inside this cohort. Results with low search-adjusted empirical p-values are less compatible with random label assignment after accounting for the full search, while results with sparse-cell warnings are fragile and hypothesis-generating.",
        "",
        "Do not interpret any pair or triple here as a diagnostic rule, predictive signature, or sufficient test. That requires a separate out-of-sample predictive analysis.",
        "",
        "## Output Files",
        "",
        "- `association_feature_inventory.csv`",
        "- `single_associations.csv`",
        "- `pair_associations.csv`",
        "- `triple_associations.csv`",
        "- `top_single_state_tables.csv`",
        "- `top_pair_state_tables.csv`",
        "- `top_triple_state_tables.csv`",
        "- `permutation_max_statistics.csv`",
        "- `age_partner_rankings.csv`",
        "- `modality_specific_rankings.csv`",
        "- `feature_frequency_top_combinations.csv`",
        "- `validation_checks.csv`",
        "- `gi_panel_missingness_corrections.csv`",
        "- `plots/`",
    ]
    (outdir / "association_analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--top-state-tables", type=int, default=25)
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(args.input_file)
    df, corrections = prepare_analysis_frame(raw_df)
    y = df[TARGET_COL].to_numpy(dtype=int)

    candidates, feature_types, inventory = infer_feature_types(df)
    label_map = {feature: display_name(feature) for feature in candidates}
    inventory.to_csv(outdir / "association_feature_inventory.csv", index=False)
    corrections.to_csv(outdir / "gi_panel_missingness_corrections.csv", index=False)

    validation = validation_checks(df, y, candidates, feature_types)
    validation.to_csv(outdir / "validation_checks.csv", index=False)
    if not bool(validation["passed"].all()):
        raise RuntimeError("Validation checks failed; see validation_checks.csv")

    candidate_sets = {}
    observed_sets = {}
    for size in [1, 2, 3]:
        combos = build_combinations(candidates, feature_types, size)
        precomputed = [
            precompute_candidate(df, tuple(row["features"]), feature_types)
            for row in combos
        ]
        candidate_sets[size] = precomputed
        observed_sets[size] = evaluate_observed(df, y, precomputed, size, feature_types, label_map)
        print(f"Prepared {len(precomputed)} size-{size} candidates", flush=True)

    exceed, max_df = permutation_test(
        df, y, candidate_sets, observed_sets, feature_types, args.permutations, args.seed
    )
    max_df.to_csv(outdir / "permutation_max_statistics.csv", index=False)

    single_df = add_permutation_columns(observed_sets[1], exceed[1], max_df, 1, args.permutations)
    pair_df = add_permutation_columns(observed_sets[2], exceed[2], max_df, 2, args.permutations)
    triple_df = add_permutation_columns(observed_sets[3], exceed[3], max_df, 3, args.permutations)

    single_df.to_csv(outdir / "single_associations.csv", index=False)
    pair_df.to_csv(outdir / "pair_associations.csv", index=False)
    triple_df.to_csv(outdir / "triple_associations.csv", index=False)

    single_tables = top_state_tables(df, y, single_df, feature_types, label_map, args.top_state_tables)
    pair_tables = top_state_tables(df, y, pair_df, feature_types, label_map, args.top_state_tables)
    triple_tables = top_state_tables(df, y, triple_df, feature_types, label_map, args.top_state_tables)
    single_tables.to_csv(outdir / "top_single_state_tables.csv", index=False)
    pair_tables.to_csv(outdir / "top_pair_state_tables.csv", index=False)
    triple_tables.to_csv(outdir / "top_triple_state_tables.csv", index=False)

    age_rankings = pd.concat(
        [
            pair_df[pair_df["contains_age"]],
            triple_df[triple_df["contains_age"]],
        ],
        ignore_index=True,
    ).sort_values("association_statistic", ascending=False)
    age_rankings.to_csv(outdir / "age_partner_rankings.csv", index=False)

    modality_df = modality_rankings(pair_df, triple_df)
    modality_df.to_csv(outdir / "modality_specific_rankings.csv", index=False)

    pair_freq = feature_frequency(pair_df)
    triple_freq = feature_frequency(triple_df)
    feature_freq = pd.concat([pair_freq, triple_freq], ignore_index=True)
    feature_freq.to_csv(outdir / "feature_frequency_top_combinations.csv", index=False)

    make_plots(outdir, single_df, pair_df, triple_df, max_df, pair_freq, triple_freq)
    write_report(
        outdir,
        args,
        df,
        inventory,
        corrections,
        validation,
        single_df,
        pair_df,
        triple_df,
        single_tables,
        pair_tables,
        triple_tables,
        age_rankings,
        modality_df,
        pair_freq,
        triple_freq,
    )

    print(f"Analysis complete. Outputs saved to {outdir}", flush=True)


if __name__ == "__main__":
    main()
