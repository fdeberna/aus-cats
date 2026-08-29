import argparse
import itertools
import json
from collections import Counter
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold


DEFAULT_INPUT = "data/cases_features_08222026.csv"
DEFAULT_OUTPUT_DIR = "results/nested_feature_strategy_08222026"
TARGET_COL = "target"
ID_COL = "Name"

BINARY_PREFIXES = ("clinical signs_", "AUS_", "CBC_", "chem_")
LAB_FLAG_COLS = [
    "cobalamin_low",
    "folate_high",
    "folate_low",
    "tli_high",
    "tli_low",
    "pli_abn",
]
CONTINUOUS_COLS = ["age"]
CATEGORICAL_COLS = ["breed_group", "procedure_clean"]

METRIC_ROWS = [
    "log_loss",
    "roc_auc",
    "brier",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
]


def infer_candidate_features(df):
    binary_cols = [
        c
        for c in df.columns
        if c.startswith(BINARY_PREFIXES) and not c.endswith("_tokens")
    ]
    planned = CONTINUOUS_COLS + LAB_FLAG_COLS + CATEGORICAL_COLS + binary_cols
    planned = [c for c in planned if c in df.columns]

    feature_types = {}
    excluded = []
    candidates = []

    for col in planned:
        nunique = df[col].nunique(dropna=True)
        if nunique < 2:
            excluded.append(
                {
                    "feature": col,
                    "reason": "excluded_no_variation",
                    "n_unique": nunique,
                    "n_missing": int(df[col].isna().sum()),
                }
            )
            continue

        if col in CONTINUOUS_COLS:
            feature_types[col] = "continuous"
        elif col in CATEGORICAL_COLS:
            feature_types[col] = "categorical"
        else:
            feature_types[col] = "binary"
        candidates.append(col)

    if "male" in df.columns and "male" not in planned:
        excluded.append(
            {
                "feature": "male",
                "reason": "excluded_not_requested_and_no_variation_in_new_file",
                "n_unique": int(df["male"].nunique(dropna=True)),
                "n_missing": int(df["male"].isna().sum()),
            }
        )

    return candidates, feature_types, pd.DataFrame(excluded)


def feature_summary(df, candidates, feature_types, target_col):
    rows = []
    y = df[target_col]
    for col in candidates:
        row = {
            "feature": col,
            "type": feature_types[col],
            "dtype": str(df[col].dtype),
            "n": int(len(df)),
            "n_missing": int(df[col].isna().sum()),
            "missing_fraction": float(df[col].isna().mean()),
            "n_unique": int(df[col].nunique(dropna=True)),
        }
        if feature_types[col] == "binary":
            row["positive_count"] = int((df[col] == 1).sum())
            row["positive_fraction"] = float((df[col] == 1).mean())
            table = pd.crosstab(df[col], y)
            row["target_table"] = table.to_json()
            row["separation_flag"] = bool((table == 0).any().any())
        elif feature_types[col] == "categorical":
            counts = df[col].value_counts(dropna=False)
            row["levels"] = json.dumps(counts.to_dict())
            row["min_level_count"] = int(counts.min())
            row["rare_level_flag"] = bool((counts < 5).any())
        rows.append(row)
    return pd.DataFrame(rows)


def categorical_level_summary(df, categorical_cols, target_col):
    rows = []
    for col in categorical_cols:
        if col not in df.columns:
            continue
        counts = pd.crosstab(df[col], df[target_col], dropna=False)
        for level, values in counts.iterrows():
            rows.append(
                {
                    "feature": col,
                    "level": level,
                    "count": int(values.sum()),
                    "target_0": int(values.get(0, 0)),
                    "target_1": int(values.get(1, 0)),
                    "rare_level_flag": bool(values.sum() < 5),
                }
            )
    return pd.DataFrame(rows)


def binary_pair_diagnostics(df, binary_cols, target_col):
    rows = []
    for f1, f2 in itertools.combinations(binary_cols, 2):
        joint = pd.crosstab(df[f1], df[f2])
        counts = {
            f"n_{a}{b}": int(joint.loc[a, b]) if a in joint.index and b in joint.columns else 0
            for a in [0, 1]
            for b in [0, 1]
        }
        cell_target = pd.crosstab([df[f1], df[f2]], df[target_col])
        sep = False
        for _, vals in cell_target.iterrows():
            if vals.sum() > 0 and (vals.get(0, 0) == 0 or vals.get(1, 0) == 0):
                sep = True
        rows.append(
            {
                "feature_1": f1,
                "feature_2": f2,
                **counts,
                "min_joint_cell_count": min(counts.values()),
                "sparse_joint_cell_flag": bool(min(counts.values()) < 5),
                "complete_or_quasi_separation_flag": sep,
            }
        )
    return pd.DataFrame(rows)


def add_interaction_column(X, f1, f2):
    out = X.copy()
    name = interaction_name(f1, f2)
    out[name] = out[f1].fillna(0).astype(float) * out[f2].fillna(0).astype(float)
    return out, name


def interaction_name(f1, f2):
    return f"interaction__{f1}__x__{f2}"


def should_include_binary_interaction(X_train, f1, f2, min_cell):
    joint = pd.crosstab(X_train[f1], X_train[f2])
    counts = [
        int(joint.loc[a, b]) if a in joint.index and b in joint.columns else 0
        for a in [0, 1]
        for b in [0, 1]
    ]
    return min(counts) >= min_cell, min(counts), counts


def mode_or_default(series, default=0):
    modes = series.dropna().mode()
    if len(modes):
        return modes.iloc[0]
    return default


def build_design_matrices(X_train, X_apply, features, feature_types, interaction_pair=None):
    train_parts = []
    apply_parts = []
    names = []

    for feature in features:
        feature_type = feature_types[feature]

        if feature_type == "continuous":
            train_values = pd.to_numeric(X_train[feature], errors="coerce")
            apply_values = pd.to_numeric(X_apply[feature], errors="coerce")
            median = train_values.median()
            train_values = train_values.fillna(median).astype(float)
            apply_values = apply_values.fillna(median).astype(float)
            mean = train_values.mean()
            std = train_values.std(ddof=0)
            if not np.isfinite(std) or std == 0:
                std = 1.0
            train_parts.append(((train_values - mean) / std).to_numpy().reshape(-1, 1))
            apply_parts.append(((apply_values - mean) / std).to_numpy().reshape(-1, 1))
            names.append(feature)

        elif feature_type == "binary":
            fill = mode_or_default(X_train[feature], default=0)
            train_values = X_train[feature].fillna(fill).astype(float)
            apply_values = X_apply[feature].fillna(fill).astype(float)
            train_parts.append(train_values.to_numpy().reshape(-1, 1))
            apply_parts.append(apply_values.to_numpy().reshape(-1, 1))
            names.append(feature)

        elif feature_type == "categorical":
            fill = mode_or_default(X_train[feature], default="missing")
            train_values = X_train[feature].fillna(fill).astype(str)
            apply_values = X_apply[feature].fillna(fill).astype(str)
            levels = sorted(train_values.dropna().unique())
            for level in levels:
                train_parts.append((train_values == level).astype(float).to_numpy().reshape(-1, 1))
                apply_parts.append((apply_values == level).astype(float).to_numpy().reshape(-1, 1))
                names.append(f"{feature}={level}")

    if interaction_pair is not None:
        f1, f2 = interaction_pair
        fill1 = mode_or_default(X_train[f1], default=0)
        fill2 = mode_or_default(X_train[f2], default=0)
        train_values = (
            X_train[f1].fillna(fill1).astype(float)
            * X_train[f2].fillna(fill2).astype(float)
        )
        apply_values = (
            X_apply[f1].fillna(fill1).astype(float)
            * X_apply[f2].fillna(fill2).astype(float)
        )
        train_parts.append(train_values.to_numpy().reshape(-1, 1))
        apply_parts.append(apply_values.to_numpy().reshape(-1, 1))
        names.append(interaction_name(f1, f2))

    return np.hstack(train_parts), np.hstack(apply_parts), names


def fit_logistic(X_train, y_train, c_value):
    model = LogisticRegression(
        C=c_value,
        solver="lbfgs",
        max_iter=500,
        random_state=0,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_candidate(
    X_outer,
    y_outer,
    features,
    feature_types,
    inner_splits,
    c_grid,
    interaction_min_cell,
):
    features = tuple(features)
    interaction_pair = None
    interaction_reason = "not_applicable"
    min_joint_cell_count = np.nan

    if (
        len(features) == 2
        and feature_types[features[0]] == "binary"
        and feature_types[features[1]] == "binary"
    ):
        include, min_joint_cell_count, _ = should_include_binary_interaction(
            X_outer, features[0], features[1], interaction_min_cell
        )
        if include:
            interaction_pair = features
            interaction_reason = "included_binary_binary"
        else:
            interaction_reason = "omitted_sparse_joint_cells"
    elif len(features) == 2:
        interaction_reason = "omitted_non_binary_or_categorical_pair"

    c_scores = []
    for c_value in c_grid:
        losses = []
        for train_idx, val_idx in inner_splits:
            X_train = X_outer.iloc[train_idx]
            X_val = X_outer.iloc[val_idx]
            y_train = y_outer.iloc[train_idx]
            y_val = y_outer.iloc[val_idx]

            train_matrix, val_matrix, _ = build_design_matrices(
                X_train, X_val, features, feature_types, interaction_pair
            )
            try:
                model = fit_logistic(train_matrix, y_train, c_value)
                pred = model.predict_proba(val_matrix)[:, 1]
                losses.append(log_loss(y_val, pred, labels=[0, 1]))
            except Exception:
                losses.append(np.inf)

        c_scores.append((c_value, float(np.mean(losses))))

    best_c, best_inner_log_loss = min(c_scores, key=lambda item: (item[1], item[0]))
    return {
        "features": features,
        "best_c": best_c,
        "inner_log_loss": best_inner_log_loss,
        "interaction_pair": interaction_pair,
        "interaction_reason": interaction_reason,
        "min_joint_cell_count": min_joint_cell_count,
    }


def fit_predict_selected(X_train, y_train, X_test, selection, feature_types):
    features = selection["features"]
    interaction_pair = selection["interaction_pair"]
    train_matrix, test_matrix, names = build_design_matrices(
        X_train, X_test, features, feature_types, interaction_pair
    )
    model = fit_logistic(train_matrix, y_train, selection["best_c"])
    pred = model.predict_proba(test_matrix)[:, 1]
    coefs = extract_coefficients(model, names)
    return pred, coefs


def extract_coefficients(model, names):
    rows = {"intercept": float(model.intercept_[0])}
    for name, coef in zip(names, model.coef_[0]):
        rows[str(name)] = float(coef)
    return rows


def metric_dict(y_true, pred_prob, threshold):
    y_pred = (pred_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    def safe_div(num, den):
        return float(num / den) if den else np.nan

    try:
        auc = roc_auc_score(y_true, pred_prob)
    except ValueError:
        auc = np.nan

    return {
        "log_loss": float(log_loss(y_true, pred_prob, labels=[0, 1])),
        "roc_auc": float(auc),
        "brier": float(brier_score_loss(y_true, pred_prob)),
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "ppv": safe_div(tp, tp + fp),
        "npv": safe_div(tn, tn + fn),
        "threshold": threshold,
        "n": int(len(y_true)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def evaluate_strategy_candidates(
    X_outer,
    y_outer,
    combos,
    feature_types,
    inner_splits,
    c_grid,
    interaction_min_cell,
    n_jobs,
):
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(evaluate_candidate)(
            X_outer,
            y_outer,
            combo,
            feature_types,
            inner_splits,
            c_grid,
            interaction_min_cell,
        )
        for combo in combos
    )
    return min(results, key=lambda item: (item["inner_log_loss"], item["features"]))


def run_nested_cv(
    df,
    candidates,
    feature_types,
    args,
):
    X = df[candidates].copy()
    y = df[TARGET_COL].astype(int)
    ids = df[ID_COL] if ID_COL in df.columns else pd.Series(df.index, index=df.index)
    dx = df["shorthand dx"] if "shorthand dx" in df.columns else y.astype(str)

    single_combos = [(f,) for f in candidates]
    pair_combos = list(itertools.combinations(candidates, 2))
    triple_combos = (
        list(itertools.combinations(candidates, 3)) if args.include_triples else []
    )

    predictions = []
    fold_rows = []
    selected_coefficients = []

    outer_cv = RepeatedStratifiedKFold(
        n_splits=args.outer_folds,
        n_repeats=args.outer_repeats,
        random_state=args.random_seed,
    )

    total_folds = args.outer_folds * args.outer_repeats
    for fold_number, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        repeat = (fold_number - 1) // args.outer_folds + 1
        outer_fold = (fold_number - 1) % args.outer_folds + 1
        print(f"Nested CV repeat {repeat}/{args.outer_repeats}, fold {outer_fold}/{args.outer_folds}")

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        inner_cv = StratifiedKFold(
            n_splits=args.inner_folds,
            shuffle=True,
            random_state=args.random_seed + 1000 + fold_number,
        )
        inner_splits = list(inner_cv.split(X_train, y_train))

        single_sel = evaluate_strategy_candidates(
            X_train,
            y_train,
            single_combos,
            feature_types,
            inner_splits,
            args.c_grid,
            args.interaction_min_cell,
            args.n_jobs,
        )
        pair_sel = evaluate_strategy_candidates(
            X_train,
            y_train,
            pair_combos,
            feature_types,
            inner_splits,
            args.c_grid,
            args.interaction_min_cell,
            args.n_jobs,
        )

        selections = {"single": single_sel, "pair": pair_sel}
        if triple_combos:
            selections["triple"] = evaluate_strategy_candidates(
                X_train,
                y_train,
                triple_combos,
                feature_types,
                inner_splits,
                args.c_grid,
                args.interaction_min_cell,
                args.n_jobs,
            )

        fold_preds = {}
        for strategy, selection in selections.items():
            pred, coefs = fit_predict_selected(
                X_train, y_train, X_test, selection, feature_types
            )
            fold_preds[strategy] = pred
            metrics = metric_dict(y_test.to_numpy(), pred, args.threshold)
            fold_rows.append(
                {
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "strategy": strategy,
                    "selected_features": " | ".join(selection["features"]),
                    "selected_c": selection["best_c"],
                    "inner_log_loss": selection["inner_log_loss"],
                    "interaction_reason": selection["interaction_reason"],
                    "min_joint_cell_count": selection["min_joint_cell_count"],
                    **metrics,
                }
            )
            selected_coefficients.append(
                {
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "strategy": strategy,
                    "selected_features": " | ".join(selection["features"]),
                    "selected_c": selection["best_c"],
                    "coefficients_json": json.dumps(coefs),
                }
            )

        for row_pos, idx in enumerate(test_idx):
            pred_row = {
                "patient_id": ids.iloc[idx],
                "repeat": repeat,
                "outer_fold": outer_fold,
                "true_target": int(y.iloc[idx]),
                "true_diagnosis": dx.iloc[idx],
                "single_pred_prob": fold_preds["single"][row_pos],
                "pair_pred_prob": fold_preds["pair"][row_pos],
                "selected_single_feature": " | ".join(single_sel["features"]),
                "selected_single_c": single_sel["best_c"],
                "selected_pair_feature_1": pair_sel["features"][0],
                "selected_pair_feature_2": pair_sel["features"][1],
                "selected_pair_c": pair_sel["best_c"],
                "selected_pair_interaction_reason": pair_sel["interaction_reason"],
                "selected_pair_min_joint_cell_count": pair_sel["min_joint_cell_count"],
            }
            if "triple" in selections:
                triple_sel = selections["triple"]
                pred_row["triple_pred_prob"] = fold_preds["triple"][row_pos]
                pred_row["selected_triple_features"] = " | ".join(triple_sel["features"])
                pred_row["selected_triple_c"] = triple_sel["best_c"]
            predictions.append(pred_row)

    return (
        pd.DataFrame(predictions),
        pd.DataFrame(fold_rows),
        pd.DataFrame(selected_coefficients),
    )


def summarize_repeat_performance(pred_df, threshold):
    rows = []
    for repeat, g in pred_df.groupby("repeat"):
        single = metric_dict(g["true_target"].to_numpy(), g["single_pred_prob"].to_numpy(), threshold)
        pair = metric_dict(g["true_target"].to_numpy(), g["pair_pred_prob"].to_numpy(), threshold)
        row = {"repeat": repeat}
        for metric in METRIC_ROWS:
            row[f"single_{metric}"] = single[metric]
            row[f"pair_{metric}"] = pair[metric]
        row["delta_log_loss_single_minus_pair"] = single["log_loss"] - pair["log_loss"]
        row["delta_auc_pair_minus_single"] = pair["roc_auc"] - single["roc_auc"]
        row["delta_brier_single_minus_pair"] = single["brier"] - pair["brier"]
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_overall_performance(repeat_df, pred_df, threshold):
    pooled_single = metric_dict(
        pred_df["true_target"].to_numpy(),
        pred_df["single_pred_prob"].to_numpy(),
        threshold,
    )
    pooled_pair = metric_dict(
        pred_df["true_target"].to_numpy(),
        pred_df["pair_pred_prob"].to_numpy(),
        threshold,
    )

    rows = []
    for strategy, pooled in [("single", pooled_single), ("pair", pooled_pair)]:
        row = {"strategy": strategy}
        for metric in METRIC_ROWS:
            values = repeat_df[f"{strategy}_{metric}"]
            row[f"{metric}_pooled"] = pooled[metric]
            row[f"{metric}_repeat_mean"] = float(values.mean())
            row[f"{metric}_repeat_sd"] = float(values.std(ddof=1))
            row[f"{metric}_repeat_p025"] = float(values.quantile(0.025))
            row[f"{metric}_repeat_p975"] = float(values.quantile(0.975))
        rows.append(row)

    delta_rows = []
    for metric_col in [
        "delta_log_loss_single_minus_pair",
        "delta_auc_pair_minus_single",
        "delta_brier_single_minus_pair",
    ]:
        values = repeat_df[metric_col]
        delta_rows.append(
            {
                "quantity": metric_col,
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "p025": float(values.quantile(0.025)),
                "p50": float(values.quantile(0.5)),
                "p975": float(values.quantile(0.975)),
                "fraction_positive": float((values > 0).mean()),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(delta_rows)


def selection_frequency_tables(fold_df):
    single = fold_df[fold_df["strategy"] == "single"]
    single_counts = single["selected_features"].value_counts().reset_index()
    single_counts.columns = ["feature", "times_selected"]
    single_counts["fraction_selected"] = single_counts["times_selected"] / len(single)

    pair = fold_df[fold_df["strategy"] == "pair"].copy()
    pair[["feature_1", "feature_2"]] = pair["selected_features"].str.split(
        " \\| ", expand=True
    )
    pair_counts = (
        pair.groupby(["feature_1", "feature_2"])
        .size()
        .reset_index(name="times_selected")
        .sort_values("times_selected", ascending=False)
    )
    pair_counts["fraction_selected"] = pair_counts["times_selected"] / len(pair)

    feature_counter = Counter()
    for _, row in pair.iterrows():
        feature_counter[row["feature_1"]] += 1
        feature_counter[row["feature_2"]] += 1
    pair_feature_counts = pd.DataFrame(
        [
            {"feature": feature, "times_in_selected_pair": count}
            for feature, count in feature_counter.items()
        ]
    ).sort_values("times_in_selected_pair", ascending=False)
    pair_feature_counts["fraction_of_selected_pairs"] = (
        pair_feature_counts["times_in_selected_pair"] / len(pair)
    )

    return single_counts, pair_counts, pair_feature_counts


def exploratory_ranking(
    df,
    candidates,
    feature_types,
    combo_size,
    args,
):
    X = df[candidates]
    y = df[TARGET_COL].astype(int)
    cv = StratifiedKFold(
        n_splits=args.exploratory_folds,
        shuffle=True,
        random_state=args.random_seed + 4242 + combo_size,
    )
    splits = list(cv.split(X, y))
    combos = list(itertools.combinations(candidates, combo_size))
    rows = Parallel(n_jobs=args.n_jobs, prefer="threads")(
        delayed(evaluate_candidate)(
            X,
            y,
            combo,
            feature_types,
            splits,
            args.c_grid,
            args.interaction_min_cell,
        )
        for combo in combos
    )

    out_rows = []
    for row in rows:
        pred, coefs = fit_predict_selected(X, y, X, row, feature_types)
        apparent = metric_dict(y.to_numpy(), pred, args.threshold)
        out_rows.append(
            {
                "features": " | ".join(row["features"]),
                "feature_1": row["features"][0],
                "feature_2": row["features"][1] if len(row["features"]) > 1 else "",
                "feature_3": row["features"][2] if len(row["features"]) > 2 else "",
                "best_c": row["best_c"],
                "cv_log_loss": row["inner_log_loss"],
                "apparent_log_loss": apparent["log_loss"],
                "apparent_auc": apparent["roc_auc"],
                "apparent_brier": apparent["brier"],
                "interaction_reason": row["interaction_reason"],
                "min_joint_cell_count": row["min_joint_cell_count"],
                "coefficients_json": json.dumps(coefs),
            }
        )
    return pd.DataFrame(out_rows).sort_values("cv_log_loss")


def make_plots(outdir, repeat_df, single_freq, pair_freq, pred_df):
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.hist(repeat_df["delta_log_loss_single_minus_pair"], bins=12, edgecolor="black")
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Delta log loss: single - pair")
    plt.ylabel("Repeats")
    plt.title("Pair vs single log-loss improvement")
    plt.tight_layout()
    plt.savefig(plot_dir / "delta_log_loss_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(repeat_df["delta_auc_pair_minus_single"], bins=12, edgecolor="black")
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Delta AUC: pair - single")
    plt.ylabel("Repeats")
    plt.title("Pair vs single AUC improvement")
    plt.tight_layout()
    plt.savefig(plot_dir / "delta_auc_distribution.png", dpi=150)
    plt.close()

    top_single = single_freq.head(12).sort_values("times_selected")
    plt.figure(figsize=(8, 5))
    plt.barh(top_single["feature"], top_single["times_selected"])
    plt.xlabel("Times selected")
    plt.title("Top selected single features")
    plt.tight_layout()
    plt.savefig(plot_dir / "top_single_selection_frequencies.png", dpi=150)
    plt.close()

    top_pair = pair_freq.head(12).copy()
    top_pair["pair"] = top_pair["feature_1"] + " + " + top_pair["feature_2"]
    top_pair = top_pair.sort_values("times_selected")
    plt.figure(figsize=(9, 5))
    plt.barh(top_pair["pair"], top_pair["times_selected"])
    plt.xlabel("Times selected")
    plt.title("Top selected pairs")
    plt.tight_layout()
    plt.savefig(plot_dir / "top_pair_selection_frequencies.png", dpi=150)
    plt.close()

    bins = np.linspace(0, 1, 8)
    calib_rows = []
    for strategy, col in [("single", "single_pred_prob"), ("pair", "pair_pred_prob")]:
        tmp = pd.DataFrame({"prob": pred_df[col], "target": pred_df["true_target"]})
        tmp["bin"] = pd.cut(tmp["prob"], bins=bins, include_lowest=True)
        grouped = tmp.groupby("bin", observed=False)
        for _, g in grouped:
            if len(g) == 0:
                continue
            calib_rows.append(
                {
                    "strategy": strategy,
                    "mean_predicted_probability": float(g["prob"].mean()),
                    "observed_event_fraction": float(g["target"].mean()),
                    "n": int(len(g)),
                }
            )
    calib = pd.DataFrame(calib_rows)
    calib.to_csv(outdir / "calibration_bins.csv", index=False)

    plt.figure(figsize=(5, 5))
    for strategy, g in calib.groupby("strategy"):
        plt.plot(
            g["mean_predicted_probability"],
            g["observed_event_fraction"],
            marker="o",
            label=strategy,
        )
    plt.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed fraction target=1")
    plt.title("Calibration by probability bin")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "calibration_comparison.png", dpi=150)
    plt.close()


def write_report(
    outdir,
    df,
    candidates,
    feature_summary_df,
    class_balance,
    fold_df,
    repeat_df,
    overall_df,
    delta_df,
    single_freq,
    pair_freq,
    pair_feature_freq,
    exploratory_single,
    exploratory_pair,
    args,
):
    best_single = single_freq.iloc[0].to_dict() if len(single_freq) else {}
    best_pair = pair_freq.iloc[0].to_dict() if len(pair_freq) else {}
    deltas = delta_df.set_index("quantity")
    single_perf = overall_df[overall_df["strategy"] == "single"].iloc[0]
    pair_perf = overall_df[overall_df["strategy"] == "pair"].iloc[0]

    rare_count = int(
        (
            (feature_summary_df["type"] == "binary")
            & (
                (feature_summary_df["positive_count"] < args.rare_binary_count)
                | (
                    (feature_summary_df["n"] - feature_summary_df["positive_count"])
                    < args.rare_binary_count
                )
            )
        ).sum()
    )

    lines = [
        "# Nested CV Feature Strategy Analysis",
        "",
        "## Dataset",
        "",
        f"- Input file: `{args.input_file}`",
        f"- Rows: {len(df)}",
        f"- Target column: `{TARGET_COL}`; target=1 is low grade lymphoma.",
        f"- Class balance: {class_balance.to_dict(orient='records')}",
        f"- Candidate predictors used after documented exclusions: {len(candidates)}",
        f"- Singles evaluated per outer fold: {len(candidates)}",
        f"- Pairs evaluated per outer fold: {comb(len(candidates), 2)}",
        f"- Triples evaluated: {'yes' if args.include_triples else 'no'}",
        "",
        "The candidate set uses `age`, lab abnormality flags, the requested binary clinical/AUS/CBC/chem features, and `breed_group` plus `procedure_clean`. Raw text, diagnosis labels, outcomes after diagnosis, dates, IDs, raw lab values, token helper columns, and no-variation fields were excluded.",
        "",
        "## Validation Method",
        "",
        f"Repeated nested stratified CV was used with {args.outer_repeats} repeats of {args.outer_folds}-fold outer CV and {args.inner_folds}-fold inner CV. Within each outer training fold, all single features and all pairs were evaluated by inner-CV log loss, including ridge penalty selection over C values "
        f"{args.c_grid}. The selected model was then refit on the full outer training fold and evaluated once on the untouched outer test fold.",
        "",
        "Binary-by-binary interactions were included only when all four joint cells in the outer training fold had at least "
        f"{args.interaction_min_cell} observations. Other pair interactions were omitted to avoid adding unstable parameters for age or multi-level categorical variables.",
        "",
        "Threshold-dependent metrics used a fixed probability threshold of "
        f"{args.threshold:.2f}; no threshold was optimized on test data.",
        "",
        "## Out-of-Sample Performance",
        "",
        "| Strategy | Log loss mean | AUC mean | Brier mean | Sensitivity | Specificity |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| Single | {single_perf['log_loss_repeat_mean']:.3f} | {single_perf['roc_auc_repeat_mean']:.3f} | {single_perf['brier_repeat_mean']:.3f} | {single_perf['sensitivity_repeat_mean']:.3f} | {single_perf['specificity_repeat_mean']:.3f} |",
        f"| Pair | {pair_perf['log_loss_repeat_mean']:.3f} | {pair_perf['roc_auc_repeat_mean']:.3f} | {pair_perf['brier_repeat_mean']:.3f} | {pair_perf['sensitivity_repeat_mean']:.3f} | {pair_perf['specificity_repeat_mean']:.3f} |",
        "",
        "Positive delta log loss and Brier values mean the pair strategy improved over the single-feature strategy. Positive delta AUC means the pair strategy had higher AUC.",
        "",
        "| Difference | Mean | 2.5% | 97.5% | Fraction positive |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Log loss: single - pair | {deltas.loc['delta_log_loss_single_minus_pair', 'mean']:.4f} | {deltas.loc['delta_log_loss_single_minus_pair', 'p025']:.4f} | {deltas.loc['delta_log_loss_single_minus_pair', 'p975']:.4f} | {deltas.loc['delta_log_loss_single_minus_pair', 'fraction_positive']:.2f} |",
        f"| AUC: pair - single | {deltas.loc['delta_auc_pair_minus_single', 'mean']:.4f} | {deltas.loc['delta_auc_pair_minus_single', 'p025']:.4f} | {deltas.loc['delta_auc_pair_minus_single', 'p975']:.4f} | {deltas.loc['delta_auc_pair_minus_single', 'fraction_positive']:.2f} |",
        f"| Brier: single - pair | {deltas.loc['delta_brier_single_minus_pair', 'mean']:.4f} | {deltas.loc['delta_brier_single_minus_pair', 'p025']:.4f} | {deltas.loc['delta_brier_single_minus_pair', 'p975']:.4f} | {deltas.loc['delta_brier_single_minus_pair', 'fraction_positive']:.2f} |",
        "",
        "The uncertainty intervals above are empirical quantiles across repeated outer-CV runs. These repeats are not fully independent because the same patients appear across repeats, so the intervals should be read as a stability/sensitivity summary rather than formal independent-sample confidence intervals.",
        "",
        "## Selection Stability",
        "",
        f"- Most frequently selected single feature: `{best_single.get('feature', '')}` selected {best_single.get('times_selected', 0)} times.",
        f"- Most frequently selected pair: `{best_pair.get('feature_1', '')}` + `{best_pair.get('feature_2', '')}` selected {best_pair.get('times_selected', 0)} times.",
        f"- Top feature appearing in selected pairs: `{pair_feature_freq.iloc[0]['feature'] if len(pair_feature_freq) else ''}`.",
        "",
        "If selection frequencies are spread across many features or pairs, the apparent winner should be treated cautiously because small perturbations of the data change the selected model.",
        "",
        "## Sparse Diagnostics",
        "",
        f"- Rare binary predictors flagged: {rare_count}.",
        "- See `sparse_cell_diagnostics.csv`, `feature_diagnostic_summary.csv`, and `categorical_level_counts.csv` for details.",
        "",
        "## Exploratory Full-Data Ranking",
        "",
        "The full-data rankings are descriptive and optimistically biased because the same dataset is used to search many features/pairs and rank the winners. Use the nested-CV comparison above for the primary out-of-sample inference.",
        "",
        "Top exploratory single features by full-data CV log loss:",
        "",
        markdown_table(exploratory_single.head(10)[["features", "cv_log_loss", "best_c"]]),
        "",
        "Top exploratory pairs by full-data CV log loss:",
        "",
        markdown_table(exploratory_pair.head(10)[["features", "cv_log_loss", "best_c", "interaction_reason"]]),
        "",
        "## Files",
        "",
        "- `feature_diagnostic_summary.csv`",
        "- `class_balance.csv`",
        "- `binary_feature_prevalence.csv`",
        "- `categorical_level_counts.csv`",
        "- `sparse_cell_diagnostics.csv`",
        "- `nested_cv_overall_performance.csv`",
        "- `nested_cv_delta_summary.csv`",
        "- `nested_cv_repeat_level_performance.csv`",
        "- `nested_cv_fold_level_results.csv`",
        "- `nested_cv_out_of_fold_predictions.csv`",
        "- `single_feature_selection_frequency.csv`",
        "- `pair_selection_frequency.csv`",
        "- `feature_frequency_within_selected_pairs.csv`",
        "- `exploratory_single_feature_ranking.csv`",
        "- `exploratory_pair_ranking.csv`",
        "- `plots/`",
    ]
    (outdir / "nested_feature_strategy_report.md").write_text("\n".join(lines), encoding="utf-8")


def markdown_table(df):
    if df.empty:
        return "_Not generated._"
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--outer-repeats", type=int, default=20)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--exploratory-folds", type=int, default=5)
    parser.add_argument("--include-triples", action="store_true")
    parser.add_argument("--skip-exploratory", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-seed", type=int, default=20260822)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--interaction-min-cell", type=int, default=5)
    parser.add_argument("--rare-binary-count", type=int, default=5)
    parser.add_argument(
        "--c-grid",
        type=float,
        nargs="+",
        default=[0.03, 0.1, 0.3, 1.0, 3.0],
    )
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_file)
    candidates, feature_types, exclusions = infer_candidate_features(df)
    binary_cols = [c for c in candidates if feature_types[c] == "binary"]
    categorical_cols = [c for c in candidates if feature_types[c] == "categorical"]

    class_balance = (
        df.groupby([TARGET_COL, "shorthand dx"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    class_balance["fraction"] = class_balance["count"] / len(df)
    class_balance.to_csv(outdir / "class_balance.csv", index=False)

    diagnostics = feature_summary(df, candidates, feature_types, TARGET_COL)
    diagnostics.to_csv(outdir / "feature_diagnostic_summary.csv", index=False)
    exclusions.to_csv(outdir / "predictor_exclusions.csv", index=False)

    binary_prevalence = diagnostics[diagnostics["type"] == "binary"][
        [
            "feature",
            "positive_count",
            "positive_fraction",
            "n_missing",
            "separation_flag",
        ]
    ].copy()
    binary_prevalence["rare_binary_flag"] = (
        (binary_prevalence["positive_count"] < args.rare_binary_count)
        | ((len(df) - binary_prevalence["positive_count"]) < args.rare_binary_count)
    )
    binary_prevalence.to_csv(outdir / "binary_feature_prevalence.csv", index=False)

    categorical_levels = categorical_level_summary(df, categorical_cols, TARGET_COL)
    categorical_levels.to_csv(outdir / "categorical_level_counts.csv", index=False)

    pair_diag = binary_pair_diagnostics(df, binary_cols, TARGET_COL)
    pair_diag.to_csv(outdir / "sparse_cell_diagnostics.csv", index=False)

    pd.DataFrame(
        {
            "quantity": [
                "n_rows",
                "n_candidates",
                "n_single_models_per_outer_fold",
                "n_pair_models_per_outer_fold",
                "n_triple_models_per_outer_fold",
            ],
            "value": [
                len(df),
                len(candidates),
                len(candidates),
                comb(len(candidates), 2),
                comb(len(candidates), 3) if args.include_triples else 0,
            ],
        }
    ).to_csv(outdir / "analysis_dimensions.csv", index=False)

    pred_df, fold_df, coef_df = run_nested_cv(df, candidates, feature_types, args)
    pred_df.to_csv(outdir / "nested_cv_out_of_fold_predictions.csv", index=False)
    fold_df.to_csv(outdir / "nested_cv_fold_level_results.csv", index=False)
    coef_df.to_csv(outdir / "nested_cv_selected_model_coefficients.csv", index=False)

    repeat_df = summarize_repeat_performance(pred_df, args.threshold)
    repeat_df.to_csv(outdir / "nested_cv_repeat_level_performance.csv", index=False)
    overall_df, delta_df = summarize_overall_performance(repeat_df, pred_df, args.threshold)
    overall_df.to_csv(outdir / "nested_cv_overall_performance.csv", index=False)
    delta_df.to_csv(outdir / "nested_cv_delta_summary.csv", index=False)

    single_freq, pair_freq, pair_feature_freq = selection_frequency_tables(fold_df)
    single_freq.to_csv(outdir / "single_feature_selection_frequency.csv", index=False)
    pair_freq.to_csv(outdir / "pair_selection_frequency.csv", index=False)
    pair_feature_freq.to_csv(
        outdir / "feature_frequency_within_selected_pairs.csv", index=False
    )

    if args.skip_exploratory:
        exploratory_single = pd.DataFrame(columns=["features", "cv_log_loss", "best_c"])
        exploratory_pair = pd.DataFrame(
            columns=["features", "cv_log_loss", "best_c", "interaction_reason"]
        )
    else:
        print("Running exploratory single-feature ranking")
        exploratory_single = exploratory_ranking(df, candidates, feature_types, 1, args)
        exploratory_single.to_csv(
            outdir / "exploratory_single_feature_ranking.csv", index=False
        )

        print("Running exploratory pair ranking")
        exploratory_pair = exploratory_ranking(df, candidates, feature_types, 2, args)
        exploratory_pair.to_csv(outdir / "exploratory_pair_ranking.csv", index=False)

    make_plots(outdir, repeat_df, single_freq, pair_freq, pred_df)
    write_report(
        outdir,
        df,
        candidates,
        diagnostics,
        class_balance,
        fold_df,
        repeat_df,
        overall_df,
        delta_df,
        single_freq,
        pair_freq,
        pair_feature_freq,
        exploratory_single,
        exploratory_pair,
        args,
    )

    print(f"Analysis complete. Outputs saved to {outdir}")


if __name__ == "__main__":
    main()
