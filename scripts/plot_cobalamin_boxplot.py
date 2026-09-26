import argparse
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("data/cases_cleaned_08222026.csv")
DEFAULT_PNG = Path("results/cobalamin_boxplot.png")
DEFAULT_PDF = Path("results/cobalamin_boxplot.pdf")
DEFAULT_SUMMARY = Path("results/cobalamin_censored_summary.csv")
COBALAMIN_COL = "cobalamin (290-1500)"
DIAGNOSIS_COL = "shorthand dx"
LOWER_REPORTING_LIMIT = 150.0
REPORTING_LIMIT = 1000.0
RANDOM_SEED = 20260926


def read_xlsx_first_sheet_without_openpyxl(path):
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    with ZipFile(path) as workbook:
        shared_strings = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            shared_root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            for item in shared_root.findall("main:si", ns):
                shared_strings.append(
                    "".join(text.text or "" for text in item.findall(".//main:t", ns))
                )

        sheet_root = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in sheet_root.findall(".//main:sheetData/main:row", ns):
            values = []
            for cell in row.findall("main:c", ns):
                value = cell.find("main:v", ns)
                value = "" if value is None else value.text
                if cell.attrib.get("t") == "s" and value != "":
                    value = shared_strings[int(value)]
                values.append(value)
            if values:
                rows.append(values)

    return pd.DataFrame(rows)


def load_raw_table(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xlsm"}:
        try:
            return pd.read_excel(path, header=None)
        except ImportError:
            return read_xlsx_first_sheet_without_openpyxl(path)

    raise ValueError(f"Unsupported input format: {path}")


def parse_numeric(series):
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(r"[^0-9.+-]", "", regex=True)
        .replace("", pd.NA)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def normalize_diagnosis(value):
    diagnosis = str(value).strip().lower()
    if diagnosis == "ibd":
        return "IBD"
    if diagnosis in {"lymphoma", "low grade lymphoma", "low-grade lymphoma"}:
        return "LGAL"
    return pd.NA


def load_cobalamin_data(path, lower_reporting_limit, upper_reporting_limit):
    raw = load_raw_table(path)

    if {DIAGNOSIS_COL, COBALAMIN_COL}.issubset(raw.columns):
        df = raw[[DIAGNOSIS_COL, COBALAMIN_COL]].copy()
        df.columns = ["diagnosis_raw", "recorded_cobalamin"]
    else:
        if raw.shape[1] < 2:
            raise ValueError(f"Expected at least two columns in {path}")

        df = raw.iloc[:, :2].copy()
        df.columns = ["diagnosis_raw", "recorded_cobalamin"]
        df = df.dropna(how="all")

        first_row = df.iloc[0].astype(str).str.lower().str.strip().tolist()
        has_header = "diagnosis" in first_row[0] or "cobalamin" in first_row[1]
        if has_header:
            df = df.iloc[1:].copy()

    df["diagnosis"] = df["diagnosis_raw"].apply(normalize_diagnosis)
    df["recorded_cobalamin"] = parse_numeric(df["recorded_cobalamin"])
    df = df.dropna(subset=["diagnosis", "recorded_cobalamin"]).copy()

    df["cobalamin_lower_censored"] = df["recorded_cobalamin"] <= lower_reporting_limit
    df["cobalamin_upper_censored"] = df["recorded_cobalamin"] >= upper_reporting_limit
    df["cobalamin_censored"] = (
        df["cobalamin_lower_censored"] | df["cobalamin_upper_censored"]
    )
    df["plot_cobalamin"] = np.where(
        df["cobalamin_lower_censored"],
        lower_reporting_limit,
        np.where(
            df["cobalamin_upper_censored"],
            upper_reporting_limit,
            df["recorded_cobalamin"],
        ),
    )

    if df.empty:
        raise ValueError(f"No IBD or LGAL cats with cobalamin values found in {path}")

    return df


def format_ng_l(value):
    if float(value).is_integer():
        return f"{int(value)} ng/L"
    return f"{value:.1f} ng/L"


def censored_median_display(
    values, lower_censored, upper_censored, lower_reporting_limit, upper_reporting_limit
):
    values = pd.Series(values).reset_index(drop=True)
    lower_censored = pd.Series(lower_censored).reset_index(drop=True)
    upper_censored = pd.Series(upper_censored).reset_index(drop=True)
    exact = values[~lower_censored & ~upper_censored].sort_values().reset_index(drop=True)

    n_total = len(values)
    n_lower = int(lower_censored.sum())
    n_exact = len(exact)
    if n_total == 0:
        return ""

    def value_at_rank(rank):
        if rank < n_lower:
            return "lower", None
        exact_index = rank - n_lower
        if exact_index < n_exact:
            return "exact", exact.iloc[exact_index]
        return "upper", None

    if n_total % 2 == 1:
        kind, value = value_at_rank(n_total // 2)
        if kind == "lower":
            return f"<={int(lower_reporting_limit)} ng/L"
        if kind == "upper":
            return f">={int(upper_reporting_limit)} ng/L"
        return format_ng_l(value)

    left = value_at_rank(n_total // 2 - 1)
    right = value_at_rank(n_total // 2)
    if left[0] == right[0] == "lower":
        return f"<={int(lower_reporting_limit)} ng/L"
    if left[0] == right[0] == "upper":
        return f">={int(upper_reporting_limit)} ng/L"
    if left[0] == right[0] == "exact":
        return format_ng_l((left[1] + right[1]) / 2)
    if left[0] == "exact" and right[0] == "exact":
        return format_ng_l((left[1] + right[1]) / 2)
    if left[0] == "lower" and right[0] == "exact":
        return "not identifiable due to lower censoring"
    if left[0] == "exact" and right[0] == "upper":
        return "not identifiable due to upper censoring"
    return "not identifiable due to censoring"


def summarize_cobalamin(df, lower_reporting_limit, upper_reporting_limit):
    rows = []
    for diagnosis in ["IBD", "LGAL"]:
        group = df[df["diagnosis"] == diagnosis]
        n_total = len(group)
        n_lower_censored = int(group["cobalamin_lower_censored"].sum())
        n_upper_censored = int(group["cobalamin_upper_censored"].sum())
        n_censored = n_lower_censored + n_upper_censored
        n_uncensored = n_total - n_censored
        percent_censored = 100 * n_censored / n_total if n_total else 0
        median_display = censored_median_display(
            group["recorded_cobalamin"],
            group["cobalamin_lower_censored"],
            group["cobalamin_upper_censored"],
            lower_reporting_limit,
            upper_reporting_limit,
        )

        rows.append(
            {
                "diagnosis": diagnosis,
                "n_total_with_cobalamin": n_total,
                "n_uncensored": n_uncensored,
                "n_lower_censored": n_lower_censored,
                "n_upper_censored": n_upper_censored,
                "n_censored": n_censored,
                "percent_censored": round(percent_censored, 1),
                "median_display": median_display,
            }
        )

    return pd.DataFrame(rows)


def make_censored_jitter_plot(
    df, summary, png_path, pdf_path, lower_reporting_limit, upper_reporting_limit
):
    rng = np.random.default_rng(RANDOM_SEED)
    positions = {"IBD": 0, "LGAL": 1}
    colors = {"IBD": "#78A6C8", "LGAL": "#E3A857"}

    fig, ax = plt.subplots(figsize=(6.8, 5.0), constrained_layout=True)

    for diagnosis in ["IBD", "LGAL"]:
        group = df[df["diagnosis"] == diagnosis].copy()
        x_center = positions[diagnosis]
        jitter = rng.uniform(-0.13, 0.13, size=len(group))
        group["x"] = x_center + jitter

        uncensored = group[~group["cobalamin_censored"]]
        lower_censored = group[group["cobalamin_lower_censored"]]
        upper_censored = group[group["cobalamin_upper_censored"]]

        ax.scatter(
            uncensored["x"],
            uncensored["plot_cobalamin"],
            marker="o",
            s=42,
            facecolor=colors[diagnosis],
            edgecolor="#222222",
            linewidth=0.6,
            alpha=0.82,
            zorder=3,
        )
        ax.scatter(
            lower_censored["x"],
            lower_censored["plot_cobalamin"],
            marker="v",
            s=82,
            facecolor="#5A5A5A",
            edgecolor="#222222",
            linewidth=0.7,
            alpha=0.95,
            zorder=4,
        )
        ax.scatter(
            upper_censored["x"],
            upper_censored["plot_cobalamin"],
            marker="^",
            s=82,
            facecolor="#B43C2F",
            edgecolor="#222222",
            linewidth=0.7,
            alpha=0.95,
            zorder=4,
        )

    lower_limit_label = f"Lower reporting limit: <={int(lower_reporting_limit)} ng/L"
    upper_limit_label = f"Upper reporting limit: >={int(upper_reporting_limit)} ng/L"
    ax.axhline(
        lower_reporting_limit,
        color="#777777",
        linestyle="--",
        linewidth=1.0,
        zorder=1,
    )
    ax.axhline(
        upper_reporting_limit,
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        zorder=1,
    )
    ax.text(
        -0.46,
        lower_reporting_limit + 22,
        lower_limit_label,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2},
    )
    ax.text(
        0.5,
        upper_reporting_limit + 45,
        upper_limit_label,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2},
    )

    tick_labels = []
    for diagnosis in ["IBD", "LGAL"]:
        row = summary[summary["diagnosis"] == diagnosis].iloc[0]
        tick_labels.append(
            f"{diagnosis}\n"
            f"n = {row['n_total_with_cobalamin']}, "
            f"censored = {row['n_censored']}"
        )

    measured_handle = mlines.Line2D(
        [],
        [],
        color="#222222",
        marker="o",
        linestyle="None",
        markersize=7,
        markerfacecolor="#A9C5DA",
        label="measured value",
    )
    lower_censored_handle = mlines.Line2D(
        [],
        [],
        color="#222222",
        marker="v",
        linestyle="None",
        markersize=8,
        markerfacecolor="#5A5A5A",
        label=f"censored at <={int(lower_reporting_limit)} ng/L",
    )
    upper_censored_handle = mlines.Line2D(
        [],
        [],
        color="#222222",
        marker="^",
        linestyle="None",
        markersize=8,
        markerfacecolor="#B43C2F",
        label=f"censored at >={int(upper_reporting_limit)} ng/L",
    )

    ax.legend(
        handles=[measured_handle, lower_censored_handle, upper_censored_handle],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncols=3,
        frameon=False,
        fontsize=9,
    )
    ax.set_title("Serum cobalamin by diagnosis")
    ax.set_xlabel("Diagnosis")
    ax.set_ylabel("Serum cobalamin (ng/L)")
    ax.set_xticks([positions["IBD"], positions["LGAL"]])
    ax.set_xticklabels(tick_labels)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(lower_reporting_limit - 60, upper_reporting_limit + 110)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)


def verify_censoring(df, lower_reporting_limit, upper_reporting_limit):
    lower_triangle_count = int(df["cobalamin_lower_censored"].sum())
    upper_triangle_count = int(df["cobalamin_upper_censored"].sum())
    lower_coded_count = int((df["recorded_cobalamin"] <= lower_reporting_limit).sum())
    upper_coded_count = int((df["recorded_cobalamin"] >= upper_reporting_limit).sum())
    missing_count = int(df["recorded_cobalamin"].isna().sum())

    if lower_triangle_count != lower_coded_count:
        raise AssertionError("Lower-censored marker count does not match boundary values.")
    if upper_triangle_count != upper_coded_count:
        raise AssertionError("Upper-censored marker count does not match top-coded values.")
    if missing_count:
        raise AssertionError("Missing cobalamin values remained after filtering.")
    if (
        df.loc[~df["cobalamin_lower_censored"], "recorded_cobalamin"]
        <= lower_reporting_limit
    ).any():
        raise AssertionError("A lower boundary value was treated as uncensored.")
    if (
        df.loc[~df["cobalamin_upper_censored"], "recorded_cobalamin"]
        >= upper_reporting_limit
    ).any():
        raise AssertionError("An upper boundary value was treated as uncensored.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a censoring-aware cobalamin jitter plot by diagnosis. "
            "Values recorded at the reporting limit are plotted as censored markers."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--lower-reporting-limit", type=float, default=LOWER_REPORTING_LIMIT
    )
    parser.add_argument("--upper-reporting-limit", type=float, default=REPORTING_LIMIT)
    args = parser.parse_args()

    df = load_cobalamin_data(
        args.input, args.lower_reporting_limit, args.upper_reporting_limit
    )
    verify_censoring(df, args.lower_reporting_limit, args.upper_reporting_limit)

    summary = summarize_cobalamin(
        df, args.lower_reporting_limit, args.upper_reporting_limit
    )
    make_censored_jitter_plot(
        df,
        summary,
        args.png,
        args.pdf,
        args.lower_reporting_limit,
        args.upper_reporting_limit,
    )

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False)

    print(f"Saved PNG to {args.png}")
    print(f"Saved PDF to {args.pdf}")
    print(f"Saved summary to {args.summary}")
    print(summary.to_string(index=False))
    print(
        "Verified censored markers match recorded values "
        f"<={int(args.lower_reporting_limit)} and "
        f">={int(args.upper_reporting_limit)} ng/L; missing values were excluded."
    )


if __name__ == "__main__":
    main()
