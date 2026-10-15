"""
02_baseline_characteristics.py
==============================
Baseline characteristics table comparing Depression_Event = 0 vs = 1.

Statistical tests:
  - Binary categorical (2 levels)   : chi-square test
  - Multi-category categorical (3+)  : Mann-Whitney U test
  - Numeric variables                : independent two-sample t-test

Reporting:
  - Categorical: n (%)
  - Numeric: median (IQR)

Output:
  - baseline_table.csv / .xlsx in config.BASELINE_OUTPUT_DIR
  - Appends a "Baseline_Table" sheet to combined_analysis_results.xlsx if it exists.
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import config as C
from utils import (
    ensure_dir, format_pvalue, median_iqr_str, freq_percent_str,
)

warnings.filterwarnings("ignore")


def main():
    out_dir = ensure_dir(C.BASELINE_OUTPUT_DIR)

    # Load final analysis dataset
    print("Loading data...")
    df = pd.read_excel(C.FINAL_DATASET_PATH)

    # Split by event
    group0 = df[df[C.EVENT_COL] == 0]
    group1 = df[df[C.EVENT_COL] == 1]
    n0, n1 = len(group0), len(group1)
    print(f"No_Depression: n={n0}, Depression: n={n1}")

    # Predictor columns = everything except ID / event / time
    predictor_cols = [c for c in df.columns
                      if c not in (C.ID_COL, C.EVENT_COL, C.TIME_COL)]

    # Classify variables
    binary_categorical, multi_categorical, numeric_vars = [], [], []
    for col in predictor_cols:
        n_unique = df[col].nunique()
        if n_unique <= 1:
            continue
        if (pd.api.types.is_integer_dtype(df[col])
                and n_unique <= C.CATEGORICAL_MAX_UNIQUE):
            if n_unique == 2:
                binary_categorical.append(col)
            else:
                multi_categorical.append(col)
        else:
            numeric_vars.append(col)

    print(f"Binary categorical: {len(binary_categorical)}; "
          f"Multi-category: {len(multi_categorical)}; Numeric: {len(numeric_vars)}")

    rows = []

    # Header row with sample sizes
    rows.append({
        "Variable": "", "Category": "", "Label": "",
        "No_Depression": f"n = {n0}",
        "Depression": f"n = {n1}",
        "P_Value": "",
    })

    # --- Binary categorical: chi-square ---
    print("\n--- Binary categorical (chi-square) ---")
    for var in binary_categorical:
        levels = sorted(df[var].dropna().unique())
        table = [[(group0[var] == lv).sum(), (group1[var] == lv).sum()] for lv in levels]
        try:
            chi2, p_val, _, _ = stats.chi2_contingency(np.array(table))
        except Exception:
            p_val = np.nan

        rows.append({
            "Variable": var, "Category": "", "Label": "n (%)",
            "No_Depression": "", "Depression": "",
            "P_Value": format_pvalue(p_val),
        })
        for lv in levels:
            rows.append({
                "Variable": "", "Category": str(lv), "Label": str(lv),
                "No_Depression": freq_percent_str((group0[var] == lv).sum(), n0),
                "Depression": freq_percent_str((group1[var] == lv).sum(), n1),
                "P_Value": "",
            })
        print(f"  {var}: p={format_pvalue(p_val)}")

    # --- Multi-category: Mann-Whitney U ---
    print("\n--- Multi-category (Mann-Whitney U) ---")
    for var in multi_categorical:
        levels = sorted(df[var].dropna().unique())
        try:
            _, p_val = stats.mannwhitneyu(
                group0[var].dropna(), group1[var].dropna(), alternative="two-sided"
            )
        except Exception:
            p_val = np.nan

        rows.append({
            "Variable": var, "Category": "", "Label": "n (%)",
            "No_Depression": "", "Depression": "",
            "P_Value": format_pvalue(p_val),
        })
        for lv in levels:
            rows.append({
                "Variable": "", "Category": str(lv), "Label": str(lv),
                "No_Depression": freq_percent_str((group0[var] == lv).sum(), n0),
                "Depression": freq_percent_str((group1[var] == lv).sum(), n1),
                "P_Value": "",
            })
        print(f"  {var}: p={format_pvalue(p_val)}")

    # --- Numeric: t-test, report median (IQR) ---
    print("\n--- Numeric (t-test, median [IQR]) ---")
    for var in numeric_vars:
        try:
            _, p_val = stats.ttest_ind(
                group0[var].dropna(), group1[var].dropna(), equal_var=True
            )
        except Exception:
            p_val = np.nan

        rows.append({
            "Variable": var, "Category": "", "Label": "Median (IQR)",
            "No_Depression": median_iqr_str(group0[var]),
            "Depression": median_iqr_str(group1[var]),
            "P_Value": format_pvalue(p_val),
        })
        print(f"  {var}: p={format_pvalue(p_val)}")

    baseline_df = pd.DataFrame(rows)

    # Save
    csv_path = os.path.join(out_dir, "baseline_table.csv")
    xlsx_path = os.path.join(out_dir, "baseline_table.xlsx")
    baseline_df.to_csv(csv_path, index=False)
    baseline_df.to_excel(xlsx_path, index=False, sheet_name="Baseline_Table")
    print(f"\nSaved:\n  {csv_path}\n  {xlsx_path}")

    # Append to combined workbook if present
    combined_xlsx = os.path.join(out_dir, "combined_analysis_results.xlsx")
    if os.path.exists(combined_xlsx):
        existing = pd.read_excel(combined_xlsx, sheet_name=None)
        with pd.ExcelWriter(combined_xlsx, engine="openpyxl") as writer:
            for sn, sdf in existing.items():
                sdf.to_excel(writer, sheet_name=sn, index=False)
            baseline_df.to_excel(writer, sheet_name="Baseline_Table", index=False)
        print(f"Appended Baseline_Table to {combined_xlsx}")

    print(f"\nTotal rows: {len(baseline_df)} | "
          f"binary={len(binary_categorical)}, multi={len(multi_categorical)}, "
          f"numeric={len(numeric_vars)}")
    print("Done.")


if __name__ == "__main__":
    main()
