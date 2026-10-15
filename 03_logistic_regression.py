"""
03_logistic_regression.py
=========================
Univariate and multivariable logistic regression of Depression_Event on
baseline predictors.

Pipeline:
  1. Load the final analysis dataset (config.FINAL_DATASET_PATH).
  2. Drop zero-variance columns.
  3. Run univariate logistic regression for each predictor (OR, 95% CI, p).
  4. Enter variables with univariate p < 0.1 into a multivariable model.
  5. Save results (CSV + Excel) to config.LOGISTIC_OUTPUT_DIR. 
"""

import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

import config as C
from utils import ensure_dir, format_pvalue

warnings.filterwarnings("ignore")

# Predictor lists (mirrors the numeric + categorical sets used in analysis)
NUMERIC_COLS = [
    "Baseline_Age",
    "Baseline_Cigarettes",
    "Baseline_IADL_Limitation_5Items",
    "Baseline_Frailty_5Items",
    "Baseline_ADL_Limitation_6Items",
    "Baseline_Mobility_Limitation_7Items",
    "Baseline_Lower_Limb_Limitation",
    "Baseline_Upper_Limb_Limitation",
    "Baseline_Immediate_Word_Recall_Score",
    "Baseline_Serial_Subtraction_Correct",
    "Baseline_Orientation_Score",
    "Baseline_Household_Total_Debts",
    "Baseline_Individual_Pension_Income",
    "Baseline_Individual_Transfer_Income",
    "Baseline_Individual_Other_Income",
    "Baseline_Number_Of_Living_Siblings",
    "Baseline_Transfer_From_Children",
    "Baseline_Transfer_To_Children",
    "Baseline_Transfer_From_Parents",
    "Baseline_Transfer_To_Other_Relatives",
    "Baseline_Mother_Warmth_Score",
    "Baseline_Individual_Total_Income",
    "Baseline_Total_Transfer_Income",
    "Baseline_Net_Transfer",
    "Baseline_Total_Word_Recall_Score",
    "Baseline_Deficit_Accumulation_Index",
    "Baseline_Individual_Earned_Income",
    "Baseline_Transfer_To_Parents",
]

# Categorical predictors = all Pattern_* columns + Gender (if present)


def _fit_logit(df: pd.DataFrame, cols, y):
    """Fit statsmodels Logit on df[cols] and return a results dict per column."""
    X = sm.add_constant(df[cols])
    model = sm.Logit(y, X).fit(disp=0)
    params = model.params
    ci = model.conf_int()
    pvals = model.pvalues
    out = []
    for col in cols:
        out.append({
            "Variable": col,
            "OR": float(np.exp(params[col])),
            "CI_Lower": float(np.exp(ci.loc[col, 0])),
            "CI_Upper": float(np.exp(ci.loc[col, 1])),
            "P_Value": float(pvals[col]),
            "P_Value_Formatted": format_pvalue(pvals[col]),
            "OR_95CI": f"{np.exp(params[col]):.3f} "
                       f"({np.exp(ci.loc[col, 0]):.3f}-{np.exp(ci.loc[col, 1]):.3f})",
        })
    return model, pd.DataFrame(out)


def main():
    out_dir = ensure_dir(C.LOGISTIC_OUTPUT_DIR)

    # 1. Load data
    df = pd.read_excel(C.FINAL_DATASET_PATH)
    print(f"Loaded data: {df.shape}")

    # 2. Drop zero-variance columns
    var = df.var(numeric_only=True)
    zv = var[var == 0].index.tolist()
    if zv:
        print(f"Dropping {len(zv)} zero-variance columns: {zv}")
        df = df.drop(columns=zv)

    # Ensure age is integer
    if "Baseline_Age" in df.columns:
        df["Baseline_Age"] = df["Baseline_Age"].round().astype(int)

    # Build predictor list
    numeric_cols = [c for c in NUMERIC_COLS if c in df.columns]
    categorical_cols = [c for c in df.columns if c.startswith("Pattern_")]
    if "Gender" in df.columns:
        categorical_cols.append("Gender")
    categorical_cols = [c for c in categorical_cols
                        if c not in (C.ID_COL, C.EVENT_COL, C.TIME_COL)]

    predictor_cols = numeric_cols + categorical_cols
    y = df[C.EVENT_COL]
    print(f"Predictors: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")

    # 3. Univariate logistic regression
    print("\nRunning univariate logistic regression...")
    uni_rows = []
    for i, col in enumerate(predictor_cols, 1):
        try:
            _, res_df = _fit_logit(df, [col], y)
            uni_rows.append(res_df.iloc[0].to_dict())
        except Exception as e:
            print(f"  [WARN] {col}: {e}")
        if i % 10 == 0:
            print(f"  ... {i}/{len(predictor_cols)}")

    uni_df = pd.DataFrame(uni_rows).sort_values("P_Value").reset_index(drop=True)
    uni_df.round(6).to_csv(os.path.join(out_dir, "univariate_logistic.csv"), index=False)
    uni_df.round(6).to_excel(os.path.join(out_dir, "univariate_logistic.xlsx"),
                             index=False, sheet_name="Univariate")
    print(f"Univariate done: {len(uni_df)} variables")

    # 4. Multivariable logistic regression (p < 0.1 inclusion)
    sig_vars = uni_df.loc[uni_df["P_Value"] < 0.1, "Variable"].tolist()
    print(f"\nSelected {len(sig_vars)} variables for multivariable model (p < 0.1)")

    if sig_vars:
        try:
            model, multi_df = _fit_logit(df, sig_vars, y)
            multi_df = multi_df.sort_values("P_Value").reset_index(drop=True)
            multi_df.round(6).to_csv(
                os.path.join(out_dir, "multivariable_logistic.csv"), index=False)
            multi_df.round(6).to_excel(
                os.path.join(out_dir, "multivariable_logistic.xlsx"),
                index=False, sheet_name="Multivariable")
            with open(os.path.join(out_dir, "multivariable_model_summary.txt"), "w",
                      encoding="utf-8") as f:
                f.write(str(model.summary()))
            print("Multivariable done.")
        except Exception as e:
            print(f"[ERROR] multivariable model failed: {e}")
    else:
        print("No variables met p < 0.1 threshold; skipping multivariable model.")

    # 5. Summary
    summary = pd.DataFrame({
        "Step": [
            "Load final dataset",
            "Drop zero-variance columns",
            "Univariate logistic regression",
            "Multivariable logistic regression (p<0.1)",
        ],
        "Status": [
            f"{df.shape[0]} rows x {df.shape[1]} cols",
            f"Dropped {len(zv)} columns",
            f"{len(uni_df)} variables tested",
            f"{len(sig_vars)} variables entered",
        ],
    })
    summary.to_csv(os.path.join(out_dir, "logistic_summary.csv"), index=False)
    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
