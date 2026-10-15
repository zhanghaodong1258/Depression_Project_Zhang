"""
Survival Analysis for Depression Risk Prediction
=================================================
Merged and refactored from:
  - 02-1.COX.md        (Cox regression, forest plots, model evaluation)
  - 02-2.Survival_analysis.md (variable definitions, clean_data, fixed
                        get_groups_and_filter with index alignment, log-rank,
                        Kaplan-Meier curves, univariate / multivariate Cox)
  - 02-3.md            (median-split mapping, Schoenfeld PH testing, ROC /
                        calibration / DCA evaluation, C-index)

Pipeline order:
  1. Data loading & cleaning
  2. Median-split mapping for numeric variables (descriptive artifact)
  3. Log-rank tests + individual Kaplan-Meier curves
  4. Combined Kaplan-Meier grids (significant vs. non-significant)
  5. Univariate Cox regression + Schoenfeld proportional-hazards test
  6. Multivariate Cox regression (significant + PH-passing variables)
  7. Forest plots (univariate & multivariate HR)
  8. Multivariate model evaluation: C-index, time-dependent ROC,
     calibration, decision curve analysis
  9. Save all tables and figures to config.SURVIVAL_OUTPUT_DIR
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, proportional_hazard_test
from lifelines.utils import concordance_index
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve

import config
from utils import (
    setup_matplotlib,
    format_pvalue,
    save_dual_format,
    ensure_dir,
    calculate_dca,
)

warnings.filterwarnings("ignore")


# ============================================================================
# 1. Variable Definitions
# ============================================================================
# Outcome / time columns (from config)
TIME_COL = config.TIME_COL
EVENT_COL = config.EVENT_COL
ID_COL = config.ID_COL

# Numeric (continuous) predictors: median-split into two groups for KM/log-rank.
NUMERIC_VARS = [
    "Baseline_Age",
    "Baseline_ADL_Limitation_6Items",
    "Baseline_IADL_Limitation_5Items",
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
    "Baseline_Frailty_5Items",
    "Baseline_Deficit_Accumulation_Index_Wave2",
]

# Categorical predictors: use the original (already integer-encoded) groups.
CATEGORICAL_VARS = [
    "Gender",
    "Pattern_Measured_BMI",
    "Pattern_Individual_Other_Income",
    "Pattern_Measured_Weight",
    "Pattern_Household_Total_Assets",
    "Pattern_Walking_Speed",
    "Pattern_Mobility_Limitation_7Items",
    "Pattern_Transfer_To_Parents",
    "Pattern_Average_Systolic_Blood_Pressure",
    "Pattern_Transfer_To_Other_Relatives",
    "Pattern_Average_Diastolic_Blood_Pressure",
    "Pattern_Orientation_Score",
    "Pattern_Transfer_To_Children",
    "Pattern_Peak_Flow",
    "Pattern_Transfer_From_Parents",
    "Pattern_Measured_Waist_Circumference",
    "Pattern_Immediate_Word_Recall_Score",
    "Pattern_Household_Total_Debts",
    "Pattern_Individual_Earned_Income",
    "Pattern_Measured_Height",
    "Pattern_Serial_Subtraction_Correct",
    "Pattern_ADL_Limitation_6Items",
    "Baseline_Cigarettes",
    "Pattern_Cigarettes",
    "Pattern_IADL_Limitation_5Items",
    "Pattern_Upper_Limb_Limitation",
    "Pattern_Household_Total_Income",
    "Pattern_Delayed_Word_Recall_Score",
    "Pattern_Five_Times_Chair_Stand_Seconds",
    "Pattern_Lower_Limb_Limitation",
    "Pattern_Dominant_Hand_Grip_Strength",
    "Pattern_Individual_Pension_Income",
    "Pattern_Transfer_From_Children",
    "Pattern_Individual_Transfer_Income",
    "Pattern_Number_Of_Living_Siblings",
    "Baseline_Individual_Earned_Income",
    "Baseline_Transfer_To_Parents",
    "Pattern_Total_Transfer_Income",
    "Pattern_Net_Transfer",
    "Pattern_Total_Word_Recall_Score",
    "Pattern_Frailty_5Items",
    "Pattern_Deficit_Accumulation_Index",
]

INDEPENDENT_VARS = list(dict.fromkeys(NUMERIC_VARS + CATEGORICAL_VARS))

# Evaluation time points (years) for the multivariate model.
EVAL_TIME_POINTS = config.TIME_POINTS

# Group colors (low = reference / blue, high = exposed / red).
COLOR_LOW = config.COLOR_LOW_RISK
COLOR_HIGH = config.COLOR_HIGH_RISK
COLOR_REF = "#7f7f7f"

# Output directories
OUTPUT_DIR = ensure_dir(config.SURVIVAL_OUTPUT_DIR)
PLOT_DIR = ensure_dir(os.path.join(OUTPUT_DIR, "plots"))
KM_SINGLE_DIR = ensure_dir(os.path.join(PLOT_DIR, "km_single"))
KM_COMBINED_DIR = ensure_dir(os.path.join(PLOT_DIR, "km_combined"))
COX_PLOT_DIR = ensure_dir(os.path.join(PLOT_DIR, "cox_model"))


# ============================================================================
# 2. Data Loading & Cleaning
# ============================================================================
def clean_data(df):
    """
    Coerce every column (except ID) to numeric and drop rows with missing
    follow-up time or event status. The ID column keeps its original type.
    """
    df = df.copy()
    for c in df.columns:
        if c != ID_COL:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[TIME_COL, EVENT_COL])
    return df


def build_median_split_mapping(df, available_vars):
    """
    Record the median split threshold for each numeric variable. This is a
    descriptive artifact: the data itself is NOT overwritten, but the same
    thresholds are reused in get_groups_and_filter so the mapping and the
    plotted groups stay consistent.
    """
    rows = []
    for var in available_vars:
        if var not in NUMERIC_VARS:
            continue
        median_val = df[var].median()
        rows.append({
            "variable": var,
            "type": "numeric",
            "median_value": median_val,
            "group_0_definition": f"<= {median_val:.6f}",
            "group_1_definition": f"> {median_val:.6f}",
        })
    return pd.DataFrame(rows)


# ============================================================================
# 3. Grouping Function (fixed version with index alignment)
# ============================================================================
def get_groups_and_filter(var_name, series, df):
    """
    Return (groups, filtered_df) with strict index alignment.

    - Numeric variables: median split into 0 (<= median) vs 1 (> median).
    - Categorical variables: keep the two most common/earliest groups so the
      comparison is always two-group.

    The returned groups Series and filtered_df share the same index so that
    boolean masks (groups == 0 / groups == 1) can be applied directly to
    filtered_df without a misalignment bug.
    """
    series = series.dropna()
    if var_name in NUMERIC_VARS:
        med = series.median()
        groups = (series > med).astype(int)
        filtered_df = df.loc[series.index]
        return groups, filtered_df

    # Categorical: keep the first two sorted groups for a clean two-group test.
    unique_vals = sorted(series.unique())
    if len(unique_vals) >= 2:
        mask = series.isin(unique_vals[:2])
        filtered_series = series[mask]
        groups = filtered_series.map({unique_vals[0]: 0, unique_vals[1]: 1})
        filtered_df = df.loc[filtered_series.index]
        return groups, filtered_df

    groups = series.astype(int)
    return groups, df.loc[series.index]


# ============================================================================
# 4. Log-Rank Tests + Individual Kaplan-Meier Curves
# ============================================================================
def run_logrank_and_km(df, available_vars):
    """
    Run a log-rank test for every variable and draw an individual KM curve.
    Returns the log-rank results DataFrame.
    """
    kmf = KaplanMeierFitter()
    records = []

    for var in available_vars:
        tmp = df[[var, TIME_COL, EVENT_COL]].copy()
        groups, filtered_tmp = get_groups_and_filter(var, tmp[var], tmp)
        groups = groups.dropna()
        filtered_tmp = filtered_tmp.loc[groups.index]

        if len(filtered_tmp) < 20 or groups.nunique() != 2:
            continue
        g0 = groups == 0
        g1 = groups == 1
        if g0.sum() < 5 or g1.sum() < 5:
            continue

        # Log-rank test
        result = logrank_test(
            filtered_tmp.loc[g0, TIME_COL], filtered_tmp.loc[g1, TIME_COL],
            event_observed_A=filtered_tmp.loc[g0, EVENT_COL],
            event_observed_B=filtered_tmp.loc[g1, EVENT_COL],
        )
        p_val = result.p_value

        records.append({
            "variable": var,
            "variable_type": "Numeric" if var in NUMERIC_VARS else "Categorical",
            "group_0_n": int(g0.sum()),
            "group_1_n": int(g1.sum()),
            "group_0_events": int(filtered_tmp.loc[g0, EVENT_COL].sum()),
            "group_1_events": int(filtered_tmp.loc[g1, EVENT_COL].sum()),
            "logrank_statistic": result.test_statistic,
            "p_value": p_val,
            "p_label": f"p = {p_val:.4f}" if p_val >= 0.001 else "p < 0.001",
            "significant": "Yes" if p_val < 0.05 else "No",
        })

        # Individual KM curve
        fig, ax = plt.subplots(figsize=(7, 5))
        kmf.fit(filtered_tmp.loc[g0, TIME_COL], filtered_tmp.loc[g0, EVENT_COL],
                label="Group 0")
        kmf.plot_survival_function(ax=ax, color=COLOR_LOW, ci_show=False, linewidth=2)
        kmf.fit(filtered_tmp.loc[g1, TIME_COL], filtered_tmp.loc[g1, EVENT_COL],
                label="Group 1")
        kmf.plot_survival_function(ax=ax, color=COLOR_HIGH, ci_show=False, linewidth=2)

        ax.text(
            0.95, 0.05, format_pvalue(p_val) if p_val >= 0.001 else "p < 0.001",
            ha="right", va="bottom", transform=ax.transAxes,
            fontsize=12, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=4),
        )
        ax.set_xlabel("Follow-up Time (years)")
        ax.set_ylabel("Depression-free Survival Probability")
        ax.set_title(f"Kaplan-Meier Curve: {var}")
        ax.legend(loc="lower left")
        ax.set_ylim(0, 1.05)
        save_dual_format(fig, os.path.join(KM_SINGLE_DIR, f"km_{var}"))

    return pd.DataFrame(records)


def plot_combined_km_grid(df, logrank_df, available_vars):
    """Draw combined KM grids split by significance (p < 0.05 vs. >= 0.05)."""
    kmf = KaplanMeierFitter()

    def _grid(vars_list, title, filename):
        n = len(vars_list)
        if n == 0:
            return
        n_cols = min(4, n)
        n_rows = int(np.ceil(n / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 3.5, n_rows * 3.2))
        axes = np.array(axes).flatten()

        for i, var in enumerate(vars_list):
            ax = axes[i]
            row = logrank_df[logrank_df.variable == var].iloc[0]
            tmp = df[[var, TIME_COL, EVENT_COL]].copy()
            groups, filtered_tmp = get_groups_and_filter(var, tmp[var], tmp)
            groups = groups.dropna()
            filtered_tmp = filtered_tmp.loc[groups.index]
            g0, g1 = groups == 0, groups == 1

            kmf.fit(filtered_tmp.loc[g0, TIME_COL], filtered_tmp.loc[g0, EVENT_COL])
            kmf.plot_survival_function(ax=ax, color=COLOR_LOW, ci_show=False, linewidth=1.5)
            kmf.fit(filtered_tmp.loc[g1, TIME_COL], filtered_tmp.loc[g1, EVENT_COL])
            kmf.plot_survival_function(ax=ax, color=COLOR_HIGH, ci_show=False, linewidth=1.5)

            ax.text(0.95, 0.05, row["p_label"], ha="right", va="bottom",
                    transform=ax.transAxes, fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.8))
            ax.set_title(var, fontsize=9)
            ax.set_ylim(0, 1.05)
            ax.legend().remove()

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        fig.suptitle(title, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_dual_format(fig, os.path.join(KM_COMBINED_DIR, filename))

    sig = logrank_df[logrank_df.significant == "Yes"]["variable"].tolist()
    nonsig = logrank_df[logrank_df.significant == "No"]["variable"].tolist()
    _grid(sig, "Significant Survival Curves (p < 0.05)", "combined_significant")
    _grid(nonsig, "Non-significant Survival Curves (p >= 0.05)", "combined_nonsignificant")


# ============================================================================
# 5. Univariate Cox Regression + PH Assumption Test
# ============================================================================
def run_univariate_cox(df, available_vars):
    """
    Fit one Cox PH model per variable. For each model record the HR (95% CI),
    the Wald p-value, and the Schoenfeld-residual proportional-hazards p-value
    (rank-transformed time).
    """
    records = []
    for var in available_vars:
        try:
            tmp = df[[var, TIME_COL, EVENT_COL]].dropna()
            if len(tmp) < 20 or tmp[var].nunique() <= 1:
                continue

            cph = CoxPHFitter()
            cph.fit(tmp, duration_col=TIME_COL, event_col=EVENT_COL)
            summary = cph.summary

            # Schoenfeld-based PH test (clean per-covariate p-value).
            ph_p = np.nan
            try:
                ph_result = proportional_hazard_test(cph, tmp, time_transform="rank")
                ph_summary = ph_result.summary
                if var in ph_summary.index:
                    ph_p = float(ph_summary.loc[var, "p"])
            except Exception:
                ph_p = np.nan

            records.append({
                "variable": var,
                "variable_type": "Numeric" if var in NUMERIC_VARS else "Categorical",
                "HR": float(summary.loc[var, "exp(coef)"]),
                "HR_lower_95": float(summary.loc[var, "exp(coef) lower 95%"]),
                "HR_upper_95": float(summary.loc[var, "exp(coef) upper 95%"]),
                "p_value": float(summary.loc[var, "p"]),
                "ph_test_p": ph_p,
                "ph_pass": "Yes" if (not np.isnan(ph_p) and ph_p >= 0.05) else "No",
            })
        except Exception:
            continue

    return pd.DataFrame(records)


# ============================================================================
# 6. Multivariate Cox Regression
# ============================================================================
def run_multivariate_cox(df, univariate_df):
    """
    Fit a multivariate Cox model on variables that are significant (p < 0.05)
    in univariate analysis AND pass the PH assumption test.
    Returns (multivariate_df, fitted_cph, analysis_df).
    """
    selected = univariate_df[
        (univariate_df.p_value < 0.05) & (univariate_df.ph_pass == "Yes")
    ]["variable"].tolist()

    multi_df = pd.DataFrame()
    cph_multi = None
    tmp_multi = None

    if len(selected) >= 1:
        try:
            tmp_multi = df[selected + [TIME_COL, EVENT_COL]].dropna()
            cph_multi = CoxPHFitter()
            cph_multi.fit(tmp_multi, duration_col=TIME_COL, event_col=EVENT_COL)
            summary = cph_multi.summary

            rows = []
            for var in selected:
                rows.append({
                    "variable": var,
                    "HR": float(summary.loc[var, "exp(coef)"]),
                    "HR_lower_95": float(summary.loc[var, "exp(coef) lower 95%"]),
                    "HR_upper_95": float(summary.loc[var, "exp(coef) upper 95%"]),
                    "p_value": float(summary.loc[var, "p"]),
                })
            multi_df = pd.DataFrame(rows)
        except Exception:
            cph_multi = None
            tmp_multi = None

    return multi_df, cph_multi, tmp_multi


# ============================================================================
# 7. Forest Plots
# ============================================================================
def plot_forest(cox_df, title, filename):
    """
    Forest plot of hazard ratios with 95% CI and p-value annotations.
    Handles NaN / non-finite values gracefully.
    """
    if len(cox_df) == 0:
        return
    df = cox_df.sort_values("HR", ascending=True).reset_index(drop=True)
    n = len(df)

    fig, ax = plt.subplots(figsize=(11, max(5, n * 0.55)))
    y_pos = np.arange(n)

    xerr = [
        (df["HR"] - df["HR_lower_95"]).values,
        (df["HR_upper_95"] - df["HR"]).values,
    ]
    ax.errorbar(df["HR"], y_pos, xerr=xerr, fmt="o", color=COLOR_LOW,
                capsize=4, linewidth=1.5, markersize=5)
    ax.axvline(1, color=COLOR_REF, linestyle="--", linewidth=1.2)

    max_hr = df["HR_upper_95"].replace([np.inf, -np.inf], np.nan).dropna().max()
    annotation_x = max(max_hr, 2.0) * 1.05

    for i, row in df.iterrows():
        p_str = format_pvalue(row["p_value"])
        ax.text(0.0, i, row["variable"], va="center", ha="left", fontsize=9)
        ax.text(annotation_x, i,
                f"HR={row['HR']:.3f} ({row['HR_lower_95']:.3f}-{row['HR_upper_95']:.3f})\n"
                f"p={p_str}",
                va="center", fontsize=8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    ax.set_xlabel("Hazard Ratio (HR) with 95% CI")
    ax.set_title(title, fontweight="bold")
    ax.set_xlim(left=0, right=annotation_x * 1.25)
    ax.grid(True, linestyle="--", alpha=0.3, axis="x")
    plt.tight_layout()
    save_dual_format(fig, os.path.join(COX_PLOT_DIR, filename))


# ============================================================================
# 8. Multivariate Model Evaluation (C-index, ROC, Calibration, DCA)
# ============================================================================
def evaluate_multivariate_model(cph_multi, tmp_multi):
    """
    Compute C-index and draw time-dependent ROC, calibration, and DCA curves
    for the multivariate Cox model across config.TIME_POINTS.
    Returns a metrics DataFrame.
    """
    metrics_rows = []

    # C-index on the multivariate modeling dataset
    c_index = concordance_index(
        tmp_multi[TIME_COL],
        -cph_multi.predict_partial_hazard(tmp_multi),
        tmp_multi[EVENT_COL],
    )
    metrics_rows.append({"metric": "C_index", "value": c_index})

    # --- Time-dependent ROC (combined on one axis) ---
    fig, ax = plt.subplots(figsize=(7, 6))
    risk_scores = cph_multi.predict_partial_hazard(tmp_multi)
    for t in EVAL_TIME_POINTS:
        mask = (tmp_multi[TIME_COL] <= t) & (tmp_multi[EVENT_COL] == 1)
        y_true = np.where(mask, 1, 0)
        if y_true.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true, risk_scores)
        auc = roc_auc_score(y_true, risk_scores)
        ax.plot(fpr, tpr, linewidth=2, label=f"{t} years (AUC={auc:.3f})")
        metrics_rows.append({"metric": f"AUC_{t}y", "value": auc})
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("Time-dependent ROC (Multivariate Cox)")
    ax.legend(loc="lower right")
    save_dual_format(fig, os.path.join(COX_PLOT_DIR, "multivariate_roc"))

    # --- Calibration curves (one panel per time point) ---
    n_t = len(EVAL_TIME_POINTS)
    n_cols = min(3, n_t)
    n_rows = int(np.ceil(n_t / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = np.array(axes).flatten()
    for i, t in enumerate(EVAL_TIME_POINTS):
        ax = axes[i]
        surv_prob = cph_multi.predict_survival_function(tmp_multi, times=[t]).T.iloc[:, 0]
        risk_prob = 1 - surv_prob
        mask = (tmp_multi[TIME_COL] <= t) & (tmp_multi[EVENT_COL] == 1)
        y_true = np.where(mask, 1, 0)
        if y_true.sum() == 0:
            ax.set_title(f"{t} years (no events)")
            continue
        prob_true, prob_pred = calibration_curve(y_true, risk_prob, n_bins=10)
        ax.plot(prob_pred, prob_true, marker="o", linewidth=2, color=COLOR_LOW, label="Model")
        ax.plot([0, 1], [0, 1], "--", color=COLOR_REF, label="Perfect")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Observed")
        ax.set_title(f"Calibration at {t} years")
        ax.legend(loc="lower right", fontsize=8)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    save_dual_format(fig, os.path.join(COX_PLOT_DIR, "multivariate_calibration"))

    # --- Decision curve analysis (combined on one axis) ---
    fig, ax = plt.subplots(figsize=(7, 6))
    thresholds = np.linspace(0.01, 0.99, 100)
    for t in EVAL_TIME_POINTS:
        surv_prob = cph_multi.predict_survival_function(tmp_multi, times=[t]).T.iloc[:, 0]
        risk_prob = 1 - surv_prob
        mask = (tmp_multi[TIME_COL] <= t) & (tmp_multi[EVENT_COL] == 1)
        y_true = np.where(mask, 1, 0)
        if y_true.sum() == 0:
            continue
        thr, nb = calculate_dca(y_true, risk_prob, thresholds)
        ax.plot(thr, nb, linewidth=2, label=f"{t} years")
    ax.plot(thresholds, np.zeros_like(thresholds), "k--", linewidth=1, label="Treat none")
    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Benefit")
    ax.set_title("Decision Curve Analysis (Multivariate Cox)")
    ax.legend(loc="upper right")
    save_dual_format(fig, os.path.join(COX_PLOT_DIR, "multivariate_dca"))

    return pd.DataFrame(metrics_rows)


# ============================================================================
# 9. Main
# ============================================================================
def main():
    setup_matplotlib()

    # Load and clean data
    df = pd.read_excel(config.FINAL_DATASET_PATH)
    df = clean_data(df)

    missing = [v for v in INDEPENDENT_VARS if v not in df.columns]
    available = [v for v in INDEPENDENT_VARS if v in df.columns]
    assert TIME_COL not in available, "Time column used as a predictor."
    assert EVENT_COL not in available, "Event column used as a predictor."

    # Median-split mapping (descriptive)
    mapping_df = build_median_split_mapping(df, available)
    mapping_df.to_excel(os.path.join(OUTPUT_DIR, "00_numeric_median_split_mapping.xlsx"),
                        index=False)

    # Log-rank + individual KM
    logrank_df = run_logrank_and_km(df, available)
    logrank_df.to_excel(os.path.join(OUTPUT_DIR, "01_logrank_results.xlsx"), index=False)

    # Combined KM grids
    plot_combined_km_grid(df, logrank_df, available)

    # Univariate Cox + PH
    univariate_df = run_univariate_cox(df, available)
    univariate_df.to_excel(os.path.join(OUTPUT_DIR, "02_univariate_cox_ph.xlsx"),
                           index=False)

    # Forest plot (univariate)
    plot_forest(univariate_df, "Univariate Cox Regression", "forest_univariate")

    # Multivariate Cox
    multi_df, cph_multi, tmp_multi = run_multivariate_cox(df, univariate_df)
    multi_df.to_excel(os.path.join(OUTPUT_DIR, "03_multivariate_cox.xlsx"), index=False)

    eval_metrics = pd.DataFrame()
    if cph_multi is not None:
        plot_forest(multi_df, "Multivariate Cox Regression", "forest_multivariate")
        eval_metrics = evaluate_multivariate_model(cph_multi, tmp_multi)
        eval_metrics.to_excel(os.path.join(OUTPUT_DIR, "04_model_evaluation_metrics.xlsx"),
                              index=False)

    # Summary
    summary = pd.DataFrame({
        "Step": [
            "Median-split mapping",
            "Log-rank tests + KM curves",
            "Univariate Cox + PH test",
            "Multivariate Cox",
            "Model evaluation (C-index/ROC/DCA)",
        ],
        "Status": [
            f"Completed ({len(mapping_df)} numeric vars)",
            f"Completed ({len(logrank_df)} vars)",
            f"Completed ({len(univariate_df)} vars)",
            f"Completed ({len(multi_df)} vars)" if len(multi_df) else "Skipped",
            "Completed" if len(eval_metrics) else "Skipped",
        ],
    })
    summary.to_excel(os.path.join(OUTPUT_DIR, "05_analysis_summary.xlsx"), index=False)


if __name__ == "__main__":
    main()
