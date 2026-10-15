"""
10_ensemble_evaluation.py
================================================================================
Ensemble-model evaluation and plotting.

Reads the ensemble predictions produced by 09_ensemble_modeling.py and
generates the publication figures and summary tables:

  * ROC curves for every ensemble model (all time points, train + test)
  * Side-by-side metrics comparison line plot (Accuracy / Precision / Recall /
    F1 / AUC / Brier) with the overall mean reference line 
  * Calibration curves
  * Hyper-parameter impact plot (how F1 changes with each tuned parameter)
  * Summary table: best ensemble model per (dataset, time point)

Inputs (read from config.ENSEMBLE_OUTPUT_DIR):
  - ensemble_all_predictions.csv       long-format ensemble predictions
  - ensemble_metrics.csv               per-dataset / model / timepoint metrics
  - optimized_ensemble_predictions.csv best tuned-ensemble predictions
  - tuning_results.csv                 full grid-search results from 09

Outputs are written to config.ENSEMBLE_OUTPUT_DIR in PNG + PDF.
"""

import os
import gc
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_curve, roc_auc_score, calibration_curve

import config as CFG
from utils import (
    setup_matplotlib, calculate_dca, save_dual_format, ensure_dir,
    bootstrap_auc,
)

warnings.filterwarnings("ignore")

RMB_COLORS = ["#be0f2d", "#cf273c", "#cb364a", "#eb4035", "#d55f6f", "#f5abb7"]
TIME_COLORS = {t: c for t, c in zip(
    CFG.TIME_POINTS,
    ["#d32f2f", "#388e3c", "#1976d2", "#f57c00", "#7b1fa2"])}
DATASET_STYLES_ROC = {
    "Train": {"linestyle": "-", "linewidth": 1.8},
    "Test": {"linestyle": "--", "linewidth": 1.8},
}
DATASET_STYLES_LINE = {
    "Train": {"marker": "o", "linestyle": "-", "linewidth": 2.0,
              "markersize": 6, "markeredgecolor": "black"},
    "Test": {"marker": "s", "linestyle": "--", "linewidth": 2.0,
             "markersize": 6, "markeredgecolor": "black"},
}
METRICS = ["Accuracy", "Precision", "Recall", "F1", "AUC", "Brier"]


def _ensemble_info(df):
    """Sorted list of (category, name) for ensemble rows."""
    names = sorted(df.loc[df["Algorithm_Category"] == "ensemble",
                          "Algorithm_Name"].unique())
    return [("ensemble", n) for n in names]


def plot_roc_grid(ens_df, out_dir):
    """ROC subplot grid, one per ensemble model."""
    info = _ensemble_info(ens_df)
    n = len(info)
    if n == 0:
        return
    n_cols = min(4, n)
    n_rows = int(np.ceil(n / n_cols))
    fig = plt.figure(figsize=(3.6 * n_cols, 3.2 * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.45, wspace=0.55)

    for idx, (_, name) in enumerate(info):
        ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
        sub = ens_df[ens_df["Algorithm_Name"] == name]
        for t in CFG.TIME_POINTS:
            a_col, p_col = f"Actual_{t}y", f"Pred_{t}y"
            if a_col not in sub.columns:
                continue
            for dataset in ["Train", "Test"]:
                d = sub[sub["Dataset"] == dataset]
                y = d[a_col].values
                p = d[p_col].values
                ok = ~np.isnan(y) & ~np.isnan(p)
                y, p = y[ok].astype(int), p[ok]
                if len(y) == 0 or len(np.unique(y)) < 2:
                    continue
                fpr, tpr, _ = roc_curve(y, p)
                auc = roc_auc_score(y, p)
                _, lo, hi = bootstrap_auc(y, p,
                                          n_bootstraps=CFG.N_BOOTSTRAPS,
                                          random_state=CFG.RANDOM_STATE)
                ax.plot(fpr, tpr, color=TIME_COLORS[t],
                        **DATASET_STYLES_ROC[dataset],
                        label=f"{t}y {dataset} AUC={auc:.3f} [{lo:.2f}-{hi:.2f}]")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1)
        ax.set_title(name.replace("_", "\n"), fontsize=8, fontweight="bold")
        ax.set_xlabel("False Positive Rate", fontsize=8)
        ax.set_ylabel("True Positive Rate", fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=5, loc="lower right", framealpha=0.9)

    fig.suptitle("ROC Curves: Ensemble Models (Train + Test, All Time Points)",
                 fontsize=12, y=0.99, fontweight="bold")
    save_dual_format(fig, os.path.join(out_dir, "ensemble_roc_curves"))
    plt.close(fig)
    gc.collect()


def plot_metrics_comparison(metrics_df, out_dir):
    """Six side-by-side metric line plots, Train vs Test, mean reference line."""
    datasets = sorted(metrics_df["Dataset"].unique())
    fig = plt.figure(figsize=(2.2 * len(METRICS), 3.2))
    gs = GridSpec(1, len(METRICS), figure=fig, wspace=0.12)
    for i, metric in enumerate(METRICS):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor("#f8f9fa")
        sub = metrics_df[metrics_df["Metric"] == metric] \
            if "Metric" in metrics_df.columns else None
        if sub is None:
            # metrics_df is in wide (per-row) form; reshape on the fly.
            for j, dataset in enumerate(datasets):
                d = metrics_df[(metrics_df["Dataset"] == dataset)] \
                    .sort_values("Time_Point_Years")
                if not d.empty:
                    ax.plot(d["Time_Point_Years"], d[metric],
                            color=RMB_COLORS[j],
                            **DATASET_STYLES_LINE[dataset], label=dataset)
            mean_val = metrics_df[metric].mean()
        else:
            mean_val = sub["Value"].mean()
            for j, dataset in enumerate(datasets):
                d = sub[sub["Dataset"] == dataset].sort_values("Year")
                ax.plot(d["Year"], d["Value"], color=RMB_COLORS[j],
                        **DATASET_STYLES_LINE[dataset], label=dataset)
        ax.axhline(mean_val, color="#666666", linestyle=":", linewidth=1.5)
        ax.set_title(metric, fontsize=10, fontweight="bold")
        ax.set_xticks(CFG.TIME_POINTS)
        ax.set_xticklabels([f"{t}y" for t in CFG.TIME_POINTS], fontsize=8)
        if i == 0:
            ax.set_ylabel("Score", fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.set_ylim(0, 1.02)
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=9)
    fig.suptitle("Ensemble Performance Metrics: Train vs Test",
                 fontsize=12, y=1.02, fontweight="bold")
    save_dual_format(fig, os.path.join(out_dir, "ensemble_metrics_comparison"))
    plt.close(fig)
    gc.collect()


def plot_dca_grid(ens_df, out_dir):
    """DCA net-benefit subplot grid for ensemble models."""
    info = _ensemble_info(ens_df)
    n = len(info)
    if n == 0:
        return
    n_cols = min(4, n)
    n_rows = int(np.ceil(n / n_cols))
    fig = plt.figure(figsize=(3.6 * n_cols, 3.0 * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.45, wspace=0.55)

    for idx, (_, name) in enumerate(info):
        ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
        sub = ens_df[ens_df["Algorithm_Name"] == name]
        ax.axhline(0, color="black", linewidth=1, alpha=0.7)
        for t in CFG.TIME_POINTS:
            a_col, p_col = f"Actual_{t}y", f"Pred_{t}y"
            if a_col not in sub.columns:
                continue
            for dataset in ["Train", "Test"]:
                d = sub[sub["Dataset"] == dataset]
                y = d[a_col].values
                p = d[p_col].values
                ok = ~np.isnan(y) & ~np.isnan(p)
                y, p = y[ok].astype(int), p[ok]
                if len(y) == 0 or len(np.unique(y)) < 2:
                    continue
                thr, nb = calculate_dca(y, p)
                ax.plot(thr, nb, color=TIME_COLORS[t],
                        **DATASET_STYLES_ROC[dataset], label=f"{t}y {dataset}")
        ax.set_title(name.replace("_", "\n"), fontsize=8, fontweight="bold")
        ax.set_xlabel("Threshold Probability", fontsize=8)
        ax.set_ylabel("Net Benefit", fontsize=8)
        ax.set_xlim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=5, loc="upper right", framealpha=0.9)

    fig.suptitle("Decision Curve Analysis: Ensemble Models",
                 fontsize=12, y=0.99, fontweight="bold")
    save_dual_format(fig, os.path.join(out_dir, "ensemble_dca_curves"))
    plt.close(fig)
    gc.collect()


def plot_calibration_grid(ens_df, out_dir):
    """Calibration subplot grid for ensemble models."""
    info = _ensemble_info(ens_df)
    n = len(info)
    if n == 0:
        return
    n_cols = min(4, n)
    n_rows = int(np.ceil(n / n_cols))
    fig = plt.figure(figsize=(3.6 * n_cols, 3.0 * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.40, wspace=0.55)

    for idx, (_, name) in enumerate(info):
        ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
        sub = ens_df[ens_df["Algorithm_Name"] == name]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.6, linewidth=1.2)
        for t in CFG.TIME_POINTS:
            a_col, p_col = f"Actual_{t}y", f"Pred_{t}y"
            if a_col not in sub.columns:
                continue
            for dataset in ["Train", "Test"]:
                d = sub[sub["Dataset"] == dataset]
                y = d[a_col].values
                p = d[p_col].values
                ok = ~np.isnan(y) & ~np.isnan(p)
                y, p = y[ok].astype(int), p[ok]
                if len(y) == 0 or len(np.unique(y)) < 2:
                    continue
                pt, pp = calibration_curve(y, p, n_bins=10, strategy="uniform")
                ax.plot(pp, pt, color=TIME_COLORS[t], marker="o",
                        markersize=4, **DATASET_STYLES_ROC[dataset],
                        label=f"{t}y {dataset}")
        ax.set_title(name.replace("_", "\n"), fontsize=8, fontweight="bold")
        ax.set_xlabel("Predicted Probability", fontsize=8)
        ax.set_ylabel("Observed Frequency", fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=5, loc="upper left", framealpha=0.9)

    fig.suptitle("Calibration Curves: Ensemble Models",
                 fontsize=12, y=0.99, fontweight="bold")
    save_dual_format(fig, os.path.join(out_dir, "ensemble_calibration_curves"))
    plt.close(fig)
    gc.collect()


def plot_parameter_impact(tuning_df, out_dir):
    """Mean +/- SD validation F1 vs each tuned hyper-parameter."""
    if tuning_df is None or tuning_df.empty:
        return
    params = [
        ("weight_exponent", "Weight Exponent"),
        ("weight_threshold", "Weight Threshold"),
        ("normalize_weights", "Normalize Weights"),
        ("clip_weights", "Clip Weights"),
    ]
    target = "Validation_Avg_F1"
    fig = plt.figure(figsize=(3.2 * len(params), 3.2))
    gs = GridSpec(1, len(params), figure=fig, wspace=0.15)
    best_val = tuning_df[target].max()
    for i, (pname, plabel) in enumerate(params):
        ax = fig.add_subplot(gs[0, i])
        if pname == "normalize_weights":
            vals = [False, True]
            labels = ["False", "True"]
        elif pname == "clip_weights":
            vals = sorted(tuning_df[pname].astype(str).unique())
            labels = [v.strip("()").replace(", ", "-") for v in vals]
        else:
            vals = sorted(tuning_df[pname].unique())
            labels = [str(v) for v in vals]
        means, stds = [], []
        for v in vals:
            sub = tuning_df[tuning_df[pname] == v]
            means.append(sub[target].mean())
            stds.append(sub[target].std())
            ax.scatter([i] * len(sub), sub[target],
                       color=RMB_COLORS[0], alpha=0.2, s=20, zorder=1)
        ax.errorbar(range(len(vals)), means, yerr=stds, color=RMB_COLORS[3],
                    linewidth=2, marker="o", markersize=6, capsize=3, zorder=2)
        ax.axhline(best_val, color=RMB_COLORS[0], linestyle="--",
                   linewidth=1.5, alpha=0.8)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel(plabel, fontsize=9)
        if i == 0:
            ax.set_ylabel("Validation Avg F1", fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Hyper-parameter Impact on Validation F1",
                 fontsize=12, y=0.99, fontweight="bold")
    save_dual_format(fig, os.path.join(out_dir, "ensemble_parameter_impact"))
    plt.close(fig)
    gc.collect()


def best_ensemble_summary(metrics_df):
    """Best ensemble model per (dataset, time point) by AUC."""
    rows = []
    for (dataset, t), grp in metrics_df.groupby(["Dataset", "Time_Point_Years"]):
        best = grp.sort_values("AUC", ascending=False).iloc[0]
        rows.append({
            "Dataset": dataset,
            "Time_Point_Years": t,
            "Best_Ensemble": best["Algorithm_Name"],
            "AUC": round(best["AUC"], 4),
            "F1": round(best["F1"], 4),
            "Brier": round(best["Brier"], 4),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    setup_matplotlib()
    out_dir = ensure_dir(CFG.ENSEMBLE_OUTPUT_DIR)
    print("=" * 60)
    print("Ensemble-model evaluation and plotting")
    print("=" * 60)

    pred_path = os.path.join(out_dir, "ensemble_all_predictions.csv")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(
            f"Ensemble predictions not found: {pred_path}\n"
            "Run 09_ensemble_modeling.py first.")
    ens_df = pd.read_csv(pred_path)

    metrics_path = os.path.join(out_dir, "ensemble_metrics.csv")
    metrics_df = pd.read_csv(metrics_path) if os.path.exists(metrics_path) \
        else pd.DataFrame()

    # Summary table.
    if not metrics_df.empty:
        summary = best_ensemble_summary(metrics_df)
        summary.to_csv(os.path.join(out_dir, "best_ensemble_per_timepoint.csv"),
                       index=False)
        print(f"Best-ensemble summary:\n{summary.to_string(index=False)}\n")
        plot_metrics_comparison(metrics_df, out_dir)

    # Plots.
    print("Plotting ensemble ROC curves ...")
    plot_roc_grid(ens_df, out_dir) 

    print("Plotting ensemble calibration curves ...")
    plot_calibration_grid(ens_df, out_dir)

    tuning_path = os.path.join(out_dir, "tuning_results.csv")
    if os.path.exists(tuning_path):
        print("Plotting hyper-parameter impact ...")
        tuning_df = pd.read_csv(tuning_path)
        plot_parameter_impact(tuning_df, out_dir)

    print("=" * 60)
    print(f"Done. Ensemble evaluation outputs saved to:\n  {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
