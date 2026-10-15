"""
08_single_model_evaluation.py
================================================================================
Single-model evaluation and plotting.

Reads per-model prediction CSVs produced by scripts 05-07 (machine learning,
deep learning, and lightweight/open-reasoning models), evaluates every model on
every time point for both the train and test splits, and generates:

  * Combined ROC curves (one panel per algorithm, all time points + both datasets)
  * Combined metrics comparison (line plots across time; grouped bar charts) 
  * Calibration curves
  * Cross-algorithm Spearman correlation heatmaps
  * Summary tables (best model per time point; detailed / average metrics)

Expected input files (one per model sub-directory, discovered automatically):
  - predictions.csv  OR  prediction_results.csv
  Each CSV is expected to contain at least:
      ID, Dataset, Actual_2y, Pred_2y, Actual_4y, Pred_4y, ...
  where Actual_*y is the binary event indicator and Pred_*y is the predicted
  probability from that model.

Outputs are written to config.SINGLE_EVAL_OUTPUT_DIR in both PNG and PDF.
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
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, roc_auc_score, calibration_curve
from scipy.stats import spearmanr

import config as CFG
from utils import (
    setup_matplotlib, calculate_metrics, bootstrap_auc, calculate_dca,
    save_dual_format, ensure_dir, format_pvalue,
)

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Category layout: maps each output directory to a display category name.
# ----------------------------------------------------------------------------
DATA_DIR_CATEGORY = [
    (CFG.ML_OUTPUT_DIR, "machine_learning"),
    (CFG.DL_OUTPUT_DIR, "deep_learning"),
    (CFG.LIGHTWEIGHT_OUTPUT_DIR, "open_light_reasoning_Model"),
]
CATEGORY_ORDER = [cat for _, cat in DATA_DIR_CATEGORY]

METRICS = ["Accuracy", "Precision", "Recall", "F1", "AUC", "Brier"]

# RMB-100 inspired sequential palette (kept from the original plotting scripts).
RMB_COLORS = ["#be0f2d", "#cf273c", "#cb364a", "#eb4035", "#d55f6f", "#f5abb7"]
RMB_CMAP = LinearSegmentedColormap.from_list("rmb100", RMB_COLORS[::-1], N=256)
RMB_CMAP_P = LinearSegmentedColormap.from_list("rmb100_p", RMB_COLORS, N=256)

CATEGORY_BG_COLORS = {
    "machine_learning": "#ffebee",
    "deep_learning": "#fff3e0",
    "open_light_reasoning_Model": "#e8f5e9",
}

# Distinct color per follow-up time point. Built dynamically from TIME_POINTS.
_TIME_PALETTE = ["#d32f2f", "#388e3c", "#1976d2", "#f57c00", "#7b1fa2"]
TIME_COLORS = {t: _TIME_PALETTE[i % len(_TIME_PALETTE)]
               for i, t in enumerate(CFG.TIME_POINTS)}

DATASET_STYLES_LINE = {
    "Train": {"marker": "o", "linestyle": "-", "linewidth": 2.0,
              "markersize": 6, "markeredgecolor": "black", "markeredgewidth": 0.6},
    "Test": {"marker": "s", "linestyle": "--", "linewidth": 2.0,
             "markersize": 6, "markeredgecolor": "black", "markeredgewidth": 0.6},
}
DATASET_STYLES_ROC = {
    "Train": {"linestyle": "-", "linewidth": 1.8},
    "Test": {"linestyle": "--", "linewidth": 1.8},
}


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def read_all_predictions():
    """
    Walk the ML / DL / lightweight output directories, read every model's
    prediction CSV (predictions.csv preferred, else prediction_results.csv),
    tag each row with its category and model name, and return one combined frame.
    """
    all_data = []
    algorithm_info = []

    for base_dir, category in DATA_DIR_CATEGORY:
        if not os.path.isdir(base_dir):
            print(f"[warn] category directory not found: {base_dir}")
            continue

        subdirs = sorted(d for d in os.listdir(base_dir)
                         if os.path.isdir(os.path.join(base_dir, d)))
        for subdir in subdirs:
            sub_path = os.path.join(base_dir, subdir)
            candidates = [
                os.path.join(sub_path, "predictions.csv"),
                os.path.join(sub_path, "prediction_results.csv"),
                os.path.join(sub_path, "evaluation_metrics.csv"),
                os.path.join(sub_path, "metrics.csv"),
            ]
            pred_file = next((p for p in candidates if os.path.exists(p)), None)
            if pred_file is None:
                continue
            try:
                df = pd.read_csv(pred_file)
                df["Algorithm_Category"] = category
                df["Algorithm_Name"] = subdir
                all_data.append(df)
                algorithm_info.append((category, subdir))
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[warn] could not read {pred_file}: {exc}")

    if not all_data:
        raise FileNotFoundError(
            "No prediction CSVs found under the model output directories. "
            "Run scripts 05-07 first."
        )

    combined = pd.concat(all_data, ignore_index=True)
    print(f"Loaded predictions for {len(algorithm_info)} models, "
          f"{len(combined)} rows.")
    return combined, algorithm_info


def _ordered_algorithm_info(combined_df):
    """Return (category, model_name) tuples in CATEGORY_ORDER, sorted within."""
    info = []
    for category in CATEGORY_ORDER:
        names = sorted(combined_df.loc[
            combined_df["Algorithm_Category"] == category, "Algorithm_Name"
        ].unique())
        info.extend((category, n) for n in names)
    return info


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def evaluate_all_models(combined_df, algorithm_info):
    """Compute per (dataset, category, model, time point) classification metrics."""
    rows = []
    datasets = sorted(combined_df["Dataset"].unique())

    for dataset in datasets:
        ds_df = combined_df[combined_df["Dataset"] == dataset]
        for category, algo in algorithm_info:
            sub = ds_df[(ds_df["Algorithm_Category"] == category) &
                        (ds_df["Algorithm_Name"] == algo)]
            if sub.empty:
                continue
            for t in CFG.TIME_POINTS:
                a_col, p_col = f"Actual_{t}y", f"Pred_{t}y"
                if a_col not in sub.columns or p_col not in sub.columns:
                    continue
                valid = sub.dropna(subset=[a_col, p_col])
                if valid.empty:
                    continue
                y_true = valid[a_col].values.astype(int)
                y_prob = valid[p_col].values
                y_pred = (y_prob >= 0.5).astype(int)
                m = calculate_metrics(y_true, y_pred, y_prob)
                rows.append({
                    "Dataset": dataset,
                    "Algorithm_Category": category,
                    "Algorithm_Name": algo,
                    "Time_Point_Years": t,
                    **m,
                })

    results = pd.DataFrame(rows)
    avg = (results.groupby(["Dataset", "Algorithm_Category", "Algorithm_Name"],
                           as_index=False)[METRICS].mean())
    return results, avg


def best_model_per_timepoint(results_df):
    """For each (dataset, time point) pick the model with the highest test AUC."""
    rows = []
    for (dataset, t), grp in results_df.groupby(["Dataset", "Time_Point_Years"]):
        best = grp.sort_values("AUC", ascending=False).iloc[0]
        rows.append({
            "Dataset": dataset,
            "Time_Point_Years": t,
            "Best_Model": best["Algorithm_Name"],
            "Category": best["Algorithm_Category"],
            "AUC": round(best["AUC"], 4),
            "F1": round(best["F1"], 4),
            "Brier": round(best["Brier"], 4),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def plot_performance_line(results_df, algorithm_info, out_dir):
    """Line plot grid: rows = metrics, columns = models; Train vs Test curves."""
    datasets = sorted(results_df["Dataset"].unique())
    n_algo = len(algorithm_info)
    n_metric = len(METRICS)
    if n_algo == 0:
        return

    fig = plt.figure(figsize=(2.2 * n_algo, 2.6 * n_metric))
    gs = GridSpec(n_metric, n_algo, figure=fig, hspace=0.32, wspace=0.08)

    for i, metric in enumerate(METRICS):
        for j, (category, algo) in enumerate(algorithm_info):
            ax = fig.add_subplot(gs[i, j])
            ax.set_facecolor(CATEGORY_BG_COLORS.get(category, "#ffffff"))
            sub = results_df[(results_df["Algorithm_Category"] == category) &
                             (results_df["Algorithm_Name"] == algo)]
            if sub.empty:
                ax.axis("off")
                continue
            for dataset in datasets:
                d = sub[sub["Dataset"] == dataset].sort_values("Time_Point_Years")
                if d.empty:
                    continue
                ax.plot(d["Time_Point_Years"], d[metric],
                        color=RMB_COLORS[j % len(RMB_COLORS)],
                        **DATASET_STYLES_LINE.get(dataset, DATASET_STYLES_LINE["Test"]))
            if i == 0:
                ax.set_title(algo.replace("_", "\n"), fontsize=8, pad=6,
                             fontweight="bold")
            if j == 0:
                ax.set_ylabel(metric, fontsize=9)
            else:
                ax.set_yticklabels([])
            if i == n_metric - 1:
                ax.set_xlabel("Years", fontsize=9)
            else:
                ax.set_xticklabels([])
            ax.set_xticks(CFG.TIME_POINTS)
            ax.set_ylim(0, 1.02)
            ax.spines[["top", "right"]].set_visible(False)

    # Top category headers
    col_cursor = 0
    for category in CATEGORY_ORDER:
        n_cat = sum(1 for c, _ in algorithm_info if c == category)
        if n_cat:
            ax_top = fig.add_subplot(gs[0, col_cursor:col_cursor + n_cat])
            ax_top.text(0.5, 1.12, category.replace("_", " ").title(),
                        ha="center", va="bottom", fontsize=10, fontweight="bold")
            ax_top.axis("off")
            col_cursor += n_cat

    handles = [
        plt.Line2D([0], [0], color="black", **DATASET_STYLES_LINE["Test"]),
        plt.Line2D([0], [0], color="black", **DATASET_STYLES_LINE["Train"]),
    ]
    fig.legend(handles, ["Test", "Train"], loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=9)
    fig.suptitle("Performance Comparison: Train vs Test",
                 fontsize=12, y=0.99, fontweight="bold")
    save_dual_format(fig, os.path.join(out_dir, "performance_comparison_lines"))
    plt.close(fig)
    gc.collect()


def plot_roc_grid(combined_df, algorithm_info, out_dir):
    """One ROC subplot per model; overlaid curves for every time point + dataset."""
    n_algo = len(algorithm_info)
    if n_algo == 0:
        return
    n_cols = min(4, n_algo)
    n_rows = int(np.ceil(n_algo / n_cols))
    fig = plt.figure(figsize=(3.6 * n_cols, 3.2 * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.45, wspace=0.55)

    for idx, (category, algo) in enumerate(algorithm_info):
        ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
        ax.set_facecolor(CATEGORY_BG_COLORS.get(category, "#ffffff"))
        sub = combined_df[(combined_df["Algorithm_Category"] == category) &
                          (combined_df["Algorithm_Name"] == algo)]
        if sub.empty:
            ax.axis("off")
            continue
        for t in CFG.TIME_POINTS:
            a_col, p_col = f"Actual_{t}y", f"Pred_{t}y"
            if a_col not in sub.columns:
                continue
            for dataset in ["Train", "Test"]:
                d = sub[sub["Dataset"] == dataset]
                y = d[a_col].values
                p = d[p_col].values
                mask = ~np.isnan(y) & ~np.isnan(p)
                y, p = y[mask].astype(int), p[mask]
                if len(y) == 0 or len(np.unique(y)) < 2:
                    continue
                fpr, tpr, _ = roc_curve(y, p)
                auc = roc_auc_score(y, p)
                _, ci_lo, ci_hi = bootstrap_auc(y, p,
                                                n_bootstraps=CFG.N_BOOTSTRAPS,
                                                random_state=CFG.RANDOM_STATE)
                ax.plot(fpr, tpr, color=TIME_COLORS[t],
                        **DATASET_STYLES_ROC[dataset],
                        label=f"{t}y {dataset} AUC={auc:.3f} "
                              f"[{ci_lo:.2f}-{ci_hi:.2f}]")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1)
        ax.set_title(algo.replace("_", "\n"), fontsize=8, fontweight="bold")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("False Positive Rate", fontsize=8)
        ax.set_ylabel("True Positive Rate", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=5, loc="lower right", framealpha=0.9)

    fig.suptitle("ROC Curves by Model (Train + Test, All Time Points)",
                 fontsize=12, y=0.99, fontweight="bold")
    save_dual_format(fig, os.path.join(out_dir, "roc_curves_by_model"))
    plt.close(fig)
    gc.collect()


def plot_metrics_bar(avg_df, algorithm_info, out_dir):
    """Grouped bar chart: one row per metric, Train vs Test across models."""
    n_algo = len(algorithm_info)
    if n_algo == 0:
        return
    fig = plt.figure(figsize=(1.6 * n_algo, 1.7 * len(METRICS)))
    gs = GridSpec(len(METRICS), 1, figure=fig, hspace=0.45)
    x = np.arange(n_algo)
    width = 0.38

    for row, metric in enumerate(METRICS):
        ax = fig.add_subplot(gs[row, 0])
        train_vals, test_vals = [], []
        for category, algo in algorithm_info:
            tr = avg_df[(avg_df["Algorithm_Category"] == category) &
                        (avg_df["Algorithm_Name"] == algo) &
                        (avg_df["Dataset"] == "Train")][metric].values
            te = avg_df[(avg_df["Algorithm_Category"] == category) &
                        (avg_df["Algorithm_Name"] == algo) &
                        (avg_df["Dataset"] == "Test")][metric].values
            train_vals.append(tr[0] if len(tr) else np.nan)
            test_vals.append(te[0] if len(te) else np.nan)
        ax.bar(x - width / 2, train_vals, width, label="Train",
               color="#d32f2f", alpha=0.8)
        ax.bar(x + width / 2, test_vals, width, label="Test",
               color="#1976d2", alpha=0.8)
        ax.set_ylabel(metric, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(-0.6, n_algo - 0.4)
        ax.spines[["top", "right"]].set_visible(False)
        if row == len(METRICS) - 1:
            ax.set_xticks(x)
            ax.set_xticklabels([a.replace("_", "\n") for _, a in algorithm_info],
                               fontsize=6, rotation=45, ha="right")
        else:
            ax.set_xticklabels([])

    fig.suptitle("Average Metrics: Train vs Test",
                 fontsize=12, y=0.99, fontweight="bold")
    save_dual_format(fig, os.path.join(out_dir, "average_metrics_bar"))
    plt.close(fig)
    gc.collect()


def plot_dca_grid(combined_df, algorithm_info, out_dir):
    """DCA net-benefit curves, one subplot per model."""
    n_algo = len(algorithm_info)
    if n_algo == 0:
        return
    n_cols = min(4, n_algo)
    n_rows = int(np.ceil(n_algo / n_cols))
    fig = plt.figure(figsize=(3.6 * n_cols, 3.0 * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.45, wspace=0.55)

    for idx, (category, algo) in enumerate(algorithm_info):
        ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
        ax.set_facecolor(CATEGORY_BG_COLORS.get(category, "#ffffff"))
        sub = combined_df[(combined_df["Algorithm_Category"] == category) &
                          (combined_df["Algorithm_Name"] == algo)]
        if sub.empty:
            ax.axis("off")
            continue
        ax.axhline(0, color="black", linewidth=1, alpha=0.7, label="Treat None")
        for t in CFG.TIME_POINTS:
            a_col, p_col = f"Actual_{t}y", f"Pred_{t}y"
            if a_col not in sub.columns:
                continue
            for dataset in ["Train", "Test"]:
                d = sub[sub["Dataset"] == dataset]
                y = d[a_col].values
                p = d[p_col].values
                mask = ~np.isnan(y) & ~np.isnan(p)
                y, p = y[mask].astype(int), p[mask]
                if len(y) == 0 or len(np.unique(y)) < 2:
                    continue
                thr, nb = calculate_dca(y, p)  # negative NB clipped to 0
                ax.plot(thr, nb, color=TIME_COLORS[t],
                        **DATASET_STYLES_ROC[dataset], label=f"{t}y {dataset}")
        ax.set_title(algo.replace("_", "\n"), fontsize=8, fontweight="bold")
        ax.set_xlabel("Threshold Probability", fontsize=8)
        ax.set_ylabel("Net Benefit", fontsize=8)
        ax.set_xlim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=5, loc="upper right", framealpha=0.9)

    fig.suptitle("Decision Curve Analysis: Train vs Test",
                 fontsize=12, y=0.99, fontweight="bold")
    save_dual_format(fig, os.path.join(out_dir, "dca_curves_by_model"))
    plt.close(fig)
    gc.collect()


def plot_calibration_grid(combined_df, algorithm_info, out_dir):
    """Calibration curves, one subplot per model."""
    n_algo = len(algorithm_info)
    if n_algo == 0:
        return
    n_cols = min(4, n_algo)
    n_rows = int(np.ceil(n_algo / n_cols))
    fig = plt.figure(figsize=(3.6 * n_cols, 3.0 * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.40, wspace=0.55)

    for idx, (category, algo) in enumerate(algorithm_info):
        ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
        ax.set_facecolor(CATEGORY_BG_COLORS.get(category, "#ffffff"))
        sub = combined_df[(combined_df["Algorithm_Category"] == category) &
                          (combined_df["Algorithm_Name"] == algo)]
        if sub.empty:
            ax.axis("off")
            continue
        ax.plot([0, 1], [0, 1], "k--", alpha=0.6, linewidth=1.2,
                label="Perfect")
        for t in CFG.TIME_POINTS:
            a_col, p_col = f"Actual_{t}y", f"Pred_{t}y"
            if a_col not in sub.columns:
                continue
            for dataset in ["Train", "Test"]:
                d = sub[sub["Dataset"] == dataset]
                y = d[a_col].values
                p = d[p_col].values
                mask = ~np.isnan(y) & ~np.isnan(p)
                y, p = y[mask].astype(int), p[mask]
                if len(y) == 0 or len(np.unique(y)) < 2:
                    continue
                prob_true, prob_pred = calibration_curve(
                    y, p, n_bins=10, strategy="uniform")
                ax.plot(prob_pred, prob_true, color=TIME_COLORS[t],
                        marker="o", markersize=4,
                        **DATASET_STYLES_ROC[dataset], label=f"{t}y {dataset}")
        ax.set_title(algo.replace("_", "\n"), fontsize=8, fontweight="bold")
        ax.set_xlabel("Predicted Probability", fontsize=8)
        ax.set_ylabel("Observed Frequency", fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=5, loc="upper left", framealpha=0.9)

    fig.suptitle("Calibration Curves: Train vs Test",
                 fontsize=12, y=0.99, fontweight="bold")
    save_dual_format(fig, os.path.join(out_dir, "calibration_curves_by_model"))
    plt.close(fig)
    gc.collect()


def plot_correlation_heatmaps(combined_df, algorithm_info, out_dir):
    """Split Spearman correlation heatmaps (lower triangle = rho, upper = p)."""
    corr_dir = ensure_dir(os.path.join(out_dir, "cross_algorithm_correlation"))
    names = [a for _, a in algorithm_info]
    cat_of = {a: c for c, a in algorithm_info}
    n = len(names)

    for t in CFG.TIME_POINTS:
        p_col = f"Pred_{t}y"
        if p_col not in combined_df.columns:
            continue
        pivot = combined_df.pivot_table(index="ID", columns="Algorithm_Name",
                                        values=p_col, aggfunc="mean").dropna()
        present = [nm for nm in names if nm in pivot.columns]
        pivot = pivot[present]
        m = len(present)
        if m < 2:
            continue

        corr_mat = np.ones((m, m))
        p_mat = np.eye(m)
        for i in range(m):
            for j in range(i + 1, m):
                rho, pval = spearmanr(pivot.iloc[:, i], pivot.iloc[:, j])
                corr_mat[i, j] = corr_mat[j, i] = rho
                p_mat[i, j] = p_mat[j, i] = pval

        fig, ax = plt.subplots(figsize=(1.2 * m + 4, 1.2 * m + 3))
        mask_lower = np.triu(np.ones_like(corr_mat, dtype=bool))
        mask_upper = np.tril(np.ones_like(p_mat, dtype=bool))
        im_c = ax.imshow(np.ma.masked_where(mask_lower, corr_mat),
                         cmap=RMB_CMAP, vmin=0, vmax=1, aspect="equal")
        im_p = ax.imshow(np.ma.masked_where(mask_upper, p_mat),
                         cmap=RMB_CMAP_P, vmin=0, vmax=0.05, aspect="equal")
        ax.set_xticks(range(m)); ax.set_yticks(range(m))
        ax.set_xticklabels(present, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(present, fontsize=8)
        ax.set_title(f"Cross-Algorithm Spearman Correlation ({t}y)", fontsize=11)

        # Category dividers
        bounds = []
        prev = None
        for k, nm in enumerate(present):
            if prev is not None and cat_of.get(nm) != prev:
                bounds.append(k - 0.5)
            prev = cat_of.get(nm)
        for b in bounds:
            ax.axvline(b, color="black", linewidth=1.5)
            ax.axhline(b, color="black", linewidth=1.5)

        for i in range(m):
            for j in range(i):
                ax.text(j, i, f"{corr_mat[i, j]:.2f}", ha="center",
                        va="center", fontsize=6, color="white")
            for j in range(i + 1, m):
                txt = "<.001" if p_mat[i, j] < 0.001 else f"{p_mat[i, j]:.3f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=6,
                        color="white")
            ax.text(i, i, "1.00", ha="center", va="center", fontsize=6,
                    color="black")

        fig.colorbar(im_c, ax=ax, shrink=0.5, pad=0.02).set_label("Spearman r")
        fig.colorbar(im_p, ax=ax, shrink=0.5, pad=0.08).set_label("p-value")
        save_dual_format(fig, os.path.join(corr_dir, f"correlation_{t}y"))
        plt.close(fig)
        gc.collect()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    setup_matplotlib()
    out_dir = ensure_dir(CFG.SINGLE_EVAL_OUTPUT_DIR)
    print("=" * 60)
    print("Single-model evaluation and plotting")
    print("=" * 60)

    # 1. Load predictions from all model output directories.
    combined_df, algorithm_info = read_all_predictions()
    algorithm_info = _ordered_algorithm_info(combined_df)
    combined_df.to_csv(os.path.join(out_dir, "combined_all_predictions.csv"),
                       index=False)

    # 2. Metrics tables.
    results_df, avg_df = evaluate_all_models(combined_df, algorithm_info)
    results_df.round(6).to_csv(os.path.join(out_dir, "detailed_metrics_all.csv"),
                               index=False)
    avg_df.round(6).to_csv(os.path.join(out_dir, "average_metrics_all.csv"),
                           index=False)

    best = best_model_per_timepoint(results_df)
    best.to_csv(os.path.join(out_dir, "best_model_per_timepoint.csv"),
                index=False)
    print(f"Best-model summary:\n{best.to_string(index=False)}\n")

    # 3. Plots.
    print("Plotting performance line comparison ...")
    plot_performance_line(results_df, algorithm_info, out_dir)

    print("Plotting ROC grid ...")
    plot_roc_grid(combined_df, algorithm_info, out_dir)

    print("Plotting average-metric bar chart ...")
    plot_metrics_bar(avg_df, algorithm_info, out_dir) 

    print("Plotting calibration curves ...")
    plot_calibration_grid(combined_df, algorithm_info, out_dir)

    print("Plotting cross-algorithm correlation heatmaps ...")
    plot_correlation_heatmaps(combined_df, algorithm_info, out_dir)

    print("=" * 60)
    print(f"Done. All single-model evaluation outputs saved to:\n  {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
