"""
Machine Learning Models for Depression Onset Risk Prediction
=============================================================
Multi-time-point survival-based prediction using 9 traditional
classification algorithms:
    RandomForest, GaussianNB, SVM, Lasso, MLP, KNN, LightGBM, XGBoost, GBDT.

The dataset is split once into training / test sets (80/20). For every
follow-up time point in TIME_POINTS the incident-event label is derived,
both sets are scored, and ROC, confusion-matrix, calibration and
metric-trend figures are produced. 

Paths, column names, and global constants are imported from config.py;
shared evaluation and plotting helpers are imported from utils.py.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.calibration import calibration_curve
from sklearn.linear_model import Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import xgboost as xgb
from lifelines import CoxPHFitter
from matplotlib.backends.backend_pdf import PdfPages

from config import (
    SIGNIFICANT_VARS_PATH, ML_OUTPUT_DIR,
    ID_COL, TIME_COL, EVENT_COL, TIME_POINTS,
    TEST_SIZE, RANDOM_STATE,
)
from utils import (
    setup_matplotlib, calculate_metrics, safe_predict_proba,
    bootstrap_auc, calculate_dca, save_dual_format, ensure_dir,
)


# ============================================================================
# Model Definition
# ============================================================================
def get_models(random_state=RANDOM_STATE):
    """Instantiate the 9 traditional ML classifiers."""
    return {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "GaussianNB": GaussianNB(),
        "SVM": SVC(probability=True, random_state=random_state),
        "Lasso": Lasso(random_state=random_state, max_iter=10000),
        "MLP": MLPClassifier(random_state=random_state, max_iter=1000),
        "KNN": KNeighborsClassifier(),
        "LightGBM": lgb.LGBMClassifier(random_state=random_state, verbose=-1),
        "XGBoost": xgb.XGBClassifier(random_state=random_state, eval_metric="logloss"),
        "GBDT": GradientBoostingClassifier(random_state=random_state),
    }


# ============================================================================
# Single Model Processing
# ============================================================================
def process_single_model(
    model_name, model, X_train, X_test,
    y_time_train, y_event_train,
    y_time_test, y_event_test,
    ids, y_time, y_event, id_to_dataset,
    all_results, hr_results, binary_hr_results,
    global_plot_data
):
    """Train one model, evaluate it across TIME_POINTS, and save all figures."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_all_s = scaler.transform(np.vstack([X_train, X_test]))

    try:
        model.fit(X_train_s, y_event_train)
    except Exception:
        return

    model_dir = ensure_dir(os.path.join(ML_OUTPUT_DIR, model_name))

    sample_results = pd.DataFrame({
        ID_COL: ids,
        "Dataset": [id_to_dataset[i] for i in ids],
        TIME_COL: y_time,
        EVENT_COL: y_event,
    })

    time_metrics = []
    roc_data, cm_data, dca_data, calib_data, metrics_data = [], [], [], [], []

    for t in TIME_POINTS:
        y_train = (y_time_train <= t) & (y_event_train == 1)
        y_test = (y_time_test <= t) & (y_event_test == 1)
        sample_results[f"Actual_{t}y"] = ((y_time <= t) & (y_event == 1)).astype(int)

        p_train = np.clip(safe_predict_proba(model, X_train_s), 0, 1)
        p_test = np.clip(safe_predict_proba(model, X_test_s), 0, 1)
        p_all = np.clip(safe_predict_proba(model, X_all_s), 0, 1)
        pred_train = (p_train > 0.5).astype(int)
        pred_test = (p_test > 0.5).astype(int)
        sample_results[f"Pred_{t}y"] = p_all

        roc_data.extend([("Train", t, y_train, p_train), ("Test", t, y_test, p_test)])
        cm_data.extend([("Train", t, y_train, pred_train), ("Test", t, y_test, pred_test)])
        dca_data.extend([("Train", t, y_train, p_train), ("Test", t, y_test, p_test)])
        calib_data.extend([("Train", t, y_train, p_train), ("Test", t, y_test, p_test)])

        if len(np.unique(y_train)) > 1:
            m = calculate_metrics(y_train, pred_train, p_train)
            m.update({"Model": model_name, "Time_Point": t, "Dataset": "Train"})
            time_metrics.append(m)
            metrics_data.append(("Train", t, m))
        if len(np.unique(y_test)) > 1:
            m = calculate_metrics(y_test, pred_test, p_test)
            m.update({"Model": model_name, "Time_Point": t, "Dataset": "Test"})
            time_metrics.append(m)
            metrics_data.append(("Test", t, m))

    final_p = safe_predict_proba(model, X_all_s)
    sample_results["Final_Prediction"] = (final_p > 0.5).astype(int)

    sample_results.to_csv(os.path.join(model_dir, "prediction_results.csv"),
                          index=False, float_format="%.6f")
    pd.DataFrame(time_metrics).to_csv(os.path.join(model_dir, "evaluation_metrics.csv"),
                                      index=False, float_format="%.6f")
    all_results.extend(time_metrics)

    # 1. Metrics trend lines
    fig, ax = plt.subplots(figsize=(14, 10))
    metrics_names = ["Accuracy", "Precision", "Recall", "F1", "AUC", "Brier"]
    colors = plt.cm.tab10(np.arange(len(metrics_names)))
    for i, name in enumerate(metrics_names):
        tr_x, tr_y, te_x, te_y = [], [], [], []
        for d, t, m in metrics_data:
            if d == "Train" and name in m:
                tr_x.append(t); tr_y.append(m[name])
            if d == "Test" and name in m:
                te_x.append(t); te_y.append(m[name])
        if tr_x:
            ax.plot(tr_x, tr_y, "o-", color=colors[i], label=f"{name} (Train)",
                    linewidth=3, markersize=10)
        if te_x:
            ax.plot(te_x, te_y, "s--", color=colors[i], label=f"{name} (Test)",
                    linewidth=3, markersize=10)
    ax.set_xlabel("Time Point (Year)")
    ax.set_ylabel("Metric Value")
    ax.set_title(f"{model_name} Performance Metrics")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(TIME_POINTS)
    save_dual_format(fig, os.path.join(model_dir, "metrics_lineplot"))

    # 2. ROC curves (square)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=3)
    for d, t, y_true, y_prob in roc_data:
        if len(np.unique(y_true)) <= 1:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc, auc_l, auc_u = bootstrap_auc(y_true, y_prob)
        linestyle = "-" if d == "Train" else "--"
        ax.plot(fpr, tpr, linestyle=linestyle, linewidth=3,
                label=f"{d} {t}y (AUC={auc:.3f}, 95%CI={auc_l:.3f}-{auc_u:.3f})",
                alpha=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} ROC Curves")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_aspect("equal", adjustable="box")
    save_dual_format(fig, os.path.join(model_dir, "roc_curves"))

    # 3. Confusion matrices (2 rows x 5 columns)
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(30, 12))
    axes = axes.flatten()
    plot_idx = 0
    for d in ["Train", "Test"]:
        for t in TIME_POINTS:
            ax = axes[plot_idx]
            cm = None
            for cd, ct, yt, yp in cm_data:
                if cd == d and ct == t:
                    cm = confusion_matrix(yt, yp)
                    break
            if cm is not None:
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                            annot_kws={"size": 40},
                            xticklabels=["Negative", "Positive"],
                            yticklabels=["Negative", "Positive"])
                ax.tick_params(labelsize=32)
            ax.set_title(f"{t}y", fontsize=30)
            plot_idx += 1
    for i, d in enumerate(["Train", "Test"]):
        axes[i * 5].set_ylabel(d, fontsize=32, fontweight="bold")
    fig.suptitle(f"{model_name} Confusion Matrices", fontsize=36)
    save_dual_format(fig, os.path.join(model_dir, "confusion_matrices"))

    # 4. Decision curve analysis
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.plot([0, 1], [0, 0], "k--", label="None", linewidth=3)
    for d, t, y_true, y_prob in dca_data:
        if len(np.unique(y_true)) <= 1:
            continue
        thresholds, nb = calculate_dca(y_true, y_prob)
        mean_y = y_true.mean()
        all_nb = mean_y - thresholds * (1 - mean_y)
        ax.plot(thresholds, all_nb, linestyle=":", color="gray", alpha=0.3, linewidth=2)
        linestyle = "-" if d == "Train" else "--"
        ax.plot(thresholds, nb, linestyle=linestyle, label=f"{d} {t}y",
                alpha=0.8, linewidth=3)
    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Benefit")
    ax.set_title(f"{model_name} Decision Curve Analysis")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_ylim(-0.05, 0.2)
    save_dual_format(fig, os.path.join(model_dir, "dca_curves"))

    # 5. Calibration curves (square)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=3)
    for d, t, y_true, y_prob in calib_data:
        if len(np.unique(y_true)) <= 1:
            continue
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        linestyle = "-" if d == "Train" else "--"
        ax.plot(prob_pred, prob_true, linestyle=linestyle, marker="o",
                label=f"{d} {t}y", alpha=0.8, linewidth=3, markersize=10)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"{model_name} Calibration Curves")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_aspect("equal", adjustable="box")
    save_dual_format(fig, os.path.join(model_dir, "calibration_curves"))

    # Merge single-model figures into one PDF (re-read saved PNGs)
    with PdfPages(os.path.join(model_dir, "all_plots_merged.pdf")) as pdf:
        for plot_name in ["metrics_lineplot", "roc_curves", "confusion_matrices",
                          "dca_curves", "calibration_curves"]:
            page = plt.figure(figsize=(10, 10))
            img = plt.imread(os.path.join(model_dir, f"{plot_name}.png"))
            plt.imshow(img)
            plt.axis("off")
            pdf.savefig(page, bbox_inches="tight")
            plt.close(page)

    # Store data for global merged plots
    global_plot_data["metrics"].append((model_name, metrics_data))
    global_plot_data["roc"].append((model_name, roc_data))
    global_plot_data["cm"].append((model_name, cm_data))
    global_plot_data["dca"].append((model_name, dca_data))
    global_plot_data["calib"].append((model_name, calib_data))

    # ---- Hazard ratio calculation (Cox PH) ----
    binary_pred = (final_p > 0.5).astype(int)
    train_mask = np.array([id_to_dataset[i] == "Train" for i in ids])
    test_mask = np.array([id_to_dataset[i] == "Test" for i in ids])

    try:
        cox = CoxPHFitter(penalizer=0.5)
        cox_df = pd.DataFrame({TIME_COL: y_time, EVENT_COL: y_event, "Pred_Prob": final_p})
        cox.fit(cox_df, duration_col=TIME_COL, event_col=EVENT_COL)
        hr_results.append({
            "Model": model_name,
            "HR": cox.summary.loc["Pred_Prob", "exp(coef)"],
            "HR_Lower": cox.summary.loc["Pred_Prob", "exp(coef) lower 95%"],
            "HR_Upper": cox.summary.loc["Pred_Prob", "exp(coef) upper 95%"],
            "P_Value": cox.summary.loc["Pred_Prob", "p"],
        })
    except Exception:
        pass

    def calc_hr(mask):
        try:
            df = pd.DataFrame({TIME_COL: y_time[mask],
                               EVENT_COL: y_event[mask],
                               "Binary": binary_pred[mask]})
            cox = CoxPHFitter(penalizer=0.5)
            cox.fit(df, duration_col=TIME_COL, event_col=EVENT_COL)
            return (
                cox.summary.loc["Binary", "exp(coef)"],
                cox.summary.loc["Binary", "exp(coef) lower 95%"],
                cox.summary.loc["Binary", "exp(coef) upper 95%"],
                cox.summary.loc["Binary", "p"],
            )
        except Exception:
            return np.nan, np.nan, np.nan, np.nan

    tr_hr, tr_l, tr_u, tr_p = calc_hr(train_mask)
    te_hr, te_l, te_u, te_p = calc_hr(test_mask)
    binary_hr_results.append({
        "Model": model_name,
        "Train_HR": tr_hr, "Train_L": tr_l, "Train_U": tr_u, "Train_P": tr_p,
        "Test_HR": te_hr, "Test_L": te_l, "Test_U": te_u, "Test_P": te_p,
    })


# ============================================================================
# Global Merged Plots
# ============================================================================
def merge_global_plots(plot_data, binary_hr_df):
    """Create side-by-side comparison plots across all models plus HR forest plot."""
    # 1. Metrics merge (one panel per model)
    fig, axes = plt.subplots(3, 3, figsize=(40, 40))
    axes = axes.flatten()
    metric_names = ["Accuracy", "Precision", "Recall", "F1", "AUC", "Brier"]
    colors = plt.cm.tab10(np.arange(6))
    for i, (name, data) in enumerate(plot_data["metrics"]):
        ax = axes[i]
        for j, m_name in enumerate(metric_names):
            tr_x, tr_y, te_x, te_y = [], [], [], []
            for d, t, m in data:
                if d == "Train" and m_name in m:
                    tr_x.append(t); tr_y.append(m[m_name])
                if d == "Test" and m_name in m:
                    te_x.append(t); te_y.append(m[m_name])
            if tr_x:
                ax.plot(tr_x, tr_y, "o-", c=colors[j], lw=3, ms=12, label=f"{m_name} Train")
            if te_x:
                ax.plot(te_x, te_y, "s--", c=colors[j], lw=3, ms=12, label=f"{m_name} Test")
        ax.set_title(name, fontsize=36)
        ax.set_xlabel("Time (Year)", fontsize=36)
        ax.set_ylabel("Value", fontsize=36)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(TIME_POINTS)
        ax.tick_params(labelsize=36)
        ax.legend(fontsize=36, bbox_to_anchor=(1, 1))
    fig.suptitle("All Models Performance Metrics", fontsize=44)
    save_dual_format(fig, os.path.join(ML_OUTPUT_DIR, "all_models_metrics_merged"))

    # 2. ROC merge
    fig, axes = plt.subplots(3, 3, figsize=(40, 40))
    axes = axes.flatten()
    for i, (name, data) in enumerate(plot_data["roc"]):
        ax = axes[i]
        ax.plot([0, 1], [0, 1], "k--", lw=3)
        for d, t, y, p in data:
            if len(np.unique(y)) <= 1:
                continue
            fpr, tpr, _ = roc_curve(y, p)
            auc, l, u = bootstrap_auc(y, p)
            ax.plot(fpr, tpr, lw=3, label=f"{d} {t}y AUC={auc:.2f}({l:.2f}-{u:.2f})")
        ax.set_title(name, fontsize=36)
        ax.set_xlabel("FPR", fontsize=36)
        ax.set_ylabel("TPR", fontsize=36)
        ax.set_aspect("equal")
        ax.legend(fontsize=36, bbox_to_anchor=(1, 1))
        ax.tick_params(labelsize=36)
    fig.suptitle("All Models ROC Curves", fontsize=44)
    save_dual_format(fig, os.path.join(ML_OUTPUT_DIR, "all_models_roc_merged"))

    # 3. DCA merge
    fig, axes = plt.subplots(3, 3, figsize=(40, 40))
    axes = axes.flatten()
    for i, (name, data) in enumerate(plot_data["dca"]):
        ax = axes[i]
        ax.plot([0, 1], [0, 0], "k--", lw=3)
        for d, t, y, p in data:
            if len(np.unique(y)) <= 1:
                continue
            th, nb = calculate_dca(y, p)
            ax.plot(th, nb, lw=3, label=f"{d} {t}y")
        ax.set_title(name, fontsize=36)
        ax.set_xlabel("Threshold", fontsize=36)
        ax.set_ylabel("Net Benefit", fontsize=36)
        ax.set_ylim(-0.05, 0.2)
        ax.legend(fontsize=36, bbox_to_anchor=(1, 1))
        ax.tick_params(labelsize=36)
    fig.suptitle("All Models DCA", fontsize=44)
    save_dual_format(fig, os.path.join(ML_OUTPUT_DIR, "all_models_dca_merged"))

    # 4. Binary risk-group HR forest plot
    fig, ax = plt.subplots(figsize=(20, 14))
    models = binary_hr_df["Model"].values[::-1]
    y = np.arange(len(models))
    tr_hr = binary_hr_df["Train_HR"].values[::-1]
    tr_err = [tr_hr - binary_hr_df["Train_L"].values[::-1],
              binary_hr_df["Train_U"].values[::-1] - tr_hr]
    te_hr = binary_hr_df["Test_HR"].values[::-1]
    te_err = [te_hr - binary_hr_df["Test_L"].values[::-1],
              binary_hr_df["Test_U"].values[::-1] - te_hr]

    ax.errorbar(tr_hr[~np.isnan(tr_hr)], y[~np.isnan(tr_hr)] - 0.15,
                xerr=np.array(tr_err)[:, ~np.isnan(tr_hr)],
                fmt="o", c="#1f77b4", label="Train Set", capsize=12, ms=16, lw=4)
    ax.errorbar(te_hr[~np.isnan(te_hr)], y[~np.isnan(te_hr)] + 0.15,
                xerr=np.array(te_err)[:, ~np.isnan(te_hr)],
                fmt="s", c="#ff7f0e", label="Test Set", capsize=12, ms=16, lw=4)

    ax.axvline(1, c="gray", ls="--", lw=3)
    ax.set_yticks(y, models)
    ax.set_xlabel("Hazard Ratio (95% CI)", fontsize=36)
    ax.set_ylabel("Model", fontsize=36)
    ax.set_title("Binary Risk Group Hazard Ratio", fontsize=42)
    ax.legend(fontsize=32)
    ax.set_xscale("log")
    ax.tick_params(labelsize=30)
    save_dual_format(fig, os.path.join(ML_OUTPUT_DIR, "binary_hr_forest_plot"))


# ============================================================================
# Main
# ============================================================================
def main():
    setup_matplotlib()
    ensure_dir(ML_OUTPUT_DIR)

    df = pd.read_excel(SIGNIFICANT_VARS_PATH).dropna()
    X = df.drop([ID_COL, TIME_COL, EVENT_COL], axis=1).values
    y_time = df[TIME_COL].values
    y_event = df[EVENT_COL].values
    ids = df[ID_COL].values

    X_train, X_test, yt_train, yt_test, ye_train, ye_test, ids_train, ids_test = train_test_split(
        X, y_time, y_event, ids,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_event,
    )

    id_map = {i: "Train" for i in ids_train} | {i: "Test" for i in ids_test}

    models = get_models()
    all_results, hr_res, binary_hr = [], [], []
    plot_data = {"roc": [], "cm": [], "dca": [], "calib": [], "metrics": []}

    for name, model in models.items():
        process_single_model(
            name, model, X_train, X_test,
            yt_train, ye_train,
            yt_test, ye_test,
            ids, y_time, y_event, id_map,
            all_results, hr_res, binary_hr,
            plot_data,
        )

    pd.DataFrame(all_results).to_csv(
        os.path.join(ML_OUTPUT_DIR, "all_models_combined_metrics.csv"),
        index=False, float_format="%.6f")
    pd.DataFrame(binary_hr).to_csv(
        os.path.join(ML_OUTPUT_DIR, "all_models_binary_hr_results.csv"),
        index=False, float_format="%.6f")

    merge_global_plots(plot_data, pd.DataFrame(binary_hr))


if __name__ == "__main__":
    main()
