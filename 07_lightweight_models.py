"""
Lightweight Models for Depression Onset Risk Prediction
========================================================
Multi-time-point survival-based prediction using 6 ultra-lightweight
PyTorch models:
    nanoTabPFN, LimiX2M, SERSAL, TabDistill, TabICLv2, Tables2Traces.

The dataset is split once into training / test sets (80/20). Every model is
trained with a small batch size and aggressive CUDA / garbage-collection
memory management.  

Paths, column names, and global constants are imported from config.py;
shared evaluation and plotting helpers are imported from utils.py.
"""

import os
import gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.calibration import calibration_curve
from matplotlib.backends.backend_pdf import PdfPages

from config import (
    SIGNIFICANT_VARS_PATH, LIGHTWEIGHT_OUTPUT_DIR,
    ID_COL, TIME_COL, EVENT_COL,
    TEST_SIZE, RANDOM_STATE,
)
from utils import (
    setup_matplotlib, calculate_metrics, safe_predict_proba,
    bootstrap_auc, calculate_dca, save_dual_format, ensure_dir,
)

# Aggressive memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32, garbage_collection_threshold:0.8"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MPLBACKEND"] = "Agg"
torch.cuda.empty_cache()
gc.collect()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training hyper-parameters (small batch for memory-constrained lightweight models)
BATCH_SIZE = 32
EPOCHS = 35
LEARNING_RATE = 1e-3

# Lightweight stage evaluates four follow-up horizons (config.TIME_POINTS uses five).
TIME_POINTS = [2, 4, 6, 8]


# ============================================================================
# Ultra-Lightweight Model Definitions
# ============================================================================
class NanoTabPFN(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1, num_heads=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.bn_input = nn.BatchNorm1d(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 2,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.bn_input(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.classifier(x)


class LimiX2M(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 16)
        self.random_effects = nn.Linear(input_dim, 8)
        self.classifier = nn.Sequential(
            nn.Linear(24, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        linear_out = self.linear(x)
        random_out = self.random_effects(x)
        combined = torch.cat([linear_out, random_out], dim=1)
        return self.classifier(combined)


class SERSAL(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)


class TabDistill(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class TabICLv2(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.feature_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Softmax(dim=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        attention_weights = self.feature_attention(x)
        weighted_features = x * attention_weights
        return self.classifier(weighted_features)


class Tables2Traces(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv = nn.Conv1d(1, 8, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        return self.classifier(x)


# ============================================================================
# Unified Model Wrapper
# ============================================================================
class TorchModelWrapper:
    """Adapter that gives any nn.Module the sklearn fit / predict_proba API,
    with batched inference and explicit memory release."""

    def __init__(self, model_class, **kwargs):
        self.model_class = model_class
        self.kwargs = kwargs
        self.model = None

    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=device).view(-1, 1)

        loader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            BATCH_SIZE, shuffle=True, pin_memory=False, num_workers=0,
        )

        self.model = self.model_class(input_dim=X.shape[1], **self.kwargs).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        self.model.train()
        for epoch in range(EPOCHS):
            for x, yb in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.model(x), yb)
                loss.backward()
                optimizer.step()

            if epoch % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        return self

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            batch_size = 32
            probs = []
            for i in range(0, X.shape[0], batch_size):
                batch = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
                prob_batch = self.model(batch).cpu().numpy().ravel()
                probs.append(prob_batch)
                del batch
                gc.collect()
            prob = np.concatenate(probs)
        return np.column_stack([1 - prob, prob])


# ============================================================================
# Local Helper
# ============================================================================
def safe_remove_legend(ax):
    """Remove a matplotlib legend from an axis if one exists."""
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()


# ============================================================================
# Single Model Processing (labels fixed: test set uses ye_test)
# ============================================================================
def process_single_model(model_name, model, X_train, X_test,
                         yt_train, ye_train, yt_test, ye_test,
                         all_ids, all_time, all_event, id_map,
                         all_results, plot_data, all_predictions,
                         feature_names):
    """Train one lightweight model, evaluate it across TIME_POINTS, save outputs."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_all_scaled = scaler.transform(np.vstack([X_train, X_test]))

    try:
        model.fit(X_train_scaled, ye_train)
    except Exception:
        plt.close("all")
        torch.cuda.empty_cache()
        gc.collect()
        return

    model_dir = ensure_dir(os.path.join(LIGHTWEIGHT_OUTPUT_DIR, model_name))

    metrics_list = []
    roc_data, dca_data, calib_data = [], [], []

    pred_df = pd.DataFrame({
        ID_COL: all_ids,
        "Dataset": [id_map[i] for i in all_ids],
        TIME_COL: all_time,
        EVENT_COL: all_event,
    })

    final_predictions = safe_predict_proba(model, X_all_scaled)
    pred_df["Final_Prediction"] = final_predictions

    n_train = len(X_train)
    n_test = len(X_test)
    n_total = len(pred_df)

    for t in TIME_POINTS:
        # ROOT-CAUSE FIX (preserved):
        # Previously the test set labels were incorrectly derived from ye_train.
        # The test incident-event label now correctly uses ye_test.
        y_train_actual = (yt_train <= t) & (ye_train == 1)
        y_test_actual = (yt_test <= t) & (ye_test == 1)

        # Shape validation safeguards
        assert len(y_train_actual) == n_train, \
            f"Train label shape mismatch: {len(y_train_actual)} vs {n_train}"
        assert len(y_test_actual) == n_test, \
            f"Test label shape mismatch: {len(y_test_actual)} vs {n_test}"

        p_train = safe_predict_proba(model, X_train_scaled)
        p_test = safe_predict_proba(model, X_test_scaled)

        roc_data.extend([("Train", t, y_train_actual, p_train),
                         ("Test", t, y_test_actual, p_test)])
        dca_data.extend([("Train", t, y_train_actual, p_train),
                         ("Test", t, y_test_actual, p_test)])
        calib_data.extend([("Train", t, y_train_actual, p_train),
                           ("Test", t, y_test_actual, p_test)])

        for dataset, y_true, y_prob in [("Train", y_train_actual, p_train),
                                        ("Test", y_test_actual, p_test)]:
            if len(np.unique(y_true)) > 1:
                y_pred = (y_prob > 0.5).astype(int)
                metrics = calculate_metrics(y_true, y_pred, y_prob)
                metrics.update({"Model": model_name, "Time": t, "Dataset": dataset})
                metrics_list.append(metrics)
                all_results.append(metrics)

        all_actual = np.concatenate([y_train_actual, y_test_actual])
        all_pred = np.concatenate([p_train, p_test])

        # Final shape validation before assignment
        assert len(all_actual) == n_total, \
            f"Combined actual shape mismatch: {len(all_actual)} vs {n_total}"
        assert len(all_pred) == n_total, \
            f"Combined prediction shape mismatch: {len(all_pred)} vs {n_total}"

        pred_df[f"Actual_{t}y"] = all_actual.astype(int)
        pred_df[f"Pred_{t}y"] = all_pred

    pd.DataFrame(metrics_list).to_csv(
        os.path.join(model_dir, "metrics.csv"), index=False, float_format="%.6f")
    pred_df.to_csv(os.path.join(model_dir, "predictions.csv"),
                   index=False, float_format="%.6f")

    all_predictions.append((model_name, pred_df))
    plot_data["roc"].append((model_name, roc_data))
    plot_data["dca"].append((model_name, dca_data))
    plot_data["calib"].append((model_name, calib_data))

    # Individual ROC
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--")
    for dataset, t, y_true, y_prob in roc_data:
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_mean, _, _ = bootstrap_auc(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{dataset} {t}y {auc_mean:.2f}")
    ax.set_title(f"{model_name} ROC")
    ax.axis("equal")
    ax.legend(fontsize=12)
    save_dual_format(fig, os.path.join(model_dir, "roc"))

    # Individual DCA
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 0], "k--", label="Treat None")
    ax.plot([0, 1], [1, -1], "k:", label="Treat All")
    for dataset, t, y_true, y_prob in dca_data:
        if len(np.unique(y_true)) < 2:
            continue
        thresholds, nb = calculate_dca(y_true, y_prob)
        ax.plot(thresholds, nb, label=f"{dataset} {t}y")
    ax.set_title(f"{model_name} DCA")
    ax.set_ylim(0, 0.2)
    ax.legend(fontsize=12)
    save_dual_format(fig, os.path.join(model_dir, "dca"))

    # Individual calibration
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--")
    for dataset, t, y_true, y_prob in calib_data:
        if len(np.unique(y_true)) < 2:
            continue
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=5)
        ax.plot(prob_pred, prob_true, "o-", label=f"{dataset} {t}y")
    ax.set_title(f"{model_name} Calibration")
    ax.axis("equal")
    ax.legend(fontsize=12)
    save_dual_format(fig, os.path.join(model_dir, "calibration"))

    # Merge individual plots into a single PDF (re-read saved PNGs)
    with PdfPages(os.path.join(model_dir, "all_plots_combined.pdf")) as pdf:
        for plot_name in ["roc", "dca", "calibration"]:
            page = plt.figure(figsize=(6, 6))
            img = plt.imread(os.path.join(model_dir, f"{plot_name}.png"))
            plt.imshow(img)
            plt.axis("off")
            pdf.savefig(page, bbox_inches="tight")
            plt.close(page)
            gc.collect()

    plt.close("all")
    torch.cuda.empty_cache()
    gc.collect()


# ============================================================================
# Incremental Plotting
# ============================================================================
def plot_metrics_trends_incremental(metrics_df, output_path):
    """2x3 panels of each metric across follow-up time for all models."""
    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC", "Brier"]
    y_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC", "Brier Score"]
    palette = sns.color_palette("tab10", n_colors=6)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, (metric, y_label) in enumerate(zip(metrics, y_labels)):
        ax = axes[i]
        sns.lineplot(
            data=metrics_df, x="Time", y=metric, hue="Model", style="Dataset",
            markers=True, dashes={"Train": (1, 0), "Test": (2, 2)},
            palette=palette, ax=ax, linewidth=2, markersize=10,
        )
        ax.set_title(f"{metric} Over Time", fontsize=20)
        ax.set_xlabel("Follow-up Time (Years)", fontsize=18)
        ax.set_ylabel(y_label, fontsize=18)
        ax.set_xticks(TIME_POINTS)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", alpha=0.5)
        safe_remove_legend(ax)
        gc.collect()

    handles, labels = axes[0].get_legend_handles_labels()
    if handles and labels:
        fig.legend(handles, labels, loc="lower center", ncol=3,
                   fontsize=16, bbox_to_anchor=(0.5, 0.01))

    plt.subplots_adjust(bottom=0.12)
    save_dual_format(fig, output_path)


def plot_merged_incremental(plot_data, plot_type, title, output_path):
    """2x3 merged panels for ROC, DCA, or calibration across all models."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    all_handles, all_labels = [], []

    for i, (model_name, data) in enumerate(plot_data[plot_type]):
        ax = axes[i]
        if plot_type == "roc":
            ax.plot([0, 1], [0, 1], "k--")
            for dataset, t, y_true, y_prob in data:
                if len(np.unique(y_true)) < 2:
                    continue
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                auc_mean, _, _ = bootstrap_auc(y_true, y_prob)
                line, = ax.plot(fpr, tpr, label=f"{dataset} {t}y {auc_mean:.2f}")
                if i == 0:
                    all_handles.append(line)
                    all_labels.append(f"{dataset} {t}y {auc_mean:.2f}")
            ax.axis("equal")
        elif plot_type == "dca":
            line1, = ax.plot([0, 1], [0, 0], "k--", label="Treat None")
            line2, = ax.plot([0, 1], [1, -1], "k:", label="Treat All")
            if i == 0:
                all_handles.extend([line1, line2])
                all_labels.extend(["Treat None", "Treat All"])
            for dataset, t, y_true, y_prob in data:
                if len(np.unique(y_true)) < 2:
                    continue
                thresholds, nb = calculate_dca(y_true, y_prob)
                line, = ax.plot(thresholds, nb, label=f"{dataset} {t}y")
                if i == 0:
                    all_handles.append(line)
                    all_labels.append(f"{dataset} {t}y")
            ax.set_ylim(0, 0.2)
        else:
            ax.plot([0, 1], [0, 1], "k--")
            for dataset, t, y_true, y_prob in data:
                if len(np.unique(y_true)) < 2:
                    continue
                prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=5)
                line, = ax.plot(prob_pred, prob_true, "o-", label=f"{dataset} {t}y")
                if i == 0:
                    all_handles.append(line)
                    all_labels.append(f"{dataset} {t}y")
            ax.axis("equal")

        ax.set_title(model_name, fontsize=20)
        safe_remove_legend(ax)
        gc.collect()

    if all_handles and all_labels:
        fig.legend(all_handles, all_labels, loc="lower center", ncol=2,
                   fontsize=14, bbox_to_anchor=(0.5, 0.01))

    plt.subplots_adjust(bottom=0.12)
    save_dual_format(fig, output_path)


# ============================================================================
# Global Summary
# ============================================================================
def merge_global_plots(plot_data, all_results, all_predictions):
    """Global comparison plots plus a combined prediction workbook."""
    metrics_df = pd.DataFrame(all_results)

    plot_metrics_trends_incremental(metrics_df, os.path.join(LIGHTWEIGHT_OUTPUT_DIR, "all_metrics_trends"))
    plot_merged_incremental(plot_data, "roc", "ROC Curves",
                            os.path.join(LIGHTWEIGHT_OUTPUT_DIR, "all_roc_merged"))
    plot_merged_incremental(plot_data, "dca", "DCA Curves",
                            os.path.join(LIGHTWEIGHT_OUTPUT_DIR, "all_dca_merged"))
    plot_merged_incremental(plot_data, "calib", "Calibration Curves",
                            os.path.join(LIGHTWEIGHT_OUTPUT_DIR, "all_calibration_merged"))

    metrics_df.to_csv(os.path.join(LIGHTWEIGHT_OUTPUT_DIR, "all_metrics_summary.csv"),
                      index=False, float_format="%.6f")

    if all_predictions:
        combined_df = all_predictions[0][1].copy()
        for model_name, pred_df in all_predictions[1:]:
            for col in pred_df.columns:
                if col not in [ID_COL, "Dataset", TIME_COL, EVENT_COL]:
                    combined_df[f"{model_name}_{col}"] = pred_df[col].values

        try:
            with pd.ExcelWriter(os.path.join(LIGHTWEIGHT_OUTPUT_DIR, "all_predictions_combined.xlsx"),
                                engine="openpyxl", mode="w") as writer:
                combined_df.to_excel(writer, sheet_name="Combined_Results",
                                     index=False, float_format="%.6f")
                for model_name, pred_df in all_predictions:
                    pred_df.to_excel(writer, sheet_name=model_name[:31],
                                     index=False, float_format="%.6f")
        except Exception:
            combined_df.to_csv(os.path.join(LIGHTWEIGHT_OUTPUT_DIR, "all_predictions_combined.csv"),
                               index=False, float_format="%.6f")

    gc.collect()


# ============================================================================
# Main
# ============================================================================
def main():
    setup_matplotlib()
    ensure_dir(LIGHTWEIGHT_OUTPUT_DIR)

    df = pd.read_excel(SIGNIFICANT_VARS_PATH).dropna()
    X = df.drop([ID_COL, TIME_COL, EVENT_COL], axis=1).values
    y_time = df[TIME_COL].values
    y_event = df[EVENT_COL].values
    ids = df[ID_COL].values
    feature_names = df.drop([ID_COL, TIME_COL, EVENT_COL], axis=1).columns.tolist()

    X_train, X_test, yt_train, yt_test, ye_train, ye_test, ids_train, ids_test = train_test_split(
        X, y_time, y_event, ids,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_event,
    )

    id_map = {i: "Train" for i in ids_train} | {i: "Test" for i in ids_test}

    models = {
        "nanoTabPFN": TorchModelWrapper(NanoTabPFN),
        "LimiX_2M": TorchModelWrapper(LimiX2M),
        "SERSAL": TorchModelWrapper(SERSAL),
        "TabDistill": TorchModelWrapper(TabDistill),
        "TabICLv2": TorchModelWrapper(TabICLv2),
        "Tables2Traces": TorchModelWrapper(Tables2Traces),
    }

    all_results = []
    all_predictions = []
    plot_data = {"roc": [], "dca": [], "calib": []}

    for model_name, model in models.items():
        process_single_model(
            model_name, model, X_train, X_test,
            yt_train, ye_train,
            yt_test, ye_test,
            ids, y_time, y_event, id_map,
            all_results, plot_data, all_predictions, feature_names,
        )

    merge_global_plots(plot_data, all_results, all_predictions)


if __name__ == "__main__":
    main()
