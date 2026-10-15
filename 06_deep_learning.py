"""
Deep Learning Models for Depression Onset Risk Prediction
==========================================================
Multi-time-point survival-based prediction using 9 PyTorch deep models:
    DeepMLP, WideDeep, DeepFM, NFM, DCN, xDeepFM, AutoInt, FTTransformer, TabNet.

The dataset is split once into training / test sets (80/20). Every model is
trained with mini-batch gradient descent (BCELoss, Adam) and evaluated at the
follow-up time points in TIME_POINTS. 

Paths, column names, and global constants are imported from config.py;
shared evaluation and plotting helpers are imported from utils.py.
"""

import os
import numpy as np
import pandas as pd
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
    SIGNIFICANT_VARS_PATH, DL_OUTPUT_DIR,
    ID_COL, TIME_COL, EVENT_COL, TIME_POINTS,
    TEST_SIZE, RANDOM_STATE,
)
from utils import (
    setup_matplotlib, calculate_metrics, safe_predict_proba,
    bootstrap_auc, calculate_dca, save_dual_format, ensure_dir,
)

# Memory optimization for CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.cuda.empty_cache()

# Training hyper-parameters
BATCH_SIZE = 256
EPOCHS = 35
LEARNING_RATE = 1e-3

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# PyTorch Model Wrapper
# ============================================================================
class TorchModelWrapper:
    """Adapter that gives any nn.Module the sklearn fit / predict_proba API."""

    def __init__(self, model_class, **kwargs):
        self.model_class = model_class
        self.kwargs = kwargs
        self.model = None

    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=device).view(-1, 1)
        loader = DataLoader(TensorDataset(X_tensor, y_tensor), BATCH_SIZE, shuffle=True)

        self.model = self.model_class(input_dim=X.shape[1], **self.kwargs).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        self.model.train()
        for _ in range(EPOCHS):
            for x, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(x), yb)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            prob = self.model(
                torch.tensor(X, dtype=torch.float32, device=device)
            ).cpu().numpy().ravel()
        return np.column_stack([1 - prob, prob])


# ============================================================================
# PyTorch Models
# ============================================================================
class DeepMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class WideDeep(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.wide = nn.Linear(input_dim, 1)
        self.deep = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        self.out = nn.Linear(33, 1)

    def forward(self, x):
        wide_out = self.wide(x)
        deep_out = self.deep(x)
        combined = torch.cat([wide_out, deep_out], dim=1)
        return torch.sigmoid(self.out(combined))


class DeepFM(nn.Module):
    def __init__(self, input_dim, embed_dim=8):
        super().__init__()
        self.emb = nn.Parameter(torch.randn(input_dim, embed_dim))
        self.linear = nn.Linear(input_dim, 1)
        self.deep = nn.Sequential(
            nn.Linear(input_dim * embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        emb = x.unsqueeze(-1) * self.emb
        fm_term = 0.5 * (torch.sum(emb, dim=1) ** 2 - torch.sum(emb ** 2, dim=1)).sum(dim=1, keepdim=True)
        deep_term = self.deep(emb.flatten(start_dim=1))
        return torch.sigmoid(self.linear(x) + fm_term + deep_term)


class NFM(nn.Module):
    def __init__(self, input_dim, embed_dim=8):
        super().__init__()
        self.emb = nn.Parameter(torch.randn(input_dim, embed_dim))
        self.deep = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        emb = x.unsqueeze(-1) * self.emb
        bi_interaction = 0.5 * (torch.sum(emb, dim=1) ** 2 - torch.sum(emb ** 2, dim=1))
        return torch.sigmoid(self.deep(bi_interaction))


class DCN(nn.Module):
    def __init__(self, input_dim, cross_layers=2):
        super().__init__()
        self.cross_w = nn.ModuleList(
            [nn.Linear(input_dim, input_dim, bias=False) for _ in range(cross_layers)]
        )
        self.deep = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        self.out = nn.Linear(32 + input_dim, 1)

    def forward(self, x):
        cross = x
        for w in self.cross_w:
            cross = x * w(cross) + cross
        combined = torch.cat([cross, self.deep(x)], dim=1)
        return torch.sigmoid(self.out(combined))


class xDeepFM(nn.Module):
    def __init__(self, input_dim, embed_dim=8):
        super().__init__()
        self.emb = nn.Parameter(torch.randn(input_dim, embed_dim))
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        emb = x.unsqueeze(-1) * self.emb
        return torch.sigmoid(self.mlp(torch.sum(emb, dim=1)))


class AutoInt(nn.Module):
    def __init__(self, input_dim, embed_dim=8, heads=2):
        super().__init__()
        self.emb = nn.Parameter(torch.randn(input_dim, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        self.out = nn.Linear(embed_dim, 1)

    def forward(self, x):
        emb = self.emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
        attn_out, _ = self.attn(emb, emb, emb)
        return torch.sigmoid(self.out(attn_out.mean(dim=1)))


class FTTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=8):
        super().__init__()
        self.emb = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, 2, batch_first=True)
        self.trans = nn.TransformerEncoder(encoder_layer, 1)
        self.out = nn.Linear(embed_dim, 1)

    def forward(self, x):
        emb = self.emb(x).unsqueeze(1)
        trans_out = self.trans(emb).squeeze(1)
        return torch.sigmoid(self.out(trans_out))


class TabNet(nn.Module):
    def __init__(self, input_dim, dim=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.fc(x)


# ============================================================================
# Local Helper
# ============================================================================
def safe_remove_legend(ax):
    """Remove a matplotlib legend from an axis if one exists."""
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()


# ============================================================================
# Single Model Processing
# ============================================================================
def process_single_model(model_name, model, X_train, X_test,
                         yt_train, ye_train, yt_test, ye_test,
                         all_ids, all_time, all_event, id_map,
                         all_results, plot_data, all_predictions):
    """Train one deep model, evaluate it across TIME_POINTS, save figures/tables."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_all_scaled = scaler.transform(np.vstack([X_train, X_test]))

    try:
        model.fit(X_train_scaled, ye_train)
    except Exception:
        plt.close("all")
        torch.cuda.empty_cache()
        return

    model_dir = ensure_dir(os.path.join(DL_OUTPUT_DIR, model_name))

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

    for t in TIME_POINTS:
        y_train_actual = (yt_train <= t) & (ye_train == 1)
        y_test_actual = (yt_test <= t) & (ye_test == 1)

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

    # 1. ROC curve (with overall AUC annotation)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--")
    has_data = False

    overall_train_auc, overall_train_l, overall_train_u = 0.5, 0.5, 0.5
    overall_test_auc, overall_test_l, overall_test_u = 0.5, 0.5, 0.5

    if len(np.unique(ye_train)) > 1:
        p_train_overall = safe_predict_proba(model, X_train_scaled)
        overall_train_auc, overall_train_l, overall_train_u = bootstrap_auc(ye_train, p_train_overall)
    if len(np.unique(ye_test)) > 1:
        p_test_overall = safe_predict_proba(model, X_test_scaled)
        overall_test_auc, overall_test_l, overall_test_u = bootstrap_auc(ye_test, p_test_overall)

    for dataset, t, y_true, y_prob in roc_data:
        if len(np.unique(y_true)) < 2:
            continue
        has_data = True
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_mean, auc_lower, auc_upper = bootstrap_auc(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{dataset} {t}y {auc_mean:.2f}({auc_lower:.2f}-{auc_upper:.2f})")

    if has_data:
        annotation_text = (
            f"Overall Train AUC: {overall_train_auc:.2f} ({overall_train_l:.2f}-{overall_train_u:.2f})\n"
            f"Overall Test AUC: {overall_test_auc:.2f} ({overall_test_l:.2f}-{overall_test_u:.2f})"
        )
        ax.text(0.05, 0.05, annotation_text, transform=ax.transAxes, fontsize=14,
                bbox=dict(facecolor="white", alpha=0.8))
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=16)

    ax.set_title(f"{model_name} ROC")
    ax.axis("equal")
    save_dual_format(fig, os.path.join(model_dir, "roc"))

    # 2. DCA curve (y-axis 0-0.2, negative net benefit clipped to 0)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 0], "k--", label="Treat None")
    ax.plot([0, 1], [1, -1], "k:", label="Treat All")
    has_data = False
    for dataset, t, y_true, y_prob in dca_data:
        if len(np.unique(y_true)) < 2:
            continue
        has_data = True
        thresholds, nb = calculate_dca(y_true, y_prob)
        ax.plot(thresholds, nb, label=f"{dataset} {t}y")
    ax.set_title(f"{model_name} DCA")
    if has_data:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=24)
    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Benefit")
    ax.set_ylim(bottom=0, top=0.2)
    save_dual_format(fig, os.path.join(model_dir, "dca"))

    # 3. Calibration curve
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--")
    has_data = False
    for dataset, t, y_true, y_prob in calib_data:
        if len(np.unique(y_true)) < 2:
            continue
        has_data = True
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=5)
        ax.plot(prob_pred, prob_true, "o-", label=f"{dataset} {t}y")
    ax.set_title(f"{model_name} Calibration")
    if has_data:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=16)
    ax.axis("equal")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Frequency")
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

    plt.close("all")
    torch.cuda.empty_cache()


# ============================================================================
# Metrics Trend Plot
# ============================================================================
def plot_metrics_trends(metrics_df, output_dir):
    """2x3 panels of each metric across follow-up time for all models."""
    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC", "Brier"]
    y_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC", "Brier Score"]

    fig, axes = plt.subplots(2, 3, figsize=(30, 20))
    axes = axes.flatten()
    palette = sns.color_palette("tab10", n_colors=9)

    handles, labels = [], []
    for i, (metric, y_label) in enumerate(zip(metrics, y_labels)):
        ax = axes[i]
        sns.lineplot(
            data=metrics_df, x="Time", y=metric, hue="Model", style="Dataset",
            markers=True, dashes={"Train": (1, 0), "Test": (2, 2)},
            palette=palette, ax=ax, linewidth=2, markersize=12, markeredgewidth=2,
        )
        ax.set_title(f"{metric} Over Follow-up Time", fontsize=32, pad=20)
        ax.set_xlabel("Follow-up Time (Years)", fontsize=28, labelpad=15)
        ax.set_ylabel(y_label, fontsize=28, labelpad=15)
        ax.tick_params(axis="both", which="major", labelsize=24)
        ax.grid(True, linestyle="--", alpha=0.6, linewidth=2)
        ax.set_xticks(TIME_POINTS)
        ax.set_ylim(0, 1)

        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
        safe_remove_legend(ax)

    if handles and labels:
        fig.legend(handles, labels, loc="lower center", ncol=5,
                   fontsize=24, bbox_to_anchor=(0.5, 0.02))

    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    save_dual_format(fig, os.path.join(output_dir, "all_metrics_trends"))


# ============================================================================
# Global Merged Plots
# ============================================================================
def merge_global_plots(plot_data, all_results, all_predictions):
    """Global comparison plots plus a combined prediction workbook."""
    metrics_df = pd.DataFrame(all_results)

    # 1. Metrics trend
    plot_metrics_trends(metrics_df, DL_OUTPUT_DIR)

    # 2. Merged ROC curves
    fig, axes = plt.subplots(3, 3, figsize=(24, 24))
    axes = axes.flatten()
    all_handles, all_labels = [], []
    for i, (model_name, data) in enumerate(plot_data["roc"]):
        ax = axes[i]
        ax.plot([0, 1], [0, 1], "k--")
        for dataset, t, y_true, y_prob in data:
            if len(np.unique(y_true)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_mean, auc_lower, auc_upper = bootstrap_auc(y_true, y_prob)
            line, = ax.plot(fpr, tpr,
                            label=f"{dataset} {t}y {auc_mean:.2f}({auc_lower:.2f}-{auc_upper:.2f})")
            if i == 0:
                all_handles.append(line)
                all_labels.append(f"{dataset} {t}y {auc_mean:.2f}({auc_lower:.2f}-{auc_upper:.2f})")
        ax.set_title(model_name, fontsize=36)
        ax.axis("equal")
        safe_remove_legend(ax)

    if all_handles and all_labels:
        fig.legend(all_handles, all_labels, loc="lower center", ncol=2,
                   fontsize=20, bbox_to_anchor=(0.5, 0.01))
    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    save_dual_format(fig, os.path.join(DL_OUTPUT_DIR, "all_roc_merged"))

    # 3. Merged DCA curves
    fig, axes = plt.subplots(3, 3, figsize=(24, 24))
    axes = axes.flatten()
    all_handles, all_labels = [], []
    for i, (model_name, data) in enumerate(plot_data["dca"]):
        ax = axes[i]
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
        ax.set_title(model_name, fontsize=36)
        ax.set_xlabel("Threshold Probability")
        ax.set_ylabel("Net Benefit")
        ax.set_ylim(bottom=0, top=0.2)
        safe_remove_legend(ax)

    if all_handles and all_labels:
        fig.legend(all_handles, all_labels, loc="lower center", ncol=2,
                   fontsize=20, bbox_to_anchor=(0.5, 0.01))
    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    save_dual_format(fig, os.path.join(DL_OUTPUT_DIR, "all_dca_merged"))

    # 4. Merged calibration curves
    fig, axes = plt.subplots(3, 3, figsize=(24, 24))
    axes = axes.flatten()
    all_handles, all_labels = [], []
    for i, (model_name, data) in enumerate(plot_data["calib"]):
        ax = axes[i]
        ax.plot([0, 1], [0, 1], "k--")
        for dataset, t, y_true, y_prob in data:
            if len(np.unique(y_true)) < 2:
                continue
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=5)
            line, = ax.plot(prob_pred, prob_true, "o-", label=f"{dataset} {t}y")
            if i == 0:
                all_handles.append(line)
                all_labels.append(f"{dataset} {t}y")
        ax.set_title(model_name, fontsize=36)
        ax.axis("equal")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Observed Frequency")
        safe_remove_legend(ax)

    if all_handles and all_labels:
        fig.legend(all_handles, all_labels, loc="lower center", ncol=2,
                   fontsize=20, bbox_to_anchor=(0.5, 0.01))
    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    save_dual_format(fig, os.path.join(DL_OUTPUT_DIR, "all_calibration_merged"))

    metrics_df.to_csv(os.path.join(DL_OUTPUT_DIR, "all_metrics_summary.csv"),
                      index=False, float_format="%.6f")

    # 5. Merge all predictions into a single workbook (fall back to CSV)
    if all_predictions:
        combined_df = all_predictions[0][1].copy()
        for model_name, pred_df in all_predictions[1:]:
            for col in pred_df.columns:
                if col not in [ID_COL, "Dataset", TIME_COL, EVENT_COL]:
                    combined_df[f"{model_name}_{col}"] = pred_df[col].values

        try:
            with pd.ExcelWriter(os.path.join(DL_OUTPUT_DIR, "all_predictions_combined.xlsx"),
                                engine="xlsxwriter", mode="w") as writer:
                combined_df.to_excel(writer, sheet_name="Combined_Wide",
                                     index=False, float_format="%.6f")
                for model_name, pred_df in all_predictions:
                    pred_df.to_excel(writer, sheet_name=model_name[:31],
                                     index=False, float_format="%.6f")
        except Exception:
            combined_df.to_csv(os.path.join(DL_OUTPUT_DIR, "all_predictions_combined.csv"),
                               index=False, float_format="%.6f")


# ============================================================================
# Main
# ============================================================================
def main():
    setup_matplotlib()
    ensure_dir(DL_OUTPUT_DIR)

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

    models = {
        "Deep_MLP": TorchModelWrapper(DeepMLP),
        "Wide_Deep": TorchModelWrapper(WideDeep),
        "DeepFM": TorchModelWrapper(DeepFM),
        "NFM": TorchModelWrapper(NFM),
        "DCN": TorchModelWrapper(DCN),
        "xDeepFM": TorchModelWrapper(xDeepFM),
        "AutoInt": TorchModelWrapper(AutoInt),
        "FT_Transformer": TorchModelWrapper(FTTransformer),
        "TabNet": TorchModelWrapper(TabNet),
    }

    all_results = []
    all_predictions = []
    plot_data = {"roc": [], "dca": [], "calib": []}

    for model_name, model in models.items():
        process_single_model(
            model_name, model,
            X_train, X_test,
            yt_train, ye_train,
            yt_test, ye_test,
            ids, y_time, y_event,
            id_map, all_results, plot_data, all_predictions,
        )

    merge_global_plots(plot_data, all_results, all_predictions)


if __name__ == "__main__":
    main()
