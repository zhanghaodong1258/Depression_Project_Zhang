"""
Shared Utility Functions
========================
Common helper functions used across all analysis scripts:
  - Statistical formatting (p-values, numeric values)
  - Model evaluation metrics (AUC, accuracy, F1, Brier, etc.)
  - Bootstrap AUC confidence intervals 
  - Safe prediction probability extraction
  - Matplotlib global setup (Arial font, 300 DPI)
  - Dual-format figure saving (PNG + PDF)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, brier_score_loss
)

warnings.filterwarnings("ignore")


# ============================================================================
# Matplotlib Global Setup
# ============================================================================
def setup_matplotlib(font_size: int = 12, title_size: int = 14,
                     label_size: int = 12, tick_size: int = 10,
                     legend_size: int = 10) -> None:
    """
    Configure matplotlib global settings: Arial font, 300 DPI,
    consistent font sizes. Call once at the top of each plotting script.
    """
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": font_size,
        "axes.titlesize": title_size,
        "axes.labelsize": label_size,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "legend.fontsize": legend_size,
        "figure.titlesize": title_size + 2,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ============================================================================
# Statistical Formatting
# ============================================================================
def format_pvalue(p) -> str:
    """Format p-value: '<0.001' if p < 0.001, else 6 decimal places."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    p = float(p)
    if p < 0.001:
        return "<0.001"
    return f"{p:.6f}"


def format_numeric(val) -> str:
    """Format a numeric value: integers as int, others as 2 decimals."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    if float(val).is_integer():
        return str(int(val))
    return f"{val:.2f}"


def median_iqr_str(series: pd.Series) -> str:
    """Return 'median (Q1-Q3)' string for a numeric series."""
    series = series.dropna()
    if len(series) == 0:
        return ""
    median = series.median()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    return f"{format_numeric(median)} ({format_numeric(q1)}-{format_numeric(q3)})"


def freq_percent_str(count: int, total: int) -> str:
    """Return 'count (percentage%)' string."""
    if total == 0:
        return "0 (0.0%)"
    pct = count / total * 100
    return f"{int(count)} ({pct:.1f}%)"


# ============================================================================
# Model Evaluation
# ============================================================================
def calculate_metrics(y_true, y_pred, y_prob) -> dict:
    """
    Calculate standard binary classification metrics.

    Returns dict with: Accuracy, Precision, Recall, F1, AUC, Brier.
    """
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
        "Brier": brier_score_loss(y_true, y_prob),
    }


def safe_predict_proba(model, X) -> np.ndarray:
    """
    Safely extract positive-class prediction probabilities from a model.
    Handles models without predict_proba by falling back to predict.
    Returns a 1D array of probabilities.
    """
    try:
        proba = model.predict_proba(X)
        return proba[:, 1] if proba.shape[1] > 1 else proba.ravel()
    except Exception:
        return model.predict(X)


def bootstrap_auc(y_true, y_prob, n_bootstraps: int = 1000,
                  random_state: int = 42):
    """
    Bootstrap AUC with 95% confidence interval.

    Returns: (mean_auc, ci_lower, ci_upper)
    """
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.5, 0.5
    rng = np.random.RandomState(random_state)
    aucs = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    if not aucs:
        return 0.5, 0.5, 0.5
    return np.mean(aucs), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def calculate_dca(y_true, y_prob, thresholds=None):
    """
    Decision Curve Analysis: compute net benefit across threshold probabilities.
    Negative net benefits are clipped to 0.

    Returns: (thresholds, net_benefits)
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 100)
    net_benefits = []
    n = len(y_true)
    for th in thresholds:
        if th == 1:
            net_benefits.append(0.0)
            continue
        yp = (y_prob >= th)
        tp = np.sum((y_true == 1) & yp)
        fp = np.sum((y_true == 0) & yp)
        nb = (tp - fp * th / (1 - th)) / n
        net_benefits.append(max(nb, 0.0))
    return thresholds, np.array(net_benefits)


# ============================================================================
# Figure Saving
# ============================================================================
def save_dual_format(fig, filepath: str, dpi: int = 300) -> None:
    """
    Save a matplotlib figure in both PNG and PDF formats.
    filepath should not include an extension.
    """
    fig.savefig(f"{filepath}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{filepath}.pdf", bbox_inches="tight")
    plt.close(fig)


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist; return the path."""
    os.makedirs(path, exist_ok=True)
    return path
