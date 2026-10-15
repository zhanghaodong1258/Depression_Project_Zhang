"""
09_ensemble_modeling.py
================================================================================
Ensemble modeling built on top of the single-model predictions produced by
scripts 05-07.

This script merges the original ensembling (04-1) and ensemble hyper-parameter
optimization (04-2) into one pipeline:

  1. Load every model's prediction CSV from the three model output
     directories (config.ML_OUTPUT_DIR, config.DL_OUTPUT_DIR,
     config.LIGHTWEIGHT_OUTPUT_DIR).
  2. Build candidate ensembles:
       * Aaverage (uniform weights)
       * F1-weighted average: weights assigned by each model's average F1
         score on the validation split, positively correlated with
         performance (higher F1 -> higher weight)
       * Optimized F1-weighted average (grid-searched weight shaping)
  3. Grid-search the weighted-average hyper-parameters on the train split
     (weight_exponent, weight_threshold, normalize_weights, clip_weights,
     base_metric) and select the best configuration per ensemble combination.
  4. Evaluate every ensemble on train and test, and write:
       - ensemble_all_predictions.csv   (long-format predictions for plotting)
       - ensemble_metrics.csv            (per-dataset / model / timepoint metrics)
       - tuning_results.csv              (full grid-search results)
       - best_parameters.csv             (best hyper-parameters per combination)
       - optimized_ensemble_predictions.csv (best optimized ensemble predictions)

Expected per-model input files (one prediction CSV per model sub-directory):
  predictions.csv  OR  prediction_results.csv
Columns expected: ID, Dataset, Actual_{t}y, Pred_{t}y for each t in
config.TIME_POINTS.
"""

import os
import warnings
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                             brier_score_loss)

import config as CFG
from utils import ensure_dir, setup_matplotlib

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Ensemble candidate definitions. "algorithms" pins an explicit list,
# "category" selects every model in that category, "all" selects every model.
# ----------------------------------------------------------------------------
ENSEMBLE_COMBINATIONS = [
    {"name": "Best_3_Algorithms",
     "algorithms": ["Deep_MLP", "GaussianNB", "TabICLv2"],
     "description": "Deep_MLP + GaussianNB + TabICLv2"},
    {"name": "All_Machine_Learning", "category": "machine_learning",
     "description": "All machine learning algorithms"},
    {"name": "All_Deep_Learning", "category": "deep_learning",
     "description": "All deep learning algorithms"},
    {"name": "All_Reasoning_Models", "category": "open_light_reasoning_Model",
     "description": "All open-reasoning / lightweight models"},
    {"name": "All_Models", "all": True,
     "description": "All models combined"},
]

DATA_DIR_CATEGORY = [
    (CFG.ML_OUTPUT_DIR, "machine_learning"),
    (CFG.DL_OUTPUT_DIR, "deep_learning"),
    (CFG.LIGHTWEIGHT_OUTPUT_DIR, "open_light_reasoning_Model"),
]

# Coarse grid for the full-combination search; a finer grid can be enabled below.
# base_metric defaults to "F1" so that weights are positively correlated with
# each model's average F1 on the validation split.
TUNING_GRID = {
    "weight_exponent": [0.5, 1.0, 1.5, 2.0, 3.0],
    "weight_threshold": [0.0, 0.3, 0.5, 0.7],
    "normalize_weights": [True],
    "clip_weights": [None, (0.5, 2.0)],
    "base_metric": ["F1", "AUC", "Accuracy"],
}


# ----------------------------------------------------------------------------
# Robust CSV loading
# ----------------------------------------------------------------------------
def read_csv_robust(file_path):
    """Read a prediction CSV with auto delimiter detection and clean headers."""
    last_err = None
    for engine in ("python", "c"):
        try:
            df = pd.read_csv(file_path, sep=None, engine=engine)
            break
        except Exception as exc:  # pragma: no cover - fallback path
            last_err = exc
    else:
        for sep in (",", "\t", ";", "|"):
            try:
                df = pd.read_csv(file_path, sep=sep)
                break
            except Exception as exc:
                last_err = exc
        else:
            raise ValueError(f"Could not read {file_path}: {last_err}")
    df.columns = [str(c).strip().strip("'\"") for c in df.columns]
    return df


def load_all_data():
    """
    Load prediction CSVs from the three model output directories.

    Returns
    -------
    all_data : dict {model_name: DataFrame}
    algorithm_info : dict {model_name: {"category", "category_idx", "path"}}
    """
    all_data, algorithm_info = {}, {}
    for idx, (data_dir, category) in enumerate(DATA_DIR_CATEGORY):
        if not os.path.isdir(data_dir):
            print(f"[warn] directory not found: {data_dir}")
            continue
        subdirs = sorted(d for d in os.listdir(data_dir)
                         if os.path.isdir(os.path.join(data_dir, d)))
        for sub in subdirs:
            sub_path = os.path.join(data_dir, sub)
            candidates = [
                os.path.join(sub_path, "predictions.csv"),
                os.path.join(sub_path, "prediction_results.csv"),
            ]
            csv_file = next((p for p in candidates if os.path.exists(p)), None)
            if csv_file is None:
                continue
            try:
                df = read_csv_robust(csv_file)
                all_data[sub] = df
                algorithm_info[sub] = {
                    "category": category, "category_idx": idx, "path": csv_file,
                }
            except Exception as exc:  # pragma: no cover
                print(f"[warn] failed to load {csv_file}: {exc}")
    print(f"Loaded {len(all_data)} models.")
    return all_data, algorithm_info


# ----------------------------------------------------------------------------
# Ensemble prediction builders
# ----------------------------------------------------------------------------
def _common_ids(filtered_data):
    ids = None
    for df in filtered_data.values():
        if "ID" in df.columns:
            cur = set(df["ID"].unique())
            ids = cur if ids is None else ids & cur
    return sorted(ids) if ids else []


def federated_average(predictions_list):
    """Uniform-weight mean of per-model prediction arrays."""
    return np.mean(np.vstack(predictions_list), axis=0)


def federated_weighted(predictions_list, weights,
                       weight_exponent=1.0, weight_threshold=0.0,
                       normalize_weights=True, clip_weights=None):
    """
    Weighted average of per-model predictions with tunable weight shaping.

    Parameters
    ----------
    predictions_list : list[np.ndarray]
    weights : list[float]          Base weight per model (e.g. per-model AUC).
    weight_exponent : float         Power applied to each base weight.
    weight_threshold : float        Models with shaped weight below this drop out.
    normalize_weights : bool        Divide shaped weights by their sum.
    clip_weights : tuple | None     Optional (low, high) clipping bounds.
    """
    arr = np.vstack(predictions_list)
    w = np.power(np.asarray(weights, dtype=float), weight_exponent)
    mask = w >= weight_threshold
    if not mask.any():
        w = np.ones_like(w)
    else:
        w = np.where(mask, w, 0.0)
    if clip_weights is not None:
        w = np.clip(w, clip_weights[0], clip_weights[1])
    if normalize_weights:
        s = w.sum()
        w = w / s if s > 0 else np.ones_like(w) / len(w)
    return np.average(arr, axis=0, weights=w)


def _base_weight(y_true, y_prob, metric="AUC"):
    """Per-model scalar weight derived from a single validation metric."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    if metric == "AUC":
        return roc_auc_score(y_true, y_prob)
    y_bin = (y_prob >= 0.5).astype(int)
    if metric == "F1":
        return f1_score(y_true, y_bin, zero_division=0)
    if metric == "Accuracy":
        return accuracy_score(y_true, y_bin)
    return 0.5


def compute_f1_weights(all_data, model_names):
    """
    Compute per-model ensemble weights from the average F1 score across all
    time points on the validation (train) split.

    Weights are positively correlated with model performance:
        weight_i = avg_F1_i / sum_j(avg_F1_j)

    Models with higher average F1 receive proportionally higher weights,
    maximizing the complementary advantages of diverse base models.

    Parameters
    ----------
    all_data : dict {model_name: DataFrame}
    model_names : list[str]
        Subset of models to include.

    Returns
    -------
    weights : dict {model_name: float}
        Normalized weights summing to 1.
    avg_f1 : dict {model_name: float}
        Raw average F1 scores (before normalization).
    """
    train_data = {}
    for name, df in all_data.items():
        if "Dataset" in df.columns:
            sub = df[df["Dataset"] == "Train"].copy()
            if not sub.empty:
                train_data[name] = sub

    avg_f1 = {}
    for model in model_names:
        if model not in train_data:
            avg_f1[model] = 0.0
            continue
        d = train_data[model]
        f1s = []
        for t in CFG.TIME_POINTS:
            a_col, p_col = f"Actual_{t}y", f"Pred_{t}y"
            if a_col not in d.columns or p_col not in d.columns:
                continue
            valid = d.dropna(subset=[a_col, p_col])
            if valid.empty:
                continue
            y_true = valid[a_col].values.astype(int)
            if len(np.unique(y_true)) < 2:
                continue
            y_pred = (valid[p_col].values >= 0.5).astype(int)
            f1s.append(f1_score(y_true, y_pred, zero_division=0))
        avg_f1[model] = float(np.mean(f1s)) if f1s else 0.0

    # Normalize: positive correlation with F1, sum to 1.
    total = sum(avg_f1.values())
    if total > 0:
        weights = {m: avg_f1[m] / total for m in model_names}
    else:
        n = len(model_names)
        weights = {m: 1.0 / n for m in model_names} if n > 0 else {}
    return weights, avg_f1


def build_ensemble_table(all_data, algorithm_info, f1_weights=None, dataset=None):
    """
    Generate ensemble predictions in the same long format as single models.

    Strategies:
      - FedAvg:     uniform-weight average across models.
      - F1Weighted: weighted average using pre-computed per-model average F1
                    scores on the validation split. Weights are positively
                    correlated with performance (higher F1 -> higher weight).

    Parameters
    ----------
    f1_weights : dict {model_name: float} | None
        Pre-computed normalized F1-based weights. If None, falls back to
        per-time-point AUC weights for backward compatibility.

    Returns a DataFrame with columns: ID, Dataset, Actual_{t}y, Pred_{t}y,
    Algorithm_Category="ensemble", Algorithm_Name="{combo}_{strategy}".
    """
    filtered = {}
    for name, df in all_data.items():
        if dataset is None or "Dataset" not in df.columns:
            filtered[name] = df.copy()
        else:
            filtered[name] = df[df["Dataset"] == dataset].copy()

    common = _common_ids(filtered)
    if not common:
        return pd.DataFrame()

    rows = []
    for combo in ENSEMBLE_COMBINATIONS:
        selected = _select_models(combo, algorithm_info, filtered)
        if len(selected) < 1:
            continue

        # Build F1-based weight vector for this combination (fixed across time).
        if f1_weights is not None:
            combo_f1_w = [f1_weights.get(m, 0.0) for m in selected]
            w_sum = sum(combo_f1_w)
            if w_sum > 0:
                combo_f1_w = [w / w_sum for w in combo_f1_w]
            else:
                combo_f1_w = [1.0 / len(selected)] * len(selected)
        else:
            combo_f1_w = None

        for strategy_name, strategy in (("FedAvg", "average"),
                                       ("F1Weighted", "f1_weighted")):
            for t in CFG.TIME_POINTS:
                a_col, p_col = f"Actual_{t}y", f"Pred_{t}y"
                preds, gt = [], None
                for model in selected:
                    d = filtered[model][filtered[model]["ID"].isin(common)] \
                        .sort_values("ID")
                    if p_col not in d.columns or a_col not in d.columns:
                        continue
                    y_true = d[a_col].values
                    y_prob = d[p_col].values
                    m = ~np.isnan(y_prob)
                    preds.append(y_prob[m])
                    gt = y_true[m] if gt is None else gt
                if not preds or gt is None:
                    continue
                if strategy == "average":
                    ens_pred = federated_average(preds)
                else:
                    # F1-weighted: use pre-computed average-F1 weights (positive
                    # correlation with performance). Exponent=1 preserves linear
                    # proportionality; adjust via TUNING_GRID for sharper focus.
                    ens_pred = federated_weighted(
                        preds, combo_f1_w,
                        weight_exponent=1.0,
                        weight_threshold=0.0,
                        normalize_weights=True,
                        clip_weights=None,
                    )
                base = pd.DataFrame({
                    "ID": np.asarray(common)[~np.isnan(gt)][:len(ens_pred)],
                    "Dataset": dataset if dataset else "All",
                    a_col: gt,
                    p_col: ens_pred,
                    "Algorithm_Category": "ensemble",
                    "Algorithm_Name": f"{combo['name']}_{strategy_name}",
                })
                rows.append(base)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    # Align Actual/Pred columns across time points (fill missing with NaN).
    for t in CFG.TIME_POINTS:
        for col in (f"Actual_{t}y", f"Pred_{t}y"):
            if col not in out.columns:
                out[col] = np.nan
    return out


def _select_models(combo, algorithm_info, filtered):
    if "algorithms" in combo:
        return [m for m in combo["algorithms"] if m in filtered]
    if "category" in combo:
        return [m for m, info in algorithm_info.items()
                if info["category"] == combo["category"] and m in filtered]
    if combo.get("all"):
        return list(filtered.keys())
    return []


# ----------------------------------------------------------------------------
# Metrics for ensemble predictions
# ----------------------------------------------------------------------------
def evaluate_ensemble_predictions(ens_df):
    """Evaluate a long-format ensemble prediction table into a metrics table."""
    rows = []
    for (dataset, category, name), grp in ens_df.groupby(
            ["Dataset", "Algorithm_Category", "Algorithm_Name"]):
        for t in CFG.TIME_POINTS:
            a_col, p_col = f"Actual_{t}y", f"Pred_{t}y"
            if a_col not in grp.columns:
                continue
            d = grp.dropna(subset=[a_col, p_col])
            if d.empty:
                continue
            y = d[a_col].values.astype(int)
            p = d[p_col].values
            yhat = (p >= 0.5).astype(int)
            rows.append({
                "Dataset": dataset,
                "Algorithm_Category": category,
                "Algorithm_Name": name,
                "Time_Point_Years": t,
                "Accuracy": accuracy_score(y, yhat),
                "F1": f1_score(y, yhat, zero_division=0),
                "AUC": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
                "Brier": brier_score_loss(y, p),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Hyper-parameter optimization (grid search on train split)
# ----------------------------------------------------------------------------
def run_grid_search(all_data, algorithm_info, grid=None):
    """
    Grid-search federated-weighted hyper-parameters on the train split.

    Returns a DataFrame with one row per (combination, parameter set) and the
    average validation F1 / AUC / Accuracy across time points.
    """
    grid = grid or TUNING_GRID
    combos = list(product(grid["weight_exponent"], grid["weight_threshold"],
                          grid["normalize_weights"], grid["clip_weights"],
                          grid["base_metric"]))
    print(f"Grid-search: {len(combos)} parameter sets per combination.")

    train_data = {name: df[df["Dataset"] == "Train"].copy()
                  for name, df in all_data.items() if "Dataset" in df.columns}
    train_data = {k: v for k, v in train_data.items() if not v.empty}

    results = []
    for combo in ENSEMBLE_COMBINATIONS:
        selected = _select_models(combo, algorithm_info, train_data)
        if len(selected) < 2:
            continue
        common = _common_ids({m: train_data[m] for m in selected})
        # Pre-compute per-model base metrics once.
        base = {m: {} for m in selected}
        for m in selected:
            d = train_data[m][train_data[m]["ID"].isin(common)].sort_values("ID")
            for t in CFG.TIME_POINTS:
                a_col, p_col = f"Actual_{t}y", f"Pred_{t}y"
                y = d[a_col].values
                p = d[p_col].values
                ok = ~np.isnan(y) & ~np.isnan(p)
                base[m][t] = {
                    "AUC": _base_weight(y[ok], p[ok], "AUC"),
                    "F1": _base_weight(y[ok], p[ok], "F1"),
                    "Accuracy": _base_weight(y[ok], p[ok], "Accuracy"),
                    "y": y[ok], "p": p[ok],
                }

        for exp, thr, norm, clip, metric in combos:
            f1s, aucs, accs = [], [], []
            for t in CFG.TIME_POINTS:
                preds, weights = [], []
                gt = None
                for m in selected:
                    preds.append(base[m][t]["p"])
                    weights.append(base[m][t][metric])
                    gt = base[m][t]["y"]
                if gt is None or len(np.unique(gt)) < 2:
                    continue
                ens = federated_weighted(preds, weights,
                                         weight_exponent=exp,
                                         weight_threshold=thr,
                                         normalize_weights=norm,
                                         clip_weights=clip)
                yhat = (ens >= 0.5).astype(int)
                f1s.append(f1_score(gt, yhat, zero_division=0))
                aucs.append(roc_auc_score(gt, ens))
                accs.append(accuracy_score(gt, yhat))
            if not f1s:
                continue
            results.append({
                "Ensemble_Combination": combo["name"],
                "Number_of_Models": len(selected),
                "weight_exponent": exp,
                "weight_threshold": thr,
                "normalize_weights": norm,
                "clip_weights": str(clip),
                "base_metric": metric,
                "Validation_Avg_F1": np.mean(f1s),
                "Validation_Avg_AUC": np.mean(aucs),
                "Validation_Avg_Accuracy": np.mean(accs),
            })

    df = pd.DataFrame(results).sort_values(
        ["Ensemble_Combination", "Validation_Avg_F1"],
        ascending=[True, False]).reset_index(drop=True)
    return df


def select_best_parameters(tuning_df):
    """Pick the best hyper-parameter set per combination (by validation F1)."""
    best = {}
    for name, grp in tuning_df.groupby("Ensemble_Combination"):
        row = grp.iloc[0]
        clip_raw = row["clip_weights"]
        # Parse the stored tuple string back into a Python tuple (or None).
        clip_parsed = None
        if clip_raw not in ("None", "nan", ""):
            try:
                clip_parsed = eval(clip_raw, {"__builtins__": {}}, {})
            except Exception:
                clip_parsed = None
        best[name] = {
            "weight_exponent": float(row["weight_exponent"]),
            "weight_threshold": float(row["weight_threshold"]),
            "normalize_weights": bool(row["normalize_weights"]),
            "clip_weights": clip_parsed,
            "base_metric": row["base_metric"],
            "best_f1": float(row["Validation_Avg_F1"]),
            "best_auc": float(row["Validation_Avg_AUC"]),
        }
    return best


def generate_optimized_predictions(all_data, algorithm_info, best_params):
    """Build predictions for every combination using its tuned hyper-parameters."""
    out_rows = []
    for combo in ENSEMBLE_COMBINATIONS:
        if combo["name"] not in best_params:
            continue
        params = best_params[combo["name"]]
        for dataset in ["Train", "Test"]:
            filtered = {n: df[df["Dataset"] == dataset].copy()
                        for n, df in all_data.items() if "Dataset" in df.columns}
            selected = _select_models(combo, algorithm_info, filtered)
            if len(selected) < 2:
                continue
            common = _common_ids({m: filtered[m] for m in selected})
            base_df = pd.DataFrame()
            for t in CFG.TIME_POINTS:
                a_col, p_col = f"Actual_{t}y", f"Pred_{t}y"
                preds, weights, gt = [], [], None
                ref = None
                for m in selected:
                    d = filtered[m][filtered[m]["ID"].isin(common)] \
                        .sort_values("ID")
                    if ref is None:
                        ref = d[["ID", "Dataset"] + [a_col]].rename(
                            columns={a_col: "Actual"})
                    y = d[a_col].values
                    p = d[p_col].values
                    ok = ~np.isnan(y) & ~np.isnan(p)
                    preds.append(p[ok])
                    gt = y[ok]
                    weights.append(_base_weight(y[ok], p[ok], params["base_metric"]))
                if not preds or gt is None:
                    continue
                ens = federated_weighted(preds, weights,
                                         weight_exponent=params["weight_exponent"],
                                         weight_threshold=params["weight_threshold"],
                                         normalize_weights=params["normalize_weights"],
                                         clip_weights=params["clip_weights"])
                block = ref.copy()
                block = block[block["Actual"].notna()]
                block = block.iloc[:len(ens)]
                block["Actual"] = gt
                block = block.rename(columns={"Actual": a_col})
                block[p_col] = ens
                if base_df.empty:
                    base_df = block[["ID", "Dataset", a_col]].copy()
                else:
                    base_df[a_col] = block[a_col].values
                base_df[p_col] = block[p_col].values
            if not base_df.empty:
                base_df["Algorithm_Category"] = "ensemble"
                base_df["Algorithm_Name"] = f"{combo['name']}_FedWeighted_Optimized"
                out_rows.append(base_df)
    if not out_rows:
        return pd.DataFrame()
    return pd.concat(out_rows, ignore_index=True)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    setup_matplotlib()
    out_dir = ensure_dir(CFG.ENSEMBLE_OUTPUT_DIR)
    print("=" * 60)
    print("Ensemble modeling and hyper-parameter optimization")
    print("Weighted strategy: F1-average weights (positive correlation)")
    print("=" * 60)

    all_data, algorithm_info = load_all_data()
    if not all_data:
        raise RuntimeError("No model predictions loaded. Run scripts 05-07.")

    # 0. Compute per-model average F1 weights on the validation (train) split.
    #    These weights are positively correlated with model performance and are
    #    reused by both the baseline F1Weighted ensemble and the tuned optimizer.
    all_model_names = list(all_data.keys())
    f1_weights, avg_f1 = compute_f1_weights(all_data, all_model_names)
    f1_weight_df = pd.DataFrame({
        "Model": all_model_names,
        "Avg_F1_Validation": [avg_f1[m] for m in all_model_names],
        "Ensemble_Weight": [f1_weights[m] for m in all_model_names],
    }).sort_values("Avg_F1_Validation", ascending=False).reset_index(drop=True)
    f1_weight_df.round(6).to_csv(
        os.path.join(out_dir, "f1_weight_table.csv"), index=False)
    print("\nPer-model F1 weights (validation split):")
    print(f1_weight_df.to_string(index=False))

    # 1. Baseline ensembles (FedAvg + F1Weighted), per dataset and combined.
    ensemble_frames = []
    for dataset in ["Train", "Test", None]:
        ens = build_ensemble_table(all_data, algorithm_info,
                                   f1_weights=f1_weights, dataset=dataset)
        if not ens.empty:
            ensemble_frames.append(ens)
    ensemble_preds = pd.concat(ensemble_frames, ignore_index=True) \
        if ensemble_frames else pd.DataFrame()
    ensemble_preds.to_csv(os.path.join(out_dir, "ensemble_all_predictions.csv"),
                          index=False)

    metrics = evaluate_ensemble_predictions(ensemble_preds)
    metrics.round(6).to_csv(os.path.join(out_dir, "ensemble_metrics.csv"),
                            index=False)

    # 2. Grid-search hyper-parameters on the train split (base_metric=F1 by default).
    print("\nRunning ensemble weight grid search ...")
    tuning = run_grid_search(all_data, algorithm_info)
    tuning.round(8).to_csv(os.path.join(out_dir, "tuning_results.csv"), index=False)
    best_params = select_best_parameters(tuning)
    pd.DataFrame(best_params).T.to_csv(
        os.path.join(out_dir, "best_parameters.csv"), index_label="Ensemble")

    # 3. Generate optimized ensemble predictions.
    print("\nGenerating optimized ensemble predictions ...")
    opt = generate_optimized_predictions(all_data, algorithm_info, best_params)
    if not opt.empty:
        opt.to_csv(os.path.join(out_dir, "optimized_ensemble_predictions.csv"),
                   index=False)
        opt_metrics = evaluate_ensemble_predictions(opt)
        opt_metrics.round(6).to_csv(
            os.path.join(out_dir, "optimized_ensemble_metrics.csv"), index=False)

    print("=" * 60)
    print(f"Done. Ensemble outputs saved to:\n  {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
