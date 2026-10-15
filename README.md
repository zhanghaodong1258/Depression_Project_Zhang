# Depression Risk Prediction — Analysis Pipeline

Python codebase for the CHARLS-based depression risk prediction study.  

## Pipeline Overview

| Step | Script | Description |
|------|--------|-------------|
| 0 | `config.py` | Unified paths, column names, analysis parameters, plotting settings |
| 0 | `utils.py` | Shared helpers: metrics, bootstrap AUC, formatting, figure saving |
| 1 | `01_data_preprocessing.py` | Raw CHARLS DTA → missing filter → merge → change patterns → MICE → cohort → encoding → final dataset |
| 2 | `02_baseline_characteristics.py` | Baseline table: chi-square / Mann-Whitney U / t-test by Depression_Event |
| 3 | `03_logistic_regression.py` | Univariate + multivariable logistic regression (OR, 95% CI, p-value) |
| 4 | `04_survival_analysis.py` | Log-rank, Kaplan-Meier, univariate/multivariate Cox, PH test, forest plots, time-dependent ROC/calibration |
| 5 | `05_machine_learning.py` | 9 ML models (RF, GNB, SVM, Lasso, MLP, KNN, LightGBM, XGBoost, GBDT) |
| 6 | `06_deep_learning.py` | 9 PyTorch deep models (DeepMLP, WideDeep, DeepFM, NFM, DCN, xDeepFM, AutoInt, FTTransformer, TabNet) |
| 7 | `07_lightweight_models.py` | 6 lightweight PyTorch models (nanoTabPFN, LimiX2M, SERSAL, TabDistill, TabICLv2, Tables2Traces) |
| 8 | `08_single_model_evaluation.py` | Cross-model evaluation: ROC, metrics, calibration, best-model summary |
| 9 | `09_ensemble_modeling.py` | Ensemble strategies (average, weighted, optimized) + hyper-parameter grid search |
| 10 | `10_ensemble_evaluation.py` | Ensemble vs. single-model comparison plots and summary tables |

## Quick Start

```bash
# 1. Set your project root (or edit config.py directly)
export DEPRESSION_PROJECT_ROOT=/path/to/your/project

# 2. Place raw CHARLS .dta files in:
#    $DEPRESSION_PROJECT_ROOT/00.raw_data/

# 3. Run the pipeline in order
python 01_data_preprocessing.py
python 02_baseline_characteristics.py
python 03_logistic_regression.py
python 04_survival_analysis.py
python 05_machine_learning.py
python 06_deep_learning.py
python 07_lightweight_models.py
python 08_single_model_evaluation.py
python 09_ensemble_modeling.py
python 10_ensemble_evaluation.py
```

## Configuration

All paths and parameters live in `config.py`. Key settings:

- `PROJECT_ROOT` — root directory (override via `DEPRESSION_PROJECT_ROOT` env var)
- `RAW_DATA_DIR` — raw CHARLS `.dta` files
- `DATA_DIR` — intermediate / final datasets
- `FINAL_DATASET_PATH` — analysis-ready dataset produced by step 1
- `SIGNIFICANT_VARS_PATH` — significant-variable subset for ML/DL
- `OUTPUT_ROOT` — all results written under here, one subdirectory per step
- `ID_COL`, `TIME_COL`, `EVENT_COL` — standard column names (`ID`, `Followup_Time`, `Depression_Event`)
- `TIME_POINTS` — `[2, 4, 6, 8]` years
- `TEST_SIZE` — `0.3`, `RANDOM_STATE` — `42`
- `PLOT_DPI` — `300`, `FONT_FAMILY` — `Arial`

## Dependencies

```
pandas, numpy, scipy, scikit-learn, matplotlib, seaborn,
lifelines, lightgbm, xgboost, torch, openpyxl, statsmodels
```

## Output Structure

```
results/
├── 01_preprocessing/        # intermediate datasets, imputation reports
├── 02_baseline/              # baseline_table.csv/xlsx
├── 03_logistic/              # univariate/multivariable OR tables
├── 04_survival/              # Cox HR tables, KM plots, forest plots, plots/
├── 05_machine_learning/      # per-model subdirs + combined metrics
├── 06_deep_learning/         # per-model subdirs + combined metrics
├── 07_lightweight_models/    # per-model subdirs + combined metrics
├── 08_single_model_evaluation/  # cross-model comparison plots
├── 09_ensemble/              # ensemble predictions, tuning results
└── 10_ensemble_evaluation/   # ensemble comparison plots
```
