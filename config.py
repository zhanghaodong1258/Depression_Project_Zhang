"""
Configuration for Depression Risk Prediction Analysis Pipeline
=========================================================================
All scripts import their paths, column names, and global constants from here.
Modify the paths below to match your local environment.
"""

import os

# ============================================================================
# Path Configuration
# ============================================================================
# Root directory for all input data and output results.
# Override by setting the environment variable DEPRESSION_PROJECT_ROOT,
# or edit the default below.
PROJECT_ROOT = os.environ.get(
    "DEPRESSION_PROJECT_ZHANG",
    os.path.join(os.path.expanduser("~"), "depression_project")
)

# Raw CHARLS data directory (Stata .dta files)
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "00.raw_data")

# Intermediate / cleaned data directory
DATA_DIR = os.path.join(PROJECT_ROOT, "01.data")

# Final analysis-ready dataset (produced by preprocessing scripts)
FINAL_DATASET_PATH = os.path.join(DATA_DIR, "final_analysis_dataset.xlsx")

# Significant variables subset (used by ML / DL scripts)
SIGNIFICANT_VARS_PATH = os.path.join(DATA_DIR, "06_significant_variables_with_id.xlsx")

# Root output directory for all analysis results
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "results")

# Sub-directories for each analysis stage
PREPROCESSING_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "01_preprocessing")
BASELINE_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "02_baseline")
LOGISTIC_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "03_logistic")
SURVIVAL_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "04_survival")
ML_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "05_machine_learning")
DL_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "06_deep_learning")
LIGHTWEIGHT_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "07_lightweight_models")
SINGLE_EVAL_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "08_single_model_evaluation")
ENSEMBLE_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "09_ensemble")

# ============================================================================
# Column Name Constants
# ============================================================================
ID_COL = "ID"
TIME_COL = "Followup_Time"
EVENT_COL = "Depression_Event"

# ============================================================================
# Analysis Parameters
# ============================================================================
# Follow-up time points (years) for time-dependent prediction
TIME_POINTS = [2, 4, 6, 8]

# Train / test split
TEST_SIZE = 0.3
RANDOM_STATE = 42

# Maximum unique values for a column to be treated as categorical
CATEGORICAL_MAX_UNIQUE = 20

# Missing value threshold: drop columns with missing rate above this
MISSING_THRESHOLD = 0.2

# Bootstrap iterations for AUC confidence intervals
N_BOOTSTRAPS = 1000

# ============================================================================
# Plotting Global Settings
# ============================================================================
PLOT_DPI = 300
FONT_FAMILY = "Arial"

# Standard color palette
COLOR_LOW_RISK = "#1f77b4"
COLOR_HIGH_RISK = "#d62728"
COLOR_TRAIN = "#1f77b4"
COLOR_TEST = "#ff7f0e"