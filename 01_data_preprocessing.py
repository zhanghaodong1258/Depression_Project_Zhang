"""
01_data_preprocessing.py
=========================
Preprocessing pipeline for the CHARLS depression cohort.

Stages (in order):
  1. Read raw CHARLS Stata .dta files; drop columns with >MISSING_THRESHOLD missing.
  2. Merge cleaned datasets (left join on ID, keeping H_CHARLS_D_Data as master).
  3. Extract the harmonised 4-wave variable set and compute derived variables.
  4. Build the analysis cohort (age >= 45, available CESD-10, exclude baseline depression).
  5. Construct depression event / follow-up time columns.
  6. Classify longitudinal change patterns (Increasing / Decreasing / Fluctuating /
     Stable_Low / Stable_High) with the linear-regression slope method.
  7. MICE imputation (IterativeImputer + BayesianRidge) on numeric columns.
  8. Encode object / Pattern_* categorical columns to integer codes.
  9. Save the final analysis-ready dataset to config.FINAL_DATASET_PATH.

All paths and constants are imported from config.py; plotting / formatting helpers
from utils.py are used where applicable.
"""

import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

import config as C
from utils import ensure_dir

warnings.filterwarnings("ignore")

# ============================================================================
# Raw DTA files (from config.RAW_DATA_DIR)
# ============================================================================
RAW_DTA_FILES = [
    "H_CHARLS_D_Data.dta",
    "H_CHARLS_EOL_a.dta",
    "H_CHARLS_LH_a.dta",
]

# Harmonised 4-wave file (used for variable extraction; expected in RAW_DATA_DIR)
HARMONISED_DTA = "harmonized_charls_version_d.dta"

# Depressive-symptom cutoff on CESD-10
DEPRESSION_CUTOFF = 10

# Minimum baseline age
MIN_BASELINE_AGE = 45

# Wave years (for follow-up time calculation, years since baseline 2011)
WAVE_OFFSET_YEARS = {"Wave1": 0, "Wave2": 2, "Wave3": 4, "Wave4": 7}

# ============================================================================
# Variable extraction map (from PoC block 7)
# ============================================================================
BASE_VARIABLES = {
    "ID": "Respondent_Unique_ID",
    "ragender": "Respondent_Gender",
    "rabplace_c": "Respondent_Birth_Place",
    "raeducl": "Respondent_Education_3Categories",
    "raeduc_c": "Respondent_Education_11Categories",
    "ramomeducl": "Mother_Education_3Categories",
    "radadeducl": "Father_Education_3Categories",
    "ramomoccup_c": "Mother_Occupation_Type",
    "radadoccup_c": "Father_Occupation_Type",
    "ramischlth": "Childhood_Family_Economic_Status",
    "ramomgrela": "Relationship_With_Mother_Score",
    "radadgrela": "Relationship_With_Father_Score",
    "ramwarmm_c": "Mother_Warmth_Score",
}

WAVE1_VARIABLES = {
    "r1agey": "Wave1_Respondent_Age_Years",
    "r1mstat": "Wave1_Marital_Status_With_Partner",
    "r1mstath": "Wave1_Marital_Status_Reported",
    "r1hukou": "Wave1_Hukou_Type",
    "h1rural": "Wave1_Residence_Rural",
    "h1adebt": "Wave1_Household_Total_Debts",
    "hh1itot": "Wave1_Household_Total_Income",
    "r1itearn": "Wave1_Individual_Earned_Income",
    "r1ipena": "Wave1_Individual_Pension_Income",
    "r1igxfr": "Wave1_Individual_Transfer_Income",
    "r1iothr": "Wave1_Individual_Other_Income",
    "r1work": "Wave1_Currently_Working",
    "r1smokev": "Wave1_Ever_Smoked",
    "r1smoken": "Wave1_Current_Smoker",
    "r1smokef": "Wave1_Cigarettes_Per_Day",
    "r1drinkev": "Wave1_Ever_Drank_Alcohol",
    "r1drinkl": "Wave1_Drank_Last_Year",
    "r1drinkn_c": "Wave1_Drinking_Frequency",
    "r1drinkr_c": "Wave1_Daily_Drinking_Level",
    "r1vgact_c": "Wave1_Has_Vigorous_Exercise",
    "r1mdact_c": "Wave1_Has_Moderate_Exercise",
    "r1ltact_c": "Wave1_Has_Walking_Activity",
    "r1hibpe": "Wave1_Hypertension",
    "r1diabe": "Wave1_Diabetes",
    "r1hearte": "Wave1_Heart_Disease",
    "r1stroke": "Wave1_Stroke",
    "r1lunge": "Wave1_Chronic_Lung_Disease",
    "r1arthre": "Wave1_Arthritis",
    "r1dyslipe": "Wave1_Dyslipidemia",
    "r1livere": "Wave1_Liver_Disease",
    "r1kidneye": "Wave1_Kidney_Disease",
    "r1digeste": "Wave1_Digestive_Disease",
    "r1asthmae": "Wave1_Asthma",
    "r1memrye": "Wave1_Memory_Related_Disease",
    "r1psyche": "Wave1_Psychiatric_Emotional_Problems",
    "r1adlam_c": "Wave1_ADL_Limitation_Count_6Items",
    "r1iadlam": "Wave1_IADL_Limitation_Count_5Items",
    "r1mobilsevm": "Wave1_Mobility_Limitation_Count_7Items",
    "r1lowermobm": "Wave1_Lower_Limb_Limitation_Count",
    "r1uppermobm": "Wave1_Upper_Limb_Limitation_Count",
    "r1slfmem": "Wave1_Self_Rated_Memory_Score",
    "r1imrc": "Wave1_Immediate_Word_Recall_Score",
    "r1dlrc": "Wave1_Delayed_Word_Recall_Score",
    "r1ser7": "Wave1_Serial_Subtraction_Correct_Count",
    "r1orient": "Wave1_Orientation_Score",
    "r1satlife": "Wave1_Life_Satisfaction_Score",
    "r1lifethe": "Wave1_Experienced_Serious_Accident_Injury",
    "r1chdeathe": "Wave1_Experienced_Child_Death",
    "r1sleeprl": "Wave1_Sleep_Disturbance_Frequency",
    "r1cesd10m": "Wave1_CESD10_Depression_Score",
    "r1fearll": "Wave1_Fear_Items_For_Anxiety",
    "h1coresd": "Wave1_Lives_With_Children",
    "h1kcnt": "Wave1_Weekly_Contact_With_Children",
    "h1kcntf": "Wave1_Weekly_Face_To_Face_With_Children",
    "h1kcntpm": "Wave1_Weekly_Phone_Email_With_Children",
    "r1livsib": "Wave1_Number_Of_Living_Siblings",
    "r1hosp1y": "Wave1_Hospitalized_Last_Year",
    "r1doctor1m": "Wave1_Visited_Doctor_Last_Month",
    "r1mheight": "Wave1_Measured_Height_Meters",
    "r1mweight": "Wave1_Measured_Weight_Kg",
    "r1mbmi": "Wave1_Measured_BMI_KgM2",
    "r1mwaist": "Wave1_Measured_Waist_Circumference_Cm",
    "r1systo": "Wave1_Average_Systolic_Blood_Pressure",
    "r1diasto": "Wave1_Average_Diastolic_Blood_Pressure",
    "r1gripsum": "Wave1_Dominant_Hand_Grip_Strength_Kg",
    "r1wspeed": "Wave1_Walking_Speed_Seconds_Per_2_5m",
    "r1puff": "Wave1_Peak_Flow_Liters_Per_Min",
    "r1chr5sec": "Wave1_Five_Times_Chair_Stand_Seconds",
}

WAVE2_VARIABLES = {
    "r2agey": "Wave2_Respondent_Age_Years",
    "r2mstat": "Wave2_Marital_Status_With_Partner",
    "r2mstath": "Wave2_Marital_Status_Reported",
    "r2hukou": "Wave2_Hukou_Type",
    "h2rural": "Wave2_Residence_Rural",
    "h2adebt": "Wave2_Household_Total_Debts",
    "hh2itot": "Wave2_Household_Total_Income",
    "r2itearn": "Wave2_Individual_Earned_Income",
    "r2ipena": "Wave2_Individual_Pension_Income",
    "r2igxfr": "Wave2_Individual_Transfer_Income",
    "r2iothr": "Wave2_Individual_Other_Income",
    "r2work": "Wave2_Currently_Working",
    "r2smokev": "Wave2_Ever_Smoked",
    "r2smoken": "Wave2_Current_Smoker",
    "r2smokef": "Wave2_Cigarettes_Per_Day",
    "r2drinkev": "Wave2_Ever_Drank_Alcohol",
    "r2drinkl": "Wave2_Drank_Last_Year",
    "r2drinkn_c": "Wave2_Drinking_Frequency",
    "r2drinkr_c": "Wave2_Daily_Drinking_Level",
    "r2vgact_c": "Wave2_Has_Vigorous_Exercise",
    "r2mdact_c": "Wave2_Has_Moderate_Exercise",
    "r2ltact_c": "Wave2_Has_Walking_Activity",
    "r2hibpe": "Wave2_Hypertension",
    "r2diabe": "Wave2_Diabetes",
    "r2hearte": "Wave2_Heart_Disease",
    "r2stroke": "Wave2_Stroke",
    "r2lunge": "Wave2_Chronic_Lung_Disease",
    "r2arthre": "Wave2_Arthritis",
    "r2dyslipe": "Wave2_Dyslipidemia",
    "r2livere": "Wave2_Liver_Disease",
    "r2kidneye": "Wave2_Kidney_Disease",
    "r2digeste": "Wave2_Digestive_Disease",
    "r2asthmae": "Wave2_Asthma",
    "r2memrye": "Wave2_Memory_Related_Disease",
    "r2psyche": "Wave2_Psychiatric_Emotional_Problems",
    "r2adlam_c": "Wave2_ADL_Limitation_Count_6Items",
    "r2iadlam": "Wave2_IADL_Limitation_Count_5Items",
    "r2mobilsevm": "Wave2_Mobility_Limitation_Count_7Items",
    "r2lowermobm": "Wave2_Lower_Limb_Limitation_Count",
    "r2uppermobm": "Wave2_Upper_Limb_Limitation_Count",
    "r2imrc": "Wave2_Immediate_Word_Recall_Score",
    "r2dlrc": "Wave2_Delayed_Word_Recall_Score",
    "r2ser7": "Wave2_Serial_Subtraction_Correct_Count",
    "r2orient": "Wave2_Orientation_Score",
    "r2satlife": "Wave2_Life_Satisfaction_Score",
    "r2lifethe": "Wave2_Experienced_Serious_Accident_Injury",
    "r2chdeathe": "Wave2_Experienced_Child_Death",
    "r2sleeprl": "Wave2_Sleep_Disturbance_Frequency",
    "r2cesd10m": "Wave2_CESD10_Depression_Score",
    "r2fearll": "Wave2_Fear_Items_For_Anxiety",
    "h2coresd": "Wave2_Lives_With_Children",
    "h2kcntf": "Wave2_Weekly_Face_To_Face_With_Children",
    "r2livsib": "Wave2_Number_Of_Living_Siblings",
    "r2hosp1y": "Wave2_Hospitalized_Last_Year",
    "r2doctor1m": "Wave2_Visited_Doctor_Last_Month",
    "r2mheight": "Wave2_Measured_Height_Meters",
    "r2mweight": "Wave2_Measured_Weight_Kg",
    "r2mbmi": "Wave2_Measured_BMI_KgM2",
    "r2mwaist": "Wave2_Measured_Waist_Circumference_Cm",
    "r2systo": "Wave2_Average_Systolic_Blood_Pressure",
    "r2diasto": "Wave2_Average_Diastolic_Blood_Pressure",
    "r2gripsum": "Wave2_Dominant_Hand_Grip_Strength_Kg",
    "r2wspeed": "Wave2_Walking_Speed_Seconds_Per_2_5m",
    "r2puff": "Wave2_Peak_Flow_Liters_Per_Min",
    "r2chr5sec": "Wave2_Five_Times_Chair_Stand_Seconds",
}

WAVE3_VARIABLES = {
    "r3agey": "Wave3_Respondent_Age_Years",
    "r3mstat": "Wave3_Marital_Status_With_Partner",
    "r3mstath": "Wave3_Marital_Status_Reported",
    "r3hukou": "Wave3_Hukou_Type",
    "h3rural": "Wave3_Residence_Rural",
    "h3atotf": "Wave3_Household_Total_Assets",
    "h3adebt": "Wave3_Household_Total_Debts",
    "hh3itot": "Wave3_Household_Total_Income",
    "r3itearn": "Wave3_Individual_Earned_Income",
    "r3ipena": "Wave3_Individual_Pension_Income",
    "r3igxfr": "Wave3_Individual_Transfer_Income",
    "r3iothr": "Wave3_Individual_Other_Income",
    "r3work": "Wave3_Currently_Working",
    "r3smokev": "Wave3_Ever_Smoked",
    "r3smoken": "Wave3_Current_Smoker",
    "r3smokef": "Wave3_Cigarettes_Per_Day",
    "r3drinkev": "Wave3_Ever_Drank_Alcohol",
    "r3drinkl": "Wave3_Drank_Last_Year",
    "r3drinkn_c": "Wave3_Drinking_Frequency",
    "r3drinkr_c": "Wave3_Daily_Drinking_Level",
    "r3vgact_c": "Wave3_Has_Vigorous_Exercise",
    "r3mdact_c": "Wave3_Has_Moderate_Exercise",
    "r3ltact_c": "Wave3_Has_Walking_Activity",
    "r3hibpe": "Wave3_Hypertension",
    "r3diabe": "Wave3_Diabetes",
    "r3hearte": "Wave3_Heart_Disease",
    "r3stroke": "Wave3_Stroke",
    "r3lunge": "Wave3_Chronic_Lung_Disease",
    "r3arthre": "Wave3_Arthritis",
    "r3dyslipe": "Wave3_Dyslipidemia",
    "r3livere": "Wave3_Liver_Disease",
    "r3kidneye": "Wave3_Kidney_Disease",
    "r3digeste": "Wave3_Digestive_Disease",
    "r3asthmae": "Wave3_Asthma",
    "r3memrye": "Wave3_Memory_Related_Disease",
    "r3psyche": "Wave3_Psychiatric_Emotional_Problems",
    "r3adlam_c": "Wave3_ADL_Limitation_Count_6Items",
    "r3iadlam": "Wave3_IADL_Limitation_Count_5Items",
    "r3mobilsevm": "Wave3_Mobility_Limitation_Count_7Items",
    "r3lowermobm": "Wave3_Lower_Limb_Limitation_Count",
    "r3uppermobm": "Wave3_Upper_Limb_Limitation_Count",
    "r3imrc": "Wave3_Immediate_Word_Recall_Score",
    "r3dlrc": "Wave3_Delayed_Word_Recall_Score",
    "r3ser7": "Wave3_Serial_Subtraction_Correct_Count",
    "r3orient": "Wave3_Orientation_Score",
    "r3satlife": "Wave3_Life_Satisfaction_Score",
    "r3lifethe": "Wave3_Experienced_Serious_Accident_Injury",
    "r3chdeathe": "Wave3_Experienced_Child_Death",
    "r3sleeprl": "Wave3_Sleep_Disturbance_Frequency",
    "r3cesd10m": "Wave3_CESD10_Depression_Score",
    "r3fearll": "Wave3_Fear_Items_For_Anxiety",
    "h3coresd": "Wave3_Lives_With_Children",
    "h3kcntf": "Wave3_Weekly_Face_To_Face_With_Children",
    "r3livsib": "Wave3_Number_Of_Living_Siblings",
    "r3hosp1y": "Wave3_Hospitalized_Last_Year",
    "r3doctor1m": "Wave3_Visited_Doctor_Last_Month",
    "r3mheight": "Wave3_Measured_Height_Meters",
    "r3mweight": "Wave3_Measured_Weight_Kg",
    "r3mbmi": "Wave3_Measured_BMI_KgM2",
    "r3mwaist": "Wave3_Measured_Waist_Circumference_Cm",
    "r3systo": "Wave3_Average_Systolic_Blood_Pressure",
    "r3diasto": "Wave3_Average_Diastolic_Blood_Pressure",
    "r3gripsum": "Wave3_Dominant_Hand_Grip_Strength_Kg",
    "r3wspeed": "Wave3_Walking_Speed_Seconds_Per_2_5m",
    "r3puff": "Wave3_Peak_Flow_Liters_Per_Min",
    "r3chr5sec": "Wave3_Five_Times_Chair_Stand_Seconds",
}

WAVE4_VARIABLES = {
    "r4agey": "Wave4_Respondent_Age_Years",
    "r4mstat": "Wave4_Marital_Status_With_Partner",
    "r4mstath": "Wave4_Marital_Status_Reported",
    "r4hukou": "Wave4_Hukou_Type",
    "h4rural": "Wave4_Residence_Rural",
    "h4atotf": "Wave4_Household_Total_Assets",
    "h4adebt": "Wave4_Household_Total_Debts",
    "hh4itot": "Wave4_Household_Total_Income",
    "r4itearn": "Wave4_Individual_Earned_Income",
    "r4ipena": "Wave4_Individual_Pension_Income",
    "r4igxfr": "Wave4_Individual_Transfer_Income",
    "r4iothr": "Wave4_Individual_Other_Income",
    "r4work": "Wave4_Currently_Working",
    "r4smokev": "Wave4_Ever_Smoked",
    "r4smoken": "Wave4_Current_Smoker",
    "r4smokef": "Wave4_Cigarettes_Per_Day",
    "r4drinkev": "Wave4_Ever_Drank_Alcohol",
    "r4drinkl": "Wave4_Drank_Last_Year",
    "r4drinkn_c": "Wave4_Drinking_Frequency",
    "r4drinkr_c": "Wave4_Daily_Drinking_Level",
    "r4vgact_c": "Wave4_Has_Vigorous_Exercise",
    "r4mdact_c": "Wave4_Has_Moderate_Exercise",
    "r4ltact_c": "Wave4_Has_Walking_Activity",
    "r4hibpe": "Wave4_Hypertension",
    "r4diabe": "Wave4_Diabetes",
    "r4hearte": "Wave4_Heart_Disease",
    "r4stroke": "Wave4_Stroke",
    "r4lunge": "Wave4_Chronic_Lung_Disease",
    "r4arthre": "Wave4_Arthritis",
    "r4dyslipe": "Wave4_Dyslipidemia",
    "r4livere": "Wave4_Liver_Disease",
    "r4kidneye": "Wave4_Kidney_Disease",
    "r4digeste": "Wave4_Digestive_Disease",
    "r4asthmae": "Wave4_Asthma",
    "r4memrye": "Wave4_Memory_Related_Disease",
    "r4psyche": "Wave4_Psychiatric_Emotional_Problems",
    "r4adlam_c": "Wave4_ADL_Limitation_Count_6Items",
    "r4iadlam": "Wave4_IADL_Limitation_Count_5Items",
    "r4mobilsevm": "Wave4_Mobility_Limitation_Count_7Items",
    "r4lowermobm": "Wave4_Lower_Limb_Limitation_Count",
    "r4uppermobm": "Wave4_Upper_Limb_Limitation_Count",
    "r4imrc": "Wave4_Immediate_Word_Recall_Score",
    "r4dlrc": "Wave4_Delayed_Word_Recall_Score",
    "r4ser7": "Wave4_Serial_Subtraction_Correct_Count",
    "r4orient": "Wave4_Orientation_Score",
    "r4satlife": "Wave4_Life_Satisfaction_Score",
    "r4lifethe": "Wave4_Experienced_Serious_Accident_Injury",
    "r4chdeathe": "Wave4_Experienced_Child_Death",
    "r4sleeprl": "Wave4_Sleep_Disturbance_Frequency",
    "r4cesd10m": "Wave4_CESD10_Depression_Score",
    "r4fearll": "Wave4_Fear_Items_For_Anxiety",
    "h4coresd": "Wave4_Lives_With_Children",
    "h4kcntf": "Wave4_Weekly_Face_To_Face_With_Children",
    "r4livsib": "Wave4_Number_Of_Living_Siblings",
    "r4hosp1y": "Wave4_Hospitalized_Last_Year",
    "r4doctor1m": "Wave4_Visited_Doctor_Last_Month",
}

ALL_VARIABLES = {}
ALL_VARIABLES.update(BASE_VARIABLES)
ALL_VARIABLES.update(WAVE1_VARIABLES)
ALL_VARIABLES.update(WAVE2_VARIABLES)
ALL_VARIABLES.update(WAVE3_VARIABLES)
ALL_VARIABLES.update(WAVE4_VARIABLES)

# Longitudinal variable groups used for change-pattern classification
# (columns must exist after derived-variable computation)
CHANGE_PATTERN_GROUPS = {
    "Total_Transfer_Income": [
        "Baseline_Total_Transfer_Income",
        "Total_Transfer_Income_Wave2",
        "Total_Transfer_Income_Wave3",
        "Total_Transfer_Income_Wave4",
    ],
    "Net_Transfer": [
        "Baseline_Net_Transfer",
        "Net_Transfer_Wave2",
        "Net_Transfer_Wave3",
        "Net_Transfer_Wave4",
    ],
    "Total_Word_Recall_Score": [
        "Baseline_Total_Word_Recall_Score",
        "Total_Word_Recall_Score_Wave2",
        "Total_Word_Recall_Score_Wave3",
        "Total_Word_Recall_Score_Wave4",
    ],
    "Frailty_5Items": [
        "Baseline_Frailty_5Items",
        "Frailty_5Items_Wave2",
        "Frailty_5Items_Wave3",
    ],
    "Deficit_Accumulation_Index": [
        "Baseline_Deficit_Accumulation_Index_Wave2",
        "Deficit_Accumulation_Index_Wave3",
    ],
}

CHANGE_TIME_POINTS = {
    "Total_Transfer_Income": [0, 2, 4, 7],
    "Net_Transfer": [0, 2, 4, 7],
    "Total_Word_Recall_Score": [0, 2, 4, 7],
    "Frailty_5Items": [0, 2, 4],
    "Deficit_Accumulation_Index": [2, 4],
}


# ============================================================================
# Stage 1: Read raw DTA, drop high-missing columns
# ============================================================================
def clean_raw_dta_files():
    """Read each raw .dta, drop columns with missing > MISSING_THRESHOLD,
    save cleaned CSV/XLSX to DATA_DIR. Returns dict of {short_name: DataFrame}."""
    ensure_dir(C.DATA_DIR)
    cleaned = {}
    for fname in RAW_DTA_FILES:
        path = os.path.join(C.RAW_DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"[WARN] raw file not found, skipping: {path}")
            continue
        df = pd.read_stata(path)
        n0 = df.shape[1]
        miss_rate = df.isnull().mean()
        df = df.loc[:, miss_rate <= C.MISSING_THRESHOLD]
        print(f"[clean] {fname}: {n0} -> {df.shape[1]} columns")
        stem = os.path.splitext(fname)[0]
        df.to_csv(os.path.join(C.DATA_DIR, f"cleaned_{stem}.csv"), index=False)
        df.to_excel(
            os.path.join(C.DATA_DIR, f"cleaned_{stem}.xlsx"),
            index=False, engine="openpyxl",
        )
        cleaned[stem] = df
    return cleaned


# ============================================================================
# Stage 2: Merge cleaned datasets (left join on ID, D_Data as master)
# ============================================================================
def merge_cleaned_datasets(cleaned: dict) -> pd.DataFrame:
    """Left-join EOL and LH onto D_Data on config.ID_COL."""
    master_name = "H_CHARLS_D_Data"
    if master_name not in cleaned:
        raise FileNotFoundError(f"Master dataset {master_name} missing")
    df = cleaned[master_name].copy()
    for other in ["H_CHARLS_LH_a", "H_CHARLS_EOL_a"]:
        if other not in cleaned:
            continue
        other_df = cleaned[other].drop_duplicates(subset=[C.ID_COL], keep="first")
        suffix = f"_{other}"
        df = df.merge(other_df, on=C.ID_COL, how="left", suffixes=("", suffix))
    print(f"[merge] merged shape: {df.shape}")
    return df


# ============================================================================
# Stage 3: Extract 4-wave variables & compute derived variables
# ============================================================================
def _convert_category_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Stata category dtypes to numeric; recode CHARLS missing codes."""
    for col in df.select_dtypes(include=["category"]).columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            df[col] = df[col].cat.codes
            df[col] = df[col].replace(-1, np.nan)
    df.replace([-1, -2, -3, -8, -9], np.nan, inplace=True)
    return df


def extract_harmonised_variables() -> pd.DataFrame:
    """Read the harmonised CHARLS DTA, keep only ALL_VARIABLES columns, rename."""
    path = os.path.join(C.RAW_DATA_DIR, HARMONISED_DTA)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Harmonised DTA not found: {path}")
    df = pd.read_stata(path)
    desired = list(ALL_VARIABLES.keys())
    valid = [c for c in desired if c in df.columns]
    missing = [c for c in desired if c not in df.columns]
    if missing:
        print(f"[extract] {len(missing)} expected variables not in DTA: {missing[:5]}...")
    df = df[valid].copy()
    df = _convert_category_to_numeric(df)
    df = df.rename(columns=ALL_VARIABLES)
    print(f"[extract] extracted shape: {df.shape}")
    return df


def _safe_sum(df: pd.DataFrame, cols) -> pd.Series:
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series(np.nan, index=df.index)
    return df[existing].fillna(0).sum(axis=1)


def compute_derived_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived baseline / wave variables used downstream."""
    waves = ["Wave1", "Wave2", "Wave3", "Wave4"]

    # Household wealth = assets - debts
    for w in waves:
        assets = f"{w}_Household_Total_Assets"
        debts = f"{w}_Household_Total_Debts"
        if assets in df.columns and debts in df.columns:
            df[f"Calculated_Household_Total_Wealth_{w}"] = df[assets] - df[debts]

    # Total individual income
    for w in waves:
        parts = [f"{w}_Individual_Earned_Income", f"{w}_Individual_Pension_Income",
                 f"{w}_Individual_Transfer_Income", f"{w}_Individual_Other_Income"]
        if all(c in df.columns for c in parts):
            df[f"Calculated_Individual_Total_Income_{w}"] = (
                df[parts].fillna(0).sum(axis=1)
            )

    # Total transfer received from children + parents
    for w in waves:
        fc = f"{w}_Transfer_From_Children_Amount" if f"{w}_Transfer_From_Children_Amount" in df.columns else f"{w}_Transfer_From_Children"
        fp = f"{w}_Transfer_From_Parents_Amount" if f"{w}_Transfer_From_Parents_Amount" in df.columns else f"{w}_Transfer_From_Parents"
        if fc in df.columns and fp in df.columns:
            df[f"Calculated_Total_Transfer_Income_{w}"] = df[fc].fillna(0) + df[fp].fillna(0)

    # Net transfer = in - out
    for w in waves:
        tin = f"Calculated_Total_Transfer_Income_{w}"
        tc = f"{w}_Transfer_To_Children_Amount" if f"{w}_Transfer_To_Children_Amount" in df.columns else f"{w}_Transfer_To_Children"
        tp = f"{w}_Transfer_To_Parents_Amount" if f"{w}_Transfer_To_Parents_Amount" in df.columns else f"{w}_Transfer_To_Parents"
        to = f"{w}_Transfer_To_Other_Relatives"
        if tin in df.columns:
            df[f"Calculated_Net_Transfer_{w}"] = (
                df[tin]
                - df.get(tc, pd.Series(0, index=df.index)).fillna(0)
                - df.get(tp, pd.Series(0, index=df.index)).fillna(0)
                - df.get(to, pd.Series(0, index=df.index)).fillna(0)
            )

    # Chronic disease count
    chronic = ["Hypertension", "Diabetes", "Heart_Disease", "Stroke",
               "Chronic_Lung_Disease", "Arthritis", "Dyslipidemia",
               "Liver_Disease", "Kidney_Disease", "Digestive_Disease",
               "Asthma", "Memory_Related_Disease", "Psychiatric_Emotional_Problems"]
    for w in waves:
        cols = [f"{w}_{d}" for d in chronic]
        df[f"Calculated_Chronic_Disease_Count_{w}"] = _safe_sum(df, cols)

    # Total word recall = immediate + delayed
    for w in waves:
        im = f"{w}_Immediate_Word_Recall_Score"
        dl = f"{w}_Delayed_Word_Recall_Score"
        if im in df.columns and dl in df.columns:
            df[f"Calculated_Total_Word_Recall_Score_{w}"] = df[im].fillna(0) + df[dl].fillna(0)

    # IADL unimpaired count = 5 - limited
    for w in waves:
        col = f"{w}_IADL_Limitation_Count_5Items"
        if col in df.columns:
            df[f"Calculated_IADL_Unlimited_Count_{w}"] = 5 - df[col].fillna(0)

    # Life events (accident + child death)
    for w in waves:
        acc = f"{w}_Experienced_Serious_Accident_Injury"
        cd = f"{w}_Experienced_Child_Death"
        if acc in df.columns and cd in df.columns:
            df[f"Calculated_Life_Event_Count_{w}"] = df[acc].fillna(0) + df[cd].fillna(0)

    # Frailty 5-item index
    frailty_components = ["Measured_Weight_Kg", "Walking_Speed_Seconds_Per_2_5m",
                          "Dominant_Hand_Grip_Strength_Kg", "Has_Vigorous_Exercise",
                          "CESD10_Depression_Score"]
    for w in waves:
        cols = [f"{w}_{c}" for c in frailty_components]
        df[f"Calculated_Frailty_5Items_{w}"] = _safe_sum(df, cols)

    # Deficit accumulation index
    for w in waves:
        cc = f"Calculated_Chronic_Disease_Count_{w}"
        adl = f"{w}_ADL_Limitation_Count_6Items"
        iadl = f"{w}_IADL_Limitation_Count_5Items"
        dep = f"{w}_CESD10_Depression_Score"
        if all(c in df.columns for c in [cc, adl, iadl, dep]):
            df[f"Calculated_Deficit_Accumulation_Index_{w}"] = (
                df[cc] + df[adl] + df[iadl] + df[dep]
            ) / 4

    # Rename Wave1_* derived / baseline columns as "Baseline_*" for downstream
    baseline_renames = {}
    for col in df.columns:
        if col.startswith("Wave1_"):
            baseline_renames[col] = col.replace("Wave1_", "Baseline_")
    df = df.rename(columns=baseline_renames)

    print(f"[derived] shape after derived variables: {df.shape}")
    return df


# ============================================================================
# Stage 4-5: Cohort construction + depression event variables
# ============================================================================
def build_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """Apply age filter, drop rows without CESD, exclude baseline depression,
    and construct Depression_Event / Followup_Time columns."""
    # Need at least one non-null CESD-10
    cesd_cols = [f"Wave{w}_CESD10_Depression_Score" for w in (1, 2, 3, 4)
                 if f"Wave{w}_CESD10_Depression_Score" in df.columns]
    df = df.dropna(subset=cesd_cols, how="all").copy()

    # Age filter
    age_col = "Wave1_Respondent_Age_Years"
    if age_col in df.columns:
        df = df[df[age_col] >= MIN_BASELINE_AGE].copy()

    # Exclude baseline depression (Wave1 CESD-10 >= cutoff)
    w1_cesd = "Wave1_CESD10_Depression_Score"
    if w1_cesd in df.columns:
        df = df[df[w1_cesd] < DEPRESSION_CUTOFF].copy()

    # Per-wave depression status
    for w in (2, 3, 4):
        col = f"Wave{w}_CESD10_Depression_Score"
        out = f"Depression_Wave{w}"
        if col in df.columns:
            df[out] = (df[col] >= DEPRESSION_CUTOFF).astype(float)
            df.loc[df[col].isnull(), out] = np.nan

    # Time to first onset (years since 2011 baseline)
    def _first_time(row):
        for w, yr in ((2, 2), (3, 4), (4, 7)):
            c = f"Depression_Wave{w}"
            if c in row.index and row[c] == 1:
                return yr
        return np.nan

    df["Time_To_First_Depression"] = df.apply(_first_time, axis=1)
    df[C.EVENT_COL] = df["Time_To_First_Depression"].notnull().astype(int)
    df[C.TIME_COL] = df["Time_To_First_Depression"].fillna(7)

    # Baseline depression flag (always 0 after exclusion; kept for downstream compat)
    df["Depression_Baseline"] = 0

    print(f"[cohort] analytic sample: {df.shape[0]} rows, {df.shape[1]} cols; "
          f"events={int(df[C.EVENT_COL].sum())}")
    return df


# ============================================================================
# Stage 6: Change-pattern classification
# ============================================================================
def classify_change_pattern(values, times, baseline_value=None, ref_median=None):
    """Classify a longitudinal trajectory into one of:
    Increasing / Decreasing / Fluctuating / Stable_Low / Stable_High.

    Uses OLS slope against time; slope threshold = 0.1 * std / time_span.
    If slope is within +/- threshold, inspect adjacent changes for fluctuation,
    otherwise classify Stable_Low / Stable_High by comparison to ref_median.
    Returns np.nan if fewer than 2 valid observations.
    """
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    valid = ~pd.isnull(values)
    vv = values[valid]
    tt = times[valid]
    if len(vv) < 2:
        return np.nan

    slope, _, _, _, _ = linregress(tt, vv)
    std_val = np.std(vv)
    if std_val == 0:
        slope_thr = 0.0
    else:
        slope_thr = 0.1 * std_val / (tt.max() - tt.min())

    if slope > slope_thr:
        return "Increasing"
    if slope < -slope_thr:
        return "Decreasing"

    changes = np.diff(vv)
    pos = int((changes > 0).sum())
    neg = int((changes < 0).sum())
    if pos > 0 and neg > 0:
        if np.mean(np.abs(changes)) > 0.2 * std_val:
            return "Fluctuating"

    base = baseline_value if baseline_value is not None else vv[0]
    if pd.isnull(base):
        return np.nan
    if ref_median is not None and not pd.isnull(ref_median):
        return "Stable_Low" if base <= ref_median else "Stable_High"
    return "Stable_Low" if base <= np.median(vv) else "Stable_High"


def add_change_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """For each variable group in CHANGE_PATTERN_GROUPS, compute Pattern_<Group>."""
    for group, cols in CHANGE_PATTERN_GROUPS.items():
        present = [c for c in cols if c in df.columns]
        if len(present) < 2:
            continue
        times = CHANGE_TIME_POINTS[group][: len(present)]
        # reference median = median of baseline column
        base_col = present[0]
        ref_median = df[base_col].median()
        patterns = []
        for _, row in df.iterrows():
            vals = row[present].values
            patterns.append(
                classify_change_pattern(vals, times, baseline_value=row[base_col],
                                        ref_median=ref_median)
            )
        df[f"Pattern_{group}"] = patterns
    print(f"[patterns] shape: {df.shape}")
    return df


# ============================================================================
# Stage 7: MICE imputation
# ============================================================================
def mice_impute(df: pd.DataFrame, max_iter: int = 20) -> pd.DataFrame:
    """IterativeImputer (BayesianRidge) on numeric columns; non-numeric left as-is."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_num = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if not num_cols:
        return df
    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=max_iter,
        random_state=C.RANDOM_STATE,
        initial_strategy="median",
    )
    num_imputed = pd.DataFrame(
        imputer.fit_transform(df[num_cols]),
        columns=num_cols, index=df.index,
    )
    out = pd.concat([num_imputed, df[non_num]], axis=1)
    print(f"[mice] remaining missing after imputation: {out.isnull().sum().sum()}")
    return out


# ============================================================================
# Stage 8: Categorical encoding (object / Pattern_* columns -> integer codes)
# ============================================================================
def encode_categoricals(df: pd.DataFrame) -> tuple:
    """Map object-dtype and Pattern_* columns to integer codes (NaN -> 0).
    Returns (encoded_df, mapping_dict)."""
    cat_cols = [c for c in df.columns if df[c].dtype == object or c.startswith("Pattern_")]
    mappings = {}
    out = df.copy()
    for col in cat_cols:
        uniques = out[col].dropna().unique()
        mapping = {v: i + 1 for i, v in enumerate(uniques)}
        mapping[np.nan] = 0
        out[col] = out[col].map(mapping)
        mappings[col] = mapping
    print(f"[encode] encoded {len(cat_cols)} categorical columns")
    return out, mappings


# ============================================================================
# Stage 9: FIXED post-processing (gender merge, age, cigarettes, 3-sigma)
# ============================================================================
def _extract_numeric_gender(x):
    if pd.isnull(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return int(x)
    m = re.search(r"(\d+)", str(x))
    return int(m.group(1)) if m else np.nan


def fixed_postprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """FIXED version (from 01-2.Logistics.md, second block).

    Fixes applied vs. the buggy first version:
      - uses convert_categoricals=False when reading the DTA (correct parameter name)
      - adds a regex-based numeric extractor for any residual string gender values
      - imputes missing Gender with the mode instead of leaving NaN
    """
    # 9a. Merge Gender from raw DTA
    dta_path = os.path.join(C.RAW_DATA_DIR, "H_CHARLS_D_Data.dta")
    if os.path.exists(dta_path) and C.ID_COL in df.columns:
        dta = pd.read_stata(dta_path, columns=[C.ID_COL, "ragender"],
                            convert_categoricals=False)
        dta = dta.rename(columns={"ragender": "Gender"})
        dta[C.ID_COL] = dta[C.ID_COL].astype(np.int64)
        df[C.ID_COL] = df[C.ID_COL].astype(np.int64)
        df = df.merge(dta, on=C.ID_COL, how="left")
        df["Gender"] = df["Gender"].apply(_extract_numeric_gender)
        gmode = df["Gender"].mode()[0] if not df["Gender"].mode().empty else 1
        n_miss = int(df["Gender"].isnull().sum())
        df["Gender"] = df["Gender"].fillna(gmode).astype(np.int32)
        print(f"[postfix] imputed {n_miss} missing Gender with mode={gmode}")
    else:
        df["Gender"] = 1  # fallback placeholder

    # 9b. Baseline_Age -> integer
    if "Baseline_Age" in df.columns:
        df["Baseline_Age"] = pd.to_numeric(df["Baseline_Age"], errors="coerce").round().astype(np.int32)

    # 9c. Baseline_Cigarettes: shift to positive + log + cap at 50
    if "Baseline_Cigarettes" in df.columns:
        df["Baseline_Cigarettes"] = pd.to_numeric(
            df["Baseline_Cigarettes"], errors="coerce"
        )
        mn = df["Baseline_Cigarettes"].min()
        if pd.notnull(mn) and mn < 0:
            df["Baseline_Cigarettes"] += abs(mn)
        df["Baseline_Cigarettes"] = np.log(df["Baseline_Cigarettes"] + 1e-6)
        df["Baseline_Cigarettes"] = df["Baseline_Cigarettes"].clip(upper=50)

    # 9d. 3-sigma truncation on all numeric columns (skip ID / outcome / time)
    skip = {C.ID_COL, C.EVENT_COL, C.TIME_COL}
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in skip]
    n_trunc = 0
    for col in num_cols:
        mu = df[col].mean()
        sd = df[col].std()
        if pd.isnull(sd) or sd == 0:
            continue
        lo, hi = mu - 3 * sd, mu + 3 * sd
        n_trunc += int(((df[col] < lo) | (df[col] > hi)).sum())
        df[col] = df[col].clip(lower=lo, upper=hi)
    print(f"[postfix] 3-sigma truncated {n_trunc} outlying cells across {len(num_cols)} columns")
    return df


# ============================================================================
# Main
# ============================================================================
def main():
    ensure_dir(C.PREPROCESSING_OUTPUT_DIR)
    ensure_dir(C.DATA_DIR)

    # 1-2. Raw DTA cleaning + merge (only needed if cleaned CSVs are absent)
    cleaned_path = os.path.join(C.DATA_DIR, "merged_CHARLS_D_only.csv")
    if os.path.exists(cleaned_path):
        print(f"[main] reusing merged cleaned data: {cleaned_path}")
    else:
        cleaned = clean_raw_dta_files()
        merged = merge_cleaned_datasets(cleaned)
        merged.to_csv(cleaned_path, index=False)

    # 3. Extract harmonised 4-wave variables & derived variables
    df = extract_harmonised_variables()
    df = compute_derived_variables(df)

    # 4-5. Cohort + event variables
    df = build_cohort(df)

    # 6. Change patterns
    df = add_change_patterns(df)

    # 7. MICE imputation
    df = mice_impute(df)

    # 8. Categorical encoding
    df, mappings = encode_categoricals(df)
    # Save mapping table
    mapping_rows = []
    for col, m in mappings.items():
        for k, v in m.items():
            mapping_rows.append({"Variable": col, "Original_Value": k, "Numeric_Value": v})
    pd.DataFrame(mapping_rows).to_csv(
        os.path.join(C.PREPROCESSING_OUTPUT_DIR, "category_mappings.csv"), index=False
    )

    # 9. FIXED post-processing
    df = fixed_postprocessing(df)

    # 10. Save final dataset
    df.to_excel(C.FINAL_DATASET_PATH, index=False, engine="openpyxl")
    df.to_csv(C.FINAL_DATASET_PATH.replace(".xlsx", ".csv"), index=False)
    print(f"[main] final dataset saved: {C.FINAL_DATASET_PATH}  shape={df.shape}")


if __name__ == "__main__":
    main()
