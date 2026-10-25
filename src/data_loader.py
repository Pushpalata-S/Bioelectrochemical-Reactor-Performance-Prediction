"""Data loading, caching, feature domain extraction, and missingness profiling."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    ELEC_DATA_PATH,
    WW_DATA_PATH,
    CAT_FEATURES,
    NUM_FEATURES,
    TARGET_ELEC,
    TARGET_WW,
)


@st.cache_data(show_spinner=False)
def load_electricity_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load and format the electricity generation dataset."""
    target_path = Path(path) if path else ELEC_DATA_PATH
    if not target_path.exists():
        # Fallback to local data folder if relative path shifted
        fallback = Path("data") / "CSV for elec.csv"
        if fallback.exists():
            target_path = fallback
        else:
            raise FileNotFoundError(f"Electricity dataset not found at {target_path}")

    df = pd.read_csv(target_path)
    df["Used_for_training"] = df["Used_for_training"].astype(bool)
    return df


@st.cache_data(show_spinner=False)
def load_wastewater_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load and format the wastewater treatment dataset."""
    target_path = Path(path) if path else WW_DATA_PATH
    if not target_path.exists():
        fallback = Path("data") / "CSV for waste water.csv"
        if fallback.exists():
            target_path = fallback
        else:
            raise FileNotFoundError(f"Wastewater dataset not found at {target_path}")

    df = pd.read_csv(target_path)
    df["Used_for_training"] = df["Used_for_training"].astype(bool)
    return df


@st.cache_data(show_spinner=False)
def load_both_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load both datasets simultaneously with error handling."""
    return load_electricity_data(), load_wastewater_data()


@st.cache_data(show_spinner=False)
def get_merged_dataset() -> pd.DataFrame:
    """
    Merge electricity and wastewater datasets on common reactor identifiers and features.
    Maintains all observations (R001 to R068) with dual targets and predictions.
    """
    df_elec = load_electricity_data()
    df_ww = load_wastewater_data()

    common_cols = ["Row_ID", "Study_ID"] + CAT_FEATURES + NUM_FEATURES
    
    # Select distinct prediction columns
    elec_cols = [
        "Row_ID",
        "Actual_power_density_mW_m2",
        "Used_for_training",
        "Predicted_power_density_mW_m2_final_model",
        "Predicted_power_density_mW_m2_out_of_fold",
    ]
    ww_cols = [
        "Row_ID",
        "Actual_COD_removal_pct",
        "Used_for_training",
        "Predicted_COD_removal_pct_final_model",
        "Predicted_COD_removal_pct_out_of_fold",
    ]

    df_e_sub = df_elec[elec_cols].rename(columns={"Used_for_training": "Used_in_Electricity_Train"})
    df_w_sub = df_ww[ww_cols].rename(columns={"Used_for_training": "Used_in_Wastewater_Train"})

    merged = df_elec[common_cols].copy()
    merged = merged.merge(df_e_sub, on="Row_ID", how="left")
    merged = merged.merge(df_w_sub, on="Row_ID", how="left")

    merged["Has_Electricity_Label"] = merged[TARGET_ELEC].notna() & (merged[TARGET_ELEC] > 0)
    merged["Has_Wastewater_Label"] = merged[TARGET_WW].notna()
    merged["Has_Both_Labels"] = merged["Has_Electricity_Label"] & merged["Has_Wastewater_Label"]

    return merged


@st.cache_data(show_spinner=False)
def get_dataset_statistics() -> Dict:
    """Calculate exact dataset summary metrics and missingness profiles."""
    df_e = load_electricity_data()
    df_w = load_wastewater_data()
    merged = get_merged_dataset()

    total_records = len(df_e)
    unique_studies = merged["Study_ID"].nunique()
    
    elec_labelled = int(merged["Has_Electricity_Label"].sum())
    ww_labelled = int(merged["Has_Wastewater_Label"].sum())
    both_labelled = int(merged["Has_Both_Labels"].sum())

    # Category cardinalities (across all 67 reactors)
    cardinalities = {
        col: int(merged[col].dropna().nunique()) for col in CAT_FEATURES
    }

    # Missingness table for shared input features
    missingness = []
    for col in CAT_FEATURES + NUM_FEATURES:
        missing_count = int(merged[col].isna().sum())
        missing_pct = round((missing_count / total_records) * 100, 1)
        missingness.append({
            "Feature": col,
            "Type": "Categorical" if col in CAT_FEATURES else "Continuous",
            "Present": total_records - missing_count,
            "Missing": missing_count,
            "Missing_Pct": missing_pct,
            "Status": "Included in Default ML"
        })

    # Add the excluded physics features documented in report
    missingness.extend([
        {"Feature": "pH", "Type": "Continuous", "Present": 3, "Missing": 64, "Missing_Pct": 95.5, "Status": "Excluded (>95% missing)"},
        {"Feature": "Temperature_C", "Type": "Continuous", "Present": 2, "Missing": 65, "Missing_Pct": 97.0, "Status": "Excluded (>97% missing)"},
        {"Feature": "External_Resistance_Ohm", "Type": "Continuous", "Present": 3, "Missing": 64, "Missing_Pct": 95.5, "Status": "Excluded (>95% missing)"},
        {"Feature": "Reactor_Volume_L", "Type": "Continuous", "Present": 8, "Missing": 59, "Missing_Pct": 88.1, "Status": "Excluded (>88% missing)"},
    ])

    return {
        "total_records": total_records,
        "unique_studies": unique_studies,
        "elec_labelled": elec_labelled,
        "elec_unlabelled": total_records - elec_labelled,
        "ww_labelled": ww_labelled,
        "ww_unlabelled": total_records - ww_labelled,
        "both_labelled": both_labelled,
        "cardinalities": cardinalities,
        "missingness_df": pd.DataFrame(missingness),
    }


def get_feature_domains() -> Dict[str, List]:
    """Extract valid categorical options and continuous boundaries for user inputs."""
    merged = get_merged_dataset()
    domains = {}
    for col in CAT_FEATURES:
        vals = sorted([str(x) for x in merged[col].dropna().unique() if str(x).strip()])
        domains[col] = vals

    # Continuous ranges
    cod_vals = merged["Initial_COD_mg_L"].dropna()
    hrt_vals = merged["HRT_h"].dropna()

    domains["Initial_COD_mg_L"] = {
        "min": float(cod_vals.min()),
        "max": float(cod_vals.max()),
        "median": float(cod_vals.median()),
        "mean": float(cod_vals.mean()),
    }
    domains["HRT_h"] = {
        "min": float(hrt_vals.min()),
        "max": float(hrt_vals.max()),
        "median": float(hrt_vals.median()),
        "mean": float(hrt_vals.mean()),
    }
    return domains
