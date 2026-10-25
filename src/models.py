"""Machine learning pipelines, cached model estimators, inference routines, and benchmark records."""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import streamlit as st

from src.config import (
    CAT_FEATURES,
    NUM_FEATURES,
    LOG10_COLS,
    ELEC_LOG_MIN,
    ELEC_LOG_MAX,
    WW_MIN_PCT,
    WW_MAX_PCT,
    ELEC_BENCHMARK_DATA,
    WW_BENCHMARK_DATA,
    ELEC_FEATURE_IMPORTANCE,
    WW_FEATURE_IMPORTANCE,
    LEAKAGE_COMPARISON_DATA,
    ALL_FEATURES_IMPACT_DATA,
)
from src.data_loader import load_electricity_data, load_wastewater_data


def prepare_features(df: pd.DataFrame, num_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Deterministic transforms: convert categoricals to object, apply log10
    to skewed continuous columns (Initial_COD_mg_L, HRT_h).
    """
    if num_cols is None:
        num_cols = NUM_FEATURES
    
    cols = CAT_FEATURES + num_cols
    out = df.reindex(columns=cols).copy()

    for c in CAT_FEATURES:
        out[c] = out[c].astype("object").where(out[c].notna(), np.nan)
        
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        if c in LOG10_COLS:
            out[c] = np.log10(out[c].where(out[c] > 0))
            
    return out


def make_preprocessor(num_cols: Optional[List[str]] = None) -> ColumnTransformer:
    """Construct ColumnTransformer with median imputation, scaling, and infrequent-handling OHE."""
    if num_cols is None:
        num_cols = NUM_FEATURES
        
    num_pipeline = Pipeline([
        ("imp", SimpleImputer(strategy="median", add_indicator=True)),
        ("sc", StandardScaler()),
    ])
    
    cat_pipeline = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=3, sparse_output=False)),
    ])
    
    return ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, CAT_FEATURES),
    ])


def build_pipeline(estimator, num_cols: Optional[List[str]] = None) -> Pipeline:
    """Encapsulate preprocessor and estimator within a TransformedTargetRegressor with StandardScaler."""
    if num_cols is None:
        num_cols = NUM_FEATURES
    return Pipeline([
        ("prep", make_preprocessor(num_cols)),
        ("reg", TransformedTargetRegressor(regressor=clone(estimator), transformer=StandardScaler())),
    ])


@st.cache_resource(show_spinner=False)
def get_trained_electricity_model():
    """
    Fit and cache the best electricity model (Random Forest, 300 trees)
    trained on log10(power density) across the 40 labelled rows.
    """
    df_el = load_electricity_data()
    train_df = df_el[df_el["Used_for_training"]].reset_index(drop=True)
    
    y = np.log10(train_df["Actual_power_density_mW_m2"].to_numpy(float))
    X = prepare_features(train_df, NUM_FEATURES)
    
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42)
    pipeline = build_pipeline(rf, NUM_FEATURES)
    pipeline.fit(X, y)
    return pipeline


@st.cache_resource(show_spinner=False)
def get_trained_wastewater_model():
    """
    Fit and cache the best wastewater ML model (Gaussian Process with RBF + WhiteKernel)
    trained on COD removal % across the 55 labelled rows.
    """
    df_ww = load_wastewater_data()
    train_df = df_ww[df_ww["Used_for_training"]].reset_index(drop=True)
    
    y = train_df["Actual_COD_removal_pct"].to_numpy(float)
    X = prepare_features(train_df, NUM_FEATURES)
    
    gp = GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * RBF(3.0) + WhiteKernel(0.5),
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=42,
    )
    pipeline = build_pipeline(gp, NUM_FEATURES)
    pipeline.fit(X, y)
    return pipeline


def predict_electricity_single(
    wastewater_type: str,
    configuration: str,
    anode_material: str,
    cathode_material: str,
    initial_cod: float,
    hrt: float,
) -> Dict[str, Union[float, str]]:
    """Predict area-based power density (mW/m2) for a single reactor specification."""
    model = get_trained_electricity_model()
    row_df = pd.DataFrame([{
        "Wastewater_Type": wastewater_type,
        "MFC_Configuration": configuration,
        "Anode_Material": anode_material,
        "Cathode_Material": cathode_material,
        "Initial_COD_mg_L": initial_cod,
        "HRT_h": hrt,
    }])
    X = prepare_features(row_df, NUM_FEATURES)
    
    raw_pred_log = float(model.predict(X)[0])
    clipped_log = float(np.clip(raw_pred_log, ELEC_LOG_MIN, ELEC_LOG_MAX))
    pred_physical = float(round(10 ** clipped_log, 3))

    return {
        "pred_power_density_mW_m2": pred_physical,
        "pred_log10": round(clipped_log, 3),
        "raw_log10": round(raw_pred_log, 3),
        "model_name": "Random Forest Regressor (300 trees)",
        "unit": "mW/m²",
        "log_unit": "log₁₀(mW/m²)",
    }


def predict_wastewater_single(
    wastewater_type: str,
    configuration: str,
    anode_material: str,
    cathode_material: str,
    initial_cod: float,
    hrt: float,
) -> Dict[str, Union[float, str]]:
    """Predict COD removal % for a single reactor specification."""
    model = get_trained_wastewater_model()
    row_df = pd.DataFrame([{
        "Wastewater_Type": wastewater_type,
        "MFC_Configuration": configuration,
        "Anode_Material": anode_material,
        "Cathode_Material": cathode_material,
        "Initial_COD_mg_L": initial_cod,
        "HRT_h": hrt,
    }])
    X = prepare_features(row_df, NUM_FEATURES)
    
    raw_pred = float(model.predict(X)[0])
    clipped_pred = float(round(np.clip(raw_pred, WW_MIN_PCT, WW_MAX_PCT), 2))

    return {
        "pred_cod_removal_pct": clipped_pred,
        "raw_pred": round(raw_pred, 2),
        "model_name": "Gaussian Process Regressor (RBF + WhiteKernel)",
        "unit": "%",
    }


def predict_scenario_batch(scenarios_df: pd.DataFrame) -> pd.DataFrame:
    """Compute predictions for both electricity and wastewater across a table of scenarios."""
    df_out = scenarios_df.copy()
    
    # Electricity prediction
    elec_model = get_trained_electricity_model()
    X_el = prepare_features(scenarios_df, NUM_FEATURES)
    log_preds = elec_model.predict(X_el)
    clipped_log = np.clip(log_preds, ELEC_LOG_MIN, ELEC_LOG_MAX)
    df_out["Pred_Power_Density_mW_m2"] = np.round(10 ** clipped_log, 2)
    
    # Wastewater prediction
    ww_model = get_trained_wastewater_model()
    X_ww = prepare_features(scenarios_df, NUM_FEATURES)
    ww_preds = ww_model.predict(X_ww)
    df_out["Pred_COD_Removal_pct"] = np.round(np.clip(ww_preds, WW_MIN_PCT, WW_MAX_PCT), 2)
    
    return df_out


def get_historical_row_data(row_id: str) -> Optional[Dict]:
    """Retrieve ground truth and predictions for an existing record in the dataset."""
    df_el = load_electricity_data()
    df_ww = load_wastewater_data()
    
    match_el = df_el[df_el["Row_ID"] == row_id]
    match_ww = df_ww[df_ww["Row_ID"] == row_id]
    
    if match_el.empty or match_ww.empty:
        return None
        
    rel = match_el.iloc[0]
    rww = match_ww.iloc[0]
    
    return {
        "Row_ID": row_id,
        "Study_ID": rel.get("Study_ID", "Unknown"),
        "Wastewater_Type": rel.get("Wastewater_Type", "Unknown"),
        "MFC_Configuration": rel.get("MFC_Configuration", "Unknown"),
        "Anode_Material": rel.get("Anode_Material", "Unknown"),
        "Cathode_Material": rel.get("Cathode_Material", "Unknown"),
        "Initial_COD_mg_L": rel.get("Initial_COD_mg_L", np.nan),
        "HRT_h": rel.get("HRT_h", np.nan),
        
        # Electricity
        "Actual_Power_Density": rel.get("Actual_power_density_mW_m2", np.nan),
        "Final_Power_Density": rel.get("Predicted_power_density_mW_m2_final_model", np.nan),
        "OOF_Power_Density": rel.get("Predicted_power_density_mW_m2_out_of_fold", np.nan),
        "Used_in_Elec_Train": bool(rel.get("Used_for_training", False)),
        
        # Wastewater
        "Actual_COD_Removal": rww.get("Actual_COD_removal_pct", np.nan),
        "Final_COD_Removal": rww.get("Predicted_COD_removal_pct_final_model", np.nan),
        "OOF_COD_Removal": rww.get("Predicted_COD_removal_pct_out_of_fold", np.nan),
        "Used_in_WW_Train": bool(rww.get("Used_for_training", False)),
    }


def get_benchmark_df(task: str) -> pd.DataFrame:
    """Return benchmark DataFrame for electricity or wastewater."""
    if task == "electricity":
        return pd.DataFrame(ELEC_BENCHMARK_DATA)
    elif task == "wastewater":
        return pd.DataFrame(WW_BENCHMARK_DATA)
    raise ValueError("Task must be 'electricity' or 'wastewater'")


def get_feature_importance_df(task: str) -> pd.DataFrame:
    """Return permutation feature importance DataFrame."""
    if task == "electricity":
        return pd.DataFrame(ELEC_FEATURE_IMPORTANCE)
    elif task == "wastewater":
        return pd.DataFrame(WW_FEATURE_IMPORTANCE)
    raise ValueError("Task must be 'electricity' or 'wastewater'")


def get_leakage_df() -> pd.DataFrame:
    return pd.DataFrame(LEAKAGE_COMPARISON_DATA)


def get_all_features_impact_df() -> pd.DataFrame:
    return pd.DataFrame(ALL_FEATURES_IMPACT_DATA)
