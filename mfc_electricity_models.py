#!/usr/bin/env python3
"""
MFC dashboard  -  MODE 1: ELECTRICITY GENERATION
Target (y): area-based power density, mW/m2 (column Power_Density_mW_per_m2_std)
  Models are trained on log10(power density) because it spans ~4 orders of magnitude.
  Volume-based power density (W/m3) has only 12 rows and is NOT pooled with area-based.

What it does
  1. Loads the 3 cleaned CSVs (features | targets | provenance) and joins them by row order.
  2. Compares 9+ regression models with study-grouped, repeated 5-fold CV
     (rows from the same paper/CV_Group are never split between train and test).
  3. Saves a model-comparison table, out-of-fold predictions, a final-model prediction CSV
     for every row, permutation importance, plots, and (optionally) what-if predictions.

Run:
  python mfc_electricity_models.py --data_dir "path/to/Project"
  python mfc_electricity_models.py --data_dir "path/to/Project" --scenarios my_scenarios.csv
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------- settings
TARGET = "Power_Density_mW_per_m2_std"
CAT = ["Wastewater_Type", "MFC_Configuration", "Anode_Material", "Cathode_Material"]
NUM = ["Initial_COD_mg_L", "HRT_h"]
# pH, Temperature, External resistance (>95 % missing) and Reactor volume (88 % missing)
# are left out by default (cleaning report, Output 7). Use --all_features to include them.
NUM_EXTRA = ["pH", "Temperature_C", "External_Resistance_Ohm", "Reactor_Volume_L"]
LOG10_COLS = ["Initial_COD_mg_L", "HRT_h", "External_Resistance_Ohm", "Reactor_Volume_L"]
LOG_MIN, LOG_MAX = -1.5, 4.5      # clip predictions to ~0.03 - 30 000 mW/m2
OUT = "outputs_electricity"  # (a "_random" suffix is added for --cv random)


# ----------------------------------------------------------------------------- data
def find_file(data_dir, name):
    for d in (data_dir, data_dir / "Project", Path(".")):
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(f"{name} not found in {data_dir}. Use --data_dir.")


def load(data_dir, exclude_caveats):
    X = pd.read_csv(find_file(data_dir, "MFC_ML_ready_10features.csv"))
    T = pd.read_csv(find_file(data_dir, "MFC_targets_NOT_features.csv"))
    P = pd.read_csv(find_file(data_dir, "MFC_provenance_metadata.csv"))
    assert len(X) == len(T) == len(P), "The three CSVs must have the same row order/length"
    df = X.copy()
    df["Row_ID"], df["Study_ID"], df["CV_Group"] = P["Row_ID"], P["Study_ID"], P["CV_Group"]
    df["y"] = T[TARGET]
    df["Target_Caveat"] = T["Target_Caveat"]
    keep = df["y"].notna() & (df["y"] > 0)
    if exclude_caveats:                      # drop soluble-COD / integrated-system rows
        keep &= df["Target_Caveat"].isna()
    df["Labelled"] = keep
    return df


def prepare(df, num_cols):
    """Deterministic (non-fitted) transforms: log10 on skewed numeric columns."""
    out = df.reindex(columns=CAT + num_cols).copy()
    for c in CAT:
        out[c] = out[c].astype("object").where(out[c].notna(), np.nan)
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        if c in LOG10_COLS:
            out[c] = np.log10(out[c].where(out[c] > 0))
    return out


def make_preprocessor(num_cols):
    num = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                    ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
                    ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                         min_frequency=3, sparse_output=False))])
    return ColumnTransformer([("num", num, num_cols), ("cat", cat, CAT)])


# ----------------------------------------------------------------------------- models
def get_models(seed):
    m = {
        "Baseline (mean)": DummyRegressor(strategy="mean"),
        "Ridge": RidgeCV(alphas=np.logspace(-2, 3, 30)),
        "ElasticNet": ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=20000),
        "KNN": KNeighborsRegressor(n_neighbors=5, weights="distance"),
        "SVR (RBF)": SVR(C=3.0, epsilon=0.1),
        "Gaussian Process": GaussianProcessRegressor(
            ConstantKernel(1.0) * RBF(3.0) + WhiteKernel(0.5),
            normalize_y=True, n_restarts_optimizer=2, random_state=seed),
        "Random Forest": RandomForestRegressor(300, min_samples_leaf=2, random_state=seed),
        "Extra Trees": ExtraTreesRegressor(300, min_samples_leaf=2, random_state=seed),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=2, subsample=0.8,
            random_state=seed),
    }
    try:
        from xgboost import XGBRegressor
        m["XGBoost"] = XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05,
                                    subsample=0.8, random_state=seed, verbosity=0)
    except Exception:
        pass
    try:
        from lightgbm import LGBMRegressor
        m["LightGBM"] = LGBMRegressor(n_estimators=200, max_depth=2, learning_rate=0.05,
                                      min_child_samples=3, subsample=0.8, random_state=seed,
                                      verbose=-1)
    except Exception:
        pass
    return m


def build(est, num_cols):
    return Pipeline([("prep", make_preprocessor(num_cols)),
                     ("reg", TransformedTargetRegressor(regressor=clone(est),
                                                        transformer=StandardScaler()))])


# ----------------------------------------------------------------------------- CV + metrics
def grouped_folds(groups, k, rng):
    """One random study-grouped k-fold split -> list of (train_idx, test_idx)."""
    uniq = np.array(sorted(set(groups)))
    rng.shuffle(uniq)
    fold_of = {g: i % k for i, g in enumerate(uniq)}
    f = np.array([fold_of[g] for g in groups])
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]


def metrics(y, p):
    """y, p are log10(power density in mW/m2). RMSE/MAE are in log10 units
    (0.30 = factor 2, 0.48 = factor 3, 1.0 = factor 10)."""
    return dict(R2=r2_score(y, p), RMSE=np.sqrt(mean_squared_error(y, p)),
                MAE=mean_absolute_error(y, p),
                MAE_mW_m2=mean_absolute_error(10 ** y, 10 ** p),
                Within_3x_pct=100 * np.mean(np.abs(y - p) <= np.log10(3)))


def cross_validate(models, Xp, y, groups, k, repeats, seed, num_cols):
    n = len(y)
    oof = {name: np.zeros((repeats, n)) for name in models}
    for r in range(repeats):
        folds = grouped_folds(groups, k, np.random.default_rng(seed + r))
        for name, est in models.items():
            for tr, te in folds:
                mdl = build(est, num_cols)
                mdl.fit(Xp.iloc[tr], y[tr])
                oof[name][r, te] = mdl.predict(Xp.iloc[te])
        print(f"  repeat {r + 1}/{repeats} done")
    rows = []
    for name in models:
        per = pd.DataFrame([metrics(y, np.clip(oof[name][r], LOG_MIN, LOG_MAX)) for r in range(repeats)])
        row = {"Model": name}
        for c in per.columns:
            row[f"{c}_mean"], row[f"{c}_sd"] = per[c].mean(), per[c].std()
        rows.append(row)
    res = pd.DataFrame(rows).sort_values("RMSE_mean").reset_index(drop=True)
    res.insert(0, "Rank", res.index + 1)
    return res, oof


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=".", help="folder with the 3 MFC CSV files")
    ap.add_argument("--scenarios", default=None, help="CSV of what-if designs to predict")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv", choices=["group", "random"], default="group",
                    help="group = study-grouped CV (honest, default); "
                         "random = plain KFold (optimistic, same-paper leakage)")
    ap.add_argument("--all_features", action="store_true",
                    help="also use pH, temperature, ext. resistance, reactor volume")
    ap.add_argument("--exclude_caveats", action="store_true",
                    help="drop soluble-COD / integrated-system rows from training")
    a = ap.parse_args()

    out = Path(OUT + ("_random" if a.cv == "random" else ""))
    out.mkdir(exist_ok=True)
    num_cols = NUM + (NUM_EXTRA if a.all_features else [])
    df = load(Path(a.data_dir), a.exclude_caveats)
    lab = df[df["Labelled"]].reset_index(drop=True)
    y_raw = lab["y"].to_numpy(float)
    y = np.log10(y_raw)                      # models are trained on log10(power density)
    Xp = prepare(lab, num_cols)
    groups = lab["CV_Group"].astype(str).to_numpy()
    if a.cv == "random":
        groups = np.arange(len(lab)).astype(str)
    print(f"Training rows: {len(lab)} from {len(set(groups))} studies | target = {TARGET}")
    print(f"Features: {CAT + num_cols}")

    models = get_models(a.seed)
    res, oof = cross_validate(models, Xp, y, groups, a.folds, a.repeats, a.seed, num_cols)
    res.round(3).to_csv(out / "model_comparison_electricity.csv", index=False)
    print("\n=== Grouped CV results (sorted by RMSE; R2 / RMSE / MAE in log10 units) ===")
    print(res[["Rank", "Model", "R2_mean", "R2_sd", "RMSE_mean", "MAE_mean",
               "MAE_mW_m2_mean", "Within_3x_pct_mean"]].round(3).to_string(index=False))

    base_r2 = res.loc[res.Model == "Baseline (mean)", "R2_mean"].iloc[0]
    cand = res[res.Model != "Baseline (mean)"]
    best = cand.iloc[0]["Model"]
    if cand.iloc[0]["R2_mean"] < 0.3 and a.cv == "group":
        print("\nWARNING: best model has R2 < 0.3 on unseen studies (R2 <= 0 = no better than "
              "predicting the mean). Treat predictions as exploratory/screening only.")
    print(f"\nBest non-baseline model: {best}")

    # ---- out-of-fold predictions for every model (honest, unseen-study predictions)
    oofdf = lab[["Row_ID", "Study_ID", "CV_Group"] + CAT + num_cols].copy()
    oofdf["Actual_power_density_mW_m2"] = y_raw
    for name in models:
        oofdf[f"OOF_{name}_mW_m2"] = (10 ** np.clip(oof[name].mean(axis=0), LOG_MIN, LOG_MAX)).round(3)
    oofdf.to_csv(out / "oof_predictions_all_models_electricity.csv", index=False)

    # ---- final model (best) fitted on all labelled rows -> prediction for EVERY row
    final = build(models[best], num_cols).fit(Xp, y)
    allp = df[["Row_ID", "Study_ID"] + CAT + num_cols].copy()
    allp["Actual_power_density_mW_m2"] = df["y"]
    allp["Used_for_training"] = df["Labelled"]
    allp["Predicted_power_density_mW_m2_final_model"] = (10 ** np.clip(
        final.predict(prepare(df, num_cols)), LOG_MIN, LOG_MAX)).round(3)
    allp["Predicted_power_density_mW_m2_out_of_fold"] = np.nan
    allp.loc[df.index[df["Labelled"]], "Predicted_power_density_mW_m2_out_of_fold"] = (
        10 ** np.clip(oof[best].mean(axis=0), LOG_MIN, LOG_MAX)).round(3)
    allp["Model"] = best
    allp.to_csv(out / "predictions_electricity_best_model.csv", index=False)

    # ---- permutation importance (in-sample, indicates what the model relies on)
    imp = permutation_importance(final, Xp, y, n_repeats=30, random_state=a.seed,
                                 scoring="neg_root_mean_squared_error")
    pd.DataFrame({"Feature": Xp.columns, "Importance_RMSE_increase": imp.importances_mean,
                  "SD": imp.importances_std}).sort_values(
        "Importance_RMSE_increase", ascending=False).round(3).to_csv(
        out / "feature_importance_electricity.csv", index=False)

    # ---- plots
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    r = res.iloc[::-1]
    ax[0].barh(r["Model"], r["R2_mean"], xerr=r["R2_sd"], color="#2a7f62")
    ax[0].axvline(0, color="k", lw=0.8)
    ax[0].set_xlabel("R² (study-grouped CV)")
    ax[0].set_title("Electricity: log10 power density - model comparison")
    p = np.clip(oof[best].mean(axis=0), LOG_MIN, LOG_MAX)
    ax[1].scatter(10 ** y, 10 ** p, c="#2a7f62")
    lim = [10 ** LOG_MIN, 10 ** LOG_MAX]
    ax[1].plot(lim, lim, "k--", lw=0.8)
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("Actual power density (mW/m²)")
    ax[1].set_ylabel("Predicted (out-of-fold, mW/m²)")
    ax[1].set_title(f"Best model: {best}")
    plt.tight_layout()
    plt.savefig(out / "electricity_model_comparison.png", dpi=150)

    # ---- what-if scenarios
    scen_path = Path(a.scenarios) if a.scenarios else None
    if scen_path is None:
        tmpl = df[df["Labelled"]].dropna(subset=["Initial_COD_mg_L", "HRT_h"]).head(3)
        tmpl = tmpl.reindex(columns=CAT + num_cols)
        tmpl.to_csv(out / "scenarios_template_electricity.csv", index=False)
        print(f"\nEdit {out/'scenarios_template_electricity.csv'} and re-run with "
              f"--scenarios <file> for what-if predictions.")
    else:
        sc = pd.read_csv(scen_path)
        pred = pd.DataFrame({name: 10 ** np.clip(build(est, num_cols).fit(Xp, y)
                                              .predict(prepare(sc, num_cols)), LOG_MIN, LOG_MAX)
                             for name, est in models.items() if name != "Baseline (mean)"})
        sc["Predicted_power_density_mW_m2_best_model"] = pred[best].round(3).values
        sc["Best_model"] = best
        for name in pred.columns:
            sc[f"Pred_{name}_mW_m2"] = pred[name].round(3).values
        sc.to_csv(out / "scenario_predictions_electricity.csv", index=False)
        print(f"Saved {out/'scenario_predictions_electricity.csv'}")

    print(f"\nAll outputs saved in ./{out}/")


if __name__ == "__main__":
    main()
