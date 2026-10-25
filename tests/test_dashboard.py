"""Comprehensive test suite for Bioelectrochemical Reactor Dashboard."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure root directory is on python path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.data_loader import (
    load_electricity_data,
    load_wastewater_data,
    get_merged_dataset,
    get_dataset_statistics,
    get_feature_domains,
)
from src.models import (
    get_trained_electricity_model,
    get_trained_wastewater_model,
    predict_electricity_single,
    predict_wastewater_single,
    predict_scenario_batch,
    get_historical_row_data,
    get_benchmark_df,
    get_feature_importance_df,
    get_leakage_df,
    get_all_features_impact_df,
)
from src.plots import (
    plot_missingness,
    plot_parity,
    plot_residuals,
    plot_benchmark_r2,
    plot_benchmark_rmse,
    plot_feature_importance,
    plot_data_leakage,
    plot_target_distribution,
    plot_bivariate,
    plot_scenario_tradeoff,
)
from src.pages.scenarios import DEFAULT_SCENARIOS


def test_data_loader():
    print("--- Testing Data Loader ---")
    df_e = load_electricity_data()
    df_w = load_wastewater_data()
    assert len(df_e) == 67, f"Expected 67 rows, got {len(df_e)}"
    assert len(df_w) == 67, f"Expected 67 rows, got {len(df_w)}"
    
    merged = get_merged_dataset()
    assert len(merged) == 67, f"Expected 67 rows in merged, got {len(merged)}"
    
    stats = get_dataset_statistics()
    assert stats["total_records"] == 67
    assert stats["elec_labelled"] == 40
    assert stats["ww_labelled"] == 55
    assert stats["both_labelled"] == 32
    
    domains = get_feature_domains()
    assert len(domains["Wastewater_Type"]) > 0
    assert len(domains["MFC_Configuration"]) > 0
    print("Data Loader: PASS (67 total, 40 elec, 55 ww, 32 both)")


def test_models():
    print("--- Testing Models & Inference ---")
    m_el = get_trained_electricity_model()
    m_ww = get_trained_wastewater_model()
    assert m_el is not None
    assert m_ww is not None
    
    # Test R001 prediction
    res_el = predict_electricity_single(
        wastewater_type="Sewage (raw)",
        configuration="Tubular",
        anode_material="Graphite felt",
        cathode_material="Carbon cloth with activated carbon catalyst",
        initial_cod=450.0,
        hrt=12.0,
    )
    assert 20.0 < res_el["pred_power_density_mW_m2"] < 25.0, f"Expected ~21.95, got {res_el['pred_power_density_mW_m2']}"
    print(f"Electricity R001 prediction: {res_el['pred_power_density_mW_m2']} mW/m2 (Target ~21.95)")
    
    res_ww = predict_wastewater_single(
        wastewater_type="Sewage (raw)",
        configuration="Tubular",
        anode_material="Graphite felt",
        cathode_material="Carbon cloth with activated carbon catalyst",
        initial_cod=450.0,
        hrt=12.0,
    )
    assert 50.0 < res_ww["pred_cod_removal_pct"] < 60.0, f"Expected ~55.76, got {res_ww['pred_cod_removal_pct']}"
    print(f"Wastewater R001 prediction: {res_ww['pred_cod_removal_pct']}% (Target ~55.76)")
    
    # Test historical lookup
    hist_r1 = get_historical_row_data("R001")
    assert hist_r1["Final_Power_Density"] == 21.95
    assert hist_r1["Final_COD_Removal"] == 55.76
    assert hist_r1["Actual_COD_Removal"] == 17.0
    print("Historical Row Lookup: PASS")


def test_scenarios():
    print("--- Testing Scenario Batch Engine ---")
    sc_df = pd.DataFrame(DEFAULT_SCENARIOS)
    batch_res = predict_scenario_batch(sc_df)
    assert len(batch_res) == 4
    assert "Pred_Power_Density_mW_m2" in batch_res.columns
    assert "Pred_COD_Removal_pct" in batch_res.columns
    print(batch_res[["Scenario_Name", "Pred_Power_Density_mW_m2", "Pred_COD_Removal_pct"]].to_string())
    print("Scenario Batch Engine: PASS")


def test_plots():
    print("--- Testing Plot Generation ---")
    stats = get_dataset_statistics()
    df_e = load_electricity_data()
    df_w = load_wastewater_data()
    bench_el = get_benchmark_df("electricity")
    bench_ww = get_benchmark_df("wastewater")
    imp_el = get_feature_importance_df("electricity")
    imp_ww = get_feature_importance_df("wastewater")
    leak_df = get_leakage_df()
    sc_df = predict_scenario_batch(pd.DataFrame(DEFAULT_SCENARIOS))
    
    fig1 = plot_missingness(stats["missingness_df"])
    fig2 = plot_parity(df_e, "Actual_power_density_mW_m2", "Predicted_power_density_mW_m2_out_of_fold", "Parity", "Act", "Pred", is_log=True)
    fig3 = plot_parity(df_w, "Actual_COD_removal_pct", "Predicted_COD_removal_pct_out_of_fold", "Parity", "Act", "Pred", is_log=False)
    fig4 = plot_residuals(df_e, "Actual_power_density_mW_m2", "Predicted_power_density_mW_m2_out_of_fold", "Res", "Act", "Res", is_log=True)
    fig5 = plot_benchmark_r2(bench_el, "Electricity")
    fig6 = plot_benchmark_rmse(bench_ww, "Wastewater", "% pts")
    fig7 = plot_feature_importance(imp_el, "Electricity")
    fig8 = plot_data_leakage(leak_df)
    fig9 = plot_target_distribution(df_e, "Actual_power_density_mW_m2", "Dist", "mW/m2", is_log=True)
    fig10 = plot_bivariate(df_e, "Initial_COD_mg_L", "Actual_power_density_mW_m2", "MFC_Configuration", "Biv", "COD", "Power", True, True)
    fig11 = plot_scenario_tradeoff(sc_df)
    
    for i, fig in enumerate([fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11], 1):
        assert fig is not None, f"Fig {i} is None"
    print("Plot Generation: PASS (11 Plotly figures tested successfully)")


if __name__ == "__main__":
    print("========================================")
    print("RUNNING BIOELECTROCHEMICAL DASHBOARD TESTS")
    print("========================================")
    test_data_loader()
    test_models()
    test_scenarios()
    test_plots()
    print("========================================")
    print("ALL TESTS PASSED SUCCESSFULLY! [OK]")
    print("========================================")
