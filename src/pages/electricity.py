"""Page B: Electricity Generation Prediction (Area-Based Power Density, mW/m2)."""
import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import get_feature_domains, load_electricity_data
from src.models import (
    predict_electricity_single,
    get_historical_row_data,
    get_feature_importance_df,
)
from src.plots import plot_feature_importance


def render_electricity_page():
    st.markdown("""
        <div style="margin-bottom: 1.25rem;">
            <span class="sci-badge sci-badge-teal">MODE 1: ELECTRICITY GENERATION</span>
            <span class="sci-badge sci-badge-amber">TARGET: POWER DENSITY (mW/m²)</span>
            <h2 style="color: #0A192F; margin-top: 0.5rem; margin-bottom: 0.25rem;">
                Electricity Generation Prediction Interface
            </h2>
            <p style="color: #64748B;">
                Predict area-based power density ($P_{\text{area}}$, $\\text{mW/m}^2$) using the benchmark-winning 
                Random Forest Regressor ($R^2 = +0.05$, Lowest RMSE = $1.02\\ \\log_{10}$ units).
            </p>
        </div>
    """, unsafe_allow_html=True)

    domains = get_feature_domains()
    df_el = load_electricity_data()

    # Mode Selector: New Design vs Inspect Historical Dataset Row
    view_mode = st.radio(
        "Input Mode:",
        ["🔬 Define Custom Reactor Specification", "📂 Load Historical Experimental Record (R001 - R068)"],
        horizontal=True,
    )

    # Defaults
    default_ww = "Sewage (raw)"
    default_cfg = "Tubular"
    default_anode = "Graphite felt"
    default_cathode = "Carbon cloth with activated carbon catalyst"
    default_cod = 500.0
    default_hrt = 24.0
    selected_row_data = None

    if view_mode.startswith("📂"):
        st.markdown("#### Select Historical Reactor Row")
        row_options = df_el["Row_ID"].tolist()
        selected_row_id = st.selectbox(
            "Select Row_ID to populate configuration:",
            row_options,
            index=0,
            help="Pulls exact design parameters and experimental ground truth from the curated database.",
        )
        selected_row_data = get_historical_row_data(selected_row_id)
        if selected_row_data:
            default_ww = selected_row_data["Wastewater_Type"] if pd.notna(selected_row_data["Wastewater_Type"]) else domains["Wastewater_Type"][0]
            default_cfg = selected_row_data["MFC_Configuration"] if pd.notna(selected_row_data["MFC_Configuration"]) else domains["MFC_Configuration"][0]
            default_anode = selected_row_data["Anode_Material"] if pd.notna(selected_row_data["Anode_Material"]) else domains["Anode_Material"][0]
            default_cathode = selected_row_data["Cathode_Material"] if pd.notna(selected_row_data["Cathode_Material"]) else domains["Cathode_Material"][0]
            default_cod = float(selected_row_data["Initial_COD_mg_L"]) if pd.notna(selected_row_data["Initial_COD_mg_L"]) else 500.0
            default_hrt = float(selected_row_data["HRT_h"]) if pd.notna(selected_row_data["HRT_h"]) else 24.0

    st.markdown("### 1. Reactor Design Parameters")
    with st.container():
        c1, c2 = st.columns(2)
        with c1:
            ww_list = domains["Wastewater_Type"]
            ww_index = ww_list.index(default_ww) if default_ww in ww_list else 0
            ww_input = st.selectbox("Wastewater Substrate Feed:", ww_list, index=ww_index)

            anode_list = domains["Anode_Material"]
            anode_index = anode_list.index(default_anode) if default_anode in anode_list else 0
            anode_input = st.selectbox("Anode Electrode Material:", anode_list, index=anode_index)

            cod_input = st.number_input(
                "Initial COD Concentration (mg/L):",
                min_value=1.0,
                max_value=100000.0,
                value=float(default_cod),
                step=50.0,
                help="Substrate chemical oxygen demand concentration. Model applies log10 transformation.",
            )

        with c2:
            cfg_list = domains["MFC_Configuration"]
            cfg_index = cfg_list.index(default_cfg) if default_cfg in cfg_list else 0
            cfg_input = st.selectbox("MFC Architecture / Configuration:", cfg_list, index=cfg_index)

            cathode_list = domains["Cathode_Material"]
            cathode_index = cathode_list.index(default_cathode) if default_cathode in cathode_list else 0
            cathode_input = st.selectbox("Cathode Material & Catalyst:", cathode_list, index=cathode_index)

            hrt_input = st.number_input(
                "Hydraulic Retention Time - HRT (hours):",
                min_value=0.1,
                max_value=1000.0,
                value=float(default_hrt),
                step=1.0,
                help="Contact / residence time inside the reactor. Model applies log10 transformation.",
            )

    # Optional / Secondary Physical Features Expander
    with st.expander("ℹ️ Secondary Electrochemical Parameters (pH, Temp, R_ext, Volume)"):
        st.markdown("""
        <div class="sci-alert">
            <strong>Why are these omitted from the default model?</strong><br>
            Across published review papers, <code>pH</code> (95.5% missing), <code>Temperature_C</code> (97.0% missing), 
            <code>External_Resistance_Ohm</code> (95.5% missing), and <code>Reactor_Volume_L</code> (88.1% missing) are rarely reported together. 
            Imputing these features on the small labelled sample (N=40) added severe noise and caused cross-validated R² to collapse from +0.049 to -0.215.
        </div>
        """, unsafe_allow_html=True)
        ec1, ec2 = st.columns(2)
        with ec1:
            st.text_input("Operating Temperature (°C):", value="25.0 (Standard Room Temp)", disabled=True)
            st.text_input("Influent pH:", value="7.0 (Neutral Buffer)", disabled=True)
        with ec2:
            st.text_input("External Load Resistor (Ω):", value="1000 Ω (Typical Unmeasured Range)", disabled=True)
            st.text_input("Working Volume (L):", value="0.5 L (Typical Bench Scale)", disabled=True)

    # Submission & Prediction
    st.markdown("### 2. Reactor Performance Prediction")
    
    col_pred, col_diag = st.columns([1.1, 0.9])
    
    with col_pred:
        res = predict_electricity_single(
            wastewater_type=ww_input,
            configuration=cfg_input,
            anode_material=anode_input,
            cathode_material=cathode_input,
            initial_cod=cod_input,
            hrt=hrt_input,
        )

        st.markdown(f"""
        <div class="sci-card" style="border-left: 4px solid #0D9488;">
            <div class="metric-label">Predicted Area-Based Power Density</div>
            <div class="metric-value" style="color: #0D9488;">
                {res['pred_power_density_mW_m2']:,.2f} <span style="font-size: 1.1rem; font-weight: 500; color: #475569;">mW/m²</span>
            </div>
            <div class="metric-sub" style="margin-top: 0.4rem;">
                <strong>log₁₀ Scale Output:</strong> {res['pred_log10']:.3f} log₁₀(mW/m²) &nbsp;|&nbsp; 
                <strong>Estimator:</strong> {res['model_name']}
            </div>
            <hr style="margin: 0.75rem 0; border-color: #E2E8F0;">
            <div style="font-size: 0.85rem; color: #64748B;">
                <strong>Scientific Status:</strong> Research Screening Estimate.<br>
                <em>This is an empirical regression projection based on 40 literature observations. It is not an experimentally guaranteed value.</em>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Contextual Spectrum Bar
        val = res["pred_power_density_mW_m2"]
        st.markdown(f"""
        <div style="background: #F1F5F9; border-radius: 8px; padding: 0.75rem; margin-top: 0.5rem; font-size: 0.85rem;">
            <strong>Contextual Benchmark Spectrum:</strong><br>
            • Training Minimum: <code>1.7 mW/m²</code><br>
            • Training Median: <code>110.0 mW/m²</code><br>
            • Naive Mean Baseline: <code>408.2 mW/m²</code><br>
            • Training Maximum: <code>13,000.0 mW/m²</code>
        </div>
        """, unsafe_allow_html=True)

        if selected_row_data:
            st.markdown("#### Historical Record Comparison")
            act = selected_row_data["Actual_Power_Density"]
            oof = selected_row_data["OOF_Power_Density"]
            fin = selected_row_data["Final_Power_Density"]
            used_el = selected_row_data["Used_in_Elec_Train"]

            act_str = f"<b>{round(act, 2)} mW/m²</b>" if pd.notna(act) else "<span style='color:#94A3B8;'>Unmeasured in original study</span>"
            oof_str = f"<b>{round(oof, 2)} mW/m²</b>" if pd.notna(oof) else "<span style='color:#94A3B8;'>Not in training fold</span>"
            fin_str = f"<b>{round(fin, 2)} mW/m²</b>" if pd.notna(fin) else "N/A"
            used_str = "<span style='color:#0D9488;'>Yes (N=40)</span>" if used_el else "<span style='color:#D97706;'>No (Unlabelled Row)</span>"

            st.markdown(f"""
            <div class="sci-card" style="background: #F8FAFC;">
                <strong>Record {selected_row_data['Row_ID']} ({selected_row_data['Study_ID']}):</strong><br>
                • <strong>Experimental Ground Truth:</strong> {act_str}<br>
                • <strong>Honest Out-Of-Fold Prediction:</strong> {oof_str}<br>
                • <strong>Final In-Sample Prediction:</strong> {fin_str}<br>
                • <strong>Used in Model Training:</strong> {used_str}
            </div>
            """, unsafe_allow_html=True)

    with col_diag:
        st.markdown("#### Feature Importance & Sensitivity")
        st.caption("Permutation importance: Average RMSE increase in log₁₀ space when feature values are permuted.")
        imp_df = get_feature_importance_df("electricity")
        fig_imp = plot_feature_importance(imp_df, "Electricity Generation")
        st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("""
        <div class="sci-alert-warning" style="font-size: 0.85rem;">
            <strong>Uncertainty Boundary:</strong><br>
            The model's cross-validated RMSE is <strong>1.02 log₁₀ units</strong>. 
            Because 10<sup>1.02</sup> ≈ 10.5, predictions carry an uncertainty span of up to an order of magnitude. 
            Use predictions to rank configurations rather than sizing electrical infrastructure.
        </div>
        """, unsafe_allow_html=True)
