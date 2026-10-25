"""Page C: Wastewater Treatment Prediction (COD Removal Percentage, %)."""
import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import get_feature_domains, load_wastewater_data
from src.models import (
    predict_wastewater_single,
    get_historical_row_data,
    get_feature_importance_df,
)
from src.plots import plot_feature_importance


def render_wastewater_page():
    st.markdown("""
        <div style="margin-bottom: 1.25rem;">
            <span class="sci-badge sci-badge-blue">MODE 2: WASTEWATER TREATMENT</span>
            <span class="sci-badge sci-badge-amber">TARGET: COD REMOVAL (%)</span>
            <h2 style="color: #0A192F; margin-top: 0.5rem; margin-bottom: 0.25rem;">
                Wastewater Treatment Prediction Interface
            </h2>
            <p style="color: #64748B;">
                Predict Chemical Oxygen Demand (COD) removal efficiency ($COD_{\text{removal}}\\%$) using the 
                Gaussian Process Regressor ($R^2 = -0.09$, $\\text{RMSE} = 18.82\\%\\ \\text{pts}$).
            </p>
        </div>
    """, unsafe_allow_html=True)

    domains = get_feature_domains()
    df_ww = load_wastewater_data()

    # Input Mode Selector
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
    default_cod = 450.0
    default_hrt = 12.0
    selected_row_data = None

    if view_mode.startswith("📂"):
        st.markdown("#### Select Historical Reactor Row")
        row_options = df_ww["Row_ID"].tolist()
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
            default_cod = float(selected_row_data["Initial_COD_mg_L"]) if pd.notna(selected_row_data["Initial_COD_mg_L"]) else 450.0
            default_hrt = float(selected_row_data["HRT_h"]) if pd.notna(selected_row_data["HRT_h"]) else 12.0

    st.markdown("### 1. Reactor Operating Specifications")
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
                help="Influent organic strength. Higher concentrations typically correlate with higher absolute removal but lower retention efficiencies.",
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
                help="Contact duration of wastewater within the reactor. Log10 transformed by pipeline.",
            )

    # Scientific Caution Callout
    st.markdown("""
    <div class="sci-alert-warning">
        <strong>⚠️ Model Generalization Limitation Notice:</strong><br>
        Under rigorous study-grouped 5-fold cross-validation, <strong>no machine learning model beat simply predicting the dataset mean (78.5%)</strong>. 
        The Gaussian Process Regressor achieved the lowest RMSE (18.82% pts) among ML models by shrinking uncertain predictions towards the global prior mean. 
        Estimates should be treated strictly as screening heuristics, not precision kinetic simulations.
    </div>
    """, unsafe_allow_html=True)

    # Prediction Section
    st.markdown("### 2. Treatment Performance Prediction")
    
    col_pred, col_diag = st.columns([1.1, 0.9])
    
    with col_pred:
        res = predict_wastewater_single(
            wastewater_type=ww_input,
            configuration=cfg_input,
            anode_material=anode_input,
            cathode_material=cathode_input,
            initial_cod=cod_input,
            hrt=hrt_input,
        )

        st.markdown(f"""
        <div class="sci-card" style="border-left: 4px solid #2563EB;">
            <div class="metric-label">Predicted COD Removal Efficiency</div>
            <div class="metric-value" style="color: #2563EB;">
                {res['pred_cod_removal_pct']:.2f}%
            </div>
            <div class="metric-sub" style="margin-top: 0.4rem;">
                <strong>Raw Model Output:</strong> {res['raw_pred']:.2f}% &nbsp;|&nbsp; 
                <strong>Estimator:</strong> {res['model_name']}
            </div>
            <hr style="margin: 0.75rem 0; border-color: #E2E8F0;">
            <div style="font-size: 0.85rem; color: #64748B;">
                <strong>Physical Bounds:</strong> Enforced in [0%, 100%].<br>
                <strong>Evaluation Benchmark:</strong> Study-Grouped RMSE = 18.8% pts, MAE = 14.1% pts.<br>
                <em>Naive Mean Baseline comparison: 18.52% pts RMSE.</em>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if selected_row_data:
            st.markdown("#### Historical Record Comparison")
            act = selected_row_data["Actual_COD_Removal"]
            oof = selected_row_data["OOF_COD_Removal"]
            fin = selected_row_data["Final_COD_Removal"]
            used_ww = selected_row_data["Used_in_WW_Train"]

            act_str = f"<b>{round(act, 2)}%</b>" if pd.notna(act) else "<span style='color:#94A3B8;'>Unmeasured in original study</span>"
            oof_str = f"<b>{round(oof, 2)}%</b>" if pd.notna(oof) else "<span style='color:#94A3B8;'>Not in training fold</span>"
            fin_str = f"<b>{round(fin, 2)}%</b>" if pd.notna(fin) else "N/A"
            used_str = "<span style='color:#0D9488;'>Yes (N=55)</span>" if used_ww else "<span style='color:#D97706;'>No (Unlabelled Row)</span>"

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
        st.markdown("#### Permutation Feature Importance")
        st.caption("Average increase in test RMSE (% points) when feature column is randomly shuffled.")
        imp_df = get_feature_importance_df("wastewater")
        fig_imp = plot_feature_importance(imp_df, "Wastewater Treatment (COD Removal)")
        st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("""
        <div class="sci-card" style="background: #F8FAFC; font-size: 0.85rem; color: #334155;">
            <strong>Why did complex tree algorithms overfit?</strong><br>
            With 55 training observations and over 40 one-hot encoded categories, gradient boosting algorithms 
            (XGBoost, Gradient Boosting) aggressively partitioned noise in the training set, causing test R² to drop to -0.45. 
            The Gaussian Process regressor regularized variance by shrinking test predictions toward the prior mean.
        </div>
        """, unsafe_allow_html=True)
