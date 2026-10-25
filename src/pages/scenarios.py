"""Page F: Reactor Design Scenario Explorer & Multi-Objective Comparison."""
import pandas as pd
import streamlit as st

from src.data_loader import get_feature_domains
from src.models import predict_scenario_batch
from src.plots import plot_scenario_tradeoff


DEFAULT_SCENARIOS = [
    {
        "Scenario_Name": "Scenario 1: Brewery Wastewater (Tubular AC)",
        "Wastewater_Type": "Brewery wastewater",
        "MFC_Configuration": "Tubular",
        "Anode_Material": "Graphite felt",
        "Cathode_Material": "Carbon cloth with activated carbon catalyst",
        "Initial_COD_mg_L": 2500.0,
        "HRT_h": 48.0,
    },
    {
        "Scenario_Name": "Scenario 2: Swine Waste (Single-Chamber Pt)",
        "Wastewater_Type": "Swine wastewater",
        "MFC_Configuration": "Single-chamber",
        "Anode_Material": "Carbon felt",
        "Cathode_Material": "Carbon paper with Pt catalyst",
        "Initial_COD_mg_L": 12000.0,
        "HRT_h": 72.0,
    },
    {
        "Scenario_Name": "Scenario 3: Domestic Sewage (Dual-Chamber)",
        "Wastewater_Type": "Domestic wastewater",
        "MFC_Configuration": "Dual-chamber",
        "Anode_Material": "Carbon brush",
        "Cathode_Material": "Carbon cloth with Pt catalyst",
        "Initial_COD_mg_L": 450.0,
        "HRT_h": 12.0,
    },
    {
        "Scenario_Name": "Scenario 4: Synthetic Acetate (Nutrient-Recovery)",
        "Wastewater_Type": "Synthetic wastewater",
        "MFC_Configuration": "Nutrient-recovery cell (CEM+AEM, multi-chamber)",
        "Anode_Material": "Carbon brush (heat-treated)",
        "Cathode_Material": "Carbon cloth with Pt catalyst",
        "Initial_COD_mg_L": 600.0,
        "HRT_h": 16.5,
    },
]


def render_scenarios_page():
    st.markdown("""
        <div style="margin-bottom: 1.25rem;">
            <span class="sci-badge sci-badge-teal">DECISION SUPPORT</span>
            <span class="sci-badge sci-badge-amber">MULTI-SCENARIO PROFILING</span>
            <h2 style="color: #0A192F; margin-top: 0.5rem; margin-bottom: 0.25rem;">
                Reactor Design Scenario Explorer
            </h2>
            <p style="color: #64748B;">
                Simultaneously project electricity generation and wastewater COD removal across candidate reactor configurations 
                to compare trade-offs and guide experimental pilot design.
            </p>
        </div>
    """, unsafe_allow_html=True)

    domains = get_feature_domains()

    # Session State for Scenarios
    if "scenarios_list" not in st.session_state:
        st.session_state["scenarios_list"] = [s.copy() for s in DEFAULT_SCENARIOS]

    st.markdown("### 1. Configure Candidate Reactor Scenarios")

    # Add custom scenario expander
    with st.expander("➕ Define New Custom Scenario", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            new_name = st.text_input("Scenario Name:", value=f"Custom Design {len(st.session_state['scenarios_list']) + 1}")
            new_ww = st.selectbox("Wastewater Substrate:", domains["Wastewater_Type"], key="new_ww")
            new_anode = st.selectbox("Anode Material:", domains["Anode_Material"], key="new_anode")
            new_cod = st.number_input("Initial COD (mg/L):", min_value=1.0, max_value=100000.0, value=1000.0, step=50.0, key="new_cod")
        with c2:
            new_cfg = st.selectbox("MFC Configuration:", domains["MFC_Configuration"], key="new_cfg")
            new_cathode = st.selectbox("Cathode Material & Catalyst:", domains["Cathode_Material"], key="new_cathode")
            new_hrt = st.number_input("HRT (hours):", min_value=0.1, max_value=1000.0, value=24.0, step=1.0, key="new_hrt")

        if st.button("Add Scenario to Comparison Table", type="primary"):
            st.session_state["scenarios_list"].append({
                "Scenario_Name": new_name,
                "Wastewater_Type": new_ww,
                "MFC_Configuration": new_cfg,
                "Anode_Material": new_anode,
                "Cathode_Material": new_cathode,
                "Initial_COD_mg_L": new_cod,
                "HRT_h": new_hrt,
            })
            st.success(f"Added '{new_name}' to scenario roster!")
            st.rerun()

    # Reset button
    col_reset, _ = st.columns([1, 4])
    with col_reset:
        if st.button("🔄 Reset to Default 4 Scenarios"):
            st.session_state["scenarios_list"] = [s.copy() for s in DEFAULT_SCENARIOS]
            st.rerun()

    # Batch prediction calculation
    scenarios_df = pd.DataFrame(st.session_state["scenarios_list"])
    results_df = predict_scenario_batch(scenarios_df)

    # Ranking & Multi-objective sorting
    st.markdown("### 2. Comparative Scenario Evaluation")

    sort_criteria = st.selectbox(
        "Rank Scenarios By:",
        [
            "⚡ Highest Predicted Power Density (mW/m²)",
            "💧 Highest Predicted COD Removal (%)",
            "⚖️ Balanced Score (Normalized 50% Power + 50% COD)",
        ],
        index=0,
    )

    # Compute ranking
    if "Highest Predicted Power Density" in sort_criteria:
        ranked_df = results_df.sort_values("Pred_Power_Density_mW_m2", ascending=False).reset_index(drop=True)
    elif "Highest Predicted COD Removal" in sort_criteria:
        ranked_df = results_df.sort_values("Pred_COD_Removal_pct", ascending=False).reset_index(drop=True)
    else:
        # Normalized balanced score
        p_norm = (results_df["Pred_Power_Density_mW_m2"] - results_df["Pred_Power_Density_mW_m2"].min()) / (
            results_df["Pred_Power_Density_mW_m2"].max() - results_df["Pred_Power_Density_mW_m2"].min() + 1e-6
        )
        c_norm = (results_df["Pred_COD_Removal_pct"] - results_df["Pred_COD_Removal_pct"].min()) / (
            results_df["Pred_COD_Removal_pct"].max() - results_df["Pred_COD_Removal_pct"].min() + 1e-6
        )
        results_df["Balanced_Score"] = (0.5 * p_norm + 0.5 * c_norm).round(3)
        ranked_df = results_df.sort_values("Balanced_Score", ascending=False).reset_index(drop=True)

    ranked_df.insert(0, "Rank", range(1, len(ranked_df) + 1))

    # Display clean table
    show_cols = [
        "Rank",
        "Scenario_Name",
        "Pred_Power_Density_mW_m2",
        "Pred_COD_Removal_pct",
        "Wastewater_Type",
        "MFC_Configuration",
        "Anode_Material",
        "Cathode_Material",
        "Initial_COD_mg_L",
        "HRT_h",
    ]
    renamed = {
        "Pred_Power_Density_mW_m2": "Pred. Power (mW/m²)",
        "Pred_COD_Removal_pct": "Pred. COD Rem. (%)",
        "Initial_COD_mg_L": "COD (mg/L)",
        "HRT_h": "HRT (h)",
    }
    st.dataframe(ranked_df[show_cols].rename(columns=renamed), use_container_width=True, hide_index=True)

    # Visual Trade-Off Chart
    st.markdown("### 3. Power Generation vs. COD Removal Trade-off Map")
    fig_tradeoff = plot_scenario_tradeoff(ranked_df)
    st.plotly_chart(fig_tradeoff, use_container_width=True)

    # Engineering Limitations Disclaimer
    st.markdown("""
    <div class="sci-alert-warning">
        <strong>⚠️ Chemical Engineering Design Limitations:</strong><br>
        1. <strong>Causality vs Correlation:</strong> Higher predicted power does not establish that a given material caused the improvement; high performance in literature is often co-linear with unmeasured parameters such as small electrode spacing or high buffer conductivity.<br>
        2. <strong>Capital & Operating Economics:</strong> Platinum catalysts typically deliver high power densities in predictions, but have prohibitive capital expenditure compared to activated carbon.<br>
        3. <strong>Ohmic Losses:</strong> Real-world reactors require electrolyte conductivity management to prevent excessive internal resistance.
    </div>
    """, unsafe_allow_html=True)
