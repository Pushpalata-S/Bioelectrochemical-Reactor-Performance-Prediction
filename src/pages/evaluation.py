"""Page D: Model Performance, Cross-Study Benchmarking & Data Leakage Diagnostics."""
import pandas as pd
import streamlit as st

from src.data_loader import load_electricity_data, load_wastewater_data
from src.models import (
    get_benchmark_df,
    get_feature_importance_df,
    get_leakage_df,
)
from src.plots import (
    plot_parity,
    plot_residuals,
    plot_benchmark_r2,
    plot_benchmark_rmse,
    plot_feature_importance,
    plot_data_leakage,
)


def render_evaluation_page():
    st.markdown("""
        <div style="margin-bottom: 1.25rem;">
            <span class="sci-badge sci-badge-teal">EMPIRICAL BENCHMARK</span>
            <span class="sci-badge sci-badge-amber">STUDY-GROUPED CROSS-VALIDATION</span>
            <h2 style="color: #0A192F; margin-top: 0.5rem; margin-bottom: 0.25rem;">
                Scientific Model Evaluation & Benchmark Engine
            </h2>
            <p style="color: #64748B;">
                Rigorous multi-algorithm comparison across 11 regression estimators with 10-repeat 5-fold 
                study-grouped validation and empirical data leakage diagnostics.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Critical Methodology Callout
    st.markdown("""
    <div class="sci-alert">
        <strong>Validation Rule Applied:</strong><br>
        Models are evaluated strictly on <strong>Out-Of-Fold (OOF) predictions from Study-Grouped Cross-Validation</strong>. 
        Rows originating from the same research paper / laboratory (<code>CV_Group</code>) were never split between train and test folds. 
        This assesses true generalization to unseen scientific studies rather than memorizing laboratory-specific experimental setups.
    </div>
    """, unsafe_allow_html=True)

    tab_eval_elec, tab_eval_ww, tab_leakage = st.tabs([
        "⚡ Mode 1: Electricity Models Benchmark",
        "💧 Mode 2: Wastewater Models Benchmark",
        "🕵️ Literature Data Leakage Diagnostic",
    ])

    # TAB 1: Electricity Benchmark
    with tab_eval_elec:
        st.markdown("### Electricity Generation: 11-Model Benchmark")
        st.caption("Target: Area-based Power Density (log₁₀ mW/m²). N = 40 labelled rows across 35 independent studies.")
        
        bench_el = get_benchmark_df("electricity")
        
        # Display benchmark metrics table
        display_cols = ["Rank", "Model", "R2_mean", "R2_sd", "RMSE_mean", "MAE_mean", "MAE_physical", "Within_3x_pct", "Status"]
        renamed_cols = {
            "R2_mean": "R² (Mean)",
            "R2_sd": "R² (SD)",
            "RMSE_mean": "RMSE (log₁₀)",
            "MAE_mean": "MAE (log₁₀)",
            "MAE_physical": "MAE (mW/m²)",
            "Within_3x_pct": "Within 3x Factor (%)",
        }
        st.dataframe(bench_el[display_cols].rename(columns=renamed_cols), use_container_width=True, hide_index=True)

        col_r2, col_rmse = st.columns(2)
        with col_r2:
            fig_r2_el = plot_benchmark_r2(bench_el, "Electricity Generation")
            st.plotly_chart(fig_r2_el, use_container_width=True)
        with col_rmse:
            fig_rmse_el = plot_benchmark_rmse(bench_el, "Electricity Generation", "log₁₀ units")
            st.plotly_chart(fig_rmse_el, use_container_width=True)

        st.markdown("---")
        st.markdown("### Parity & Residual Diagnostics (Random Forest)")
        
        df_el = load_electricity_data()
        pred_source_el = st.radio(
            "Evaluation Prediction Set (Electricity):",
            ["Honest Out-Of-Fold Predictions (Study-Grouped CV)", "Final Model In-Sample Predictions (All Labelled)"],
            horizontal=True,
            key="pred_source_el_radio",
        )
        
        col_actual = "Actual_power_density_mW_m2"
        if "Out-Of-Fold" in pred_source_el:
            col_pred = "Predicted_power_density_mW_m2_out_of_fold"
            title_p = "Electricity Parity: Actual vs. Out-Of-Fold Predicted (Honest Cross-Study Generalization)"
            title_r = "Electricity Residuals (log₁₀ Actual - log₁₀ OOF Predicted)"
        else:
            col_pred = "Predicted_power_density_mW_m2_final_model"
            title_p = "Electricity Parity: Actual vs. In-Sample Final Model Predicted (Overly Optimistic)"
            title_r = "Electricity Residuals (log₁₀ Actual - log₁₀ Final Model Predicted)"
            st.warning("⚠️ In-sample predictions evaluate performance on the exact data used for fitting. They do not represent generalization to new literature.")

        pcol1, pcol2 = st.columns(2)
        with pcol1:
            fig_par_el = plot_parity(
                df_el,
                col_actual,
                col_pred,
                title_p,
                "Actual Power Density (mW/m²)",
                "Predicted Power Density (mW/m²)",
                is_log=True,
                task="electricity",
            )
            st.plotly_chart(fig_par_el, use_container_width=True)
        with pcol2:
            fig_res_el = plot_residuals(
                df_el,
                col_actual,
                col_pred,
                title_r,
                "Actual log₁₀(Power Density)",
                "Residual Error (log₁₀ units)",
                is_log=True,
            )
            st.plotly_chart(fig_res_el, use_container_width=True)

    # TAB 2: Wastewater Benchmark
    with tab_eval_ww:
        st.markdown("### Wastewater Treatment: 11-Model Benchmark")
        st.caption("Target: COD Removal Efficiency (%). N = 55 labelled rows across 47 independent studies.")
        
        bench_ww = get_benchmark_df("wastewater")
        display_ww_cols = ["Rank", "Model", "R2_mean", "R2_sd", "RMSE_mean", "MAE_mean", "Within_10pts_pct", "Status"]
        renamed_ww_cols = {
            "R2_mean": "R² (Mean)",
            "R2_sd": "R² (SD)",
            "RMSE_mean": "RMSE (% pts)",
            "MAE_mean": "MAE (% pts)",
            "Within_10pts_pct": "Within ±10% pts (%)",
        }
        st.dataframe(bench_ww[display_ww_cols].rename(columns=renamed_ww_cols), use_container_width=True, hide_index=True)

        col_r2_w, col_rmse_w = st.columns(2)
        with col_r2_w:
            fig_r2_ww = plot_benchmark_r2(bench_ww, "Wastewater Treatment")
            st.plotly_chart(fig_r2_ww, use_container_width=True)
        with col_rmse_w:
            fig_rmse_ww = plot_benchmark_rmse(bench_ww, "Wastewater Treatment", "% pts")
            st.plotly_chart(fig_rmse_ww, use_container_width=True)

        st.markdown("---")
        st.markdown("### Parity & Residual Diagnostics (Gaussian Process)")
        
        df_ww = load_wastewater_data()
        pred_source_ww = st.radio(
            "Evaluation Prediction Set (Wastewater):",
            ["Honest Out-Of-Fold Predictions (Study-Grouped CV)", "Final Model In-Sample Predictions (All Labelled)"],
            horizontal=True,
            key="pred_source_ww_radio",
        )
        
        col_actual_w = "Actual_COD_removal_pct"
        if "Out-Of-Fold" in pred_source_ww:
            col_pred_w = "Predicted_COD_removal_pct_out_of_fold"
            title_p_w = "Wastewater Parity: Actual vs. Out-Of-Fold Predicted (Honest Generalization)"
            title_r_w = "Wastewater Residuals (Actual - OOF Predicted, % pts)"
        else:
            col_pred_w = "Predicted_COD_removal_pct_final_model"
            title_p_w = "Wastewater Parity: Actual vs. In-Sample Final Model Predicted (In-Sample Fit)"
            title_r_w = "Wastewater Residuals (Actual - Final Model Predicted, % pts)"
            st.warning("⚠️ In-sample predictions mask generalization failures by evaluating training data.")

        pcol1_w, pcol2_w = st.columns(2)
        with pcol1_w:
            fig_par_ww = plot_parity(
                df_ww,
                col_actual_w,
                col_pred_w,
                title_p_w,
                "Actual COD Removal (%)",
                "Predicted COD Removal (%)",
                is_log=False,
                task="wastewater",
            )
            st.plotly_chart(fig_par_ww, use_container_width=True)
        with pcol2_w:
            fig_res_ww = plot_residuals(
                df_ww,
                col_actual_w,
                col_pred_w,
                title_r_w,
                "Actual COD Removal (%)",
                "Residual Error (% pts)",
                is_log=False,
            )
            st.plotly_chart(fig_res_ww, use_container_width=True)

    # TAB 3: Data Leakage Diagnostic
    with tab_leakage:
        st.markdown("### 🕵️ Scientific Investigation: The Data Leakage Effect")
        st.markdown("""
        Many published papers report surprisingly high machine learning performance ($R^2 = 0.85\text{--}0.95$) 
        on meta-analyses of Microbial Fuel Cells. 
        
        **How is this possible when primary electrochemical variables are missing?**
        The answer is **Data Leakage from Random Train-Test Splits**:
        - In a random split, rows from the same research paper are distributed into both training and testing sets.
        - The model does not learn universal bioelectrochemical mechanisms; it simply memorizes the specific baseline offset 
          of that laboratory's setup (electrode spacing, pump rate, synthetic recipe).
        - When tested on other rows from the same paper, it achieves an artificially inflated test score.
        """)

        leak_df = get_leakage_df()
        fig_leak = plot_data_leakage(leak_df)
        st.plotly_chart(fig_leak, use_container_width=True)

        st.markdown("#### Quantitative Empirical Demonstration")
        st.dataframe(leak_df, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="sci-alert-warning">
            <strong>Key Finding:</strong><br>
            • For Electricity with Random Forest: R² rose from <strong>+0.049</strong> to <strong>+0.133</strong> (+171% artificial boost).<br>
            • For Electricity with XGBoost: R² flipped from <strong>-0.045</strong> to <strong>+0.117</strong>.<br>
            • For Wastewater with Gaussian Process: R² flipped from <strong>-0.091</strong> to <strong>+0.007</strong>, deceptively appearing to beat the baseline.<br>
            This proves that high scores reported in literature without study grouping are methodological artifacts.
        </div>
        """, unsafe_allow_html=True)
