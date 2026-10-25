"""Page E: Interactive Dataset Explorer and Visual Analytics."""
import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import get_merged_dataset, load_electricity_data, load_wastewater_data
from src.plots import plot_target_distribution, plot_bivariate
from src.config import CAT_FEATURES, NUM_FEATURES, TARGET_ELEC, TARGET_WW


def render_explorer_page():
    st.markdown("""
        <div style="margin-bottom: 1.25rem;">
            <span class="sci-badge sci-badge-teal">CURATED DATABASE</span>
            <span class="sci-badge sci-badge-blue">67 REACTOR CONFIGURATIONS</span>
            <h2 style="color: #0A192F; margin-top: 0.5rem; margin-bottom: 0.25rem;">
                Interactive Dataset Explorer & Multi-Parametric Profiling
            </h2>
            <p style="color: #64748B;">
                Explore raw literature records, filter across operational features, inspect ground truth vs. model predictions, 
                and export filtered scientific data.
            </p>
        </div>
    """, unsafe_allow_html=True)

    try:
        merged_df = get_merged_dataset()
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        return

    # Dataset View Selection
    dataset_view = st.radio(
        "Select Dataset View:",
        ["🔬 Merged Dual-Target View (67 Reactors)", "⚡ Electricity Generation Only", "💧 Wastewater Treatment Only"],
        horizontal=True,
    )

    if "Electricity" in dataset_view:
        df_display = load_electricity_data()
    elif "Wastewater" in dataset_view:
        df_display = load_wastewater_data()
    else:
        df_display = merged_df.copy()

    # Sidebar / Filter Expanders
    st.markdown("### 1. Multi-Parametric Filtering")
    with st.expander("🔍 Filter Controls (Categorical & Continuous)", expanded=True):
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            all_ww = sorted([str(x) for x in df_display["Wastewater_Type"].dropna().unique()])
            sel_ww = st.multiselect("Wastewater Substrates:", all_ww, default=[])

            all_cfg = sorted([str(x) for x in df_display["MFC_Configuration"].dropna().unique()])
            sel_cfg = st.multiselect("MFC Configurations:", all_cfg, default=[])

        with fcol2:
            all_anode = sorted([str(x) for x in df_display["Anode_Material"].dropna().unique()])
            sel_anode = st.multiselect("Anode Electrode Materials:", all_anode, default=[])

            all_cathode = sorted([str(x) for x in df_display["Cathode_Material"].dropna().unique()])
            sel_cathode = st.multiselect("Cathode Electrode Materials:", all_cathode, default=[])

        with fcol3:
            # Training status filter
            train_col = "Used_in_Electricity_Train" if "Electricity" in dataset_view else ("Used_in_Wastewater_Train" if "Wastewater" in dataset_view else "Has_Both_Labels")
            if train_col in df_display.columns:
                train_opts = st.multiselect(
                    "Training Status:",
                    ["Labelled / In Training Set", "Unlabelled / Missing Ground Truth"],
                    default=[],
                )
            else:
                train_opts = []

            # Range sliders
            cod_min = float(df_display["Initial_COD_mg_L"].dropna().min() or 0.0)
            cod_max = float(df_display["Initial_COD_mg_L"].dropna().max() or 60000.0)
            sel_cod_range = st.slider(
                "Initial COD Range (mg/L):",
                min_value=float(np.floor(cod_min)),
                max_value=float(np.ceil(cod_max)),
                value=(float(np.floor(cod_min)), float(np.ceil(cod_max))),
            )

    # Apply filters
    filtered_df = df_display.copy()
    if sel_ww:
        filtered_df = filtered_df[filtered_df["Wastewater_Type"].astype(str).isin(sel_ww)]
    if sel_cfg:
        filtered_df = filtered_df[filtered_df["MFC_Configuration"].astype(str).isin(sel_cfg)]
    if sel_anode:
        filtered_df = filtered_df[filtered_df["Anode_Material"].astype(str).isin(sel_anode)]
    if sel_cathode:
        filtered_df = filtered_df[filtered_df["Cathode_Material"].astype(str).isin(sel_cathode)]
    
    # Filter COD
    cod_mask = (filtered_df["Initial_COD_mg_L"].isna()) | (
        (filtered_df["Initial_COD_mg_L"] >= sel_cod_range[0]) & (filtered_df["Initial_COD_mg_L"] <= sel_cod_range[1])
    )
    filtered_df = filtered_df[cod_mask]

    # Metrics summary for filtered slice
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.metric("Filtered Records", f"{len(filtered_df)} / {len(df_display)}")
    with mcol2:
        studies = filtered_df["Study_ID"].nunique() if "Study_ID" in filtered_df.columns else "N/A"
        st.metric("Unique Studies", str(studies))
    with mcol3:
        if TARGET_ELEC in filtered_df.columns:
            elec_cnt = filtered_df[TARGET_ELEC].dropna().count()
            st.metric("Electricity Labels", f"{elec_cnt} rows")
        else:
            st.metric("Electricity Labels", "—")
    with mcol4:
        if TARGET_WW in filtered_df.columns:
            ww_cnt = filtered_df[TARGET_WW].dropna().count()
            st.metric("Wastewater Labels", f"{ww_cnt} rows")
        else:
            st.metric("Wastewater Labels", "—")

    # Table View
    st.markdown("### 2. Tabular Data Inspector")
    st.caption("Includes both experimental measurements and model projections. Unmeasured targets are shown as nulls (empty).")

    # Search box
    search_query = st.text_input("Search dataset (Study ID, material keyword, wastewater keyword):", "")
    if search_query:
        query = search_query.lower()
        mask = filtered_df.astype(str).apply(lambda row: query in " ".join(row).lower(), axis=1)
        display_table = filtered_df[mask]
    else:
        display_table = filtered_df

    st.dataframe(display_table, use_container_width=True, height=360)

    # Download Button
    csv_bytes = display_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv_bytes,
        file_name="mfc_filtered_reactor_data.csv",
        mime="text/csv",
        help="Export current filtered view to local CSV file.",
    )

    # Data Visualizations Section
    st.markdown("---")
    st.markdown("### 3. Distribution & Multi-Variable Relationships")
    
    viz_tab1, viz_tab2 = st.tabs(["📊 Target Distributions", "📈 Bivariate Relationships"])

    with viz_tab1:
        vcol1, vcol2 = st.columns(2)
        with vcol1:
            if TARGET_ELEC in filtered_df.columns and filtered_df[TARGET_ELEC].dropna().count() > 0:
                fig_dist_el = plot_target_distribution(
                    filtered_df,
                    TARGET_ELEC,
                    "Electricity: Power Density Distribution (Log Scale)",
                    "Power Density (mW/m²)",
                    is_log=True,
                )
                st.plotly_chart(fig_dist_el, use_container_width=True)
            else:
                st.info("No power density measurements in current filtered selection.")
        with vcol2:
            if TARGET_WW in filtered_df.columns and filtered_df[TARGET_WW].dropna().count() > 0:
                fig_dist_ww = plot_target_distribution(
                    filtered_df,
                    TARGET_WW,
                    "Wastewater: COD Removal Efficiency Distribution",
                    "COD Removal (%)",
                    is_log=False,
                )
                st.plotly_chart(fig_dist_ww, use_container_width=True)
            else:
                st.info("No COD removal measurements in current filtered selection.")

    with viz_tab2:
        bcol1, bcol2 = st.columns(2)
        with bcol1:
            if TARGET_ELEC in filtered_df.columns and "Initial_COD_mg_L" in filtered_df.columns:
                fig_biv1 = plot_bivariate(
                    filtered_df,
                    x_col="Initial_COD_mg_L",
                    y_col=TARGET_ELEC,
                    color_col="MFC_Configuration",
                    title="Initial COD vs. Measured Power Density",
                    x_label="Initial COD (mg/L, log scale)",
                    y_label="Power Density (mW/m², log scale)",
                    log_x=True,
                    log_y=True,
                )
                st.plotly_chart(fig_biv1, use_container_width=True)
        with bcol2:
            if TARGET_WW in filtered_df.columns and "HRT_h" in filtered_df.columns:
                fig_biv2 = plot_bivariate(
                    filtered_df,
                    x_col="HRT_h",
                    y_col=TARGET_WW,
                    color_col="Wastewater_Type",
                    title="Hydraulic Retention Time (HRT) vs. COD Removal",
                    x_label="HRT (hours, log scale)",
                    y_label="COD Removal (%)",
                    log_x=True,
                    log_y=False,
                )
                st.plotly_chart(fig_biv2, use_container_width=True)
