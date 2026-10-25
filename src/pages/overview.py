"""Page A: Project Overview & Scientific Architecture."""
import streamlit as st
import pandas as pd
from src.data_loader import get_dataset_statistics, get_merged_dataset
from src.plots import plot_missingness
from src.config import ALL_FEATURES_IMPACT_DATA


def render_overview_page():
    st.markdown("""
        <div style="margin-bottom: 1.5rem;">
            <span class="sci-badge sci-badge-teal">CAPSTONE PROJECT</span>
            <span class="sci-badge sci-badge-blue">BIOELECTROCHEMICAL SYSTEMS</span>
            <span class="sci-badge sci-badge-amber">STUDY-GROUPED BENCHMARK</span>
            <h1 style="color: #0A192F; margin-top: 0.5rem; margin-bottom: 0.25rem;">
                Bioelectrochemical Reactor Performance Prediction
            </h1>
            <p style="color: #64748B; font-size: 1.05rem;">
                Machine learning-assisted design and screening platform for Microbial Fuel Cells (MFCs).
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Scientific Introduction
    with st.container():
        st.markdown("""
        <div class="sci-card">
            <h3 style="color: #0A192F; margin-top: 0;">🔬 Scientific Introduction</h3>
            <p style="color: #334155; line-height: 1.6;">
                <strong>Microbial Fuel Cells (MFCs)</strong> are bioelectrochemical bioreactors that harness the metabolic 
                machinery of electrochemically active bacteria (exoelectrogens, such as <em>Geobacter</em> and <em>Shewanella</em>) 
                to catalyze organic substrate oxidation at the anode while simultaneously reducing oxygen at the cathode. 
                This dual capability enables <strong>simultaneous wastewater organic decontamination and direct, carbon-neutral electricity recovery</strong>.
            </p>
            <p style="color: #334155; line-height: 1.6; margin-bottom: 0;">
                However, translating laboratory findings into engineering-scale designs is hindered by extreme heterogeneity across published literature. 
                This platform implements a <strong>scientifically validated, dual-mode machine learning benchmark</strong> to evaluate predictive feasibility 
                and uncover the electrochemical state variables required for cross-study generalization.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Live Dataset KPI Metrics
    stats = get_dataset_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="sci-card" style="text-align: center;">
            <div class="metric-label">Total Experimental Rows</div>
            <div class="metric-value">{stats['total_records']}</div>
            <div class="metric-sub">{stats['unique_studies']} Independent Literature Studies</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="sci-card" style="text-align: center;">
            <div class="metric-label">Electricity Labelled</div>
            <div class="metric-value" style="color: #0D9488;">{stats['elec_labelled']}</div>
            <div class="metric-sub">{stats['elec_unlabelled']} Unlabelled (Power > 0)</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="sci-card" style="text-align: center;">
            <div class="metric-label">Wastewater Labelled</div>
            <div class="metric-value" style="color: #2563EB;">{stats['ww_labelled']}</div>
            <div class="metric-sub">{stats['ww_unlabelled']} Unlabelled (COD Rem. %)</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="sci-card" style="text-align: center;">
            <div class="metric-label">Dual-Labelled Rows</div>
            <div class="metric-value" style="color: #D97706;">{stats['both_labelled']}</div>
            <div class="metric-sub">Co-occurring Targets</div>
        </div>
        """, unsafe_allow_html=True)

    # Core Scientific Finding Alert
    st.markdown("""
    <div class="sci-alert-warning">
        <strong>⚠️ Crucial Scientific Finding & Generalization Warning:</strong><br>
        Standard random train-test splitting causes severe <strong>data leakage</strong> across literature benchmarks, artificially inflating reported R² scores (+171%). 
        When evaluated honestly with <strong>Study-Grouped Cross-Validation</strong> (where entire research papers are held out), 
        <strong>no ML model outperforms the naive mean baseline for wastewater treatment (R² ≤ 0)</strong>, 
        and <strong>Random Forest yields modest predictive power for electricity (R² ≈ 0.05)</strong>. 
        This is physically dictated by >88% missingness in primary electrochemical variables (external load resistance R<sub>ext</sub>, solution conductivity, and internal resistance). 
        All predictions must be interpreted strictly as <em>exploratory screening estimates</em>.
    </div>
    """, unsafe_allow_html=True)

    # Workflow Architecture
    st.markdown("### 🔄 End-to-End Analytics & Prediction Workflow")
    st.markdown("""
    ```
    ┌─────────────────────────┐       ┌─────────────────────────┐       ┌─────────────────────────┐
    │  Literature Extraction  │       │ Preprocessing & Scaling │       │  Study-Grouped 5-Fold   │
    │  67 Observations        │ ────► │ - log10(COD), log10(HRT)│ ────► │  Repeated Cross-Val.    │
    │  Sources S2 - S5        │       │ - OHE (min_freq = 3)    │       │  Honest Generalization  │
    └─────────────────────────┘       └─────────────────────────┘       └─────────────────────────┘
                                                                                      │
                                                                                      ▼
    ┌─────────────────────────┐       ┌─────────────────────────┐       ┌─────────────────────────┐
    │  Reactor Design Screens │       │ Dual In-Sample Models   │       │ Multi-Algorithm Bench.  │
    │  - Comparative Rankings │ ◄──── │ - RF (Power Density)    │ ◄──── │ - 11 ML Estimators      │
    │  - Trade-off Analysis   │       │ - GP (COD Removal %)    │       │ - Leakage Diagnostics   │
    └─────────────────────────┘       └─────────────────────────┘       └─────────────────────────┘
    ```
    """)

    # Tabbed Explorations: Feature Space & Missingness
    tab_features, tab_missingness = st.tabs(["📋 Design Feature Architecture", "📊 Data Completeness & Excluded Physics"])

    with tab_features:
        st.markdown("#### Input Features & Cardinalities in the Curated Dataset")
        feat_col1, feat_col2 = st.columns(2)
        with feat_col1:
            st.markdown("""
            **Categorical Features (Kept in Baseline Model):**
            - **Wastewater Type** (15 observed classes): Domestic sewage, brewery, swine, synthetic acetate, dairy, landfill leachate, urine, etc.
            - **MFC Configuration** (8 observed classes): Single-chamber, Dual-chamber, Tubular, Photosynthetic/algal, Stacked, CW-MFC, etc.
            - **Anode Material** (21 observed classes): Carbon brush, Graphite felt, Carbon cloth, Biochar, Stainless steel mesh, etc.
            - **Cathode Material** (22 observed classes): Carbon cloth + Pt catalyst, Activated Carbon, Carbon brush, Stainless steel, etc.
            """)
        with feat_col2:
            st.markdown("""
            **Continuous Features (Kept in Baseline Model):**
            - **Initial COD** ($\text{mg/L}$): Organic substrate loading concentration ($50.7\%$ missing in literature, $\log_{10}$ transformed).
            - **HRT** ($\text{hours}$): Hydraulic retention / contact time ($64.2\%$ missing in literature, $\log_{10}$ transformed).
            
            **Dual Mode Targets:**
            - **Mode 1 Target:** Area-based Power Density ($\text{mW/m}^2$, $\log_{10}$ transformed; $N=40$ labelled).
            - **Mode 2 Target:** Chemical Oxygen Demand Removal ($COD_{\text{removal}}\ \%$, physical bounds $[0, 100]$; $N=55$ labelled).
            """)

    with tab_missingness:
        st.markdown("#### Missingness Profile Across Features")
        st.markdown("""
            Published literature reviews frequently omit key physical operating parameters. 
            Below is the empirical missingness distribution across the dataset:
        """)
        fig_miss = plot_missingness(stats["missingness_df"])
        st.plotly_chart(fig_miss, use_container_width=True)

        st.markdown("#### Why were pH, Temperature, $R_{ext}$, and Volume excluded?")
        st.markdown("""
        When benchmarked with all 10 features (`--all_features`), median imputation across $95\%$ missing columns 
        introduced severe synthetic noise, deteriorating model generalizability:
        """)
        impact_df = pd.DataFrame(ALL_FEATURES_IMPACT_DATA)
        st.dataframe(impact_df, use_container_width=True, hide_index=True)
