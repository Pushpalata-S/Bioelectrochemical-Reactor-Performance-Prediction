"""Configuration constants, color palettes, and benchmark reference data."""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ELEC_DATA_PATH = DATA_DIR / "CSV for elec.csv"
WW_DATA_PATH = DATA_DIR / "CSV for waste water.csv"
DOCS_DIR = BASE_DIR / "docs"

# Feature definitions matching existing ML pipelines
CAT_FEATURES = ["Wastewater_Type", "MFC_Configuration", "Anode_Material", "Cathode_Material"]
NUM_FEATURES = ["Initial_COD_mg_L", "HRT_h"]
LOG10_COLS = ["Initial_COD_mg_L", "HRT_h"]
EXTRA_FEATURES = ["pH", "Temperature_C", "External_Resistance_Ohm", "Reactor_Volume_L"]

TARGET_ELEC = "Actual_power_density_mW_m2"
TARGET_WW = "Actual_COD_removal_pct"

# Physical Bounds
ELEC_LOG_MIN = -1.5
ELEC_LOG_MAX = 4.5
ELEC_VAL_MIN = 10 ** ELEC_LOG_MIN   # ~0.0316 mW/m2
ELEC_VAL_MAX = 10 ** ELEC_LOG_MAX   # 31622.77 mW/m2

WW_MIN_PCT = 0.0
WW_MAX_PCT = 100.0

# Scientific Color Palette
COLORS = {
    "navy_dark": "#0A192F",
    "navy_card": "#1E293B",
    "navy_border": "#334155",
    "teal_primary": "#0D9488",
    "teal_light": "#14B8A6",
    "teal_bright": "#2DD4BF",
    "teal_glow": "rgba(20, 184, 166, 0.15)",
    "slate_bg": "#F8FAFC",
    "slate_light": "#F1F5F9",
    "slate_border": "#E2E8F0",
    "slate_text": "#64748B",
    "text_dark": "#0F172A",
    "amber": "#F59E0B",
    "red": "#EF4444",
    "green": "#10B981",
    "blue": "#3B82F6",
}

# Empirical Benchmark Results for Electricity Generation (Study-Grouped CV)
ELEC_BENCHMARK_DATA = [
    {"Rank": 1, "Model": "Random Forest", "R2_mean": 0.049, "R2_sd": 0.067, "RMSE_mean": 1.020, "MAE_mean": 0.804, "MAE_physical": 387.92, "Within_3x_pct": 37.25, "Status": "Recommended / Best ML"},
    {"Rank": 2, "Model": "ElasticNet", "R2_mean": 0.044, "R2_sd": 0.097, "RMSE_mean": 1.022, "MAE_mean": 0.812, "MAE_physical": 394.96, "Within_3x_pct": 35.50, "Status": "Runner-up"},
    {"Rank": 3, "Model": "KNN", "R2_mean": -0.011, "R2_sd": 0.069, "RMSE_mean": 1.052, "MAE_mean": 0.824, "MAE_physical": 397.66, "Within_3x_pct": 39.75, "Status": "Near Baseline"},
    {"Rank": 4, "Model": "Ridge", "R2_mean": -0.018, "R2_sd": 0.124, "RMSE_mean": 1.054, "MAE_mean": 0.842, "MAE_physical": 448.34, "Within_3x_pct": 34.50, "Status": "Near Baseline"},
    {"Rank": 5, "Model": "Gradient Boosting", "R2_mean": -0.028, "R2_sd": 0.112, "RMSE_mean": 1.060, "MAE_mean": 0.815, "MAE_physical": 426.61, "Within_3x_pct": 39.50, "Status": "Overfitting on small N"},
    {"Rank": 6, "Model": "LightGBM", "R2_mean": -0.041, "R2_sd": 0.107, "RMSE_mean": 1.067, "MAE_mean": 0.827, "MAE_physical": 423.07, "Within_3x_pct": 38.75, "Status": "Overfitting on small N"},
    {"Rank": 7, "Model": "XGBoost", "R2_mean": -0.045, "R2_sd": 0.142, "RMSE_mean": 1.067, "MAE_mean": 0.822, "MAE_physical": 447.26, "Within_3x_pct": 40.25, "Status": "Overfitting on small N"},
    {"Rank": 8, "Model": "Baseline (mean)", "R2_mean": -0.064, "R2_sd": 0.054, "RMSE_mean": 1.079, "MAE_mean": 0.938, "MAE_physical": 408.21, "Within_3x_pct": 21.25, "Status": "Naive Mean"},
    {"Rank": 9, "Model": "Gaussian Process", "R2_mean": -0.112, "R2_sd": 0.083, "RMSE_mean": 1.103, "MAE_mean": 0.929, "MAE_physical": 424.09, "Within_3x_pct": 22.00, "Status": "Suboptimal Kernel Scale"},
    {"Rank": 10, "Model": "SVR (RBF)", "R2_mean": -0.115, "R2_sd": 0.074, "RMSE_mean": 1.105, "MAE_mean": 0.908, "MAE_physical": 409.28, "Within_3x_pct": 30.50, "Status": "Boundary Clipping"},
    {"Rank": 11, "Model": "Extra Trees", "R2_mean": -0.340, "R2_sd": 0.166, "RMSE_mean": 1.210, "MAE_mean": 0.948, "MAE_physical": 447.94, "Within_3x_pct": 36.50, "Status": "High Leaf Variance"},
]

# Empirical Benchmark Results for Wastewater Treatment (Study-Grouped CV)
WW_BENCHMARK_DATA = [
    {"Rank": 1, "Model": "Baseline (mean)", "R2_mean": -0.056, "R2_sd": 0.024, "RMSE_mean": 18.520, "MAE_mean": 13.574, "Within_10pts_pct": 48.36, "Status": "Undefeated Baseline"},
    {"Rank": 2, "Model": "Gaussian Process", "R2_mean": -0.091, "R2_sd": 0.034, "RMSE_mean": 18.825, "MAE_mean": 14.054, "Within_10pts_pct": 44.55, "Status": "Best ML Model (Shrinks to Mean)"},
    {"Rank": 3, "Model": "SVR (RBF)", "R2_mean": -0.167, "R2_sd": 0.087, "RMSE_mean": 19.456, "MAE_mean": 14.727, "Within_10pts_pct": 41.27, "Status": "Smooth Nonlinear Fit"},
    {"Rank": 4, "Model": "Ridge", "R2_mean": -0.206, "R2_sd": 0.094, "RMSE_mean": 19.778, "MAE_mean": 15.039, "Within_10pts_pct": 39.82, "Status": "Over-penalized"},
    {"Rank": 5, "Model": "Random Forest", "R2_mean": -0.239, "R2_sd": 0.078, "RMSE_mean": 20.048, "MAE_mean": 15.769, "Within_10pts_pct": 34.73, "Status": "High Variance on Unseen"},
    {"Rank": 6, "Model": "ElasticNet", "R2_mean": -0.250, "R2_sd": 0.116, "RMSE_mean": 20.130, "MAE_mean": 15.728, "Within_10pts_pct": 37.09, "Status": "Too Sparse"},
    {"Rank": 7, "Model": "KNN", "R2_mean": -0.319, "R2_sd": 0.156, "RMSE_mean": 20.664, "MAE_mean": 15.674, "Within_10pts_pct": 41.27, "Status": "Distance Collapsed"},
    {"Rank": 8, "Model": "LightGBM", "R2_mean": -0.366, "R2_sd": 0.179, "RMSE_mean": 21.019, "MAE_mean": 16.812, "Within_10pts_pct": 33.27, "Status": "Overfits Small N"},
    {"Rank": 9, "Model": "XGBoost", "R2_mean": -0.452, "R2_sd": 0.147, "RMSE_mean": 21.692, "MAE_mean": 17.180, "Within_10pts_pct": 32.36, "Status": "Severe Overfitting"},
    {"Rank": 10, "Model": "Gradient Boosting", "R2_mean": -0.457, "R2_sd": 0.157, "RMSE_mean": 21.728, "MAE_mean": 17.155, "Within_10pts_pct": 33.45, "Status": "Severe Overfitting"},
    {"Rank": 11, "Model": "Extra Trees", "R2_mean": -0.491, "R2_sd": 0.158, "RMSE_mean": 21.976, "MAE_mean": 17.346, "Within_10pts_pct": 35.45, "Status": "Worst Performer"},
]

# Permutation Feature Importance Data (Mean increase in RMSE)
ELEC_FEATURE_IMPORTANCE = [
    {"Feature": "MFC_Configuration", "Importance": 0.310, "SD": 0.046, "Unit": "log10 units"},
    {"Feature": "Cathode_Material", "Importance": 0.201, "SD": 0.045, "Unit": "log10 units"},
    {"Feature": "Wastewater_Type", "Importance": 0.119, "SD": 0.026, "Unit": "log10 units"},
    {"Feature": "Anode_Material", "Importance": 0.117, "SD": 0.034, "Unit": "log10 units"},
    {"Feature": "Initial_COD_mg_L", "Importance": 0.050, "SD": 0.012, "Unit": "log10 units"},
    {"Feature": "HRT_h", "Importance": 0.011, "SD": 0.004, "Unit": "log10 units"},
]

WW_FEATURE_IMPORTANCE = [
    {"Feature": "MFC_Configuration", "Importance": 5.900, "SD": 0.489, "Unit": "% pts"},
    {"Feature": "Cathode_Material", "Importance": 5.655, "SD": 0.635, "Unit": "% pts"},
    {"Feature": "Initial_COD_mg_L", "Importance": 5.523, "SD": 0.429, "Unit": "% pts"},
    {"Feature": "Anode_Material", "Importance": 5.215, "SD": 0.717, "Unit": "% pts"},
    {"Feature": "Wastewater_Type", "Importance": 5.160, "SD": 0.723, "Unit": "% pts"},
    {"Feature": "HRT_h", "Importance": 4.974, "SD": 0.492, "Unit": "% pts"},
]

# Data Leakage Comparison Data (Grouped CV vs Random CV)
LEAKAGE_COMPARISON_DATA = [
    {"Task": "Electricity", "Model": "Random Forest", "Grouped_R2": 0.049, "Random_R2": 0.133, "Delta_R2": 0.084, "Impact": "+171% Artificial Boost"},
    {"Task": "Electricity", "Model": "ElasticNet", "Grouped_R2": 0.044, "Random_R2": 0.117, "Delta_R2": 0.073, "Impact": "+166% Artificial Boost"},
    {"Task": "Electricity", "Model": "XGBoost", "Grouped_R2": -0.045, "Random_R2": 0.117, "Delta_R2": 0.162, "Impact": "Negative to Positive Flip!"},
    {"Task": "Wastewater", "Model": "Gaussian Process", "Grouped_R2": -0.091, "Random_R2": 0.007, "Delta_R2": 0.098, "Impact": "Beats Baseline under Leakage"},
    {"Task": "Wastewater", "Model": "Ridge", "Grouped_R2": -0.206, "Random_R2": -0.038, "Delta_R2": 0.168, "Impact": "Major Artificial Shrinkage Gain"},
]

# High Missingness Feature Impact Comparison (--all_features test)
ALL_FEATURES_IMPACT_DATA = [
    {"Task": "Electricity", "Model": "ElasticNet", "Features_6_R2": 0.044, "Features_10_R2": -0.215, "Drop": "-0.259 (Severe Degradation)"},
    {"Task": "Electricity", "Model": "Ridge", "Features_6_R2": -0.018, "Features_10_R2": -0.403, "Drop": "-0.385 (Severe Degradation)"},
    {"Task": "Wastewater", "Model": "Extra Trees", "Features_6_R2": -0.491, "Features_10_R2": -0.550, "Drop": "-0.059 (Worsened Error)"},
    {"Task": "Wastewater", "Model": "Baseline Mean", "Features_6_R2": -0.056, "Features_10_R2": -0.056, "Drop": "Rank 1 Undefeated"},
]

# Default Custom CSS for Scientific Aesthetic
CUSTOM_CSS = """
<style>
/* Base typography and clean layout */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Header style */
.stAppHeader {
    background-color: transparent;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #0A192F;
    border-right: 1px solid #1E293B;
}

[data-testid="stSidebar"] .stMarkdown, 
[data-testid="stSidebar"] p, 
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: #E2E8F0 !important;
}

[data-testid="stSidebar"] hr {
    border-color: #1E293B;
}

/* Scientific card component */
.sci-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.sci-card:hover {
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
}

.sci-card-navy {
    background: #0F172A;
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    color: #F8FAFC;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
}

/* Pill / Badge */
.sci-badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-radius: 9999px;
    margin-right: 0.5rem;
}

.sci-badge-teal {
    background-color: rgba(13, 148, 136, 0.15);
    color: #0D9488;
    border: 1px solid rgba(13, 148, 136, 0.3);
}

.sci-badge-amber {
    background-color: rgba(245, 158, 11, 0.15);
    color: #D97706;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.sci-badge-blue {
    background-color: rgba(59, 130, 246, 0.15);
    color: #2563EB;
    border: 1px solid rgba(59, 130, 246, 0.3);
}

.sci-badge-green {
    background-color: rgba(16, 185, 129, 0.15);
    color: #059669;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

/* Metric highlight */
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.2;
    color: #0F172A;
}

.metric-label {
    font-size: 0.85rem;
    font-weight: 500;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.metric-sub {
    font-size: 0.8rem;
    color: #94A3B8;
    margin-top: 0.2rem;
}

/* Scientific callout banner */
.sci-alert {
    border-left: 4px solid #0D9488;
    background: #F0FDFA;
    padding: 0.85rem 1.15rem;
    border-radius: 0 8px 8px 0;
    margin: 1rem 0;
    color: #134E4A;
    font-size: 0.9rem;
}

.sci-alert-warning {
    border-left: 4px solid #F59E0B;
    background: #FFFBEB;
    padding: 0.85rem 1.15rem;
    border-radius: 0 8px 8px 0;
    margin: 1rem 0;
    color: #92400E;
    font-size: 0.9rem;
}

/* Dataframe & Table polish */
[data-testid="stDataFrame"] {
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    overflow: hidden;
}
</style>
"""
