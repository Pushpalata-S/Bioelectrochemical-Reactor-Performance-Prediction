# Bioelectrochemical Reactor Performance Prediction Dashboard

[![Python Version](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/Scikit--Learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![Visualization](https://img.shields.io/badge/Plotly-Interactive-teal.svg)](https://plotly.com/)
[![Scientific Validation](https://img.shields.io/badge/Validation-Study--Grouped%20CV-success.svg)](#scientific-validation-strategy)

An interactive, scientific machine learning dashboard for predicting the dual performance metrics of **Microbial Fuel Cells (MFCs)** and **Bioelectrochemical Systems (BES)**:
1. **Mode 1 (Electricity Generation):** Area-based Power Density ($P_{\text{area}}$, $\text{mW/m}^2$, on a $\log_{10}$ scale).
2. **Mode 2 (Wastewater Treatment):** Chemical Oxygen Demand Removal Efficiency ($COD_{\text{removal}}\ \%$).

Built as a chemical engineering capstone analytics platform, this dashboard benchmarks **11 machine learning algorithms** using **study-grouped repeated cross-validation** to evaluate cross-study generalization and uncover the unmeasured electrochemical variables governing bioelectrochemical reactor physics.

---

## 🔬 Scientific Background & Core Findings

### 1. What Are Microbial Fuel Cells (MFCs)?
Microbial Fuel Cells are bioelectrochemical bioreactors where exoelectrogenic microorganisms (e.g., *Geobacter sulfurreducens*, *Shewanella oneidensis*) colonize an anode electrode, oxidizing organic compounds in wastewater (measured as Chemical Oxygen Demand, COD) and transferring metabolic electrons directly to the electrode surface. Electrons flow through an external electrical load to the cathode, where terminal electron acceptors (typically dissolved $\text{O}_2$) undergo catalytic reduction:
$$\text{Anode:}\quad \text{CH}_3\text{COO}^- + 2\text{H}_2\text{O} \xrightarrow{\text{Biofilm}} 2\text{CO}_2 + 7\text{H}^+ + 8e^-$$
$$\text{Cathode (ORR):}\quad \text{O}_2 + 4\text{H}^+ + 4e^- \xrightarrow{\text{Catalyst}} 2\text{H}_2\text{O}$$

### 2. The Core Scientific Finding: Data Leakage vs. Honest Generalization
Published literature reviews frequently report high ML accuracy ($R^2 = 0.85\text{--}0.95$) by applying random train-test splits across aggregated datasets. However, our benchmark proves that **random splits cause severe data leakage**:
- When multiple data points from the same published paper are split randomly, the model memorizes laboratory-specific setup parameters (e.g., specific electrode spacing, pumping rates, baseline conductivity) rather than universal bioelectrochemical mechanisms.
- Under **Study-Grouped Cross-Validation** (where all observations from the same paper are held out in the test fold):
  - **Wastewater Treatment:** No ML model beats simply predicting the naive mean ($R^2 \le 0$). The best ML model is **Gaussian Process** ($R^2 = -0.09$, $\text{RMSE} = 18.82\%$).
  - **Electricity Generation:** **Random Forest** achieves modest predictive power ($R^2 = +0.05$, $\text{RMSE} = 1.02\ \log_{10}$ units), followed by **ElasticNet** ($R^2 = +0.04$).

### 3. Why Is Cross-Study Generalization Weak?
The models do not generalize strongly because the primary governing electrochemical state variables are **$>88\%$ missing** across published review papers:
- **External Load Resistance ($R_{ext}$, 95.5% missing):** Governs cell voltage according to the Maximum Power Transfer Theorem ($P_{\text{max}}$ occurs when $R_{ext} = R_{int}$).
- **Electrolyte Conductivity ($\kappa$, unmeasured):** Determines internal Ohmic resistance ($R_\Omega = d / \kappa A$).
- **Temperature ($97.0\%$ missing) & pH ($95.5\%$ missing):** Govern Arrhenius kinetics and proton concentration gradients.

---

## 📊 Empirical Benchmark Results

### Mode 1: Electricity Generation (Area-Based Power Density, $\log_{10}\ \text{mW/m}^2$)
*Evaluated across $N = 40$ labelled rows from 35 independent studies via 10-repeat 5-fold Study-Grouped CV:*

| Rank | Estimator | $R^2$ (Mean $\pm$ SD) | RMSE ($\log_{10}$) | MAE ($\log_{10}$) | MAE ($\text{mW/m}^2$) | Within $3\times$ Factor |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **Random Forest** | **$+0.049 \pm 0.067$** | **$1.020$** | **$0.804$** | **$387.9$** | **$37.3\%$** |
| 2 | ElasticNet | $+0.044 \pm 0.097$ | $1.022$ | $0.812$ | $395.0$ | $35.5\%$ |
| 3 | KNN | $-0.011 \pm 0.069$ | $1.052$ | $0.824$ | $397.7$ | $39.8\%$ |
| 4 | Ridge | $-0.018 \pm 0.124$ | $1.054$ | $0.842$ | $448.3$ | $34.5\%$ |
| 5 | Gradient Boosting | $-0.028 \pm 0.112$ | $1.060$ | $0.815$ | $426.6$ | $39.5\%$ |
| 6 | LightGBM | $-0.041 \pm 0.107$ | $1.067$ | $0.827$ | $423.1$ | $38.8\%$ |
| 7 | XGBoost | $-0.045 \pm 0.142$ | $1.067$ | $0.822$ | $447.3$ | $40.3\%$ |
| 8 | Baseline (Mean) | $-0.064 \pm 0.054$ | $1.079$ | $0.938$ | $408.2$ | $21.3\%$ |
| 9 | Gaussian Process | $-0.112 \pm 0.083$ | $1.103$ | $0.929$ | $424.1$ | $22.0\%$ |
| 10 | SVR (RBF) | $-0.115 \pm 0.074$ | $1.105$ | $0.908$ | $409.3$ | $30.5\%$ |
| 11 | Extra Trees | $-0.340 \pm 0.166$ | $1.210$ | $0.948$ | $447.9$ | $36.5\%$ |

### Mode 2: Wastewater Treatment (COD Removal Efficiency, $\%$)
*Evaluated across $N = 55$ labelled rows from 47 independent studies via 10-repeat 5-fold Study-Grouped CV:*

| Rank | Estimator | $R^2$ (Mean $\pm$ SD) | RMSE (% pts) | MAE (% pts) | Within $\pm 10\%$ pts |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **Baseline (Mean)** | **$-0.056 \pm 0.024$** | **$18.520$** | **$13.574$** | **$48.4\%$** |
| 2 | Gaussian Process | $-0.091 \pm 0.034$ | $18.825$ | $14.054$ | $44.5\%$ |
| 3 | SVR (RBF) | $-0.167 \pm 0.087$ | $19.456$ | $14.727$ | $41.3\%$ |
| 4 | Ridge | $-0.206 \pm 0.094$ | $19.778$ | $15.039$ | $39.8\%$ |
| 5 | Random Forest | $-0.239 \pm 0.078$ | $20.048$ | $15.769$ | $34.7\%$ |
| 6 | ElasticNet | $-0.250 \pm 0.116$ | $20.130$ | $15.728$ | $37.1\%$ |
| 7 | KNN | $-0.319 \pm 0.156$ | $20.664$ | $15.674$ | $41.3\%$ |
| 8 | LightGBM | $-0.366 \pm 0.179$ | $21.019$ | $16.812$ | $33.3\%$ |
| 9 | XGBoost | $-0.452 \pm 0.147$ | $21.692$ | $17.180$ | $32.4\%$ |
| 10 | Gradient Boosting | $-0.457 \pm 0.157$ | $21.728$ | $17.155$ | $33.5\%$ |
| 11 | Extra Trees | $-0.491 \pm 0.158$ | $21.976$ | $17.346$ | $35.5\%$ |

---

## 🖥️ Dashboard Architecture & Pages

The application is structured modularly under `src/` with 6 dedicated interactive views:

```
Bioelectrochemical Reactor Performance Prediction/
├── app.py                             # Main Streamlit application entry point & router
├── requirements.txt                   # Dependency specifications
├── README.md                          # Comprehensive scientific & technical documentation
├── data/
│   ├── CSV for elec.csv               # Electricity dataset (67 rows, ground truth, OOF & final predictions)
│   └── CSV for waste water.csv        # Wastewater dataset (67 rows, ground truth, OOF & final predictions)
├── docs/
│   ├── MFC_ML_Design_Dashboard_Technical_Report.md
│   └── we_did_this_doc.md
├── src/
│   ├── __init__.py
│   ├── config.py                      # Palette, reference benchmarks, CSS styling
│   ├── data_loader.py                 # Cached loading, feature domains, missingness
│   ├── models.py                      # Pipelines, estimators, cached inference routines
│   ├── plots.py                       # Reusable scientific Plotly chart generators
│   └── pages/
│       ├── __init__.py
│       ├── overview.py                # Page A: Project Overview & Scientific Architecture
│       ├── electricity.py             # Page B: Electricity Generation Prediction
│       ├── wastewater.py              # Page C: Wastewater Treatment Prediction
│       ├── evaluation.py              # Page D: Model Performance & Cross-Study Evaluation
│       ├── explorer.py                # Page E: Interactive Dataset Explorer
│       └── scenarios.py               # Page F: Reactor Design Scenario Explorer
└── tests/
    └── test_dashboard.py              # Comprehensive automated unit & pipeline test suite
```

### Page Breakdown:
1. **🏛️ Project Overview (Page A):** Scientific foundations, reactor mechanics, dataset KPIs (67 records, 60 studies, 40 electricity labels, 55 wastewater labels, 32 co-labelled rows), missingness profile heatmap, and the end-to-end workflow sequence.
2. **⚡ Electricity Generation Prediction (Page B):** Real-time prediction of area-based power density ($\text{mW/m}^2$) via Random Forest with contextual spectrum bars, historical record autofill (R001–R068), and permutation feature importance.
3. **💧 Wastewater Treatment Prediction (Page C):** Real-time prediction of COD removal percentage ($\%$) via Gaussian Process with physical bounds $[0\%, 100\%]$, historical record autofill, and generalization transparency callouts.
4. **📈 Model Benchmarking (Page D):** Multi-algorithm benchmark comparison tables and charts across all 11 algorithms, parity plots comparing Out-Of-Fold vs. In-Sample predictions, residual plots, and the interactive **Data Leakage Diagnostic**.
5. **🔬 Dataset Explorer (Page E):** Multi-parametric filtering across wastewater types, configurations, electrode materials, training status, and COD ranges; search and sort; distribution histograms; bivariate scatter plots; and one-click filtered CSV export.
6. **🧪 Scenario Comparison (Page F):** Multi-scenario design exploration with 4 pre-configured chemical engineering benchmarks, custom scenario additions, dual-target batch inference, multi-objective ranking, and trade-off scatter mapping.

---

## 🚀 Installation & Local Execution

### 1. Prerequisites
- Python 3.10+ (Tested on Python 3.11)
- Git

### 2. Clone Repository & Setup Environment
```bash
git clone https://github.com/Pushpalata-S/Bioelectrochemical-Reactor-Performance-Prediction.git
cd "Bioelectrochemical Reactor Performance Prediction"

# Create a virtual environment (optional but recommended)
python -m venv .venv

# Activate environment:
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 5. Run the Automated Test Suite
To verify that all loaders, pipelines, predictions, and charts operate with zero errors:
```bash
python tests/test_dashboard.py
```

---

## ⚙️ Technical Highlights
- **Zero Hallucination Guarantee:** The dashboard uses exact empirical results, out-of-fold predictions, and ground-truth records directly from `data/CSV for elec.csv` and `data/CSV for waste water.csv`.
- **Reproducible Pipelines:** Final models replicate documented model outputs with `0.0` maximum absolute difference.
- **Efficient Caching:** Utilizes `@st.cache_data` and `@st.cache_resource` to prevent redundant model fitting or file reading during user navigation.
- **Scientific Visual Theme:** Uses a restrained color palette (deep navy `#0A192F`, teal `#0D9488`, slate `#64748B`, light gray `#F8FAFC`) with standard deviation error bars, 1:1 diagonal lines, and factor error bounds.

---

## 📜 Scientific & Engineering Disclaimer
Predictions generated by this dashboard are **exploratory research screening estimates** designed to assist chemical engineers and bioelectrochemical researchers in prioritizing candidate reactor architectures. They do not constitute guaranteed experimental outputs, causal physical proofs, or commercial reactor specifications. Physical benchtop and pilot prototyping remain mandatory.
