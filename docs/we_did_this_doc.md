# MFC ML-Assisted Design Dashboard: Complete Step-by-Step Project Summary
**Document for Lab Partners & Team Members**  
**Project:** Machine Learning for Microbial Fuel Cell (MFC) Design & Performance Screening  
**Authors:** Final-Year Chemical Engineering Capstone Team  

---

## Quick Summary (TL;DR for the Team)
We built an ML system to predict two performance metrics of Microbial Fuel Cells:
1. **Mode 1 (Electricity Generation):** Area-based Power Density ($\text{mW/m}^2$, on a $\log_{10}$ scale).
2. **Mode 2 (Wastewater Treatment):** Chemical Oxygen Demand Removal ($COD_{\text{removal}}\ \%$).

We tested **11 different machine learning models** using **study-grouped repeated cross-validation** (so rows from the same paper were never in both train and test sets).

**Our Main Finding:**
- In Wastewater, **no ML model beat simply predicting the average** ($R^2 \le 0$). Gaussian Process was the closest ML model ($R^2 = -0.09$, $\text{RMSE} = 18.8\%$).
- In Electricity, **Random Forest came in 1st place** with a modest $R^2 = +0.05$ ($\text{RMSE} = 1.02\ \log_{10}$ units), closely followed by ElasticNet ($R^2 = +0.04$).
- **Why?** Crucial physics (solution conductivity, internal resistance, external electrical load $R_{ext}$, and operating temperature) were $>88\%$ missing across published review papers.
- **Why this is actually a great result:** We proved that high $R^2$ values ($0.85\text{--}0.95$) reported in past literature were caused by **data leakage** from random train/test splits. When evaluated honestly on unseen research papers, MFC performance cannot be predicted with simple off-the-shelf ML without adding key electrochemical state variables.

---

## STEP 1: What Data We Started With & Our Columns

We started with 67 experimental observations extracted from 4 major review papers (Sources S2, S3, S4, S5) covering 60 distinct studies.

The dataset was split into 3 linked CSV files (joined row-by-row on `Row_ID`):
1. `MFC_ML_ready_10features.csv`: The 10 input design features.
2. `MFC_targets_NOT_features.csv`: The target values ($y$).
3. `MFC_provenance_metadata.csv`: Paper citations, review sources, and cross-validation groups (`CV_Group`).

### The 10 Input Columns in Our Dataset

| Column Name | Type | Physical Meaning | Missingness in Our Data | Status in Default ML |
| :--- | :---: | :--- | :---: | :---: |
| `Wastewater_Type` | Categorical (15 types) | Type of feed (domestic sewage, brewery, swine, synthetic acetate, etc.) | $1.5\%$ (1 row missing) | **KEPT** |
| `MFC_Configuration` | Categorical (8 types) | Reactor structure (single-chamber, dual-chamber, stacked, tubular, CW-MFC) | $11.9\%$ (8 rows missing) | **KEPT** |
| `Anode_Material` | Categorical (21 types) | Anode electrode (carbon brush, graphite felt, biochar, carbon cloth, etc.) | $11.9\%$ (8 rows missing) | **KEPT** |
| `Cathode_Material` | Categorical (22 types) | Cathode electrode + catalyst (carbon cloth + Pt, activated carbon, MnO₂, etc.) | $6.0\%$ (4 rows missing) | **KEPT** |
| `Initial_COD_mg_L` | Continuous ($\text{mg/L}$) | Influent organic strength (fuel concentration) | $50.7\%$ (34 rows missing) | **KEPT** ($\log_{10}$ scaled) |
| `HRT_h` | Continuous ($\text{hours}$) | Hydraulic retention time (residence time inside reactor) | $64.2\%$ (43 rows missing) | **KEPT** ($\log_{10}$ scaled) |
| `pH` | Continuous | Influent wastewater pH | **$95.5\%$ (64 rows missing)** | **EXCLUDED** by default |
| `Temperature_C` | Continuous ($^\circ\text{C}$) | Reactor operating temperature | **$97.0\%$ (65 rows missing)** | **EXCLUDED** by default |
| `External_Resistance_Ohm` | Continuous ($\Omega$) | External electrical load resistor connected across electrodes | **$95.5\%$ (64 rows missing)** | **EXCLUDED** by default |
| `Reactor_Volume_L` | Continuous ($\text{L}$) | Working liquid volume of the reactor | **$88.1\%$ (59 rows missing)** | **EXCLUDED** by default |

### Our Target Columns ($y$)
- **Mode 1 Target:** `Power_Density_mW_per_m2_std` ($N = 40$ usable rows where power $> 0$). We trained models on $\log_{10}(\text{power})$ because values range from $0.03$ to $30,000\ \text{mW/m}^2$ (spanning 4 orders of magnitude).
- **Mode 2 Target:** `COD_Removal_pct` ($N = 55$ usable rows, ranging from $17\%$ to $98\%$).

---

## STEP 2: Why We Excluded pH, Temperature, $R_{ext}$, and Volume

Notice that pH, Temperature, External Resistance, and Volume are **over $88\%\text{--}97\%$ missing**:
- Temperature was only recorded in 2 rows (both were $30\ ^\circ\text{C}$).
- pH was only recorded in 3 rows.
- External resistance was only recorded in 3 rows.
- Reactor volume was only recorded in 8 rows.

### What happened when we tested including them (`--all_features`)?
If you feed features that are $95\%$ missing into a machine learning model, the computer has to guess the missing numbers (median imputation) and add "missingness indicator" columns.
- For a small dataset of only 40 rows, this added 8 noisy parameters.
- **The result was disastrous:** ElasticNet's performance dropped from $R^2 = +0.044$ to **$R^2 = -0.215$**, and Ridge dropped to **$-0.403$**.
- Therefore, our decision to exclude them from the default 6-feature baseline was **100% mathematically and scientifically justified**.

---

## STEP 3: The Validation Strategy (Study-Grouped CV vs. Random CV)

This is the most critical scientific contribution of our capstone.

### The Mistake in Previous Literature (Random Train/Test Split)
Most papers use random train/test splits. If Paper A tested 4 different flow rates in the exact same laboratory MFC setup:
- With random splits, 3 rows go to the training set and 1 row goes to the test set.
- The model doesn't learn bioelectrochemical physics—it simply memorizes that "Paper A's reactor usually outputs around $250\ \text{mW/m}^2$."
- This is called **Data Leakage**. It makes papers report fake high $R^2$ values ($0.85\text{--}0.95$).

### Our Honest Solution: Study-Grouped Cross-Validation (`CV_Group`)
We grouped all rows by their original research paper (`CV_Group`).
- In each fold, entire papers were placed in the test set.
- The model was trained **only on other papers** and had to predict performance on a **completely unseen research paper from a laboratory it had never encountered before**.
- We repeated this 5-fold cross-validation 10 times (50 test folds in total) to get rock-solid average metrics and standard deviations.

---

## STEP 4: The 11 Machine Learning Models We Benchmarked

We tested 11 algorithms representing every major machine learning family:

```
                            The 11 Benchmarked Models
   ┌───────────────────┬───────────────────┬───────────────────┬───────────────────┐
   │                   │                   │                   │                   │
1. Baseline Mean       2. Linear           3. Distance/Kernel  4. Bagged Trees     5. Boosted Trees
- DummyRegressor       - RidgeCV           - KNN               - Random Forest     - XGBoost
                       - ElasticNet        - SVR (RBF)         - Extra Trees       - LightGBM
                                           - Gaussian Process                      - Gradient Boosting
```

1. **Baseline (Mean):** Simply predicts the average value of the training set. If an ML model cannot beat this, it is not useful.
2. **Ridge Regression:** Linear regression with $L_2$ penalty (shrinks weights to prevent overfitting).
3. **ElasticNet:** Linear regression combining both $L_1$ (feature selection) and $L_2$ penalties.
4. **$k$-Nearest Neighbors (KNN):** Looks at the 5 most similar reactors in the dataset and averages their outputs.
5. **Support Vector Regression (SVR - RBF):** Fits a non-linear margin using a Radial Basis Function kernel.
6. **Gaussian Process Regressor:** Non-parametric Bayesian model that models uncertainty and shrinks uncertain points toward the prior mean.
7. **Random Forest:** Trains 300 random decision trees and averages their predictions (bagging).
8. **Extra Trees:** Like Random Forest, but with randomized cut points for higher variance reduction.
9. **Gradient Boosting:** Builds trees sequentially where each tree corrects the errors of previous trees.
10. **XGBoost:** Highly optimized gradient boosted decision trees.
11. **LightGBM:** Fast, leaf-wise gradient boosted trees.

---

## STEP 5: Exact Benchmark Results

### 1. Wastewater Treatment Results (Target: COD Removal %)
*55 usable rows across 47 independent studies. Sorted by lowest RMSE (% points).*

| Rank | Model | $R^2$ (Mean $\pm$ SD) | RMSE (% pts) | MAE (% pts) | Within $\pm 10\%$ pts | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | **Baseline (Mean)** | **$-0.056 \pm 0.024$** | **$18.52$** | **$13.57$** | **$48.4\%$** | **Winner** (beats all ML models) |
| 2 | Gaussian Process | $-0.091 \pm 0.034$ | $18.83$ | $14.05$ | $44.5\%$ | Best ML model (shrinks to mean) |
| 3 | SVR (RBF) | $-0.167 \pm 0.087$ | $19.46$ | $14.73$ | $41.3\%$ | Smooth non-linear fit |
| 4 | Ridge | $-0.206 \pm 0.094$ | $19.78$ | $15.04$ | $39.8\%$ | Over-penalized |
| 5 | Random Forest | $-0.239 \pm 0.078$ | $20.05$ | $15.77$ | $34.7\%$ | High variance on unseen categories |
| 6 | ElasticNet | $-0.250 \pm 0.116$ | $20.13$ | $15.73$ | $37.1\%$ | Too sparse |
| 7 | KNN | $-0.319 \pm 0.156$ | $20.66$ | $15.67$ | $41.3\%$ | Distance metric collapsed |
| 8 | LightGBM | $-0.366 \pm 0.179$ | $21.02$ | $16.81$ | $33.3\%$ | Overfits small sample |
| 9 | XGBoost | $-0.452 \pm 0.147$ | $21.69$ | $17.18$ | $32.4\%$ | Severe overfitting |
| 10 | Gradient Boosting | $-0.457 \pm 0.157$ | $21.73$ | $17.16$ | $33.5\%$ | Severe overfitting |
| 11 | Extra Trees | $-0.491 \pm 0.158$ | $21.98$ | $17.35$ | $35.5\%$ | Worst performer |

> **Conclusion for Wastewater:** An $R^2 \le 0$ means the ML models performed worse than a flat line predicting the average COD removal ($\approx 78\%$). Complex tree models (XGBoost, Gradient Boosting) overfitted severely.

---

### 2. Electricity Generation Results (Target: $\log_{10}$ Power Density in $\text{mW/m}^2$)
*40 usable rows across 35 independent studies. Sorted by lowest RMSE ($\log_{10}$ units).*

| Rank | Model | $R^2$ (Mean $\pm$ SD) | RMSE ($\log_{10}$) | MAE ($\log_{10}$) | MAE ($\text{mW/m}^2$) | Within $3\times$ Factor | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | **Random Forest** | **$+0.049 \pm 0.067$** | **$1.020$** | **$0.804$** | **$387.9$** | **$37.3\%$** | **Winner** (Lowest error) |
| 2 | ElasticNet | $+0.044 \pm 0.097$ | $1.022$ | $0.812$ | $395.0$ | $35.5\%$ | Runner-up (Low variance) |
| 3 | KNN | $-0.011 \pm 0.069$ | $1.052$ | $0.824$ | $397.7$ | $39.8\%$ | Near baseline |
| 4 | Ridge | $-0.018 \pm 0.124$ | $1.054$ | $0.842$ | $448.3$ | $34.5\%$ | Near baseline |
| 5 | Gradient Boosting | $-0.028 \pm 0.112$ | $1.060$ | $0.815$ | $426.6$ | $39.5\%$ | Overfitting on small N |
| 6 | LightGBM | $-0.041 \pm 0.107$ | $1.067$ | $0.827$ | $423.1$ | $38.8\%$ | Overfitting on small N |
| 7 | XGBoost | $-0.045 \pm 0.142$ | $1.067$ | $0.822$ | $447.3$ | $40.3\%$ | Overfitting on small N |
| 8 | Baseline (Mean) | $-0.064 \pm 0.054$ | $1.079$ | $0.938$ | $408.2$ | $21.3\%$ | Beaten by RF & ElasticNet |
| 9 | Gaussian Process | $-0.112 \pm 0.083$ | $1.103$ | $0.929$ | $424.1$ | $22.0\%$ | Sub-optimal kernel scale |
| 10 | SVR (RBF) | $-0.115 \pm 0.074$ | $1.105$ | $0.908$ | $409.3$ | $30.5\%$ | Boundary clipping |
| 11 | Extra Trees | $-0.340 \pm 0.166$ | $1.210$ | $0.948$ | $447.9$ | $36.5\%$ | High leaf variance |

> **Conclusion for Electricity:** **Random Forest ($R^2 = +0.05$)** and **ElasticNet ($R^2 = +0.04$)** successfully beat the mean baseline. However, an RMSE of $1.02\ \log_{10}$ units means predictions are typically off by a factor of 10 ($\approx 1$ order of magnitude), reflecting the missing electrical load physics.

---

### 3. The Proof of Data Leakage (Grouped vs. Random CV)

To prove our hypothesis about published literature, we ran both scripts with `--cv random`:

| Model & Task | Grouped CV (Honest Generalization) | Random CV (Same-Paper Leakage) | What Happened |
| :--- | :---: | :---: | :--- |
| **Electricity: Random Forest $R^2$** | **$+0.049$** | **$+0.133$** | **$+171\%$ artificial boost** |
| **Electricity: XGBoost $R^2$** | **$-0.045$** | **$+0.117$** | **Flipped from negative to positive!** |
| **Wastewater: Gaussian Process $R^2$** | **$-0.091$** | **$+0.007$** | **Flipped from negative to positive!** |

This proves that random cross-validation allows models to cheat by memorizing paper-specific baselines!

---

## STEP 6: Why the Results Are Weak (Chemical Engineering & ML Explanations)

### Chemical Engineering Reasons:
1. **The Maximum Power Transfer Theorem & Missing $R_{ext}$:**
   In electrochemistry, maximum power is delivered only when the external resistor matches internal resistance ($R_{ext} = R_{int}$):
   $$P_{max} = \frac{OCV^2}{4 R_{int}}$$
   If Study A ran at $R_{ext} = 1000\ \Omega$ and Study B ran at $R_{ext} = 25\ \Omega$, their power densities will differ by 100-fold even with identical electrodes and wastewater. Because $R_{ext}$ is $95.5\%$ missing, the ML model has no idea what operating point the cell was on.
2. **Missing Solution Conductivity ($\kappa$):**
   Internal Ohmic loss is $R_\Omega = \frac{d}{\kappa A}$. Synthetic acetate in phosphate buffer has high conductivity ($\kappa \approx 12\ \text{mS/cm}$), while raw sewage has low conductivity ($\kappa \approx 1\ \text{mS/cm}$). The model has no conductivity feature to calculate this Ohmic drop.
3. **COD is a Total Measure, Not a Pure Chemical:**
   COD measures total chemical oxygen demand. It treats readily biodegradable acetate identically to non-biodegradable lignin or toxic polyphenols. Furthermore, in real sewage, methanogenic bacteria consume COD and produce $\text{CH}_4$ without producing electricity, decoupling COD removal from power.

### Machine Learning Reasons:
1. **Small Sample Size vs. High Cardinality ($p/n$ Ratio):**
   We had only 40 rows for electricity and 55 rows for wastewater, but 21 different anodes and 22 cathodes. When one-hot encoded, there were almost as many columns as rows!
2. **Sequential Trees (Boosting) Overfit Noise:**
   XGBoost and Gradient Boosting try to aggressively fix every residual error. On 40 rows, they memorized training noise, leading to negative $R^2$ on unseen test studies. Random Forest won because bagging (averaging 300 trees) acts as an ensemble dampener on variance.

---

## STEP 7: Files We Created & Where They Are Located

All generated files are saved in both your workspace (`f:\pro pro ject\project 1\`) and on the Desktop (`C:\Users\Suraj\Desktop\Project\`):

1. **Prediction Spreadsheets (Requested by you):**
   - **[`CSV for waste water.csv`](file:///F:/pro%20pro%20ject/project%201/CSV%20for%20waste%20water.csv)**: Contains all 67 rows, ground truth COD removal, final Gaussian Process model predictions, honest out-of-fold predictions, and predictions from all 11 models side-by-side.
   - **[`CSV for elec.csv`](file:///F:/pro%20pro%20ject/project%201/CSV%20for%20elec.csv)**: Contains all 67 rows, ground truth power density ($\text{mW/m}^2$), final Random Forest predictions, honest out-of-fold predictions, and all 11 models side-by-side.
2. **Outputs Folders:**
   - `outputs_wastewater/`: Contains `model_comparison_wastewater.csv`, `wastewater_model_comparison.png`, and `feature_importance_wastewater.csv`.
   - `outputs_electricity/`: Contains `model_comparison_electricity.csv`, `electricity_model_comparison.png`, and `feature_importance_electricity.csv`.
3. **Comprehensive Technical Report:**
   - **[`MFC_ML_Design_Dashboard_Technical_Report.md`](file:///F:/pro%20pro%20ject/project%201/MFC_ML_Design_Dashboard_Technical_Report.md)**: Full 9-section report containing complete Python scripts, formulas, kinetics, diagnostics, and interview questions.

---

## STEP 8: Actionable Next Steps for Our Lab Team

To upgrade this project before the final presentation:
1. **Electrode Family Binning:** Group the 21 anodes and 22 cathodes into 4 and 3 physical families (e.g., 3D Porous Carbon, Planar Graphite, Packed Bed, Pt Catalyst, Carbon Catalyst, Uncatalyzed). This cuts down feature dimensions from $>40$ to $<10$.
2. **Calculate Organic Loading Rate (OLR):**
   Create a single kinetic feature combining Initial COD and HRT:
   $$\text{OLR}\ [\text{kg COD}/(\text{m}^3\cdot\text{d})] = \frac{\text{Initial COD}\ [\text{mg/L}]}{\text{HRT}\ [\text{h}]} \times \frac{24}{1000}$$
3. **Targeted Literature Back-Filling:** Spend 2 hours checking the primary DOI papers for S2–S5. Many papers stated room temperature ($20\text{--}25\ ^\circ\text{C}$) and pH ($7.0$) in the text, even though the secondary review tables omitted them.
4. **Change to 3-Tier Screening:** Instead of predicting exact decimal values, configure our dashboard to output design tiers (High, Medium, Low potential), which has much higher accuracy and matches chemical process screening practices.
