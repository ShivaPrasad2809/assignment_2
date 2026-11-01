# Condition Monitoring of Hydraulic Systems — Transient Stability Classification

**BITS Pilani Work Integrated Learning Programmes (WILP)**  
**M.Tech (AIML) — Machine Learning Assignment 2**

---

## Mandatory Submission Links
1. **GitHub Repository Link:** [GitHub Repository](https://github.com/ShivaPrasad2809/assignment_2)
2. **Live Streamlit App Link:** [Live Streamlit App](https://assignment2-9xyjfcmnyreaj2s562mznq.streamlit.app/)
---

## a. Problem Statement
In industrial fluid power and heavy machinery circuits, distinguishing between steady-state equilibrium and transient thermo-hydraulic instability is critical for reliable fault diagnostics and predictive maintenance. During load transitions, fluid shearing, proportional valve switching, and accumulator pre-charge dynamics introduce pressure pulsations and thermal drift that can mask or mimic component degradation.

The objective of this project is to build, evaluate, and deploy an end-to-end machine learning classification pipeline that predicts whether an industrial hydraulic test rig is operating in a **stable steady-state regime (`Stable_Flag = 0`)** or an **unstable transient regime (`Stable_Flag = 1`)** using cyclic multi-sensor statistical and higher-order moment features (Mean, Standard Deviation, and Fisher Kurtosis) extracted across 17 physical and virtual sensors.

---

## b. Dataset Description
* **Dataset Name:** Condition Monitoring of Hydraulic Systems (UCI Machine Learning Repository — ID: 447)
* **Source URL:** https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems
* **Domain:** Industrial Fluid Power, Vibro-Thermal Condition Monitoring & Mechatronics
* **Number of Instances (Cycles):** **2,205 instances** (exceeding the minimum requirement of 500 instances)
  * **Training Set (80% Stratified Split):** 1,764 cycles
  * **Holdout Test Set (`test_data.csv`, 20% Stratified Split):** 441 cycles
* **Number of Features:** **51 engineered numerical features** (exceeding the minimum requirement of 12 features)
* **Target Variable (`Stable_Flag`):** Binary classification target derived from `profile.txt` (Column 5):
  * `0` — Stable steady-state operating regime (1,449 cycles, 65.71%)
  * `1` — Static conditions not yet reached / unstable transient regime (756 cycles, 34.29%)

### Sensor Suite & Cyclic Feature Engineering
The experimental hydraulic test rig records 2,205 load cycles, each lasting 60 seconds, across **17 physical and virtual sensors** sampled at three frequencies:
1. **High-Frequency Dynamics (100 Hz — 6,000 samples/cycle):** Six pressure transducers (`PS1`–`PS6`, in bar) and one motor electrical power sensor (`EPS1`, in Watts).
2. **Medium-Frequency Dynamics (10 Hz — 600 samples/cycle):** Two volumetric flow rate meters (`FS1`, `FS2`, in L/min).
3. **Quasi-Static & RMS Dynamics (1 Hz — 60 samples/cycle):** Four oil thermocouples (`TS1`–`TS4`, in °C), one mechanical housing vibration sensor (`VS1`, in mm/s), cooling efficiency (`CE`, in %), cooling power (`CP`, in kW), and system efficiency factor (`SE`, in %).

To capture both baseline operating points and non-stationary transient spikes within each 60-second cycle, three statistical moments were extracted for every sensor (17 sensors × 3 statistics = 51 features):
* **Cycle Mean (`_mean`):** Captures the quasi-static equilibrium level of pressure, temperature, flow, and vibration.
* **Cycle Standard Deviation (`_std`):** Captures RMS dynamic fluctuation and within-cycle thermal or pressure drift.
* **Cycle Fisher Kurtosis (`_kurt`):** Captures fourth-order impulsiveness (tailedness/spikiness) caused by proportional valve switching shocks and hydraulic cavitation transients. Zero-variance constant thermal windows are mapped cleanly to `0.0`, and all 51 features are standardized via `StandardScaler` inside each classifier pipeline.

---

## c. GitHub Repository Link & Structure
* **GitHub Repository URL:** `<INSERT_YOUR_GITHUB_REPOSITORY_URL_HERE>`
* **Live Streamlit Cloud URL:** `<INSERT_YOUR_STREAMLIT_CLOUD_APP_URL_HERE>`

### Repository File Structure
* `app.py` — Interactive Streamlit web application for model evaluation on holdout/uploaded test data
* `train_models.py` — Automated UCI data ingestion, 51-feature engineering, and model training pipeline
* `requirements.txt` — Python package dependencies required for Streamlit Community Cloud deployment
* `README.md` — Project documentation, metrics comparison table, and model performance observations
* `test_data.csv` — 20% stratified holdout test dataset (441 rows × 52 columns, including `Stable_Flag`)
* `model/` — Directory containing the 5 serialized pre-trained `scikit-learn` pipelines:
  * `model/logistic_regression.joblib`
  * `model/decision_tree.joblib`
  * `model/knn.joblib`
  * `model/naive_bayes.joblib`
  * `model/svm.joblib`
  * `model/metrics_summary.csv`

---

## d. Models Used & Evaluation Metrics Comparison Table

All five classification models were trained on the same 80% stratified training split (1,764 cycles) on the BITS Virtual Lab and evaluated on the 20% holdout test dataset (`test_data.csv`, 441 cycles):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9501 | 0.9813 | 0.9216 | 0.9338 | 0.9276 | 0.8896 |
| **Decision Tree** | **0.9705** | 0.9695 | 0.9539 | 0.9603 | **0.9571** | **0.9347** |
| **KNN** | 0.9297 | 0.9696 | 0.9412 | 0.8477 | 0.8920 | 0.8426 |
| **Naive Bayes** | 0.4762 | 0.8430 | 0.3942 | **0.9868** | 0.5633 | 0.2673 |
| **Support Vector Machine** | 0.9546 | **0.9869** | **0.9852** | 0.8808 | 0.9301 | 0.8997 |

---

## e. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved strong baseline performance (**Accuracy: 0.9501, AUC: 0.9813, MCC: 0.8896**). After `StandardScaler` normalization, a linear hyperplane in the 51-dimensional feature space separates steady-state thermal equilibrium (`TS1`–`TS4` means) from transient warm-up regimes effectively, though it slightly underfits non-linear valve-switching interactions compared to tree and RBF-kernel methods. |
| **Decision Tree** | Delivered the highest overall classification accuracy (**0.9705**), F1 Score (**0.9571**), and Matthews Correlation Coefficient (**0.9347**), with well-balanced Precision (**0.9539**) and Recall (**0.9603**). Because hydraulic stability is governed by explicit physical regime thresholds (such as oil reservoir temperature `TS1_mean` reaching thermal equilibrium and pressure fluctuation `PS1_std` / `PS1_kurt` settling below transient bounds), axis-aligned recursive partitioning captures these piecewise non-linear boundaries directly without being degraded by collinear sensor channels. |
| **KNN** | Reached solid Precision (**0.9412**) and AUC (**0.9696**), but exhibited lower Recall (**0.8477**) and MCC (**0.8426**) relative to SVM and Decision Tree. In a 51-dimensional space ($d = 51$), Euclidean distance metrics begin to experience the curse of dimensionality, and fourth-order kurtosis spikes slightly stretch local neighborhood distances for borderline transient cycles. |
| **Naive Bayes** | Recorded the lowest Accuracy (**0.4762**) and MCC (**0.2673**) despite achieving the highest Recall (**0.9868**) and a moderate ranking AUC (**0.8430**). This behavior stems from two first-principles physical violations: (1) **Strong Multicollinearity**—in a closed hydraulic loop, pressure (`PS1`–`PS6`), flow (`FS1`–`FS2`), motor power (`EPS1`), and heat exchanger temperatures (`TS1`–`TS4`, `CE`, `CP`) are tightly coupled by fluid mechanics and thermodynamics, violating Naive Bayes' conditional independence assumption and causing it to double-count correlated evidence toward the unstable class (`1`); and (2) **Non-Gaussian Kurtosis Tails**—heavy-tailed kurtosis features deviate from the univariate Gaussian likelihood assumption. |
| **Support Vector Machine** | Achieved the highest **ROC-AUC (0.9869)** and highest **Precision (0.9852)** across all five classifiers, along with the second-highest Accuracy (**0.9546**) and MCC (**0.8997**). The Radial Basis Function (RBF) kernel maps the coupled thermo-hydraulic and impulsive kurtosis features into a non-linear feature space with strong L2 regularization ($C = 3.0$), yielding near-zero false positive alarms when flagging unstable operating cycles. |
| **Overall, Winner for your dataset?** | **Decision Tree Classifier** is the overall winner for discrete deployment classification (**Accuracy: 0.9705, F1: 0.9571, MCC: 0.9347**), closely followed by **Support Vector Machine (RBF)** which leads in probability ranking (**AUC: 0.9869**) and false-alarm suppression (**Precision: 0.9852**). Decision Tree wins overall because its hierarchical split rules naturally mirror the physical threshold regimes of hydraulic temperature stabilization and pressure/kurtosis damping while remaining immune to multi-sensor collinearity. |