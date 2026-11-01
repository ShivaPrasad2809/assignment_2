import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Hydraulic System Stability Classifier",
    page_icon="⚙️",
    layout="wide"
)

MODEL_DIR = "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}

@st.cache_resource
def load_all_models():
    models = {}
    for name, filename in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            st.error(f"Missing model artifact: {path}. Please run train_models.py first.")
    return models

models = load_all_models()

# Main Title & Overview
st.title("⚙️ Condition Monitoring of Hydraulic Systems")
st.markdown(
    "**M.Tech (AIML) — Machine Learning Assignment 2** | "
    "Evaluating 5 pre-trained classifiers on 51 cyclic statistical & higher-order moment features "
    "(`Mean`, `Std`, `Kurtosis`) to predict hydraulic circuit stability (`Stable_Flag`)."
)
st.markdown("---")

# Sidebar: 1. CSV Upload & Data Source Selection
st.sidebar.header("1. Test Data Upload (CSV)")
uploaded_file = st.sidebar.file_uploader(
    "Upload Holdout Test CSV (with 'Stable_Flag' column)",
    type=["csv"]
)

use_bundled = st.sidebar.checkbox(
    "Or load bundled 'test_data.csv' from repository",
    value=(uploaded_file is None)
)

test_df = None
if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Uploaded custom CSV ({len(test_df)} cycles).")
elif use_bundled:
    if os.path.exists("test_data.csv"):
        test_df = pd.read_csv("test_data.csv")
        st.sidebar.info(f"Using bundled `test_data.csv` ({len(test_df)} cycles).")
    else:
        st.sidebar.error("`test_data.csv` not found. Please upload a CSV or run train_models.py.")

# Sidebar: 2. Model Selection Dropdown
st.sidebar.markdown("---")
st.sidebar.header("2. Model Selection")
selected_model_name = st.sidebar.selectbox(
    "Select Classification Model:",
    list(MODEL_FILES.keys()),
    index=1  # Defaults to Decision Tree (Overall Winner)
)

# Guard Clauses
if test_df is None:
    st.info("👈 Please upload `test_data.csv` using the sidebar CSV uploader (or check the bundled data option) to view predictions.")
    st.stop()

if "Stable_Flag" not in test_df.columns:
    st.error("The uploaded CSV must include all 51 input features and the ground-truth target column named **`Stable_Flag`**.")
    st.stop()

if len(models) < 5:
    st.error("One or more trained models are missing from the `model/` folder.")
    st.stop()

# Separate features and target without retraining
X_test = test_df.drop(columns=["Stable_Flag"])
y_test = test_df["Stable_Flag"].astype(int)

# Compute evaluation metrics across all 5 pre-trained models
comparison_rows = []
model_preds = {}
model_probs = {}

for name, model in models.items():
    y_pred = model.predict(X_test)
    model_preds[name] = y_pred

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
    model_probs[name] = y_prob

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    comparison_rows.append({
        "ML Model Name": name,
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4)
    })

comparison_df = pd.DataFrame(comparison_rows)

# Main View Tabs
tab1, tab2, tab3 = st.tabs([
    "🎯 Selected Model Diagnostics",
    "📊 All 5 Models Comparison Table",
    "🔍 Test Dataset Preview"
])

with tab1:
    st.subheader(f"Evaluation Metrics: {selected_model_name}")

    y_pred_sel = model_preds[selected_model_name]
    y_prob_sel = model_probs[selected_model_name]

    acc_sel = accuracy_score(y_test, y_pred_sel)
    auc_sel = roc_auc_score(y_test, y_prob_sel)
    prec_sel = precision_score(y_test, y_pred_sel, zero_division=0)
    rec_sel = recall_score(y_test, y_pred_sel, zero_division=0)
    f1_sel = f1_score(y_test, y_pred_sel, zero_division=0)
    mcc_sel = matthews_corrcoef(y_test, y_pred_sel)

    # 6 Metric Display Cards
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Accuracy", f"{acc_sel:.4f}")
    c2.metric("AUC Score", f"{auc_sel:.4f}")
    c3.metric("Precision", f"{prec_sel:.4f}")
    c4.metric("Recall", f"{rec_sel:.4f}")
    c5.metric("F1 Score", f"{f1_sel:.4f}")
    c6.metric("MCC Score", f"{mcc_sel:.4f}")

    st.markdown("---")

    # Confusion Matrix & Classification Report Side-by-Side
    col_left, col_right = st.columns([1, 1.15])

    with col_left:
        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred_sel)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Stable (0)", "Unstable (1)"],
            yticklabels=["Stable (0)", "Unstable (1)"],
            annot_kws={"size": 13, "weight": "bold"},
            ax=ax
        )
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        ax.set_title(f"{selected_model_name} — Confusion Matrix", fontsize=12)
        st.pyplot(fig)
        plt.close(fig)

    with col_right:
        st.markdown("#### Classification Report")
        report_dict = classification_report(
            y_test,
            y_pred_sel,
            target_names=["Stable (0)", "Unstable (1)"],
            output_dict=True,
            zero_division=0
        )
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(
            report_df.style.format(precision=4).background_gradient(
                cmap="Blues",
                subset=["precision", "recall", "f1-score"]
            ),
            use_container_width=True
        )

with tab2:
    st.subheader("All 5 Models Performance on Test Data")
    st.markdown("Metrics calculated dynamically on the loaded test dataset without retraining:")
    st.dataframe(
        comparison_df.style.format({
            "Accuracy": "{:.4f}",
            "AUC": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1": "{:.4f}",
            "MCC": "{:.4f}"
        }).highlight_max(
            subset=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
            color="#d4edda",
            axis=0
        ),
        use_container_width=True
    )

with tab3:
    st.subheader("Uploaded / Bundled Test Data Preview")
    st.markdown(f"**Dimensions:** `{test_df.shape[0]}` instances (cycles) $\\times$ `{test_df.shape[1]}` columns (`51` features + `Stable_Flag` target).")
    st.dataframe(test_df.head(25), use_container_width=True)