import os
import ssl
import urllib.request
import zipfile
import joblib
import numpy as np
import pandas as pd
from scipy.stats import kurtosis

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# 1. Configuration & Custom Seed for Reproducibility
RANDOM_SEED = 5569
RAW_DATA_DIR = "raw_hydraulic_data"
MODEL_DIR = "model"

# Verified UCI ID 447 for Condition Monitoring of Hydraulic Systems
UCI_ZIP_URLS = [
    "https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00447/data.zip"
]

SENSORS = [
    "PS1", "PS2", "PS3", "PS4", "PS5", "PS6",
    "EPS1", "FS1", "FS2",
    "TS1", "TS2", "TS3", "TS4",
    "VS1", "CE", "CP", "SE"
]

def download_file_with_headers(url, dest_path):
    """Downloads a file using a browser User-Agent header to bypass UCI 403/404 blocks."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }
    )
    with urllib.request.urlopen(req, context=ctx, timeout=60) as response, open(dest_path, "wb") as out_file:
        while True:
            chunk = response.read(1024 * 64)
            if not chunk:
                break
            out_file.write(chunk)

def locate_sensor_files(base_dir):
    """Finds where profile.txt was extracted (in case the zip had a subfolder)."""
    for root, _, files in os.walk(base_dir):
        if "profile.txt" in files:
            return root
    return None

def download_and_extract_uci_data():
    """Downloads and extracts UCI Dataset 447 if not already present."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    existing_dir = locate_sensor_files(RAW_DATA_DIR)
    if existing_dir is not None:
        print(f"[INFO] Raw hydraulic sensor files found locally in '{existing_dir}'.")
        return existing_dir

    zip_path = os.path.join(RAW_DATA_DIR, "hydraulic_data.zip")
    downloaded = False

    for url in UCI_ZIP_URLS:
        try:
            print(f"[INFO] Downloading UCI Hydraulic Systems dataset from:\n       {url}")
            download_file_with_headers(url, zip_path)
            downloaded = True
            break
        except Exception as e:
            print(f"[WARN] Could not fetch from {url}: {e}")

    if not downloaded:
        raise RuntimeError(
            "Automated download failed. Please manually download the zip from "
            "https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems "
            "and extract its .txt files into the 'raw_hydraulic_data/' folder."
        )

    print("[INFO] Extracting sensor matrices...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DATA_DIR)
    print("[INFO] Extraction complete.")

    extracted_dir = locate_sensor_files(RAW_DATA_DIR)
    if extracted_dir is None:
        raise FileNotFoundError("Could not find 'profile.txt' inside extracted archive.")
    return extracted_dir

def build_cycle_feature_dataframe(data_dir):
    """
    Aggregates each 60-second cycle across all 17 sensors into:
      - Mean (steady-state operating level)
      - Standard Deviation (dynamic fluctuation / RMS spread)
      - Fisher Kurtosis (impulsive transient / shock spikiness)
    Total Features: 17 sensors * 3 statistics = 51 features.
    Target: Stable_Flag (0 = Stable steady-state, 1 = Unstable/transient regime).
    """
    feature_dict = {}
    print("[INFO] Engineering 51 cyclic features (Mean, Std, Kurtosis) across 2,205 cycles...")
    for sensor in SENSORS:
        file_path = os.path.join(data_dir, f"{sensor}.txt")
        raw_matrix = pd.read_csv(file_path, sep="\t", header=None).values

        cycle_mean = np.mean(raw_matrix, axis=1)
        cycle_std = np.std(raw_matrix, axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            cycle_kurt = kurtosis(raw_matrix, axis=1, fisher=True, bias=False)
            cycle_kurt = np.nan_to_num(cycle_kurt, nan=0.0, posinf=0.0, neginf=0.0)

        feature_dict[f"{sensor}_mean"] = cycle_mean
        feature_dict[f"{sensor}_std"] = cycle_std
        feature_dict[f"{sensor}_kurt"] = cycle_kurt

    df_features = pd.DataFrame(feature_dict)

    # Load ground-truth profile.txt (Column index 4 is Stable_Flag: 0 or 1)
    profile_path = os.path.join(data_dir, "profile.txt")
    profile_df = pd.read_csv(profile_path, sep="\t", header=None)
    df_features["Stable_Flag"] = profile_df.iloc[:, 4].astype(int)

    return df_features

def main():
    print("=" * 76)
    print("BITS PILANI M.TECH (AIML) - ML ASSIGNMENT 2 MODEL TRAINING PIPELINE")
    print("Dataset: UCI Condition Monitoring of Hydraulic Systems (51 Features)")
    print("=" * 76)

    data_dir = download_and_extract_uci_data()
    df = build_cycle_feature_dataframe(data_dir)
    print(f"[INFO] Engineered Dataset Shape: {df.shape[0]} instances x {df.shape[1]-1} features (+ 1 target)")
    print(f"[INFO] Target Class Counts (Stable_Flag): {df['Stable_Flag'].value_counts().to_dict()}")

    X = df.drop(columns=["Stable_Flag"])
    y = df["Stable_Flag"]

    # Stratified 80/20 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED
    )

    # Save holdout test_data.csv (with all 51 input features + Stable_Flag target column)
    test_df = X_test.copy()
    test_df["Stable_Flag"] = y_test.values
    test_df.to_csv("test_data.csv", index=False)
    print(f"[INFO] Saved holdout 'test_data.csv' ({test_df.shape[0]} rows, {test_df.shape[1]} columns).\n")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Define the 5 Required Classification Models wrapped in StandardScaler Pipelines
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1500, C=1.0, random_state=RANDOM_SEED))
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=RANDOM_SEED))
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance", metric="minkowski"))
        ]),
        "Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB(var_smoothing=1e-8))
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=3.0, gamma="scale", probability=True, random_state=RANDOM_SEED))
        ])
    }

    file_name_map = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "KNN": "knn.joblib",
        "Naive Bayes": "naive_bayes.joblib",
        "Support Vector Machine": "svm.joblib"
    }

    results = []
    for model_name, pipeline in models.items():
        pipeline.fit(X_train, y_train)

        # Save trained model artifact to model/ directory
        save_path = os.path.join(MODEL_DIR, file_name_map[model_name])
        joblib.dump(pipeline, save_path)

        # Predict on holdout test set
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        results.append({
            "ML Model Name": model_name,
            "Accuracy": round(acc, 4),
            "AUC": round(auc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "MCC": round(mcc, 4)
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(MODEL_DIR, "metrics_summary.csv"), index=False)

    print("=" * 76)
    print("EVALUATION METRICS COMPARISON TABLE (TEST SET: 441 CYCLES)")
    print("=" * 76)
    print(results_df.to_string(index=False))
    print("=" * 76)
    print("[SUCCESS] All 5 models saved to 'model/' and ready for Streamlit deployment.")

if __name__ == "__main__":
    main()