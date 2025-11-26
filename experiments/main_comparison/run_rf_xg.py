import os
import csv
import numpy as np
from uci_datasets import Dataset

# Scikit-learn imports
from sklearn.datasets import fetch_covtype
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
)

# Boosting imports
import xgboost as xgb
import lightgbm as lgb  # Added LightGBM

# Adjust import paths if needed
import sys

# Your custom modules
from src.ExplainableTreeEnsemble import ExplainableTreeEnsemble
from src.BasicMetaModel import BasicMetaModel
from src.LinearMetaModel import LinearMetaModel


# ============================================================
# DATASET-SPECIFIC CONFIG FOR METHOD A (Full Framework)
# ============================================================
DATASET_CONFIG = {
    "slice": {
        "lambda_prune": 0.5,
        "lambda_div": 0.5,
        "prune_threshold": 0.01,
        "corr_threshold": 0.92,
        "task": "regression",
    },
    "3droad": {
        "lambda_prune": 0.8,
        "lambda_div": 0.5,
        "prune_threshold": 0.01,
        "corr_threshold": 0.88,
        "task": "regression",
    },
    "kin40k": {
        "lambda_prune": 0.8,
        "lambda_div": 0.3,
        "prune_threshold": 0.01,
        "corr_threshold": 0.88,
        "task": "regression",
    },
    "covertype": {
        "lambda_prune": 0.4,
        "lambda_div": 0.5,
        "prune_threshold": 0.01,
        "corr_threshold": 0.92,
        "task": "classification",
    },
    "higgs": {
        "lambda_prune": 0.3,
        "lambda_div": 0.5,
        "prune_threshold": 0.01,
        "corr_threshold": 0.93,
        "task": "classification",
    },
}

OUT_CSV = "industry_comparison_results.csv"


# ============================================================
#               HELPER FUNCTIONS
# ============================================================

def append_to_csv(row, filename=OUT_CSV):
    """Appends a dictionary row to the CSV file."""
    write_header = not os.path.exists(filename)
    with open(filename, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def calculate_metrics(y_true, y_pred, task, y_proba=None):
    """Helper to standardize metric calculation."""
    if task == "regression":
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return {"metric": rmse, "r2": r2, "f1": None, "auc": None}
    else:
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")
        if y_proba is not None:
            try:
                # For binary classification, use probability of positive class
                if y_proba.ndim > 1 and y_proba.shape[1] > 1:
                    auc = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    auc = roc_auc_score(y_true, y_proba)
            except ValueError:
                auc = None
        else:
            auc = None
        return {"metric": acc, "r2": None, "f1": f1, "auc": auc}


# ============================================================
#           RUN COMPARISON FOR A DATASET
# ============================================================

def run_methods_for_dataset(X, y, dataset_name):
    if dataset_name not in DATASET_CONFIG:
        print(f"Skipping {dataset_name}: No config found.")
        return

    cfg = DATASET_CONFIG[dataset_name]
    task = cfg["task"]

    print(f"\n================= DATASET: {dataset_name} ({task}) =================")

    # 1. Setup Framework (Method A) - This handles data splits
    workflow = ExplainableTreeEnsemble(
        X=X,
        y=y,
        dataset_name=dataset_name,
        data_type=task,
    )
    workflow.train_base_trees()  # Trains the 200 base trees (Bagging)

    # Get the specific splits used by the framework to ensure fair comparison
    X_train = workflow.X_train
    y_train = workflow.y_train
    X_test = workflow.X_test
    y_test = workflow.y_test

    # --- Metrics for "Full Ensemble" (Bagging Baseline) ---
    _, rmse, _, r2, acc, f1 = workflow._evaluate()
    if task == "regression":
        full_metric = rmse
        full_r2 = r2
        full_f1 = None
        full_auc = None
    else:
        full_metric = acc
        full_r2 = None
        full_f1 = f1
        full_auc = workflow.auc  # Assuming workflow calculates this

    # ========================================================
    # METHOD A: Full Framework (Ours)
    # ========================================================
    print("\n----- METHOD A: Full Framework (Ours) -----")

    # Stage 1: SHAP Pruning
    basic = BasicMetaModel(data_type=task)
    basic.attach_to(workflow)
    basic.train()

    # Stage 2: Linear Optimization
    lm = LinearMetaModel(
        λ_prune=cfg["lambda_prune"],
        λ_div=cfg["lambda_div"],
        data_type=task
    )
    lm.attach_to(workflow)
    lm.train(basic.pruned_trees)
    lm.prune(
        prune_threshold=cfg["prune_threshold"],
        corr_thresh=cfg["corr_threshold"]
    )

    final_metric_A, _ = lm.evaluate()
    final_size_A = len(lm.pruned_trees) if lm.pruned_trees else 0

    if task == "regression":
        final_r2_A = lm.r2
        final_f1_A = None
        final_auc_A = None
    else:
        final_r2_A = None
        final_f1_A = lm.f1
        final_auc_A = lm.auc

    row_A = {
        "dataset": dataset_name,
        "task": task,
        "method": "Method_A_Ours",
        "n_trees": final_size_A,
        "metric (RMSE/Acc)": final_metric_A,
        "r2": final_r2_A,
        "f1": final_f1_A,
        "auc": final_auc_A
    }
    append_to_csv(row_A)

    # ========================================================
    # BASELINE: Standard Random Forest
    # ========================================================
    print("\n----- BASELINE: Standard Random Forest -----")

    n_estimators_rf = 200 # Matching base ensemble size

    if task == "regression":
        rf_model = RandomForestRegressor(
            n_estimators=n_estimators_rf,
            random_state=42,
            max_depth=8,
            # n_jobs removed or set to None for default sequential behavior if needed
            # but typically n_jobs=-1 is preferred for speed.
            # Keeping empty as per previous user request style, or add n_jobs=-1 if desired.
        )
    else:
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators_rf,
            random_state=42,
            max_depth=8,
        )

    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    y_proba_rf = None
    if task == "classification":
        y_proba_rf = rf_model.predict_proba(X_test)

    metrics_rf = calculate_metrics(y_test, y_pred_rf, task, y_proba_rf)

    row_rf = {
        "dataset": dataset_name,
        "task": task,
        "method": "RandomForest_Standard",
        "n_trees": n_estimators_rf,
        "metric (RMSE/Acc)": metrics_rf["metric"],
        "r2": metrics_rf["r2"],
        "f1": metrics_rf["f1"],
        "auc": metrics_rf["auc"]
    }
    append_to_csv(row_rf)

    # ========================================================
    # BASELINE: XGBoost
    # ========================================================
    print("\n----- BASELINE: XGBoost -----")

    n_estimators_xgb = 200

    if task == "regression":
        xgb_model = xgb.XGBRegressor(
            n_estimators=n_estimators_xgb,
            random_state=42,
            max_depth=8,
            num_parallel_tree=1
        )
    else:
        xgb_model = xgb.XGBClassifier(
            n_estimators=n_estimators_xgb,
            random_state=42,
            max_depth=8,
            num_parallel_tree=1
        )

    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)

    y_proba_xgb = None
    if task == "classification":
        y_proba_xgb = xgb_model.predict_proba(X_test)

    metrics_xgb = calculate_metrics(y_test, y_pred_xgb, task, y_proba_xgb)

    row_xgb = {
        "dataset": dataset_name,
        "task": task,
        "method": "XGBoost_Standard",
        "n_trees": n_estimators_xgb,
        "metric (RMSE/Acc)": metrics_xgb["metric"],
        "r2": metrics_xgb["r2"],
        "f1": metrics_xgb["f1"],
        "auc": metrics_xgb["auc"]
    }
    append_to_csv(row_xgb)

    # ========================================================
    # BASELINE: LightGBM
    # ========================================================
    print("\n----- BASELINE: LightGBM -----")

    n_estimators_lgb = 200

    if task == "regression":
        lgb_model = lgb.LGBMRegressor(
            n_estimators=n_estimators_lgb,
            random_state=42,
            max_depth=8,
            n_jobs=-1

        )
    else:
        lgb_model = lgb.LGBMClassifier(
            n_estimators=n_estimators_lgb,
            random_state=42,
            max_depth=8,
            n_jobs=-1
        )

    lgb_model.fit(X_train, y_train)
    y_pred_lgb = lgb_model.predict(X_test)

    y_proba_lgb = None
    if task == "classification":
        y_proba_lgb = lgb_model.predict_proba(X_test)

    metrics_lgb = calculate_metrics(y_test, y_pred_lgb, task, y_proba_lgb)

    row_lgb = {
        "dataset": dataset_name,
        "task": task,
        "method": "LightGBM_Standard",
        "n_trees": n_estimators_lgb,
        "metric (RMSE/Acc)": metrics_lgb["metric"],
        "r2": metrics_lgb["r2"],
        "f1": metrics_lgb["f1"],
        "auc": metrics_lgb["auc"]
    }
    append_to_csv(row_lgb)


# ============================================================
#                   MAIN EXECUTION
# ============================================================

def main():
    # --- Regression Datasets ---
    regression_sets = ["slice", "3droad", "kin40k"]

    for ds in regression_sets:
        try:
            data = Dataset(ds)
            X = data.x.astype(np.float32)
            y = data.y.ravel()
            run_methods_for_dataset(X, y, ds)
        except Exception as e:
            print(f"Error processing {ds}: {e}")

    # --- Classification Datasets ---
    classification_sets = ["covertype", "higgs"]

    for ds in classification_sets:
        if ds == "covertype":
            try:
                data = fetch_covtype(as_frame=False)
                X = data.data
                y = (data.target == 2).astype(int) # Binary classification task
                run_methods_for_dataset(X, y, ds)
            except Exception as e:
                print(f"Error processing {ds}: {e}")

        elif ds == "higgs":
            try:
                import kagglehub
                import pandas as pd

                path = kagglehub.dataset_download("erikbiswas/higgs-uci-dataset")
                csv_file = None
                for file in os.listdir(path):
                    if file.endswith(".csv"):
                        csv_file = os.path.join(path, file)
                    break

                if csv_file:
                    print("Loading HIGGS CSV...")
                    # Loading subset for speed/memory, adjust nrows as needed
                    df = pd.read_csv(csv_file, nrows=1000000)
                    y = df.iloc[:, 0].astype(int).values
                    X = df.iloc[:, 1:].astype("float32").values
                    run_methods_for_dataset(X, y, ds)
                else:
                    print("HIGGS CSV not found.")
            except Exception as e:
                print(f"Error processing {ds}: {e}")

if __name__ == "__main__":
    main()