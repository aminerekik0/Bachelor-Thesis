import os
import csv
import numpy as np
from uci_datasets import Dataset

from sklearn.datasets import fetch_covtype
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score, 
    roc_auc_score,
)

# adjust import paths if needed
import sys


from src.ExplainableTreeEnsemble import ExplainableTreeEnsemble
from src.BasicMetaModel import BasicMetaModel
from src.LinearMetaModel import LinearMetaModel


# ============================================================
# DATASET-SPECIFIC CONFIG FOR METHOD A (full framework)
# ============================================================
DATASET_CONFIG = {
    "slice": {
        "lambda_prune": 0.6,
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
        "lambda_prune": 1.0,
        "lambda_div": 0.3,
        "prune_threshold": 0.01,
        "corr_threshold": 0.88,
        "task": "regression",
    },
    "covertype": {
        "lambda_prune": 0.5,
        "lambda_div": 0.5,
        "prune_threshold": 0.01,
        "corr_threshold": 0.92,
        "task": "classification",
    },
    "higgs": {
        "lambda_prune": 0.4,
        "lambda_div": 0.5,
        "prune_threshold": 0.01,
        "corr_threshold": 0.93,
        "task": "classification",
    },
    # you can add "higgs" here later if you want to include it
}


OUT_CSV = "comparison_results.csv"


# ============================================================
#               HELPER FUNCTIONS
# ============================================================

def append_to_csv(row, filename=OUT_CSV):
    write_header = not os.path.exists(filename)
    with open(filename, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def get_meta_features(X, trees_list):
    """Column-stack predictions from a list of trees."""
    return np.column_stack([t.predict(X) for t in trees_list]).astype(np.float32)


def evaluate_subset(trees, workflow, data_type):
    """
    Evaluate a given subset of trees on the test set, using the same idea
    as BasicMetaModel / LinearMetaModel final evaluation.
    - For regression: linear regression on meta-train, weighted prediction on test.
    - For classification: majority vote on test.
    """
    X_test = workflow.X_test
    y_test = workflow.y_test

    # =============== REGRESSION ===============
    if data_type == "regression":
        # train meta linear model on X_train_meta
        X_train_final = get_meta_features(workflow.X_train_meta, trees)
        y_train_final = workflow.y_train_meta

        lin = LinearRegression().fit(X_train_final, y_train_final)
        w_abs = np.abs(lin.coef_)
        w = w_abs / (np.sum(w_abs) + 1e-12)

        # test predictions
        preds_matrix = get_meta_features(X_test, trees)
        final_preds = preds_matrix @ w

        rmse = np.sqrt(mean_squared_error(y_test, final_preds))
        r2 = r2_score(y_test, final_preds)
        return {"main": rmse, "r2": r2, "f1": None, "auc": None}

    # =============== CLASSIFICATION ===============
    else:
        # majority vote of tree predictions
        tree_labels = np.vstack([t.predict(X_test) for t in trees])
        from scipy.stats import mode
        mode_res = mode(tree_labels, axis=0, keepdims=False)
        final_preds = mode_res.mode

        acc = accuracy_score(y_test, final_preds)
        f1 = f1_score(y_test, final_preds, average="weighted")
        auc = roc_auc_score(y_test, final_preds)

        return {"main": acc, "r2": None, "f1": f1, "auc": auc}


def correlation_prune(trees_list, workflow, data_type, corr_thresh, importance=None):
    """
    Correlation-based pruning:
    - Compute correlations between tree predictions on X_eval_meta.
    - Greedy selection: keep high-importance trees, remove highly correlated ones.
    importance: array of scores (higher = more important); if None -> keep original order.
    """
    if len(trees_list) <= 1:
        return list(trees_list)

    X_eval = workflow.X_eval_meta
    preds = get_meta_features(X_eval, trees_list)  # shape: (n_samples, n_trees)
    corr_matrix = np.corrcoef(preds.T)
    np.fill_diagonal(corr_matrix, 0)

    n_trees = len(trees_list)
    indices = np.arange(n_trees)

    if importance is None:
        order = indices
    else:
        order = np.argsort(importance)[::-1]  # descending

    keep = []
    for idx in order:
        if not keep:
            keep.append(idx)
            continue
        # check correlation with already kept trees
        too_corr = False
        for k in keep:
            if np.abs(corr_matrix[idx, k]) > corr_thresh:
                too_corr = True
                break
        if not too_corr:
            keep.append(idx)

    kept_trees = [trees_list[i] for i in keep]
    return kept_trees


def shap_prune_on_subset(trees_subset, workflow, data_type, keep_ratio=0.3, random_state=42):
    """
    Run a SHAP-based pruning similar to BasicMetaModel, but restricted to a
    given subset of trees (used for Method C: Corr -> SHAP).
    Returns:
        pruned_trees, metrics_dict
    """
    import shap
    X_train_meta = workflow.X_train_meta
    X_eval_meta = workflow.X_eval_meta
    y_train_meta = workflow.y_train_meta
    y_eval_meta = workflow.y_eval_meta

    # Build meta-features for this subset
    X_meta_train = get_meta_features(X_train_meta, trees_subset)
    X_meta_eval = get_meta_features(X_eval_meta, trees_subset)

    # === train meta model ===
    if data_type == "regression":
        meta_model = LinearRegression()
        meta_model.fit(X_meta_train, y_train_meta)
        y_pred = meta_model.predict(X_meta_eval)
        mse = mean_squared_error(y_eval_meta, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_eval_meta, y_pred)
        meta_metrics = {"main": rmse, "r2": r2, "f1": None, "auc": None}
    else:
        meta_model = LogisticRegression(
            max_iter=2000,
            random_state=random_state,
            solver="lbfgs"
        )
        meta_model.fit(X_meta_train, y_train_meta)
        y_pred = meta_model.predict(X_meta_eval)
        acc = accuracy_score(y_eval_meta, y_pred)
        f1 = f1_score(y_eval_meta, y_pred, average="weighted")
        auc = roc_auc_score(y_eval_meta, y_pred)
        meta_metrics = {"main": acc, "r2": None, "f1": f1, "auc": auc}

    # === SHAP on subset ===
    explainer = shap.Explainer(meta_model, X_meta_eval, algorithm="linear")
    shap_result = explainer(X_meta_eval)
    shap_values = np.array(shap_result.values)

    # importance like in BasicMetaModel
    tree_importance = np.mean(np.abs(shap_values), axis=0)
    k = max(5, int(len(trees_subset) * keep_ratio))
    top_indices = np.argsort(tree_importance)[-k:][::-1]
    pruned_trees = [trees_subset[i] for i in top_indices]

    return pruned_trees, meta_metrics


# ============================================================
#           METHODS A, B, C PER DATASET
# ============================================================

def run_methods_for_dataset(X, y, dataset_name):
    cfg = DATASET_CONFIG[dataset_name]
    task = cfg["task"]

    print(f"\n================= DATASET: {dataset_name} ({task}) =================")

    # ---------- Create workflow & train base trees ----------
    print(X.shape , y.shape)
    workflow = ExplainableTreeEnsemble(
        X=X,
        y=y,
        dataset_name=dataset_name,
        data_type=task,
    )
    workflow.train_base_trees()
    full_size = len(workflow.individual_trees)

    # Full ensemble metrics
    mse, rmse, mae, r2, acc, f1 = workflow._evaluate()
    if task == "regression":
        full_metric = rmse
        full_r2 = r2
        full_f1 = None
        full_auc = None
    else:
        full_metric = acc
        full_r2 = None
        full_f1 = f1
        full_auc = workflow.auc

    # ========================================================
    #           COMMON STAGE 1: SHAP-BASED BASIC META
    #    (used by Method A and Method B)
    # ========================================================
    basic = BasicMetaModel(data_type=task)
    basic.attach_to(workflow)
    basic.train()
    pre_metric, pre_aux = basic.evaluate()

    shap_size = len(basic.pruned_trees) if basic.pruned_trees else 0

    if task == "regression":
        pre_r2 = basic.r2
        pre_f1 = None
        pre_auc = None
    else:
        pre_r2 = None
        pre_f1 = basic.f1
        pre_auc = basic.auc

    # ========================================================
    # METHOD A: Full Framework
    # ========================================================
    print("\n----- METHOD A: Full framework (Stage1 SHAP + Stage2 Linear + corr) -----")

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
        "method": "A_full_framework",
        "lambda_prune": cfg["lambda_prune"],
        "lambda_div": cfg["lambda_div"],
        "prune_threshold": cfg["prune_threshold"],
        "corr_threshold": cfg["corr_threshold"],
        "full_metric": full_metric,
        "full_r2": full_r2,
        "full_f1": full_f1,
        "full_auc": full_auc,
        "stage1_size": shap_size,
        "stage1_metric": pre_metric,
        "stage1_r2": pre_r2,
        "stage1_f1": pre_f1,
        "stage1_auc": pre_auc,
        "final_size": final_size_A,
        "final_metric": final_metric_A,
        "final_r2": final_r2_A,
        "final_f1": final_f1_A,
        "final_auc": final_auc_A,
    }
    append_to_csv(row_A)

    # ========================================================
    # METHOD B: SHAP -> Correlation (NO LinearMetaModel)
    # ========================================================
    print("\n----- METHOD B: SHAP -> Correlation (no optimization) -----")

    # correlation prune on basic.pruned_trees, using SHAP importance as order
    corr_pruned_trees_B = correlation_prune(
        basic.pruned_trees,
        workflow,
        task,
        corr_thresh=cfg["corr_threshold"],
        importance=basic.pruned_tree_weights,
    )

    final_metrics_B = evaluate_subset(corr_pruned_trees_B, workflow, task)
    final_size_B = len(corr_pruned_trees_B)

    row_B = {
        "dataset": dataset_name,
        "task": task,
        "method": "B_shap_then_corr",
        "lambda_prune": 0.0,
        "lambda_div": 0.0,
        "prune_threshold": None,
        "corr_threshold": cfg["corr_threshold"],
        "full_metric": full_metric,
        "full_r2": full_r2,
        "full_f1": full_f1,
        "full_auc": full_auc,
        "stage1_size": shap_size,           # SHAP-pruned size
        "stage1_metric": pre_metric,
        "stage1_r2": pre_r2,
        "stage1_f1": pre_f1,
        "stage1_auc": pre_auc,
        "final_size": final_size_B,
        "final_metric": final_metrics_B["main"],
        "final_r2": final_metrics_B["r2"],
        "final_f1": final_metrics_B["f1"],
        "final_auc": final_metrics_B["auc"],
    }
    append_to_csv(row_B)

    # ========================================================
    # METHOD C: Correlation -> SHAP
    # ========================================================
    print("\n----- METHOD C: Correlation -> SHAP -----")

    # Step 1: correlation pruning on ALL trees (200)
    # here we don't have importance yet, so we use equal priority (original order)
    corr_pruned_trees_C_stage1 = correlation_prune(
        workflow.individual_trees,
        workflow,
        task,
        corr_thresh=cfg["corr_threshold"],
        importance=None,
    )
    corr_stage1_size = len(corr_pruned_trees_C_stage1)

    # Step 2: SHAP pruning on this corr-pruned subset
    shap_pruned_trees_C, _ = shap_prune_on_subset(
        corr_pruned_trees_C_stage1,
        workflow,
        data_type=task,
        keep_ratio=0.3,  # same keep_ratio as BasicMetaModel
    )
    final_size_C = len(shap_pruned_trees_C)

    # Evaluate final subset
    final_metrics_C = evaluate_subset(shap_pruned_trees_C, workflow, task)

    row_C = {
        "dataset": dataset_name,
        "task": task,
        "method": "C_corr_then_shap",
        "lambda_prune": 0.0,
        "lambda_div": 0.0,
        "prune_threshold": None,
        "corr_threshold": cfg["corr_threshold"],
        "full_metric": full_metric,
        "full_r2": full_r2,
        "full_f1": full_f1,
        "full_auc": full_auc,
        "stage1_size": corr_stage1_size,    # size after corr-only
        "stage1_metric": None,              # you can fill this if you want to eval after corr-only
        "stage1_r2": None,
        "stage1_f1": None,
        "stage1_auc": None,
        "final_size": final_size_C,
        "final_metric": final_metrics_C["main"],
        "final_r2": final_metrics_C["r2"],
        "final_f1": final_metrics_C["f1"],
        "final_auc": final_metrics_C["auc"],
    }
    append_to_csv(row_C)


# ============================================================
#                   MAIN
# ============================================================

def main():
    # Regression datasets
    for ds in ["slice", "3droad", "kin40k"]:
        data = Dataset(ds)
        X = data.x.astype(np.float32)
        y = data.y.ravel()
        run_methods_for_dataset(X, y, ds)


    classification_sets = ["covertype"]

    for ds in classification_sets:
        if ds == "covertype":
            from sklearn.datasets import fetch_covtype
            data = fetch_covtype(as_frame=False)
            X = data.data
            y = (data.target == 2).astype(int)
            run_methods_for_dataset(X, y, ds)
        if ds == "higgs":
            import kagglehub
            import kagglehub
            from kagglehub import KaggleDatasetAdapter

            path = kagglehub.dataset_download("erikbiswas/higgs-uci-dataset")
            csv_file = None
            for file in os.listdir(path):
                if file.endswith(".csv"):
                    csv_file = os.path.join(path, file)
                break
            if csv_file is None:
                raise FileNotFoundError("No CSV file found in downloaded HIGGS dataset folder!")

            print("Loaded CSV:", csv_file)
            import pandas as pd

            df = pd.read_csv(csv_file, nrows=1000000)
            y = df.iloc[:, 0].astype(int).values
            X = df.iloc[:, 1:].astype("float32").values
            print("Path to dataset files:", path)
            run_methods_for_dataset(X, y, ds)




if __name__ == "__main__":
    main()
