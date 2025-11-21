import pandas as pd
import os
import sys
import numpy as np
import shap
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import mode
import warnings

from src.BasicMetaModel import BasicMetaModel
from src.ExplainableTreeEnsemble import ExplainableTreeEnsemble
from src.LinearMetaModel import LinearMetaModel

# ---------------------------------------------
# Fix import paths
# ---------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
src_path = os.path.join(project_root, "src")
sys.path.append(src_path)

warnings.filterwarnings('ignore')

# ---------------------------------------------
# Helper Functions
# ---------------------------------------------
def _get_meta_features_static(X, trees_list):
    if not trees_list:
        return np.array([]).reshape(X.shape[0], 0)
    return np.column_stack([t.predict(X) for t in trees_list]).astype(np.float32)

def evaluate_predictions(workflow, trees_list, weighted=False):
    if not trees_list:
        return None, None, None

    X_test = _get_meta_features_static(workflow.X_test, trees_list)
    X_train_meta = _get_meta_features_static(workflow.X_train_meta, trees_list)
    y_test = workflow.y_test
    y_train_meta = workflow.y_train_meta

    if weighted:
        final_model = LogisticRegression(max_iter=2000).fit(X_train_meta, y_train_meta)
        w_abs = np.abs(final_model.coef_[0])
        w_final = w_abs / (np.sum(w_abs) + 1e-12)
        tree_probs = np.vstack([t.predict_proba(workflow.X_test)[:, 1] for t in trees_list])
        y_pred_prob = (tree_probs.T @ w_final)
        y_pred = (y_pred_prob >= 0.5).astype(int)
    else:
        y_pred = mode(X_test, axis=1)[0].flatten()

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    try:
        auc = roc_auc_score(y_test, y_pred)
    except:
        auc = None
    return acc, f1, auc

def run_method_b_extended(model_b, prune_threshold, corr_thresh):
    current_trees = model_b.pruned_trees
    if not current_trees:
        return [], []

    X_train = _get_meta_features_static(model_b.workflow.X_train_meta, current_trees)
    y_train = model_b.workflow.y_train_meta

    log = LogisticRegression(random_state=42, max_iter=1000).fit(X_train, y_train)
    weights = np.mean(np.abs(log.coef_), axis=0)

    threshold_val = prune_threshold * np.max(weights)
    keep_idx = np.where(weights > threshold_val)[0]
    stage2_trees = [current_trees[i] for i in keep_idx]

    stage3_trees = stage2_trees.copy()
    if len(stage2_trees) > 1:
        X_meta = _get_meta_features_static(model_b.workflow.X_train_meta, stage2_trees)
        corr_matrix = np.corrcoef(X_meta, rowvar=False)
        np.fill_diagonal(corr_matrix, 0)
        keep_mask = np.ones(len(stage2_trees), dtype=bool)
        for i in range(len(stage2_trees)):
            if keep_mask[i]:
                for j in range(i + 1, len(stage2_trees)):
                    if keep_mask[j] and abs(corr_matrix[i, j]) > corr_thresh:
                        keep_mask[j] = False
        stage3_trees = [stage2_trees[i] for i in range(len(stage2_trees)) if keep_mask[i]]

    return stage2_trees, stage3_trees

def run_baseline_c_corr_then_shap(workflow, corr_thresh=0.9, shap_cutoff_percentile=70):
    all_trees = workflow.individual_trees
    X_train_full = _get_meta_features_static(workflow.X_train_meta, all_trees)
    n_trees = X_train_full.shape[1]

    corr_matrix = np.corrcoef(X_train_full, rowvar=False)
    np.fill_diagonal(corr_matrix, 0)
    keep_mask = np.ones(n_trees, dtype=bool)
    for i in range(n_trees):
        for j in range(i + 1, n_trees):
            if keep_mask[j] and abs(corr_matrix[i, j]) > corr_thresh:
                keep_mask[j] = False
    stage1_trees = [all_trees[i] for i in range(n_trees) if keep_mask[i]]
    if not stage1_trees:
        return []

    X_train_red = _get_meta_features_static(workflow.X_train_meta, stage1_trees)
    model_red = LogisticRegression(random_state=workflow.random_state, max_iter=1000).fit(X_train_red, workflow.y_train_meta)
    X_val_red = _get_meta_features_static(workflow.X_eval_meta, stage1_trees)
    explainer = shap.Explainer(model_red, X_val_red)
    shap_vals = explainer(X_val_red).values
    if shap_vals.ndim == 3:
        mean_shap = np.mean(np.mean(np.abs(shap_vals), axis=2), axis=0)
    else:
        mean_shap = np.mean(np.abs(shap_vals), axis=0)
    shap_threshold = np.percentile(mean_shap, shap_cutoff_percentile)
    stage2_trees = [stage1_trees[i] for i, val in enumerate(mean_shap) if val >= shap_threshold]
    return stage2_trees

def run_standard_rf_baseline(workflow):
    rf = RandomForestClassifier(n_estimators=workflow.n_trees, random_state=workflow.random_state, max_depth=8)
    rf.fit(workflow.X_train, workflow.y_train)
    y_pred = rf.predict(workflow.X_test)
    acc = accuracy_score(workflow.y_test, y_pred)
    f1 = f1_score(workflow.y_test, y_pred, average="weighted")
    try:
        auc = roc_auc_score(workflow.y_test, y_pred)
    except:
        auc = None
    return acc, f1, auc

# ---------------------------------------------
# Main Experiment Logic
# ---------------------------------------------
def run_main_comparison():
    print("Starting Grid Search Experiment (Classification only)...")

    DATASET_CONFIG = {
        "your_classification_dataset": {"corr_thresh": 0.9, "prune_threshold": 0.01}
    }

    lambda_grid = {
        "lambda_prune": [0.5],
        "lambda_div": [0.1]
    }

    keep_ratio = 0.3
    shap_cutoff_percentile = (1 - keep_ratio) * 100

    output_dir = os.path.join(current_dir, "results")
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "classification_grid_search.csv")

    results = []

    for dataset, params in DATASET_CONFIG.items():
        print(f"\n{'='*60}\nDataset: {dataset}\n{'='*60}")
        workflow = ExplainableTreeEnsemble(data_type="classification", dataset_name=dataset)
        workflow.train_base_trees()
        workflow._evaluate()

        ct = params["corr_thresh"]
        pt = params["prune_threshold"]

        # -------------------
        # Method B
        # -------------------
        model_b = BasicMetaModel(keep_ratio=keep_ratio, data_type="classification")
        model_b.attach_to(workflow)
        model_b.train()
        _, stage3_trees_b = run_method_b_extended(model_b, prune_threshold=pt, corr_thresh=ct)
        acc_b_w, f1_b_w, auc_b_w = evaluate_predictions(workflow, stage3_trees_b, weighted=True)
        acc_b_n, f1_b_n, auc_b_n = evaluate_predictions(workflow, stage3_trees_b, weighted=False)

        # -------------------
        # Grid Search over Method A
        # -------------------
        for lp in lambda_grid["lambda_prune"]:
            for ld in lambda_grid["lambda_div"]:
                model_a = LinearMetaModel(λ_prune=lp, λ_div=ld, data_type="classification")
                model_a.attach_to(workflow)
                model_a.train(pruned_trees_list=model_b.pruned_trees)
                model_a.prune(prune_threshold=pt, corr_thresh=ct)
                acc_a_w, f1_a_w, auc_a_w = evaluate_predictions(workflow, model_a.pruned_trees, weighted=True)
                acc_a_n, f1_a_n, auc_a_n = evaluate_predictions(workflow, model_a.pruned_trees, weighted=False)

                # -------------------
                # Method C
                # -------------------
        stage2_trees_c = run_baseline_c_corr_then_shap(workflow, corr_thresh=ct, shap_cutoff_percentile=shap_cutoff_percentile)
        acc_c_w, f1_c_w, auc_c_w = evaluate_predictions(workflow, stage2_trees_c, weighted=True)
        acc_c_n, f1_c_n, auc_c_n = evaluate_predictions(workflow, stage2_trees_c, weighted=False)

                # -------------------
                # Method D
                # -------------------
        acc_d, f1_d, auc_d = run_standard_rf_baseline(workflow)

        results.append({
                    "dataset": dataset,
                    "lambda_prune": lp,
                    "lambda_div": ld,
                    "acc_A_weighted": acc_a_w, "f1_A_weighted": f1_a_w, "auc_A_weighted": auc_a_w,
                    "acc_A_normal": acc_a_n, "f1_A_normal": f1_a_n, "auc_A_normal": auc_a_n,
                    "acc_B_weighted": acc_b_w, "f1_B_weighted": f1_b_w, "auc_B_weighted": auc_b_w,
                    "acc_B_normal": acc_b_n, "f1_B_normal": f1_b_n, "auc_B_normal": auc_b_n,
                    "acc_C_weighted": acc_c_w, "f1_C_weighted": f1_c_w, "auc_C_weighted": auc_c_w,
                    "acc_C_normal": acc_c_n, "f1_C_normal": f1_c_n, "auc_C_normal": auc_c_n,
                    "acc_D": acc_d, "f1_D": f1_d, "auc_D": auc_d,
                    "corr_thresh": ct,
                    "prune_thresh": pt
                })

        # Save results after each dataset
        pd.DataFrame(results).to_csv(file_path, index=False)
        print(f"Results saved to {file_path}")

    print("\nGrid search finished.")

if __name__ == "__main__":
    run_main_comparison()
