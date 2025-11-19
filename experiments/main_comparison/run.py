import pandas as pd
import os
import sys
import numpy as np
import shap
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import warnings

# Adjust imports based on your project structure
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

def run_method_b_extended(model_b, prune_threshold, corr_thresh):
    """
    Extends BasicMetaModel (Method B) to track trees after every stage.
    Returns: (metric, count_stage2, count_stage3)
    """
    # 1. Start with trees from Stage 1 (SHAP Pruned Top 30%)
    current_trees = model_b.pruned_trees
    if not current_trees:
        print("[ERROR] Method B: No trees from Stage 1.")
        return None, 0, 0

    print(f"=== [Method B] Stage 2: Weight-based pruning (thresh={prune_threshold}) ===")

    # Get features for current trees
    X_train = _get_meta_features_static(model_b.workflow.X_train_meta, current_trees)
    y_train = model_b.workflow.y_train_meta

    # Train Linear Model to get weights for pruning
    if model_b.data_type == "regression":
        lin = LinearRegression().fit(X_train, y_train)
        weights = np.abs(lin.coef_)
    else:
        log = LogisticRegression(random_state=42, max_iter=1000).fit(X_train, y_train)
        weights = np.mean(np.abs(log.coef_), axis=0)

    # Filter by weight threshold
    w_max = np.max(weights)
    threshold_val = prune_threshold * w_max
    keep_idx = np.where(weights > threshold_val)[0]

    current_trees = [current_trees[i] for i in keep_idx]
    count_stage2 = len(current_trees)
    print(f"[Method B] Stage 2 (Weight) Kept {count_stage2} trees.")

    if not current_trees:
        return None, 0, 0

    print(f"=== [Method B] Stage 3: Correlation filtering (thresh={corr_thresh}) ===")

    # Correlation Pruning
    X_meta = _get_meta_features_static(model_b.workflow.X_train_meta, current_trees)
    if X_meta.shape[1] > 1:
        corr_matrix = np.corrcoef(X_meta, rowvar=False)
        np.fill_diagonal(corr_matrix, 0)

        n = len(current_trees)
        keep_mask = np.ones(n, dtype=bool)

        # Greedy removal: preserve order
        for i in range(n):
            if keep_mask[i]:
                for j in range(i+1, n):
                    if keep_mask[j] and abs(corr_matrix[i, j]) > corr_thresh:
                        keep_mask[j] = False

        final_trees = [current_trees[i] for i in range(n) if keep_mask[i]]
        count_stage3 = len(final_trees)
        print(f"[Method B] Stage 3 (Corr) Kept {count_stage3} trees.")
    else:
        final_trees = current_trees
        count_stage3 = len(final_trees)
        print("[Method B] Stage 3 Skipped (<= 1 tree).")

    # --- Final Evaluation ---
    X_test = _get_meta_features_static(model_b.workflow.X_test, final_trees)
    X_train_final = _get_meta_features_static(model_b.workflow.X_train_meta, final_trees)
    y_test = model_b.workflow.y_test

    metric = None
    if model_b.data_type == "regression":
        final_model = LinearRegression().fit(X_train_final, y_train)
        # Normalize weights for consistent evaluation
        w_abs = np.abs(final_model.coef_)
        weights_to_use = w_abs / (np.sum(w_abs) + 1e-12)

        y_pred = X_test @ weights_to_use
        metric = mean_squared_error(y_test, y_pred)
        print(f"[Method B] Final MSE (Weighted Avg): {metric:.4f}")
    else:
        final_model = LogisticRegression(random_state=42, max_iter=1000).fit(X_train_final, y_train)
        y_pred = final_model.predict(X_test)
        metric = accuracy_score(y_test, y_pred)
        print(f"[Method B] Final Accuracy: {metric:.4f}")

    return metric, count_stage2, count_stage3

def run_baseline_c_corr_then_shap(workflow, corr_thresh=0.9, shap_cutoff_percentile=70):
    """
    Runs Method C and returns intermediate counts.
    Returns: (metric, count_stage1, count_stage2)
    """
    all_trees = workflow.individual_trees
    X_train_full = _get_meta_features_static(workflow.X_train_meta, all_trees)
    y_train = workflow.y_train_meta
    X_val_full = _get_meta_features_static(workflow.X_eval_meta, all_trees)
    n_trees = X_train_full.shape[1]
    data_type = workflow.data_type

    print(f"=== [Method C] Stage 1: Redundancy filtering (threshold={corr_thresh}) ===")
    corr_matrix = np.corrcoef(X_train_full, rowvar=False)
    np.fill_diagonal(corr_matrix, 0)

    keep_mask = np.ones(n_trees, dtype=bool)
    for i in range(n_trees):
        for j in range(i + 1, n_trees):
            if keep_mask[j] and abs(corr_matrix[i, j]) > corr_thresh:
                keep_mask[j] = False

    original_indices_stage1 = np.where(keep_mask)[0]
    trees_stage1 = [all_trees[i] for i in original_indices_stage1]
    count_stage1 = len(trees_stage1)
    print(f"[Method C] Stage 1 (Corr) Kept {count_stage1} trees.")

    if count_stage1 == 0:
        return None, 0, 0

    X_train_red = X_train_full[:, keep_mask]
    X_val_red = X_val_full[:, keep_mask]

    print(f"=== [Method C] Stage 2: SHAP pruning (Remove bottom {shap_cutoff_percentile}%) ===")

    if data_type == "regression":
        model_red = LinearRegression().fit(X_train_red, y_train)
    else:
        model_red = LogisticRegression(random_state=workflow.random_state, max_iter=1000).fit(X_train_red, y_train)

    explainer = shap.Explainer(model_red, X_val_red)
    shap_vals = explainer(X_val_red).values

    if shap_vals.ndim == 3:
        mean_shap = np.mean(np.mean(np.abs(shap_vals), axis=2), axis=0)
    else:
        mean_shap = np.mean(np.abs(shap_vals), axis=0)

    shap_threshold = np.percentile(mean_shap, shap_cutoff_percentile)
    keep_idx_shap_local = [i for i, shap_val in enumerate(mean_shap) if shap_val >= shap_threshold]

    final_keep_indices = [original_indices_stage1[i] for i in keep_idx_shap_local]
    corr_first_pruned_trees = [all_trees[i] for i in final_keep_indices]
    count_stage2 = len(corr_first_pruned_trees)
    print(f"[Method C] Stage 2 (SHAP) Kept {count_stage2} trees.")

    if not corr_first_pruned_trees:
        return None, count_stage1, 0

    # Final Eval
    X_train_final = _get_meta_features_static(workflow.X_train_meta, corr_first_pruned_trees)
    y_train_final = workflow.y_train_meta
    X_test = workflow.X_test
    y_test = workflow.y_test

    if data_type == "regression":
        final_eval_model = LinearRegression().fit(X_train_final, y_train_final)
        w_abs = np.abs(final_eval_model.coef_)
        normalized_weights = w_abs / (np.sum(w_abs) + 1e-12)
        pruned_preds_matrix = _get_meta_features_static(X_test, corr_first_pruned_trees)
        final_preds = pruned_preds_matrix @ normalized_weights
        metric = mean_squared_error(y_test, final_preds)
        print(f"[Method C] Final Pruned MSE (Weighted Avg): {metric:.4f}")
    else:
        final_eval_model = LogisticRegression(random_state=workflow.random_state, max_iter=1000).fit(X_train_final, y_train_final)
        pruned_preds_matrix = _get_meta_features_static(X_test, corr_first_pruned_trees)
        final_preds = final_eval_model.predict(pruned_preds_matrix)
        metric = accuracy_score(y_test, final_preds)

    return metric, count_stage1, count_stage2

def run_standard_rf_baseline(workflow):
    """
    Method D: Standard Random Forest on the ORIGINAL dataset.
    """
    print(f"\n--- Running Method D (Standard RF on Original Data) ---")

    if workflow.data_type == "regression":
        rf = RandomForestRegressor(n_estimators=workflow.n_trees, random_state=workflow.random_state , max_depth=np.random.choice([2, 5 , 6, 9 , 10]))
        rf.fit(workflow.X_train, workflow.y_train)
        y_pred = rf.predict(workflow.X_test)
        metric = mean_squared_error(workflow.y_test, y_pred)
        print(f"[Method D] Standard RF MSE: {metric:.4f}")
    else:
        rf = RandomForestClassifier(n_estimators=workflow.n_trees, random_state=workflow.random_state, n_jobs=-1)
        rf.fit(workflow.X_train, workflow.y_train)
        y_pred = rf.predict(workflow.X_test)
        metric = accuracy_score(workflow.y_test, y_pred)
        print(f"[Method D] Standard RF Accuracy: {metric:.4f}")

    return metric

# ---------------------------------------------
# Main Experiment Logic
# ---------------------------------------------
def run_main_comparison():
    print("Starting Main Comparison Experiment (Methods A, B, C, D)...")

    DATASET_CONFIG = {
        "3droad": {
            "corr_thresh": 0.90,
            "prune_threshold": 0.01,
            "lambda_prune": 1.0,
            "lambda_div": 1.0
        },
        "slice": {
            "corr_thresh": 0.95,
            "prune_threshold": 0.01,
            "lambda_prune": 1.0,
            "lambda_div": 0.5
        },
        "kin40k": {
            "corr_thresh": 0.90,
            "prune_threshold": 0.01,
            "lambda_prune": 1.0,
            "lambda_div": 0.5
        }
    }

    keep_ratio = 0.3
    shap_cutoff_percentile = (1 - keep_ratio) * 100

    output_dir = os.path.join(current_dir, "results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, "main_comparison_final.csv")

    results = []

    for dataset, params in DATASET_CONFIG.items():
        print(f"\n{'='*60}")
        print(f"Processing Dataset: {dataset}")
        print(f"Params: {params}")
        print(f"{'='*60}")

        workflow = ExplainableTreeEnsemble(data_type="regression", dataset_name=dataset)
        workflow.train_base_trees()
        mse_full = workflow.mse

        ct = params["corr_thresh"]
        pt = params["prune_threshold"]
        lp = params["lambda_prune"]
        ld = params["lambda_div"]

        # -------------------------------------------------
        # Method B: SHAP -> Weight -> Corr (Baseline)
        # -------------------------------------------------
        print(f"\n--- Running Method B [PT={pt}, CT={ct}] ---")
        model_b = BasicMetaModel(keep_ratio=keep_ratio)
        model_b.attach_to(workflow)

        # Stage 1 (Internal SHAP)
        model_b.train()
        trees_b_s1 = len(model_b.pruned_trees)
        mse_basic_shap_only, _ = model_b.evaluate()

        # Stages 2 & 3 (External Weight + Corr)
        mse_b, trees_b_s2, trees_b_s3 = run_method_b_extended(model_b, prune_threshold=pt, corr_thresh=ct)

        # -------------------------------------------------
        # Method A: LinearMetaModel (HRP - Ours)
        # -------------------------------------------------
        print(f"\n--- Running Method A (HRP) [LP={lp}, LD={ld}, PT={pt}, CT={ct}] ---")
        model_a = LinearMetaModel(λ_prune=lp, λ_div=ld)
        model_a.attach_to(workflow)

        # Train using the pool from Method B's Stage 1
        model_a.train(pruned_trees_list=model_b.pruned_trees)

        model_a.prune(prune_threshold=pt, corr_thresh=ct)
        mse_a, _ = model_a.evaluate()

        trees_a_s0 = trees_b_s1
        trees_a_s1 = len(model_a.final_pruned_indices) if model_a.final_pruned_indices is not None else 0
        trees_a_s2 = len(model_a.pruned_trees) if model_a.pruned_trees else 0

        # -------------------------------------------------
        # Method C: Correlation -> SHAP (Baseline)
        # -------------------------------------------------
        print(f"\n--- Running Method C (Corr -> SHAP) [CT={ct}] ---")
        mse_c, trees_c_s1, trees_c_s2 = run_baseline_c_corr_then_shap(workflow, corr_thresh=ct, shap_cutoff_percentile=shap_cutoff_percentile)

        # -------------------------------------------------
        # Method D: Standard Random Forest (Baseline)
        # -------------------------------------------------
        mse_d = run_standard_rf_baseline(workflow)

        # Store Results
        results.append({
            "dataset": dataset,
            "mse_full": mse_full,
            "trees_full": workflow.n_trees,

            # Method A (Ours)
            "mse_A": mse_a,
            "trees_A_input": trees_a_s0,
            "trees_A_stage1_weight": trees_a_s1,
            "trees_A_stage2_final": trees_a_s2,

            # Method B (Baseline Quality)
            "mse_B": mse_b,
            "trees_B_stage1_shap": trees_b_s1,
            "trees_B_stage2_weight": trees_b_s2,
            "trees_B_stage3_final": trees_b_s3,
            "mse_B_stage1_only": mse_basic_shap_only,

            # Method C (Baseline Diversity)
            "mse_C": mse_c,
            "trees_C_stage1_corr": trees_c_s1,
            "trees_C_stage2_final": trees_c_s2,

            # Method D (RF Baseline)
            "mse_D": mse_d,
            "trees_D": workflow.n_trees,

            # Config
            "corr_thresh": ct,
            "prune_thresh": pt,
            "lambda_prune": lp,
            "lambda_div": ld
        })

        df_results = pd.DataFrame(results)
        df_results.to_csv(file_path, index=False)
        print(f"Results for {dataset} saved to {file_path}")

    print(f"\nExperiment finished. All results saved to {file_path}")

if __name__ == "__main__":
    run_main_comparison()