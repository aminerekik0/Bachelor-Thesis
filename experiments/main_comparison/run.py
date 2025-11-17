import pandas as pd
import os
import sys
import numpy as np
import shap
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
from sklearn.linear_model import LinearRegression, LogisticRegression
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

def _get_meta_features_static(X, trees_list):
    if not trees_list:
        return np.array([]).reshape(X.shape[0], 0)
    return np.column_stack([t.predict(X) for t in trees_list]).astype(np.float32)

def run_baseline_b_shap_then_corr(model_b, corr_thresh=0.9):
    print(f"=== [Baseline B] Stage 2: Pruning {len(model_b.pruned_trees)} trees by correlation (threshold={corr_thresh}) ===")
    
    if not model_b.pruned_trees:
        print("[ERROR] [Baseline B] Must call train() on BasicMetaModel first.")
        return None, 0

    X_meta_train_pruned = model_b._get_meta_features(model_b.workflow.X_train_meta, model_b.pruned_trees)
    
    if X_meta_train_pruned.ndim == 1 or X_meta_train_pruned.shape[1] <= 1:
        print("[INFO] [Baseline B] Skipped (<= 1 tree).")
        shap_then_corr_trees = model_b.pruned_trees
        shap_then_corr_weights = model_b.pruned_tree_weights
    else:
        corr_matrix = np.corrcoef(X_meta_train_pruned.T)
        np.fill_diagonal(corr_matrix, 0)
        
        if corr_matrix.size == 0:
             print("[INFO] [Baseline B] No pairwise correlations to check.")
             shap_then_corr_trees = model_b.pruned_trees
             shap_then_corr_weights = model_b.pruned_tree_weights
        else:
            redundant_local_indices = set(np.unique(np.where(np.abs(corr_matrix) > corr_thresh)[0]))

            final_trees_list = []
            final_weights_list = []
            for i, (tree, weight) in enumerate(zip(model_b.pruned_trees, model_b.pruned_tree_weights)):
                if i not in redundant_local_indices:
                    final_trees_list.append(tree)
                    final_weights_list.append(weight)

            shap_then_corr_trees = final_trees_list
            shap_then_corr_weights = np.array(final_weights_list)
            print(f"[INFO] [Baseline B] Removed {len(redundant_local_indices)} trees. Final size: {len(shap_then_corr_trees)}")

    X_test = model_b.workflow.X_test
    y_test = model_b.workflow.y_test
    data_type = model_b.data_type

    if shap_then_corr_weights is None or len(shap_then_corr_weights) == 0:
        print("[WARN] [Baseline B] No trees/weights found.")
        return None, len(shap_then_corr_trees)
    
    normalized_weights = shap_then_corr_weights / (np.sum(shap_then_corr_weights) + 1e-12)

    if len(shap_then_corr_trees) == 1:
        pruned_preds_matrix = model_b._get_meta_features(X_test, shap_then_corr_trees)
        final_preds = pruned_preds_matrix.squeeze()
    else:
        if data_type == "regression":
            pruned_preds_matrix = model_b._get_meta_features(X_test, shap_then_corr_trees)
            final_preds = pruned_preds_matrix @ normalized_weights
        else:
            tree_preds = np.vstack([t.predict(X_test) for t in shap_then_corr_trees])
            final_preds = np.apply_along_axis(
                lambda x: np.bincount(x.astype(int), weights=normalized_weights).argmax(),
                axis=0,
                arr=tree_preds
            )

    if data_type == "regression":
        mse = mean_squared_error(y_test, final_preds)
        print("[Baseline B] Final (SHAP->Corr) Pruned MSE (Weighted):", mse)
        return mse, len(shap_then_corr_trees)
    else:
        acc = accuracy_score(y_test, final_preds)
        print ("[Baseline B] Final (SHAP->Corr) Pruned Metric (Weighted):", acc)
        return acc, len(shap_then_corr_trees)

def run_baseline_c_corr_then_shap(workflow, corr_thresh=0.9, shap_cutoff_percentile=70):
    all_trees = workflow.individual_trees
    X_train_full = _get_meta_features_static(workflow.X_train_meta, all_trees)
    y_train = workflow.y_train_meta
    X_val_full = _get_meta_features_static(workflow.X_eval_meta, all_trees)
    n_trees = X_train_full.shape[1]
    data_type = workflow.data_type
    
    print(f"=== [Baseline C] Stage 1: Redundancy filtering (threshold={corr_thresh}) ===")
    corr_matrix = np.corrcoef(X_train_full, rowvar=False)
    np.fill_diagonal(corr_matrix, 0)

    keep_mask = np.ones(n_trees, dtype=bool)
    for i in range(n_trees):
        for j in range(i + 1, n_trees):
            if keep_mask[j] and abs(corr_matrix[i, j]) > corr_thresh:
                keep_mask[j] = False
    
    original_indices_stage1 = np.where(keep_mask)[0]
    trees_stage1 = [all_trees[i] for i in original_indices_stage1]
    X_train_red = X_train_full[:, keep_mask]
    X_val_red = X_val_full[:, keep_mask]
    kept_after_corr = len(trees_stage1)
    print(f"Kept {kept_after_corr}/{n_trees} trees after correlation filtering\n")

    if kept_after_corr == 0:
        print("[WARN] [Baseline C] Stage 1 removed all trees.")
        return None, 0

    print(f"=== [Baseline C] Stage 2: SHAP pruning (Remove bottom {shap_cutoff_percentile}%) ===")
    
    if data_type == "regression":
        model_red = LinearRegression().fit(X_train_red, y_train)
    else:
        model_red = LogisticRegression(random_state=workflow.random_state, max_iter=1000).fit(X_train_red, y_train)

    explainer = shap.Explainer(model_red, X_val_red)
    shap_vals = explainer(X_val_red).values

    if shap_vals.ndim == 3:
        shap_vals_for_importance = np.mean(np.abs(shap_vals), axis=2)
    else:
        shap_vals_for_importance = shap_vals

    mean_shap = np.mean(np.abs(shap_vals_for_importance), axis=0)
    shap_threshold = np.percentile(mean_shap, shap_cutoff_percentile)
    
    keep_idx_shap_local = [i for i, shap_val in enumerate(mean_shap) if shap_val >= shap_threshold]
    final_keep_indices = [original_indices_stage1[i] for i in keep_idx_shap_local]
    corr_first_pruned_trees = [all_trees[i] for i in final_keep_indices]
    print(f"Kept {len(corr_first_pruned_trees)}/{kept_after_corr} trees after SHAP pruning.\n")

    X_test = workflow.X_test
    y_test = workflow.y_test

    if not corr_first_pruned_trees:
        print("[WARN] [Baseline C] No pruned trees to evaluate.")
        return None, 0

    print(f"[INFO] [Baseline C]: Re-training final model on {len(corr_first_pruned_trees)} pruned trees...")
    X_train_final = _get_meta_features_static(workflow.X_train_meta, corr_first_pruned_trees)
    y_train_final = workflow.y_train_meta
    
    if data_type == "regression":
        final_eval_model = LinearRegression().fit(X_train_final, y_train_final)
        w_abs = np.abs(final_eval_model.coef_)
        normalized_weights = w_abs / (np.sum(w_abs) + 1e-12)
        pruned_preds_matrix = _get_meta_features_static(X_test, corr_first_pruned_trees)
        if pruned_preds_matrix.ndim == 1:
            final_preds = pruned_preds_matrix.squeeze()
        else:
            final_preds = pruned_preds_matrix @ normalized_weights
        metric = mean_squared_error(y_test, final_preds)
        print("[Baseline C] Final Pruned MSE (Weighted by Retrain):", metric)
    else:
        final_eval_model = LogisticRegression(random_state=workflow.random_state, max_iter=1000).fit(X_train_final, y_train_final)
        pruned_preds_matrix = _get_meta_features_static(X_test, corr_first_pruned_trees)
        final_preds = final_eval_model.predict(pruned_preds_matrix)
        metric = accuracy_score(y_test, final_preds)
        print ("[Baseline C] Final Pruned Metric (Weighted by Retrain)", metric)
    
    return metric, len(corr_first_pruned_trees)


def run_main_comparison():
    print("Starting Main Comparison Experiment...")

    dataset_names = ["3droad", "slice"]
    
    # --- Consistent Parameters ---
    keep_ratio = 0.3
    shap_cutoff_percentile = (1 - keep_ratio) * 100
    # ---

    output_dir = os.path.join(current_dir, "results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, "main_comparison.csv")

    results = []

    for dataset in dataset_names:
        print(f"\n--- Processing Dataset: {dataset} ---")

        workflow = ExplainableTreeEnsemble(data_type="regression", dataset_name=dataset)
        workflow.train_base_trees()
        mse_full = workflow.mse
        
        if dataset == "bike":
            corr_thresh = 0.99
        elif dataset == "3droad":
            corr_thresh = 0.9
        elif dataset == "slice":
            corr_thresh = 0.95
        else:
            corr_thresh = 0.9

        print("\n--- Running Method B (BasicMetaModel: SHAP -> Corr) ---")
        model_b = BasicMetaModel(keep_ratio=keep_ratio)
        model_b.attach_to(workflow)
        model_b.train()
        mse_basic_shap_only, _ = model_b.evaluate() 

        mse_b, num_trees_b = run_baseline_b_shap_then_corr(model_b, corr_thresh=corr_thresh)

        print("\n--- Running Method A (LinearMetaModel: HRP) ---")
        model_a = LinearMetaModel()
        model_a.attach_to(workflow)
        model_a.train(pruned_trees_list=model_b.pruned_trees)
        model_a.prune(
            corr_thresh=corr_thresh
        )
        mse_a, _ = model_a.evaluate()
        num_trees_a = len(model_a.pruned_trees) if model_a.pruned_trees else 0

        print("\n--- Running Method C (BasicMetaModel: corr -> SHAP) ---")
        mse_c, num_trees_c = run_baseline_c_corr_then_shap(workflow, corr_thresh, shap_cutoff_percentile)

        results.append({
            "dataset": dataset,
            "corr_thresh": corr_thresh,
            "keep_ratio": keep_ratio,
            "mse_full": mse_full,
            "trees_full": workflow.n_trees,
            "mse_method_A": mse_a,
            "trees_method_A": num_trees_a,
            "mse_method_B": mse_b,
            "trees_method_B": num_trees_b,
            "mse_method_C": mse_c,
            "trees_method_C": num_trees_c,
            "mse_basic_shap_only": mse_basic_shap_only,
        })

        df_results = pd.DataFrame(results)
        df_results.to_csv(file_path, index=False)

    print(f"\nExperiment finished. Results saved to {file_path}")


if __name__ == "__main__":
    run_main_comparison()