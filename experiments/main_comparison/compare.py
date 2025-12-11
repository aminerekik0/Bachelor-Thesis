from experiments.main_comparison.L1_reg import L1PruningClassifier
from experiments.main_comparison.greedyPruningRegressor import ICPruningRegressor, REPruningRegressor
from experiments.main_comparison.run import correlation_prune, shap_prune_on_subset
from src.PrePruner import PrePruner
from src.MetaOptimizer import MetaOptimizer
from src.EnsembleCreator import EnsembleCreator
from uci_datasets import Dataset
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np
import pandas as pd
import warnings
import math

warnings.filterwarnings("ignore")

# ===============================================================
#  λ_prune, λ_div PER DATASET
# ===============================================================
LAMBDA_CONFIG = {
    "slice": (1.2, 0.3),
    "3droad": (1.2, 0.3),
    "kin40k": (1.2, 0.3),
    "solar": (1.2, 0.3),
    "elevators": (1.2, 0.3),
    "protein": (1.2, 0.3),
    "tamielectric": (1.2, 0.3),
    "keggundirected": (1.2, 0.3),
    "pol": (1.2, 0.3),
    "keggdirected": (1.2, 0.3),
}
CORR_THRESH_CONFIG = {
    "slice": 0.98,
    "3droad": 0.95,
    "kin40k": 0.95,
    "parkinsons": 0.95,
    "solar": 0.95,
    "elevators": 0.95,
    "protein": 0.95,
    "tamielectric": 0.95,
    "keggundirected": 0.997,
    "pol": 0.95,
    "keggdirected": 0.997,
}

# ===============================================================
# HELPER 1: Auto-Tune L1/Linear
# ===============================================================
def find_best_lambda_for_target_size(ensemble, basic_pruned_trees, target_size, tolerance=5):
    lambdas_to_try = [0.01, 0.025, 0.05, 0.1, 0.2, 0.5]
    best_lambda = 0.01
    best_diff = float('inf')

    for l_val in lambdas_to_try:
        temp_model = MetaOptimizer(mode="L1", λ_prune=l_val, λ_div=0.0, epochs=50, lr=1e-2)
        temp_model.attach_to(ensemble)
        temp_model.train(basic_pruned_trees)

        if temp_model.model is not None:
            w_abs = np.abs(temp_model.model.w.detach().numpy().squeeze())
            threshold = 0.01 * np.max(w_abs) if np.max(w_abs) > 0 else 0
            kept_count = np.sum(w_abs > threshold)

            diff = abs(kept_count - target_size)
            if diff < best_diff:
                best_diff = diff
                best_lambda = l_val
            if diff <= tolerance:
                break
    return best_lambda

# ===============================================================
# HELPER 2: Auto-Tune External L1 (The "Second L1")
# ===============================================================
def find_best_alpha_for_external_l1(ensemble, base_preds, y_meta, target_size, tolerance=5):
    alphas_to_try = [0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.58, 0.65, 0.68, 0.8]
    best_alpha = 0.005
    best_diff = float('inf')

    for alpha in alphas_to_try:
        try:
            from L1_reg import L1PruningClassifier
            method = L1PruningClassifier(l_reg=alpha)
            _, selected = method.select(base_preds, y_meta)

            size = len(selected) if selected is not None else 0

            if size == 0:
                continue

            diff = abs(size - target_size)

            if diff < best_diff:
                best_diff = diff
                best_alpha = alpha

            if diff <= tolerance:
                break

        except Exception as e:
            return 0.01

    return best_alpha

# ===============================================================
# Evaluate selected trees WITH META-WEIGHTS
# ===============================================================
def evaluate_with_meta_weights(ensemble, selected_indices):
    if len(selected_indices) == 0:
        return np.nan

    X_eval = ensemble.X_train_meta
    y_eval = ensemble.y_train_meta
    X_test = ensemble.X_test
    y_test = ensemble.y_test

    selected = [ensemble.individual_trees[i] for i in selected_indices]

    preds_eval = np.column_stack([t.predict(X_eval) for t in selected])
    preds_test = np.column_stack([t.predict(X_test) for t in selected])

    lm = LinearRegression()
    lm.fit(preds_eval, y_eval)

    w = np.abs(lm.coef_)
    w = w / (np.sum(w) + 1e-12)

    final_pred = preds_test @ w
    rmse = np.sqrt(mean_squared_error(y_test, final_pred))
    return rmse

# ===============================================================
# GLOBAL Win/Loss/Tie Computation (Standard Error)
# ===============================================================
def compute_dataset_wlt(dataset_summary):
    methods = list(set([row["method"] for row in dataset_summary]))
    WLT = {m: {"win": 0, "loss": 0, "tie": 0} for m in methods}
    datasets = set([row["dataset"] for row in dataset_summary])

    SQRT_N = math.sqrt(10)

    for ds in datasets:
        rows = [r for r in dataset_summary if r["dataset"] == ds]
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                mA = methods[i]
                mB = methods[j]
                try:
                    meanA = next(r["mean_rmse"] for r in rows if r["method"] == mA)
                    stdA = next(r["std_rmse"] for r in rows if r["method"] == mA)
                    meanB = next(r["mean_rmse"] for r in rows if r["method"] == mB)
                    stdB = next(r["std_rmse"] for r in rows if r["method"] == mB)
                except StopIteration:
                    continue

                seA = stdA / SQRT_N
                seB = stdB / SQRT_N

                if meanA + seA < meanB - seB:
                    WLT[mA]["win"] += 1
                    WLT[mB]["loss"] += 1
                elif meanA - seA > meanB + seB:
                    WLT[mA]["loss"] += 1
                    WLT[mB]["win"] += 1
                else:
                    WLT[mA]["tie"] += 1
                    WLT[mB]["tie"] += 1
    return WLT

# ===============================================================
# Run ALL methods
# ===============================================================
def run_all_methods_once(ensemble, dataset_name):

    # 1. Setup
    λ_shap, λ_div = LAMBDA_CONFIG.get(dataset_name, (1.0, 0.3))
    corr_thresh_val = CORR_THRESH_CONFIG.get(dataset_name, 0.95)
    basic = PrePruner()
    basic.attach_to(ensemble)
    basic.train()

    # Pre-calculate meta features for external baselines
    X_meta, y_meta = ensemble.X_train_meta, ensemble.y_train_meta
    base_preds = [t.predict(X_meta) for t in basic.pruned_trees]
    orginal_preds = [t.predict(X_meta) for t in ensemble.individual_trees]

    # --- METHOD 1: SHAP/Linear  ---
    print(f"\n--- [SHAP] Training Proposed Method (λ={λ_shap}) ---")
    print(f" {λ_div}=λ_div λ_prune={λ_shap}")
    linear_meta = MetaOptimizer(mode="SHAP", λ_div=λ_div, λ_prune=λ_shap, epochs=200)
    linear_meta.attach_to(ensemble)
    linear_meta.train(basic.pruned_trees)
    linear_meta.prune(prune_threshold=0.01 , corr_thresh = corr_thresh_val)

    shap_indices = [ensemble.individual_trees.index(t) for t in linear_meta.pruned_trees]
    target_size = len(shap_indices)

    # --- METHOD 2: L1/Linear  ---
    print(f"\n--- [L1/Linear] Tuning to match size ({target_size} trees) ---")
    tuned_l1 = find_best_lambda_for_target_size(ensemble, basic.pruned_trees, target_size)

    linear_meta_l1 = MetaOptimizer(mode="L1", λ_div=0.0, λ_prune=tuned_l1, epochs=200)
    linear_meta_l1.attach_to(ensemble)
    linear_meta_l1.train(basic.pruned_trees)
    linear_meta_l1.prune(prune_threshold=0.01 , corr_thresh = corr_thresh_val)
    l1_indices = [ensemble.individual_trees.index(t) for t in linear_meta_l1.pruned_trees]

    corr_pruned_trees_B = correlation_prune(
        basic.pruned_trees,
        ensemble,
        "regression",
        importance=basic.pruned_tree_weights,
        corr_thresh=0.95
    )
    method_B_idx = [ensemble.individual_trees.index(t) for t in corr_pruned_trees_B]

    corr_pruned_trees_C_stage1 = correlation_prune(
        ensemble.individual_trees,
        ensemble,
        "regression",
        corr_thresh=0.95 ,
        importance=None,
    )

    # Step 2: SHAP pruning on this corr-pruned subset
    shap_pruned_trees_C, _ = shap_prune_on_subset(
        corr_pruned_trees_C_stage1,
        ensemble,
        data_type="regression",
        keep_ratio=0.25,
    )
    method_C_idx = [ensemble.individual_trees.index(t) for t in shap_pruned_trees_C]

    # --- METHOD 3: L1 (External Class Baseline) ---
    print(f"\n--- [L1 External] Tuning to match size ({target_size} trees) ---")
    best_alpha_ext = find_best_alpha_for_external_l1(ensemble, orginal_preds, y_meta, target_size)

    try:

        l1_ext_method = L1PruningClassifier(l_reg=best_alpha_ext , task="regression")
        _, l1_ext_sel = l1_ext_method.select(orginal_preds, y_meta)

        l1_ext_indices = []
        if l1_ext_sel is not None:
            l1_ext_indices = list(l1_ext_sel)
    except Exception as e:
        print(f"[WARN] External L1 failed: {e}")
        l1_ext_indices = []

    # --- METHOD 4 & 5: RE and Individual Contribution (Greedy) ---
    re_indices = []
    ic_indices = []

    if ensemble.data_type == "regression":
        # RE (Reduced Error)
        re_method = REPruningRegressor(n_estimators=target_size)
        _, re_sel_local = re_method.select(np.array(orginal_preds), y_meta) # ensure numpy array

        if re_sel_local:
            re_indices = list(re_sel_local)

        # Individual Contribution (IC)
        ic_method = ICPruningRegressor(n_estimators=target_size)
        _, ic_sel_local = ic_method.select(np.array(orginal_preds), y_meta)

        if ic_sel_local:
            ic_indices = list(ic_sel_local)

    # --- METHOD 6: RF (Random Forest Baseline) ---
    if ensemble.data_type == "regression":
        rf = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=8)
        rf.fit(ensemble.X_train, ensemble.y_train)
        rf_preds = rf.predict(ensemble.X_test)
        rf_metric = np.sqrt(mean_squared_error(ensemble.y_test, rf_preds))
    else:
        rf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=8)
        rf.fit(ensemble.X_train, ensemble.y_train)
        rf_preds = rf.predict(ensemble.X_test)
        rf_metric = accuracy_score(ensemble.y_test, rf_preds)

    # =========================================================
    # PRINT TREE PARAMS TABLE (Method A vs RE)
    # =========================================================
    print(f"\n\n{'='*80}")
    print(f" DETAILED TREE ANALYSIS: {dataset_name}")
    print(f"{'='*80}")

   
    methods_to_inspect = {
        "Method A (SHAP/Linear)": shap_indices,
        "RE (Baseline)": re_indices
    }

    for m_name, indices in methods_to_inspect.items():
        if not indices:
            print(f"\n>> {m_name}: NO TREES SELECTED")
            continue

        print(f"\n>> {m_name} (Count: {len(indices)})")
        print(f"{'TreeID':<8} | {'Depth':<6} | {'MaxFeat':<8} | {'RndState':<10} | {'TopFeature':<10}")
        print("-" * 60)

       
        for idx in sorted(list(set(indices))):
            tree = ensemble.individual_trees[idx]

           
            d = tree.max_depth
            mf = tree.max_features
            rs = tree.random_state

           
            if hasattr(tree, 'feature_importances_'):
                top_feat_idx = np.argmax(tree.feature_importances_)
            else:
                top_feat_idx = "N/A"

            print(f"{idx:<8} | {d:<6} | {str(mf):<8} | {rs:<10} | {top_feat_idx:<10}")
        print("-" * 60)
    print("\n")

    results = {
        "SHAP/Linear": evaluate_with_meta_weights(ensemble, shap_indices),
        "MethodB_SHAP_then_Corr" : evaluate_with_meta_weights(ensemble ,method_B_idx) ,
        "MethodC_Corr_then_SHAP" : evaluate_with_meta_weights(ensemble ,method_C_idx) ,
        "L1/Linear": evaluate_with_meta_weights(ensemble, l1_indices),
        "L1": evaluate_with_meta_weights(ensemble, l1_ext_indices),
        "RE": evaluate_with_meta_weights(ensemble, re_indices),
        "Individual Contribution": evaluate_with_meta_weights(ensemble, ic_indices),
        "RF": rf_metric
    }

    sizes = {
        "SHAP/Linear": len(shap_indices),
        "MethodB_SHAP_then_Corr" : len(corr_pruned_trees_B) ,
        "MethodC_Corr_then_SHAP" : len(shap_pruned_trees_C) ,
        "L1/Linear": len(l1_indices),
        "L1": len(l1_ext_indices),
        "RE": len(re_indices),
        "Individual Contribution": len(ic_indices),
        "RF": 200
    }

    return results, sizes

# ===============================================================
# Loop Helper
# ===============================================================
def run_methods_for_dataset_10_times(X, y, dataset_name):
    print(f"\n==================================================")
    print(f" DATASET: {dataset_name}")
    print(f"==================================================")

    
    method_list = ["SHAP/Linear", "L1/Linear", "L1", "RE", "Individual Contribution", "RF", "MethodB_SHAP_then_Corr", "MethodC_Corr_then_SHAP"]

    scores = {m: [] for m in method_list}
    size_lists = {m: [] for m in method_list}

    for run in range(10):
        print(f"\n>> Run {run+1}/10")
        ensemble = EnsembleCreator(X=X, y=y, data_type="regression")
        ensemble.train_base_trees()

        one_results, one_sizes = run_all_methods_once(ensemble, dataset_name)

        for method in scores:
            scores[method].append(one_results.get(method, np.nan))
            size_lists[method].append(one_sizes.get(method, 0))

    return scores, size_lists

# ===============================================================
# MAIN
# ===============================================================
def main():
    regression_sets = [

        "KeggDirected", "KeggUndirected", "Kin40k", "Solar", "Protein", "Tamielectric", "Pol", "Slice", "3droad"
    ]

    dataset_summary_rows = []
    dataset_size_rows = []

    for ds in regression_sets:
        try:
            data = Dataset(ds)
            X = data.x.astype(np.float32)
            y = data.y.ravel()

            scores, size_lists = run_methods_for_dataset_10_times(X, y, ds)

            for method, vals in scores.items():
                dataset_summary_rows.append({
                    "dataset": ds,
                    "method": method,
                    "mean_rmse": np.nanmean(vals),
                    "std_rmse": np.nanstd(vals),
                })

            for method, sizes in size_lists.items():
                dataset_size_rows.append({
                    "dataset": ds,
                    "method": method,
                    "mean_size": np.nanmean(sizes),
                    "std_size": np.nanstd(sizes)
                })

        except Exception as e:
            print(f"Skipping {ds} due to error: {e}")

    
    global_wlt = compute_dataset_wlt(dataset_summary_rows)

    
    pd.DataFrame(dataset_summary_rows).to_csv("results/results_rmse_final_all.csv", index=False)
    pd.DataFrame(dataset_size_rows).to_csv("results/results_sizes_final_all.csv", index=False)

    rows = []
    for method, vals in global_wlt.items():
        rows.append({"method": method, **vals})

    df_wlt = pd.DataFrame(rows)
    df_wlt.to_csv("GLOBAL_win_loss_tie.csv", index=False)

    print("\n======================================")
    print(" GLOBAL WIN / LOSS / TIE SUMMARY")
    print("======================================")
    print(pd.DataFrame(rows))

if __name__ == "__main__":
    main()
