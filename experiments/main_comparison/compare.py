from experiments.main_comparison.run import correlation_prune, shap_prune_on_subset
from src.BasicMetaModel import BasicMetaModel
from src.LinearMetaModel import LinearMetaModel
from src.ExplainableTreeEnsemble import ExplainableTreeEnsemble
from uci_datasets import Dataset
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np
import pandas as pd
import warnings
import math

# --- NEW IMPORTS ---
from greedyPruningRegressor import REPruningRegressor, ICPruningRegressor

# Suppress warnings
warnings.filterwarnings("ignore")

# ===============================================================
# FIXED λ_prune, λ_div PER DATASET
# ===============================================================
LAMBDA_CONFIG = {
    "slice": (1.0, 0.3),
    "3droad": (1.0, 0.3),
    "kin40k": (1.2, 0.3),
    "parkinsons": (1.2, 0.3),
    "solar": (0.8, 0.3),
    "elevators": (1.0, 0.3),
    "protein": (1.0, 0.3),
    "tamielectric": (1.0, 0.3),
    "keggundirected": (1.2, 0.3),
    "pol": (0.8, 0.3),
    "covertype": (1.0, 0.5),
    "higgs": (0.7, 0.5),
}

# ===============================================================
# HELPER 1: Auto-Tune L1/Linear (Your PyTorch Model)
# ===============================================================
def find_best_lambda_for_target_size(ensemble, basic_pruned_trees, target_size, tolerance=5):
    lambdas_to_try = [ 0.01, 0.025, 0.05, 0.1, 0.2, 0.5]
    best_lambda = 0.01
    best_diff = float('inf')

    for l_val in lambdas_to_try:
        temp_model = LinearMetaModel(mode="L1", λ_prune=l_val, λ_div=0.0, epochs=50, lr=1e-2)
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
    alphas_to_try = [0.01, 0.05, 0.1, 0.2, 0.5 ,0.65 , 0.68 , 0.7 , 0.9]
    best_alpha = 0.005
    best_diff = float('inf')

    for alpha in alphas_to_try:
        try:
            from L1_reg import L1PruningClassifier
            method = L1PruningClassifier(l_reg=alpha)
            _, selected = method.select(base_preds, y_meta)

            size = len(selected) if selected is not None else 0
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
# Run ALL methods (one run)
# ===============================================================
def run_all_methods_once(ensemble, dataset_name):

    # 1. Setup
    λ_shap, λ_div = LAMBDA_CONFIG.get(dataset_name, (1.0, 0.3))
    basic = BasicMetaModel("regression")
    basic.attach_to(ensemble)
    basic.train()

    # Pre-calculate meta features for external baselines
    X_meta, y_meta = ensemble.X_train_meta, ensemble.y_train_meta
    base_preds = [t.predict(X_meta) for t in basic.pruned_trees]
    orginal_preds = [t.predict(X_meta) for t in ensemble.individual_trees]

    # --- METHOD 1: SHAP/Linear (Your Method) ---
    print(f"\n--- [SHAP] Training Proposed Method (λ={λ_shap}) ---")
    linear_meta = LinearMetaModel(mode="SHAP", λ_div=λ_div, λ_prune=λ_shap, epochs=200)
    linear_meta.attach_to(ensemble)
    linear_meta.train(basic.pruned_trees)
    linear_meta.prune(prune_threshold=0.01)

    shap_indices = [ensemble.individual_trees.index(t) for t in linear_meta.pruned_trees]
    target_size = len(shap_indices)

    # --- METHOD 2: L1/Linear (Your PyTorch Baseline) ---
    print(f"\n--- [L1/Linear] Tuning to match size ({target_size} trees) ---")
    tuned_l1 = find_best_lambda_for_target_size(ensemble, basic.pruned_trees, target_size)

    linear_meta_l1 = LinearMetaModel(mode="L1", λ_div=0.0, λ_prune=tuned_l1, epochs=200)
    linear_meta_l1.attach_to(ensemble)
    linear_meta_l1.train(basic.pruned_trees)
    linear_meta_l1.prune(prune_threshold=0.01)
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

    corr_stage1_size = len(corr_pruned_trees_C_stage1)


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

    from L1_reg import L1PruningClassifier

    try:
        print("best lambda" , best_alpha_ext)
        l1_ext_method = L1PruningClassifier(l_reg=best_alpha_ext , task="regression")
        _, l1_ext_sel = l1_ext_method.select(orginal_preds, y_meta)

        l1_ext_indices = []
        if l1_ext_sel:
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

    # --- Compile Results ---
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

    # Updated list of methods
    # UPDATE THIS LINE
    method_list = ["SHAP/Linear", "L1/Linear", "L1", "RE", "Individual Contribution", "RF", "MethodB_SHAP_then_Corr", "MethodC_Corr_then_SHAP"]

    scores = {m: [] for m in method_list}
    size_lists = {m: [] for m in method_list}

    for run in range(1):
        print(f"\n>> Run {run+1}/10")
        ensemble = ExplainableTreeEnsemble(X=X, y=y, data_type="regression")
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
        "kin40k",
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

    # Calculate WLT
    global_wlt = compute_dataset_wlt(dataset_summary_rows)

    # Save
    pd.DataFrame(dataset_summary_rows).to_csv("results_rmse_final_all.csv", index=False)
    pd.DataFrame(dataset_size_rows).to_csv("results_sizes_final_all.csv", index=False)

    rows = []
    for method, vals in global_wlt.items():
        rows.append({"method": method, **vals})

    print("\n======================================")
    print(" GLOBAL WIN / LOSS / TIE SUMMARY")
    print("======================================")
    print(pd.DataFrame(rows))

if __name__ == "__main__":
    main()