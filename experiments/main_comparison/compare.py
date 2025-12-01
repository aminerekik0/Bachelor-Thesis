from experiments.main_comparison.ADP import ADPPruner
from experiments.main_comparison.CwE import ConstructiveWithExploration
from experiments.main_comparison.Pw_oE import PruningWithoutExploration
from experiments.main_comparison.DREP import DREPPruner
from experiments.main_comparison.EarlyStop import EarlyStopPruning

from experiments.main_comparison.extra_pruning_methods import (
    REPruning,
    KappaPruning,
    KTPruning,
    EPRPruner
)
from experiments.main_comparison.run import correlation_prune, shap_prune_on_subset

from src.BasicMetaModel import BasicMetaModel
from src.LinearMetaModel import LinearMetaModel
from src.ExplainableTreeEnsemble import ExplainableTreeEnsemble

import os
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from uci_datasets import Dataset
import pandas as pd


# ===============================================================
# FIXED λ_prune, λ_div PER DATASET
# ===============================================================
LAMBDA_CONFIG = {
    "slice": (1.2, 0.3),
    "3droad": (1.2, 0.3),
    "kin40k": (1.5, 0.3),
    "parkinsons": (1.2, 0.3),
    "solar": (1.0, 0.3),
    "elevators": (1.3, 0.3),
    "protein": (1.5, 0.3),
    "tamielectric": (1.0, 0.3),

    "keggundirected": (1.2, 0.3),
    "pol": (1.2, 0.3),

    "covertype": (1.0, 0.5),
    "higgs": (0.7, 0.5),
}

DATASET_TYPE = {k: "regression" for k in LAMBDA_CONFIG.keys()}
DATASET_TYPE["covertype"] = "classification"
DATASET_TYPE["higgs"] = "classification"


# ===============================================================
# Evaluate selected trees WITH META-WEIGHTS
# ===============================================================
def evaluate_with_meta_weights(ensemble, selected_indices):
    if len(selected_indices) == 0:
        return None

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
# GLOBAL Win/Loss/Tie Computation (dataset-level)
# ===============================================================
def compute_dataset_wlt(dataset_summary):
    """
    Compute W/L/T using mean ± std per dataset
    """
    methods = list(set([row["method"] for row in dataset_summary]))
    WLT = {m: {"win": 0, "loss": 0, "tie": 0} for m in methods}

    datasets = set([row["dataset"] for row in dataset_summary])

    for ds in datasets:
        rows = [r for r in dataset_summary if r["dataset"] == ds]

        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                mA = methods[i]
                mB = methods[j]

                meanA = next(r["mean_rmse"] for r in rows if r["method"] == mA)
                stdA  = next(r["std_rmse"]  for r in rows if r["method"] == mA)

                meanB = next(r["mean_rmse"] for r in rows if r["method"] == mB)
                stdB  = next(r["std_rmse"]  for r in rows if r["method"] == mB)

                # Comparison using mean ± std
                if meanA + stdA < meanB - stdB:
                    WLT[mA]["win"] += 1
                    WLT[mB]["loss"] += 1
                elif meanA - stdA > meanB + stdB:
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

    λ_prune, λ_div = LAMBDA_CONFIG[dataset_name]

    basic = BasicMetaModel("regression")
    basic.attach_to(ensemble)
    basic.train()

    linear_meta = LinearMetaModel(λ_div=λ_div, λ_prune=λ_prune)
    linear_meta.attach_to(ensemble)
    linear_meta.train(basic.pruned_trees)
    linear_meta.prune()

    shap_indices = [ensemble.individual_trees.index(t) for t in linear_meta.pruned_trees]


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
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    if ensemble.data_type == "regression":
        rf = RandomForestRegressor(n_estimators=200, random_state=42 , max_depth = 8)
    else:
        rf = RandomForestClassifier(n_estimators=200, random_state=42  ,max_depth = 8)

    rf.fit(ensemble.X_train, ensemble.y_train)
    rf_preds = rf.predict(ensemble.X_test)
    if ensemble.data_type == "regression":
        rf_metric = np.sqrt(mean_squared_error(ensemble.y_test, rf_preds))
    else:
        from sklearn.metrics import accuracy_score
        rf_metric = accuracy_score(ensemble.y_test, rf_preds)

    method_C_idx = [ensemble.individual_trees.index(t) for t in shap_pruned_trees_C]


    results = {
        "SHAP/Linear": evaluate_with_meta_weights(ensemble, shap_indices) ,
        "MethodB_SHAP_then_Corr" : evaluate_with_meta_weights(ensemble ,method_B_idx) ,
        "MethodC_Corr_then_SHAP" : evaluate_with_meta_weights(ensemble ,method_C_idx) ,
        "RF" : rf_metric
    }

    methods = {
        "ADP": ADPPruner("regression"),
        "Pw-oE": PruningWithoutExploration("regression"),
        "CwE": ConstructiveWithExploration("regression"),
        "EarlyStop": EarlyStopPruning("regression"),


    }

    X_meta, y_meta = ensemble.X_train_meta, ensemble.y_train_meta
    base_preds = [t.predict(X_meta) for t in ensemble.individual_trees]

    for name, method in methods.items():
        _, sel = method.select(base_preds, y_meta)


        if not sel:
           results[name] = np.nan
        else:
            results[name] = evaluate_with_meta_weights(ensemble, sel)

    return results


# ===============================================================
# Run 10× for dataset
# ===============================================================
def run_methods_for_dataset_10_times(X, y, dataset_name):

    print(f"\n=== Training base ensemble 10× on dataset: {dataset_name} ===")

    scores = {
        "SHAP/Linear": [],
        "RF":[] ,
        "MethodB_SHAP_then_Corr" : [] ,
        "MethodC_Corr_then_SHAP" : [] ,
        "ADP": [],
        "Pw-oE": [],
        "CwE": [],
        "EarlyStop": [],

    }

    for run in range(5):
        print(f"  Run {run+1}/10")

        ensemble = ExplainableTreeEnsemble(X=X, y=y, data_type="regression")
        ensemble.train_base_trees()

        one = run_all_methods_once(ensemble, dataset_name)

        for method in scores:
            scores[method].append(one[method])

    return scores


# ===============================================================
# MAIN LOOP — GLOBAL W/L/T using mean ± std
# ===============================================================
def main():

    regression_sets = [
        "protein" ,
        "slice",
     "3droad" ,
    "kin40k",
    "parkinsons",
    "solar",
    "elevators",
    "protein",
    "tamielectric",
    "pol",
    ]

    dataset_summary_rows = []

    for ds in regression_sets:

        data = Dataset(ds)
        X = data.x.astype(np.float32)
        y = data.y.ravel()

        scores = run_methods_for_dataset_10_times(X, y, ds)

        for method, vals in scores.items():
            dataset_summary_rows.append({
                "dataset": ds,
                "method": method,
                "mean_rmse": np.nanmean(vals),  # ignore NaNs
                "std_rmse": np.nanstd(vals),    # ignore NaNs
            })

    # ===== GLOBAL W/L/T across datasets using mean ± std =====
    global_wlt = compute_dataset_wlt(dataset_summary_rows)

    # Save dataset-level summary
    df1 = pd.DataFrame(dataset_summary_rows)
    df1.to_csv("per_dataset_results_2.0.csv", index=False)

    # Save GLOBAL W/L/T summary
    rows = []
    for method, vals in global_wlt.items():
        rows.append({
            "method": method,
            "win": vals["win"],
            "loss": vals["loss"],
            "tie": vals["tie"]
        })

    df2 = pd.DataFrame(rows)
    df2.to_csv("GLOBAL_win_loss_tie.csv", index=False)

    print("\n======================================")
    print(" GLOBAL WIN / LOSS / TIE SUMMARY")
    print("======================================")
    print(df2)


if __name__ == "__main__":
    main()