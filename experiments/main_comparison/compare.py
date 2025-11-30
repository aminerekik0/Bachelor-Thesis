from experiments.main_comparison.CwE import ConstructiveWithExploration
from experiments.main_comparison.Pw_oE import PruningWithoutExploration
from experiments.main_comparison.DREP import DREPPruner
from experiments.main_comparison.EarlyStop import EarlyStopPruning

from experiments.main_comparison.extra_pruning_methods import (
    REPruning,
    KappaPruning,
    KTPruning ,
    EPRPruner
)

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
    "slice": (1.5, 0.3),
    "3droad": (1.5, 0.3),
    "kin40k": (1.5, 0.3),
    "parkinsons": (1.2, 0.3),
    "solar": (1.0, 0.3),
    "elevators": (1.3, 0.3),
    "protein": (1.5, 0.3),
    "tamielectric": (1.0, 0.3),

    # NEW regression datasets
    
    "keggundirected": (1.2, 0.3),
    "pol": (1.2, 0.3),

    # classification (kept but not executed)
    "covertype": (1.0, 0.5),
    "higgs": (0.7, 0.5),
}

# ===============================================================
# FIXED TYPE MAP
# ===============================================================
DATASET_TYPE = {
    "slice": "regression",
    "3droad": "regression",
    "kin40k": "regression",
    "parkinsons": "regression",
    "solar": "regression",
    "elevators": "regression",
    "protein": "regression",
    "tamielectric": "regression",
    "song": "regression",
    "keggundirected": "regression",
    "pol": "regression",

    "covertype": "classification",
    "higgs": "classification",
}


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
# Run ALL methods once (with ONE ensemble)
# ===============================================================
def run_all_methods_once(ensemble, dataset_name):

    λ_prune, λ_div = LAMBDA_CONFIG[dataset_name]

    # ===== SHAP Meta-Model =====
    basic = BasicMetaModel("regression")
    basic.attach_to(ensemble)
    basic.train()

    linear_meta = LinearMetaModel(λ_div=λ_div, λ_prune=λ_prune)
    linear_meta.attach_to(ensemble)
    linear_meta.train(basic.pruned_trees)
    linear_meta.prune()
    linear_meta.evaluate()
    
    shap_indices = [ensemble.individual_trees.index(t) for t in linear_meta.pruned_trees]
    results = {"SHAP/Linear": evaluate_with_meta_weights(ensemble, shap_indices)}

    # === Other pruning methods ===
    methods = {
        "CwE": ConstructiveWithExploration("regression"),
        "Pw/oE": PruningWithoutExploration("regression"),
        "EarlyStop": EarlyStopPruning("regression"),
        "Kappa": KappaPruning("regression"),
        "KT": KTPruning("regression"),
        
        
    }

    X_meta, y_meta = ensemble.X_train_meta, ensemble.y_train_meta
    base_preds = [t.predict(X_meta) for t in ensemble.individual_trees]

    for name, method in methods.items():
        _, sel = method.select(base_preds, y_meta)
        results[name] = evaluate_with_meta_weights(ensemble, sel)

    return results


# ===============================================================
# METHOD B — Run pruning 10 times AND create new base trees each time
# ===============================================================
def run_methods_for_dataset_10_times(X, y, dataset_name):

    print(f"\n=== Training base ensemble 10× for dataset: {dataset_name} ===")

    scores = {
        "SHAP/Linear": [],
        "CwE": [],
        "Pw/oE": [],
        "EarlyStop": [],
        "Kappa": [],
        "KT": [],
    }

    for run in range(10):
        print(f"[Run {run+1}/10] {dataset_name}")

        # ------------------------------------------------------
        # NEW: Create a fresh ensemble & train base trees EACH RUN
        # ------------------------------------------------------
        ensemble = ExplainableTreeEnsemble(X=X, y=y, data_type="regression")
        ensemble.train_base_trees()
        # ------------------------------------------------------

        results_once = run_all_methods_once(ensemble, dataset_name)

        for method in scores:
            scores[method].append(results_once[method])

    return scores


# ===============================================================
# MAIN LOOP
# ===============================================================
def main():

    regression_sets = [
        "protein", "tamielectric",
         "keggundirected", "pol",
        "slice", "3droad", "kin40k",
        "parkinsons", "solar", "elevators",
        
    ]

    final_rows = []

    for ds in regression_sets:

        data = Dataset(ds)
        X = data.x.astype(np.float32)
        y = data.y.ravel()

        scores = run_methods_for_dataset_10_times(X, y, ds)

        for method, values in scores.items():
            values = np.array(values)
            mean_rmse = np.mean(values)
            std_rmse = np.std(values)

            final_rows.append({
                "dataset": ds,
                "method": method,
                "mean_rmse": mean_rmse,
                "std_rmse": std_rmse
            })

            print(f"{ds} | {method}: {mean_rmse:.4f} ± {std_rmse:.4f}")

    df = pd.DataFrame(final_rows)
    df.to_csv("regression_mean_std_method_B.csv", index=False)

    print("\nSaved → regression_mean_std_method_B.csv")


if __name__ == "__main__":
    main()
