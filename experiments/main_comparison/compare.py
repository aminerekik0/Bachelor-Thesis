from experiments.main_comparison.CW_OE import ConstructiveWithoutExploration
from experiments.main_comparison.CwE import ConstructiveWithExploration
from experiments.main_comparison.PwE import PruningWithExploration
from experiments.main_comparison.Pw_oE import PruningWithoutExploration
from src.BasicMetaModel import BasicMetaModel
from src.LinearMetaModel import LinearMetaModel

import os
import numpy as np
from scipy.stats import mode
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, roc_auc_score
from uci_datasets import Dataset
from src.ExplainableTreeEnsemble import ExplainableTreeEnsemble

def evaluate_selected_trees(ensemble, selected_indices, method_name, weights=None):
    if len(selected_indices) == 0:
        print(f"[WARN] No trees selected by {method_name}.")
        return None

    X_test = ensemble.X_test
    y_test = ensemble.y_test
    selected_trees = [ensemble.individual_trees[i] for i in selected_indices]

    # Get predictions for all selected trees: Shape (n_selected, n_samples)
    preds_matrix = np.vstack([t.predict(X_test) for t in selected_trees])

    # Normalize weights if provided
    if weights is not None:
        weights = np.array(weights)
        weights = weights / (np.sum(weights) + 1e-12)

    if ensemble.data_type == "regression":
        if weights is not None:
            # Weighted Average
            final_pred = np.average(preds_matrix, axis=0, weights=weights)
        else:
            # Simple Average (Standard Bagging)
            final_pred = np.mean(preds_matrix, axis=0)

        rmse = np.sqrt(mean_squared_error(y_test, final_pred))
        print(f"{method_name} -> Trees: {len(selected_trees)}, RMSE: {rmse:.4f}")
        return {"trees": len(selected_trees), "rmse": rmse}

    else:
        # Classification
        if weights is not None:
            # Weighted Voting
            final_pred = np.apply_along_axis(
                lambda x: np.bincount(x.astype(int), weights=weights).argmax(),
                axis=0,
                arr=preds_matrix
            )
        else:
            # Majority Voting (Mode)
            from scipy.stats import mode
            final_pred = mode(preds_matrix, axis=0, keepdims=False).mode

        acc = accuracy_score(y_test, final_pred)
        f1 = f1_score(y_test, final_pred, average="weighted")
        try:
            auc = roc_auc_score(y_test, final_pred)
        except ValueError:
            auc = 0.0

        print(f"{method_name} -> Trees: {len(selected_trees)}, Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        return {"trees": len(selected_trees), "acc": acc, "f1": f1, "auc": auc}


def grid_search_best_lambdas(ensemble, basic_pruned_trees):
    """
    Performs a grid search to find the best lambda_prune and lambda_div.
    Evaluates on the ENSEMBLE'S VALIDATION SET (X_eval_meta), not the test set,
    to avoid data leakage.
    """
    print("\n--- Starting Grid Search for LinearMetaModel ---")

    # Define Grid Ranges
    prune_grid = [1.0, 1.2 ,1.4 ,1,5]
    div_grid = [0.1,0.2, 0.3, 0.5]

    best_score = float('inf') if ensemble.data_type == "regression" else -1.0
    best_params = (1.0, 0.1) # Default fallback

    # Use Validation set for Grid Search
    X_val = ensemble.X_eval_meta
    y_val = ensemble.y_eval_meta

    for lp in prune_grid:
        for ld in div_grid:
            # Train candidate model
            lm = LinearMetaModel(λ_div=ld, λ_prune=lp)
            lm.attach_to(ensemble)
            # Suppress print output during grid search if possible to keep logs clean
            try:
                lm.train(basic_pruned_trees)
                lm.prune()
            except Exception as e:
                print(f"Grid Search Error at params {lp}, {ld}: {e}")
                continue

            survivors = lm.pruned_trees
            if not survivors:
                continue

            # Extract weights
            weight_map = dict(zip(lm.initial_pruned_trees, lm.w_final))
            w = np.array([weight_map[t] for t in survivors])
            w = w / (np.sum(w) + 1e-12)

            # Predict on Validation Set
            preds = np.vstack([t.predict(X_val) for t in survivors])

            if ensemble.data_type == "regression":
                final_pred = np.average(preds, axis=0, weights=w)
                score = mean_squared_error(y_val, final_pred) # MSE

                # We want minimal MSE
                if score < best_score:
                    best_score = score
                    best_params = (lp, ld)
                    print(f"New Best: (λ_prune={lp}, λ_div={ld}) -> Val MSE: {score:.4f}")
            else:
                # Classification
                final_pred = np.apply_along_axis(
                    lambda x: np.bincount(x.astype(int), weights=w).argmax(),
                    axis=0,
                    arr=preds
                )
                score = accuracy_score(y_val, final_pred) # Accuracy

                # We want maximal Accuracy
                if score > best_score:
                    best_score = score
                    best_params = (lp, ld)
                    print(f"New Best: (λ_prune={lp}, λ_div={ld}) -> Val Acc: {score:.4f}")

    print(f"--- Grid Search Finished. Best: λ_prune={best_params[0]}, λ_div={best_params[1]} ---\n")
    return best_params


def run_all_methods(ensemble, methods_dict, λ_prune=0.5, λ_div=0.02):
    results = {}

    # --- SHAP/Linear MetaModel (YOUR METHOD) ---
    basic = BasicMetaModel(ensemble.data_type)
    basic.attach_to(ensemble)
    basic.train() # Method B stage 1

    # Instantiate LinearMetaModel with optimized lambdas
    linear_meta = LinearMetaModel(λ_div=λ_div, λ_prune=λ_prune)
    linear_meta.attach_to(ensemble)
    linear_meta.train(basic.pruned_trees)
    linear_meta.prune()

    # Identify indices
    shap_indices = [ensemble.individual_trees.index(t) for t in linear_meta.pruned_trees]

    # --- MAP WEIGHTS TO SURVIVING TREES ---
    weight_map = dict(zip(linear_meta.initial_pruned_trees, linear_meta.w_final))
    final_weights = [weight_map[t] for t in linear_meta.pruned_trees]

    # Evaluate WITH the aligned weights
    results["SHAP/Linear"] = evaluate_selected_trees(
        ensemble,
        shap_indices,
        "SHAP/Linear",
        weights=None
    )

    # --- CES / Other Methods (BASELINES) ---
    X_val, y_val = ensemble.X_train_meta, ensemble.y_train_meta
    base_preds_val = [t.predict(X_val) for t in ensemble.individual_trees]

    for name, method in methods_dict.items():
        # These methods usually just return a subset, implying equal weights
        selected_preds_val, selected_indices = method.select(base_preds_val, y_val)

        # Evaluate WITHOUT weights (Standard Bagging)
        results[name] = evaluate_selected_trees(
            ensemble,
            selected_indices,
            name,
            weights=None
        )

    return results

def run_methods_for_dataset(X, y, dataset_name):
    print(f"\n=== Dataset: {dataset_name} ===")
    data_type = "regression" if len(np.unique(y)) > 20 else "classification"

    # Train base ensemble
    ensemble = ExplainableTreeEnsemble(X=X, y=y, data_type=data_type)
    ensemble.train_base_trees()

    # Initialize CES-style methods
    methods = {
        "Cw/oE": ConstructiveWithoutExploration(data_type=data_type),
        "CwE": ConstructiveWithExploration(data_type=data_type),
        "Pw/oE": PruningWithoutExploration(data_type=data_type),
        "PwE": PruningWithExploration(data_type=data_type)
    }

    # --- GRID SEARCH STEP ---
    # 1. Train a temporary BasicMetaModel to get the initial pruned set (SHAP input)
    print("Pre-training BasicMetaModel for Grid Search inputs...")
    basic_gs = BasicMetaModel(ensemble.data_type)
    basic_gs.attach_to(ensemble)
    basic_gs.train()

    # 2. Find best parameters using Validation Set
    best_lp, best_ld = grid_search_best_lambdas(ensemble, basic_gs.pruned_trees)

    # Run all methods (Your Method will use the found optimal lambdas)
    results = run_all_methods(ensemble, methods, λ_prune=best_lp, λ_div=best_ld)
    return results

def main():
    # ------------------- REGRESSION DATASETS -------------------
    for ds in ["slice", "3droad", "kin40k"]:
        data = Dataset(ds)
        X = data.x.astype(np.float32)
        y = data.y.ravel()
        run_methods_for_dataset(X, y, ds)

    # ------------------- CLASSIFICATION DATASETS -------------------
    classification_sets = ["covertype", "higgs"]
    for ds in classification_sets:
        if ds == "covertype":
            from sklearn.datasets import fetch_covtype
            data = fetch_covtype(as_frame=False)
            X = data.data
            y = (data.target == 2).astype(int)

        if ds == "higgs":
            import kagglehub
            import pandas as pd
            path = kagglehub.dataset_download("erikbiswas/higgs-uci-dataset")
            csv_file = None
            for file in os.listdir(path):
                if file.endswith(".csv"):
                    csv_file = os.path.join(path, file)
                    break
            if csv_file is None:
                raise FileNotFoundError("No CSV file found!")

            df = pd.read_csv(csv_file, nrows=1000000)
            y = df.iloc[:, 0].astype(int).values
            X = df.iloc[:, 1:].astype("float32").values

        run_methods_for_dataset(X, y, ds)

if __name__ == "__main__":
    main()