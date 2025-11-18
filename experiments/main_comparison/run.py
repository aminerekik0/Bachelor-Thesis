import pandas as pd
import os
import sys
import numpy as np
import shap
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
from sklearn.linear_model import LinearRegression, LogisticRegression
import warnings

# ---------------------------------------------
# Fix import paths
# ---------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
src_path = os.path.join(project_root, "src")

sys.path.append(src_path)

from src.BasicMetaModel import BasicMetaModel
from src.ExplainableTreeEnsemble import ExplainableTreeEnsemble
from src.LinearMetaModel import LinearMetaModel

warnings.filterwarnings('ignore')

# ---------------------------------------------
# RQ2 (Targeted Best Parameters Run)
# ---------------------------------------------
def run_ablation_study():
    """
    Runs the extended ablation study (RQ2) using SPECIFIC optimal
    parameters for each dataset, iterating over multiple keep_ratios.
    """
    print("Starting Targeted Ablation Study (Best Parameters)...")

    # --- Define Optimal Parameters per Dataset ---
    dataset_params = {
        "3droad": {
            "lambda_prune_values": [1.0 , 0.5 , 0.2],
            "lambda_div_values":   [1.0 , 0.5 ,0.0],
            "prune_threshold":     0.01,
            "corr_threshold":      0.90
        },
        "slice": {
            "lambda_prune_values": [1.0 , 0.5 , 0.2],
            "lambda_div_values":   [1.0 , 0.5 ,0.0],
            "prune_threshold":     0.01,
            "corr_threshold":      0.95
        },
        "kin40k": {
            "lambda_prune_values": [1.0 , 0.5 , 0.2],
            "lambda_div_values":   [1.0 , 0.5 ,0.0],
            "prune_threshold":     0.01,
            "corr_threshold":      0.90
        }
    }

    # --- Comparison Parameters ---
    keep_ratio_values = [0.3]  # Now iterating over both
    # ------------------------------

    output_dir = os.path.join(current_dir, "results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, "ablation_study_best_new.csv")

    # --- Load existing results ---
    if os.path.exists(file_path):
        try:
            df_results = pd.read_csv(file_path)
            print(f"Loaded existing results from {file_path}")
        except pd.errors.EmptyDataError:
            df_results = pd.DataFrame()
    else:
        df_results = pd.DataFrame()

    for dataset, params in dataset_params.items():
        print(f"\n--- Processing Dataset: {dataset} ---")

        # Setup Stage 0 (Train Base Trees) - Done once per dataset
        workflow = ExplainableTreeEnsemble(data_type="regression", dataset_name=dataset)
        workflow.train_base_trees()
        workflow._evaluate()

        # --- CAPTURE FULL ENSEMBLE MSE ---
        mse_full_ensemble = workflow.mse
        print(f"[Info] Full Ensemble (200 Trees) MSE: {mse_full_ensemble}")

        # --- Loop over Keep Ratios ---
        for kr in keep_ratio_values:
            print(f"\n[Ablation] Creating candidate pool (keep_ratio={kr})...")

            # Stage 1: Basic Meta Model
            model_b = BasicMetaModel(keep_ratio=kr)
            model_b.attach_to(workflow)
            model_b.train()
            candidate_trees = model_b.pruned_trees
            mse_stage1_baseline, _ = model_b.evaluate()
            print(f"[Ablation] Pool created: {len(candidate_trees)} trees. Baseline MSE: {mse_stage1_baseline:.4f}")

            # Get specific params for this dataset
            lp_values = params["lambda_prune_values"]
            ld_values = params["lambda_div_values"]
            pt = params["prune_threshold"]
            ct = params["corr_threshold"]

            # Loop through the specific lambda combinations
            for lp in lp_values:
                for ld in ld_values:

                    # Define Run Name
                    if lp == 0.0 and ld == 0.0:
                        run_name = "Run 2 (No Reg)"
                    elif lp > 0.0 and ld == 0.0:
                        run_name = "Run 3 (Prune Only)"
                    elif lp == 0.0 and ld > 0.0:
                        run_name = "Run 4 (Div Only)"
                    else:
                        run_name = "Run 1 (Full Model)"

                    print(f"\n--- Training: {dataset} / KR={kr} / {run_name} (lp={lp}, ld={ld}) ---")

                    # 1. Train
                    if lp >= ld:

                       model_a = LinearMetaModel(λ_prune=lp, λ_div=ld)
                       model_a.attach_to(workflow)
                       model_a.train(pruned_trees_list=candidate_trees)

                    # 2. Prune
                       model_a.prune(prune_threshold=pt, corr_thresh=ct)


                    # 3. Evaluate
                       mse_a, _ = model_a.evaluate()

                       trees_initial = len(candidate_trees)
                       trees_after_weight_prune = len(model_a.final_pruned_indices) if model_a.final_pruned_indices is not None else 0
                       trees_final = len(model_a.pruned_trees) if model_a.pruned_trees else 0

                       print(f"   -> Result: MSE={mse_a}, Trees={trees_final} (pt={pt}, ct={ct})")
                    else :
                        continue
                    # 4. Save Results (Added keep_ratio)
                    new_row = pd.DataFrame([{
                        "dataset": dataset,
                        "keep_ratio": kr, # <-- Added keep_ratio
                        "run_name": run_name,
                        "lambda_prune_ratio": lp,
                        "lambda_div_ratio": ld,
                       # "prune_threshold": pt,
                       # "corr_threshold": ct,
                       # "mse_full_ensemble": mse_full_ensemble,
                       # "mse_stage1_baseline": mse_stage1_baseline,
                        "mse_final": mse_a,
                       # "trees_initial": trees_initial,
                        "trees_after_weight_prune": trees_after_weight_prune,
                        #"trees_final": trees_final,
                        "prune_loss_": model_a.prune_loss ,
                        "div_loss" : model_a.div_loss
                    }])

                    df_results = pd.concat([df_results, new_row], ignore_index=True)
                    df_results.to_csv(file_path, index=False, float_format='%.4f')

    print(f"\nAblation study finished. Results saved to {file_path}")

if __name__ == "__main__":
    run_ablation_study()