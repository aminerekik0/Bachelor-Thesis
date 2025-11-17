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

# --- Import your custom modules ---
# (Ensure these paths are correct)
from src.BasicMetaModel import BasicMetaModel
from src.ExplainableTreeEnsemble import ExplainableTreeEnsemble
from src.LinearMetaModel import LinearMetaModel


warnings.filterwarnings('ignore')

# ---------------------------------------------
# RQ2 (Ablation Study / Lambda Grid Search)
# ---------------------------------------------
def run_ablation_study():
    """
    Runs the ablation study (RQ2) to test different
    combinations of prune and diversity lambdas.
    """
    print("Starting Ablation Study (Grid Search)...")

    # --- Grid Search Parameters ---
    # These are the *ratios* that will be scaled by the model
    # (0.0 is included to test "No Reg" and "XXX Only" runs)
    lambda_prune_values = [0.2, 0.5, 1.0]
    lambda_div_values = [0.2 ,0.5, 1.0]
    # ------------------------------

    dataset_names = ["3droad", "slice"] # Add more datasets here

    # --- Consistent Parameters ---
    # Use keep_ratio=0.2 for the candidate pool, as discussed for RQ2
    keep_ratio_stage1 = 0.2
    # ------------------------------

    output_dir = os.path.join(current_dir, "results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, "ablation_study_results.csv")

    # --- MODIFICATION: Load existing results to append ---
    if os.path.exists(file_path):
        try:
            df_results = pd.read_csv(file_path)
            print(f"Loaded existing results from {file_path}")
        except pd.errors.EmptyDataError:
            df_results = pd.DataFrame() # File was empty
    else:
        df_results = pd.DataFrame() # File does not exist
    # -----------------------------------------------------

    for dataset in dataset_names:
        print(f"\n--- Processing Dataset: {dataset} ---")

        # --- Setup Stage 0 & 1 (Done ONCE per dataset) ---
        workflow = ExplainableTreeEnsemble(data_type="regression", dataset_name=dataset)
        workflow.train_base_trees()

        # --- Set Dataset-Specific Correlation Thresholds ---
        if dataset == "bike":
            corr_thresh = 0.99
        elif dataset == "3droad":
            corr_thresh = 0.9
        elif dataset == "slice":
            corr_thresh = 0.95
        else:
            corr_thresh = 0.9 # Default

        print(f"Using correlation threshold: {corr_thresh}")

        print(f"\n[Ablation] Creating candidate pool (keep_ratio={keep_ratio_stage1})...")
        model_b = BasicMetaModel(keep_ratio=keep_ratio_stage1)
        model_b.attach_to(workflow)
        model_b.train()
        candidate_trees = model_b.pruned_trees
        mse_stage1_baseline, _ = model_b.evaluate() # Get the baseline MSE for the 40 trees
        print(f"[Ablation] Candidate pool created: {len(candidate_trees)} trees. Baseline MSE: {mse_stage1_baseline:.4f}")

        # --- Run Grid Search (Stage 2) ---
        for lp in lambda_prune_values:
            for ld in lambda_div_values:

                # --- MODIFICATION: Check if run already exists ---
                run_exists = False
                if not df_results.empty:
                    run_exists = (
                            (df_results['dataset'] == dataset) &
                            (df_results['lambda_prune_ratio'] == lp) &
                            (df_results['lambda_div_ratio'] == ld)
                    ).any()

                if run_exists:
                    print(f"\n--- SKIPPING: {dataset} (lp_ratio={lp}, ld_ratio={ld}) --- Already in results.")
                    continue
                # ---------------------------------------------------

                # Assign a name for the table
                if lp == 0.0 and ld == 0.0:
                    run_name = "Run 2 (No Reg)"
                elif lp > 0.0 and ld == 0.0:
                    run_name = "Run 3 (Prune Only)"
                elif lp == 0.0 and ld > 0.0:
                    run_name = "Run 4 (Div Only)"
                else:
                    run_name = "Run 1 (Full Model)"

                print(f"\n--- Running: {dataset} / {run_name} (lp_ratio={lp}, ld_ratio={ld}) ---")

                # Initialize the model with the lambda ratios from the grid
                model_a = LinearMetaModel(λ_prune=lp, λ_div=ld)
                model_a.attach_to(workflow)

                # Train on the *same* candidate pool
                model_a.train(pruned_trees_list=candidate_trees)

                # Prune based on learned weights and correlation
                model_a.prune(corr_thresh=corr_thresh)

                # Evaluate the final, pruned model
                mse_a, _ = model_a.evaluate()

                # Get tree counts at each step
                trees_initial = len(candidate_trees)
                trees_after_weight_prune = len(model_a.final_pruned_indices)
                trees_final = len(model_a.pruned_trees) if model_a.pruned_trees else 0

                # --- MODIFICATION: Append new row and re-save ---
                new_row = pd.DataFrame([{
                    "dataset": dataset,
                    "run_name": run_name,
                    "lambda_prune_ratio": lp,
                    "lambda_div_ratio": ld,
                    "mse_stage1_baseline": mse_stage1_baseline,
                    "mse_final": mse_a,
                    "trees_initial": trees_initial,
                    "trees_after_weight_prune": trees_after_weight_prune,
                    "trees_final": trees_final,
                }])

                # Append this new row to the main DataFrame
                df_results = pd.concat([df_results, new_row], ignore_index=True)

                # Save the *entire* updated DataFrame to CSV
                df_results.to_csv(file_path, index=False, float_format='%.4f')
                # --------------------------------------------------

    print(f"\nAblation study finished. Results saved to {file_path}")


if __name__ == "__main__":
    # run_main_comparison()  # This is for RQ1 (Commented out)
    run_ablation_study()     # This is for RQ2 (Active)