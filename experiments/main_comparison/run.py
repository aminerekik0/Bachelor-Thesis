import pandas as pd
import os
import sys

from src.BasicMetaModel import BasicMetaModel
from src.ExplainableTreeEnsemble import ExplainableTreeEnsemble
from src.LinearMetaModel import LinearMetaModel

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "src"))
if src_dir not in sys.path:
    sys.path.append(src_dir)


def run_main_comparison():
    """
    Runs the main comparison experiment (Method A vs. B vs. C)
    and saves results to a CSV file in this directory.
    """
    print("Starting Main Comparison Experiment...")

    # --- Parameters ---
    # You can add more datasets to this list
    dataset_names = ["3droad", "slice"]

    # --- Output Path ---
    # Save results inside this experiment's folder
    output_dir = os.path.join(current_dir, "results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, "main_comparison.csv")

    results = []

    for dataset in dataset_names:
        print(f"\n--- Processing Dataset: {dataset} ---")


        workflow = ExplainableTreeEnsemble(data_type="regression", dataset_name=dataset)
        workflow.train_base_trees()
        mse_full, _ , _ ,_,_,_ = workflow._evaluate()


        if dataset == "bike":
            corr_thresh = 0.99
        elif dataset == "3droad":
            corr_thresh = 0.9
        elif dataset == "slice":
            corr_thresh = 0.95
        else:
            corr_thresh = 0.9

        # --- . Run Method B (BasicMetaModel: SHAP -> Corr) ---
        print("\n--- Running Method B (BasicMetaModel: SHAP -> Corr) ---")
        model_b = BasicMetaModel()
        model_b.attach_to(workflow)
        model_b.train()
        mse_basic_shap_only, _ = model_b.evaluate() # Get score before corr

        model_b.prune_by_correlation(corr_thresh=corr_thresh)
        mse_b, _ = model_b.evaluate_shap_then_corr()
        num_trees_b = len(model_b.shap_then_corr_trees) if model_b.shap_then_corr_trees else 0

        # --- . Run Method A (LinearMetaModel: HRP) ---
        print("\n--- Running Method A (LinearMetaModel: HRP) ---")
        model_a = LinearMetaModel()
        model_a.attach_to(workflow)
        model_a.train(pruned_trees_list=model_b.pruned_trees)
        model_a.prune(
            corr_thresh=corr_thresh
        )
        mse_a, _ = model_a.evaluate()
        num_trees_a = len(model_a.pruned_trees) if model_a.pruned_trees else 0

        # --- . Run Method C (BasicMetaModel: corr -> SHAP) ---
        print("\n--- Running Method C (BasicMetaModel: corr -> SHAP) ---")
        model_c = BasicMetaModel()
        model_c.attach_to(workflow)
        model_c.train_corr_first(corr_thresh=corr_thresh)
        mse_c = model_c.evaluate_corr_first()
        num_trees_c = len(model_c.corr_first_pruned_trees) if model_c.corr_first_pruned_trees else 0


        results.append({
            "dataset": dataset,
            "corr_thresh": corr_thresh,
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