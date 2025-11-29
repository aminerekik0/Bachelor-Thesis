import csv
import numpy as np
from uci_datasets import Dataset
import sys, os



from src.ExplainableTreeEnsemble import ExplainableTreeEnsemble
from src.BasicMetaModel import BasicMetaModel
from src.LinearMetaModel import LinearMetaModel


# ============================================================
#                 DATASET-SPECIFIC CONFIG
# ============================================================
DATASET_CONFIG = {
    "slice": {
        "lambda_prune": [0.7 , 1.0 ,1.2],
        "lambda_div":   [0.0 ,0.5 , 0.3 ,0.1],
        "prune_threshold": 0.01,
        "corr_threshold": 0.98,
    },
    "3droad": {
         "lambda_prune": [0.0 ,1.0],
        "lambda_div":   [0.0 ,0.3],
        "prune_threshold": 0.01,
        "corr_threshold": 0.9,
    },
    "kin40k": {
         "lambda_prune": [0.0 ,0.7],
        "lambda_div":   [0.0 ,0.5 ],
        "prune_threshold": 0.01,
        "corr_threshold": 0.93,
    },
    "covertype": {
         "lambda_prune": [0.0, 0.5 ],
        "lambda_div":   [0.0, 0.5 ],
        "prune_threshold": 0.01,
        "corr_threshold": 0.92,
    } , 
    "higgs": {
        "lambda_prune": [0.0, 0.5 ],
        "lambda_div":   [0.0, 0.5 ],
        "prune_threshold": 0.01,
        "corr_threshold": 0.92,
    }
    
}
# ============================================================


def append_to_csv(row, filename="results_2.0.csv"):
    write_header = not os.path.exists(filename)

    with open(filename, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_linear_grid(
    pruned_trees, workflow, data_type,
    dataset_name, full_metrics, pre_metrics,
    shap_size, full_size
):

    cfg = DATASET_CONFIG[dataset_name]

    grid_lambda_prune = cfg["lambda_prune"]
    grid_lambda_div   = cfg["lambda_div"]
    prune_threshold   = cfg["prune_threshold"]
    corr_threshold    = cfg["corr_threshold"]

    for lp in grid_lambda_prune:
        for ld in grid_lambda_div:

            model = LinearMetaModel(
                λ_prune=lp,
                λ_div=ld,
                data_type=data_type
            )

            model.attach_to(workflow)
            model.train(pruned_trees)

            model.prune(
                prune_threshold=prune_threshold,
                corr_thresh=corr_threshold
            )

            final_metric, _ = model.evaluate()

            # ================================
            #   GET FINAL EXTRA METRICS
            # ================================
            if data_type == "regression":
                final_r2  = model.r2
                final_f1  = None
                final_auc = None
            else:
                final_r2  = None
                final_f1  = model.f1
                final_auc = model.auc

            pruned_based_weighted_size = (
                len(model.kept_after_weight_pruning)
                if model.kept_after_weight_pruning else 0
            )
            linear_size = len(model.pruned_trees) if model.pruned_trees else 0

            row = {
                "dataset": dataset_name,
                "task": data_type,
                "lambda_prune": lp,
                "lambda_div": ld,
                "prune_threshold": prune_threshold,
                "corr_threshold": corr_threshold,

                # ==== FULL METRICS ====
                "full_metric": full_metrics["main"],
                "full_r2":  full_metrics["r2"],
                "full_f1":  full_metrics["f1"],
                "full_auc": full_metrics["auc"],

                # ==== PRE (SHAP) METRICS ====
                "pre_pruned_metric": pre_metrics["main"],
                "pre_r2":  pre_metrics["r2"],
                "pre_f1":  pre_metrics["f1"],
                "pre_auc": pre_metrics["auc"],

                # ==== FINAL (LINEAR) METRICS ====
                "final_pruned_metric": final_metric,
                "final_r2":  final_r2,
                "final_f1":  final_f1,
                "final_auc": final_auc,

                # ==== SIZES ====
                "full_size": full_size,
                "shap_size": shap_size,
                "pruned_weighted_size": pruned_based_weighted_size,
                "final_size": linear_size
            }

            append_to_csv(row)


def process_dataset(X, y, dataset_name, data_type):

    workflow = ExplainableTreeEnsemble(
        X=X,
        y=y,
        data_type=data_type
    )

    workflow.train_base_trees()
    full_size = len(workflow.individual_trees)

    # ---- FULL ENSEMBLE METRIC ----
    mse, rmse, mae, r2, acc, f1 = workflow._evaluate()

    # Save full metrics
    full_metrics = {
        "main": rmse if data_type == "regression" else acc,
        "r2":   r2 if data_type == "regression" else None,
        "f1":   f1 if data_type == "classification" else None,
        "auc":  workflow.auc if data_type == "classification" else None
    }

    # ---- BASIC STAGE (SHAP PRUNING) ----
    basic = BasicMetaModel(data_type=data_type)
    basic.attach_to(workflow)
    basic.train()
    pre_metric, _ = basic.evaluate()

    pre_metrics = {
        "main": pre_metric,
        "r2":   basic.r2 if data_type == "regression" else None,
        "f1":   basic.f1 if data_type == "classification" else None,
        "auc":  basic.auc if data_type == "classification" else None
    }

    shap_size = len(basic.pruned_trees) if basic.pruned_trees else 0

    # ---- LINEAR META (STAGE 2) ----
    run_linear_grid(
        pruned_trees=basic.pruned_trees,
        workflow=workflow,
        data_type=data_type,
        dataset_name=dataset_name,
        full_metrics=full_metrics,
        pre_metrics=pre_metrics,
        shap_size=shap_size,
        full_size=full_size
    )


def main():

    regression_sets = ["slice"]
    classification_sets = []

    for ds in regression_sets:
        data = Dataset(ds)
        X = data.x.astype(np.float32)
        y = data.y.ravel()
        process_dataset(X, y, ds, "regression")

    for ds in classification_sets:
        if ds == "covertype":
            from sklearn.datasets import fetch_covtype
            data = fetch_covtype(as_frame=False)
            X = data.data
            y = (data.target == 2).astype(int)
        if ds == "higgs":
            import kagglehub
            import kagglehub
            from kagglehub import KaggleDatasetAdapter

            path = kagglehub.dataset_download("erikbiswas/higgs-uci-dataset")
            csv_file = None
            for file in os.listdir(path):
                if file.endswith(".csv"):
                   csv_file = os.path.join(path, file)
                break
            if csv_file is None:
                raise FileNotFoundError("No CSV file found in downloaded HIGGS dataset folder!")
            
            print("Loaded CSV:", csv_file)
            import pandas as pd

            df = pd.read_csv(csv_file, nrows=1000000)
            y = df.iloc[:, 0].astype(int).values
            X = df.iloc[:, 1:].astype("float32").values
            print("Path to dataset files:", path) 
            process_dataset(X, y, ds, "classification")
            



if __name__ == "__main__":
    main()
