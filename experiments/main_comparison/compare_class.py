from experiments.main_comparison.run import correlation_prune, shap_prune_on_subset
from src.PrePruner import PrePruner
from src.MetaOptimizer import MetaOptimizer
from src.EnsembleCreator import EnsembleCreator
from uci_datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import warnings
import math
import sys
from Datasets import get_dataset





warnings.filterwarnings("ignore")



# ===============================================================
# EMBEDDED: GreedyPruningClassifier & Metrics
# ===============================================================

def reduced_error(i, ensemble_proba, selected_models, target):
    iproba = ensemble_proba[i,:,:]
    if len(selected_models) > 0:
        sub_proba = ensemble_proba[selected_models, :, :]
        pred = 1.0 / (1 + len(selected_models)) * (sub_proba.sum(axis=0) + iproba)
    else:
        pred = iproba
    return (pred.argmax(axis=1) != target).mean()

def complementariness(i, ensemble_proba, selected_models, target):
    iproba = ensemble_proba[i,:,:]
    if len(selected_models) > 0:
        sub_proba = ensemble_proba[selected_models, :, :]
        ensemble_wrong = (sub_proba.sum(axis=0).argmax(axis=1) != target)
    else:
        ensemble_wrong = np.ones(len(target), dtype=bool)
    candidate_correct = (iproba.argmax(axis=1) == target)
    return -1.0 * np.sum(np.logical_and(candidate_correct, ensemble_wrong))

def drep(i, ensemble_proba, selected_models, target, rho=0.25):
    if len(selected_models) == 0:
        iproba = ensemble_proba[i,:,:].argmax(axis=1)
        return (iproba != target).mean()
    else:
        sub_ensemble = ensemble_proba[selected_models, :, :]
        sproba = sub_ensemble.mean(axis=0).argmax(axis=1)
        diffs = []
        for j in range(ensemble_proba.shape[0]):
            if j not in selected_models:
                jproba = ensemble_proba[j,:,:].argmax(axis=1)
                d = (jproba == sproba).sum()
                diffs.append((j, d))
        gamma = sorted(diffs, key=lambda e: e[1], reverse=False)
        K = int(np.ceil(rho * len(gamma)))
        if K < 1: K = 1
        topidx = [idx for idx, _ in gamma[:K]]
        if i in topidx:
            iproba = ensemble_proba[i,:,:]
            pred = (sub_ensemble.sum(axis=0) + iproba).argmax(axis=1)
            return (pred != target).mean()
        else:
            return float('inf')

class GreedyPruningClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=5, metric=reduced_error, n_jobs=1):
        self.n_estimators = n_estimators
        self.metric = metric
        self.n_jobs = n_jobs
        self.selected_indices_ = []

    def _metric_wrapper(self, i, proba, selected, target):
        return (i, self.metric(i, proba, selected, target))

    def prune_(self, proba, target):
        n_received = len(proba)
        if self.n_estimators >= n_received:
            return list(range(n_received))
        not_selected = list(range(n_received))
        selected = []
        for _ in range(self.n_estimators):
            scores = Parallel(n_jobs=self.n_jobs, backend="threading")(
                delayed(self._metric_wrapper)(i, proba, selected, target)
                for i in not_selected
            )
            best_model, _ = min(scores, key=lambda e: e[1])
            not_selected.remove(best_model)
            selected.append(best_model)
        self.selected_indices_ = selected
        return selected

# ===============================================================
# CONFIG & HELPERS
# ===============================================================
LAMBDA_CONFIG ={
    "magic": (1.2, 0.3),
    "adult": (1.2, 0.3),
    "letter": (1.2, 0.3),
    "jm1": (1.2, 0.3),
    "nursery": (1.2, 0.3),
    "har": (1.2, 0.3),
    "connect": (1.2, 0.3),
    "weather": (1.2, 0.3),

}

# Correlation threshold config (example values, adjust if needed)
CORR_THRESHOLD_CONFIG = {
    "magic": 0.95,
    "adult": 0.95,
    "letter": 0.95,
    "jm1": 0.95,
    "nursery": 0.95,
    "har": 0.997,
    "connect": 0.95,
    "weather": 0.95,

}

def find_best_lambda_for_target_size(ensemble, basic_pruned_trees, target_size, tolerance=5):
    lambdas_to_try = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5]
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
            if diff <= tolerance: break
    return best_lambda

def find_best_alpha_for_external_l1(ensemble, base_preds, y_meta, target_size, tolerance=5):
    alphas_to_try = [0.001 , 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
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
            if diff <= tolerance: break
        except: return 0.01
    return best_alpha

def evaluate_with_meta_weights(ensemble, selected_indices):
    if len(selected_indices) == 0: return np.nan
    X_eval, y_eval = ensemble.X_train_meta, ensemble.y_train_meta
    X_test, y_test = ensemble.X_test, ensemble.y_test
    selected = [ensemble.individual_trees[i] for i in selected_indices]
    preds_eval = np.column_stack([t.predict(X_eval) for t in selected])
    preds_test = np.column_stack([t.predict(X_test) for t in selected])
    lm = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
    lm.fit(preds_eval, y_eval)
    return accuracy_score(y_test, lm.predict(preds_test))

def compute_dataset_wlt(dataset_summary):
    methods = list(set([row["method"] for row in dataset_summary]))
    WLT = {m: {"win": 0, "loss": 0, "tie": 0} for m in methods}
    rows = dataset_summary
    SQRT_N = math.sqrt(5)
    for ds in set([r["dataset"] for r in rows]):
        ds_rows = [r for r in rows if r["dataset"] == ds]
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                mA, mB = methods[i], methods[j]
                try:
                    meanA = next(r["mean_acc"] for r in ds_rows if r["method"] == mA)
                    stdA = next(r["std_acc"] for r in ds_rows if r["method"] == mA)
                    meanB = next(r["mean_acc"] for r in ds_rows if r["method"] == mB)
                    stdB = next(r["std_acc"] for r in ds_rows if r["method"] == mB)
                except: continue
                seA, seB = stdA/SQRT_N, stdB/SQRT_N
                if meanA - seA > meanB + seB:
                    WLT[mA]["win"] += 1; WLT[mB]["loss"] += 1
                elif meanA + seA < meanB - seB:
                    WLT[mA]["loss"] += 1; WLT[mB]["win"] += 1
                else:
                    WLT[mA]["tie"] += 1; WLT[mB]["tie"] += 1
    return WLT

# ===============================================================
# MAIN RUNNER
# ===============================================================
def run_all_methods_once(ensemble, dataset_name):
    λ_shap, λ_div = LAMBDA_CONFIG.get(dataset_name, (1.2, 0.3))
    corr_thresh = CORR_THRESHOLD_CONFIG.get(dataset_name , 0.95)
    basic = PrePruner( data_type="classification" , keep_ratio=0.25)
    basic.attach_to(ensemble)
    basic.train()
    X_meta, y_meta = ensemble.X_train_meta, ensemble.y_train_meta

    try:
        base_preds_proba = np.array([t.predict_proba(X_meta) for t in basic.pruned_trees])
        original_preds_proba = np.array([t.predict_proba(X_meta) for t in ensemble.individual_trees])
    except:
        base_preds_proba = np.zeros((len(basic.pruned_trees), len(y_meta), 2))
        original_preds_proba = np.zeros((len(ensemble.individual_trees), len(y_meta), 2))
    base_preds_flat = np.column_stack([t.predict(X_meta) for t in basic.pruned_trees])

    # 1. SHAP
    print(f"\n--- [SHAP] Training (λ={λ_shap}) ---")
    linear_meta = MetaOptimizer(mode="SHAP", λ_div=λ_div, λ_prune=λ_shap, epochs=200)
    linear_meta.attach_to(ensemble)
    linear_meta.train(basic.pruned_trees)
    linear_meta.prune(prune_threshold=0.01, corr_thresh= corr_thresh)
    shap_indices = [ensemble.individual_trees.index(t) for t in linear_meta.pruned_trees]
    target_size = len(shap_indices) if len(shap_indices) > 2 else 5

    # 2. L1/Linear
    print(f"--- [L1/Linear] Tuning to size {target_size} ---")
    tuned_l1 = find_best_lambda_for_target_size(ensemble, basic.pruned_trees, target_size)
    linear_meta_l1 = MetaOptimizer(mode="L1", λ_div=0.0, λ_prune=tuned_l1, epochs=200)
    linear_meta_l1.attach_to(ensemble)
    linear_meta_l1.train(basic.pruned_trees)
    linear_meta_l1.prune(prune_threshold=0.01 ,  corr_thresh= corr_thresh)
    l1_indices = [ensemble.individual_trees.index(t) for t in linear_meta_l1.pruned_trees]

    # 3. L1 External
    print(f"--- [L1 External] Tuning to size {target_size} ---")
    best_alpha = find_best_alpha_for_external_l1(ensemble, original_preds_proba, y_meta, target_size)
    try:
        from L1_reg import L1PruningClassifier
        l1_ext = L1PruningClassifier(l_reg=best_alpha)
        _, l1_ext_sel = l1_ext.select(original_preds_proba, y_meta)
        if l1_ext_sel:
            l1_ext_idx = list(l1_ext_sel)
            print("len(l1_ext_sel)",len(l1_ext_sel))
        else:
            l1_ext_idx = []
    except Exception as e:
        print(f"[WARN] External L1 failed: {e}")
        l1_ext_idx = []

    # 4. Greedy Methods
    re_m = GreedyPruningClassifier(n_estimators=target_size, metric=reduced_error)
    re_sel = re_m.prune_(original_preds_proba, y_meta)
    re_idx = list(re_sel)

    ic_m = GreedyPruningClassifier(n_estimators=target_size, metric=complementariness)
    ic_sel = ic_m.prune_(original_preds_proba, y_meta)
    ic_idx = list(ic_sel)

    drep_m = GreedyPruningClassifier(n_estimators=target_size, metric=drep)
    drep_sel = drep_m.prune_(original_preds_proba, y_meta)
    drep_idx = list(drep_sel)

    # 5. RF Baseline
    rf = RandomForestClassifier(n_estimators=200, max_depth=8)
    rf.fit(ensemble.X_train, ensemble.y_train)
    rf_acc = accuracy_score(ensemble.y_test, rf.predict(ensemble.X_test))



    corr_pruned_trees_B = correlation_prune(

        basic.pruned_trees,

        ensemble,

        "classification",

        importance=basic.pruned_tree_weights,

        corr_thresh=corr_thresh

    )

    method_B_idx = [ensemble.individual_trees.index(t) for t in corr_pruned_trees_B]



    corr_pruned_trees_C_stage1 = correlation_prune(

        ensemble.individual_trees,

        ensemble,

        "classification",

        corr_thresh=corr_thresh ,

        importance=None,

    )

    corr_stage1_size = len(corr_pruned_trees_C_stage1)


    # Step 2: SHAP pruning on this corr-pruned subset

    shap_pruned_trees_C, _ = shap_prune_on_subset(

        corr_pruned_trees_C_stage1,

        ensemble,

        data_type="classification",

        keep_ratio=0.25,

    )
    method_C_idx = [ensemble.individual_trees.index(t) for t in shap_pruned_trees_C]

    results = {
        "SHAP/Linear": evaluate_with_meta_weights(ensemble, shap_indices),
        "L1/Linear": evaluate_with_meta_weights(ensemble, l1_indices),
        "L1": evaluate_with_meta_weights(ensemble, l1_ext_idx),
        "corr -> SHAP" : evaluate_with_meta_weights(ensemble,method_C_idx ),
        "SHAP -> corr" : evaluate_with_meta_weights(ensemble,method_B_idx ),
        "RE": evaluate_with_meta_weights(ensemble, re_idx),
        "DREP": evaluate_with_meta_weights(ensemble, drep_idx),
        "RF": rf_acc
    }
    sizes = {
        "SHAP/Linear": len(shap_indices), "L1/Linear": len(l1_indices),
        "L1": len(l1_ext_idx), "corr -> SHAP" : len(method_C_idx), "SHAP -> corr" : len(method_B_idx), "RE": len(re_idx),
        "DREP": len(drep_idx), "RF": 200
    }
    return results, sizes

def main():
    datasets = [
        "covtype" ,"higgs", "magic",
    "adult",
    "letter",    "jm1",
    "nursery",
    "har",
    "connect",
    "weather",

    ]
    summary_rows = []
    size_rows = []
    for ds in datasets:
        try:
            if ds == "covtype":
                from sklearn.datasets import fetch_covtype
                data = fetch_covtype(as_frame=False)
                X = data.data
                y = (data.target == 2).astype(int)

            elif ds == "higgs":
                import os
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
                import pandas as pd
                df = pd.read_csv(csv_file, nrows=1000000)
                y = df.iloc[:, 0].astype(int).values
                X = df.iloc[:, 1:].astype("float32").values
            else:
                print(f"\n=== DATASET: {ds} ===")
                X, y = get_dataset(ds)  # Get X, y unsplit

            # Run ensemble methods
            scores, size_lists = run_methods_for_dataset_10_times(X, y, ds)
            for m, vals in scores.items():
                summary_rows.append({"dataset": ds, "method": m, "mean_acc": np.nanmean(vals), "std_acc": np.nanstd(vals)})
            for m, vals in size_lists.items():
                size_rows.append({"dataset": ds, "method": m, "mean_size": np.nanmean(vals)})
        except Exception as e:
            print(f"Skipping {ds}: {e}")

    wlt = compute_dataset_wlt(summary_rows)
    pd.DataFrame(summary_rows).to_csv("results/results_acc_final_class4.0.csv", index=False)

    pd.DataFrame(size_rows).to_csv("results/results_size_final_class_4.0.csv", index=False)

    rows = []
    for method, vals in wlt.items():
        rows.append({"method": method, **vals})

    df_wlt = pd.DataFrame(rows)
    df_wlt.to_csv("GLOBAL_win_loss_tie4.0.csv", index=False)

    print("\n======================================")
    print(" GLOBAL WIN / LOSS / TIE SUMMARY")
    print("======================================")
    print(df_wlt)

def run_methods_for_dataset_10_times(X, y, dataset_name):
    print(f"\n=== DATASET: {dataset_name} ===")
    methods = ["SHAP/Linear", "L1/Linear", "L1", "corr -> SHAP" , "SHAP -> corr" ,"RE", "DREP", "RF"]
    scores = {m: [] for m in methods}
    sizes = {m: [] for m in methods}
    for run in range(5):
        print(f">> Run {run+1}/5")
        ensemble = EnsembleCreator(X=X, y=y, data_type="classification")
        ensemble.train_base_trees()
        res, sz = run_all_methods_once(ensemble, dataset_name)
        for m in methods:
            scores[m].append(res.get(m, np.nan))
            sizes[m].append(sz.get(m, 0))
    return scores, sizes

if __name__ == "__main__":
    main()