import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from copy import deepcopy

# Import your custom modules
from src.BasicMetaModel import BasicMetaModel
from src.ExplainableTreeEnsemble import ExplainableTreeEnsemble
from src.LinearMetaModel import LinearMetaModel

class DREPPruner(BaseEstimator, ClassifierMixin):
    """
    Diversity Regularized Ensemble Pruning (DREP) - Soft Voting Implementation.

    Updated to use probability averaging (Soft Voting) instead of hard labels.
    This prevents the 'Voting Valley of Death' (accuracy drop at N=2) and
    allows the ensemble to grow naturally.

    Parameters:
    -----------
    base_classifiers : list
        List of fitted scikit-learn estimators (must support predict_proba).
    rho : float
        Tradeoff parameter in (0, 1). Controls the size of the candidate subset
        considered for diversity.
    """
    def __init__(self, base_classifiers, rho=0.05):
        self.base_classifiers = base_classifiers
        self.rho = rho
        self.selected_indices_ = []
        self.selected_classifiers_ = []

    def fit(self, X_val, y_val):
        """
        Runs DREP algorithm using Soft Voting (Probabilities) on validation data.
        """
        n_classifiers = len(self.base_classifiers)
        n_samples = X_val.shape[0]

        # 1. Pre-compute PROBABILITIES for all classifiers
        # Shape: (n_samples, n_classifiers)
        # We use the probability of Class 1
        all_probs = np.zeros((n_samples, n_classifiers))
        for i, clf in enumerate(self.base_classifiers):
            # predict_proba returns [prob_0, prob_1]
            all_probs[:, i] = clf.predict_proba(X_val)[:, 1]

        # 2. Find the single best classifier (lowest error)
        errors = []
        for i in range(n_classifiers):
            # Threshold probability at 0.5 to get labels
            pred_labels = (all_probs[:, i] >= 0.5).astype(int)
            errors.append(1 - accuracy_score(y_val, pred_labels))

        H_indices = list(range(n_classifiers))
        best_init_idx = np.argmin(errors)

        # Initialize H* with the best classifier
        H_star_indices = [H_indices[best_init_idx]]
        H_indices.remove(H_indices[best_init_idx])

        current_best_error = min(errors)

        # 3. Greedy Loop
        while len(H_indices) > 0:
            # Current ensemble average probability (Soft Vote)
            current_ensemble_prob = np.mean(all_probs[:, H_star_indices], axis=1)

            # --- Diversity Check ---
            # We calculate diversity using the correlation of centered probabilities
            # This is the "soft" equivalent of Equation 9 in the paper
            diffs = []
            for idx in H_indices:
                # Center the probabilities (prob - 0.5)
                h_prime = all_probs[:, idx] - 0.5
                h_ens   = current_ensemble_prob - 0.5

                # Dot product implies correlation. Lower value = Higher Diversity.
                diversity_score = np.mean(h_prime * h_ens)
                diffs.append((diversity_score, idx))

            # Sort by diversity score (ascending)
            diffs.sort(key=lambda x: x[0])

            # Select Candidates (Top rho% most diverse)
            k = max(1, int(np.ceil(self.rho * len(H_indices))))
            candidates = [x[1] for x in diffs[:k]]

            # --- Accuracy Check ---
            # From the diverse candidates, pick the one that improves accuracy the most
            best_candidate = None
            lowest_new_error = float('inf')

            for cand_idx in candidates:
                # Tentatively form new ensemble
                trial_indices = H_star_indices + [cand_idx]

                # Soft Vote Average of the trial ensemble
                avg_prob = np.mean(all_probs[:, trial_indices], axis=1)
                pred_label = (avg_prob >= 0.5).astype(int)

                error = 1 - accuracy_score(y_val, pred_label)

                if error < lowest_new_error:
                    lowest_new_error = error
                    best_candidate = cand_idx

            # --- Stopping Condition ---
            # We use <= to allow the ensemble to traverse plateaus in the error surface
            if lowest_new_error <= current_best_error:
                current_best_error = lowest_new_error
                H_star_indices.append(best_candidate)
                H_indices.remove(best_candidate)
            else:
                # Stop if adding a tree strictly increases error
                break

        # Finalize Selection
        self.selected_indices_ = H_star_indices
        self.selected_classifiers_ = [self.base_classifiers[i] for i in H_star_indices]
        return self

    def predict(self, X):
        """Predict using the pruned ensemble (Soft Majority Voting)."""
        if not self.selected_classifiers_:
            raise RuntimeError("You must fit the pruner first.")

        # Collect probability predictions
        probs = []
        for clf in self.selected_classifiers_:
            probs.append(clf.predict_proba(X)[:, 1])

        # Average Probabilities
        avg_prob = np.mean(probs, axis=0)

        # Threshold at 0.5
        return (avg_prob >= 0.5).astype(int)


# Example Usage
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import kagglehub
    import pandas as pd
    import os

    # 1. Load Data
    path = kagglehub.dataset_download("erikbiswas/higgs-uci-dataset")
    csv_file = None
    for file in os.listdir(path):
        if file.endswith(".csv"):
            csv_file = os.path.join(path, file)
        break

    if csv_file:
        print("Loading HIGGS CSV...")
        # Using 1M rows as requested
        df = pd.read_csv(csv_file, nrows=1000000)
        y = df.iloc[:, 0].astype(int).values
        X = df.iloc[:, 1:].astype("float32").values
    else:
        # Fallback to Covertype if HIGGS fails
        print("HIGGS CSV not found. Loading Covertype...")
        from sklearn.datasets import fetch_covtype
        data = fetch_covtype(as_frame=False)
        X = data.data
        y = (data.target == 2).astype(int)

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 2. Train Base Ensemble
    workflow = ExplainableTreeEnsemble(X=X, y=y, data_type="classification", dataset_name="higgs")
    workflow._prepare_data()
    workflow.train_base_trees()

    # 3. Apply DREP Pruning (Soft Version)
    # Create a validation set specifically for pruning
    X_drep_train, X_drep_val, y_drep_train, y_drep_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    drep = DREPPruner(base_classifiers=workflow.individual_trees, rho=0.05)
    drep.fit(X_drep_val, y_drep_val)

    # Evaluate DREP
    y_pred_pruned = drep.predict(X_test)
    acc_pruned = accuracy_score(y_test, y_pred_pruned)

    print(f"DREP Pruned Ensemble Size: {len(drep.selected_indices_)}")
    print(f"DREP Pruned Accuracy: {acc_pruned:.4f}")

    # 4. Meta Model Processing
    basic = BasicMetaModel(data_type="classification")
    basic.attach_to(workflow)
    basic.train()

    lm = LinearMetaModel(data_type="classification")
    lm.attach_to(workflow)

    # CRITICAL FIX: Pass the DREP-selected trees to the Linear Meta Model
    lm.train(basic.pruned_trees)

    lm.prune()
    lm.evaluate()

    final_size_A = len(lm.pruned_trees) if lm.pruned_trees else 0

    print(f"Final LMM Ensemble Size: {final_size_A}")
    print(f"Final LMM Accuracy: {lm.acc:.4f}")