import sys
import gc
import numpy as np
from copy import deepcopy

class KappaPruner:
    """
    Implements Kappa Pruning for Ensemble Learning (Margineantu & Dietterich, 1997).

    This class selects a subset of classifiers from an ensemble by prioritizing
    pairs with the lowest Cohen's Kappa (lowest agreement), thereby maximizing diversity.
    """

    def __init__(self, n_classifiers_to_keep=5):
        """
        Args:
            n_classifiers_to_keep (int): The target number of classifiers to select (M).
        """
        self.n_to_keep = n_classifiers_to_keep

    def _check_zero(self, val, tol=1e-9):
        """Helper to avoid division by zero."""
        if abs(val) < tol:
            return tol
        return val

    def _calculate_kappa_statistic(self, pred_a, pred_b, y_true):
        """
        Calculates the Kappa statistic for multi-class classification between two classifiers.

        Math:
            kappa = (theta1 - theta2) / (1 - theta2)
        """
        # Ensure inputs are numpy arrays
        ha = np.array(pred_a)
        hb = np.array(pred_b)
        y = np.array(y_true)

        # Identify all unique classes across ground truth and predictions
        classes = np.unique(np.concatenate([y, ha, hb]))
        n_classes = len(classes)
        m = len(y) # number of samples

        # 1. Construct Contingency Table (Confusion Matrix between A and B)
        # Cij[i, j] = count where A predicted class i AND B predicted class j
        Cij = np.zeros((n_classes, n_classes))

        # Note: This loop can be optimized with sklearn.metrics.confusion_matrix
        # but is kept explicit here for clarity and independence.
        for i in range(n_classes):
            for j in range(n_classes):
                Cij[i, j] = np.sum((ha == classes[i]) & (hb == classes[j]))

        # 2. Calculate Theta 1 (Observed Agreement)
        # Sum of diagonal elements divided by m
        theta1 = np.trace(Cij) / float(m)

        # 3. Calculate Theta 2 (Chance Agreement) - THE FIX
        # Sum of (marginal_row * marginal_col)
        row_sums = np.sum(Cij, axis=1) # How often Classifier A predicted class i
        col_sums = np.sum(Cij, axis=0) # How often Classifier B predicted class i

        theta2 = np.sum(row_sums * col_sums) / (float(m) ** 2)

        # 4. Calculate Kappa
        denominator = 1.0 - theta2
        ans = (theta1 - theta2) / self._check_zero(denominator)

        return ans

    def prune(self, predictions_list, y_true):
        """
        Main pruning function.

        Args:
            predictions_list (list or np.array): Shape (n_classifiers, n_samples).
                                                 List of predictions from each base learner.
            y_true (list or np.array): Shape (n_samples,). Ground truth labels.

        Returns:
            selected_preds (list): List of prediction arrays for the selected classifiers.
            selected_indices (list): Indices of the selected classifiers from the original list.
        """
        predictions = np.array(predictions_list)
        n_classifiers = predictions.shape[0]

        if n_classifiers < self.n_to_keep:
            print(f"Warning: Total classifiers ({n_classifiers}) is less than requested prune size ({self.n_to_keep}). Returning all.")
            return predictions_list, list(range(n_classifiers))

        # --- Step 1: Initialize Pairwise Kappa Matrix ---
        # We fill with maxsize because we want to find the MINIMUM Kappa (lowest agreement)
        K_matrix = np.full((n_classifiers, n_classifiers), sys.maxsize, dtype=float)

        # Calculate Kappa for every unique pair
        for i in range(n_classifiers - 1):
            for j in range(i + 1, n_classifiers):
                kappa_val = self._calculate_kappa_statistic(predictions[i], predictions[j], y_true)
                K_matrix[i, j] = kappa_val

        # --- Step 2: Selection Loop ---
        selected_mask = np.zeros(n_classifiers, dtype=bool)
        n_selected = 0

        while n_selected < self.n_to_keep:
            # Find the pair indices (i, j) with the absolute minimum Kappa in the matrix
            # This represents the most diverse pair available
            min_idx = np.unravel_index(np.argmin(K_matrix), K_matrix.shape)
            row, col = min_idx

            # If the matrix is fully maxsize, we have no more valid pairs
            if K_matrix[row, col] == sys.maxsize:
                break

            # Logic: If we pick a pair, we mark them as selected.
            # We then "remove" them from the matrix so we don't pick the same pair again
            # by setting their row/cols to infinity.

            # Check how many new classifiers this pair adds
            new_picks = 0
            if not selected_mask[row]: new_picks += 1
            if not selected_mask[col]: new_picks += 1

            # Stop if adding both would exceed our limit (optional logic, depending on strictness)
            if n_selected + new_picks > self.n_to_keep:
                # If we only need 1 more but the pair gives 2, we force pick the first one
                if n_selected < self.n_to_keep and not selected_mask[row]:
                    selected_mask[row] = True
                    n_selected += 1
                break

            # Mark as selected
            if not selected_mask[row]:
                selected_mask[row] = True
                n_selected += 1

            if not selected_mask[col]:
                selected_mask[col] = True
                n_selected += 1

            # "Remove" these from future consideration in the matrix
            # We set the entire row/col of the picked items to maxsize
            # so they aren't the source of the *next* minimum finding.
            K_matrix[row, :] = sys.maxsize
            K_matrix[:, row] = sys.maxsize
            K_matrix[:, col] = sys.maxsize
            K_matrix[col, :] = sys.maxsize

        # --- Step 3: Return Results ---
        selected_indices = np.where(selected_mask)[0].tolist()
        selected_preds = predictions[selected_indices].tolist()

        gc.collect()
        return deepcopy(selected_preds), selected_indices


if __name__ == "__main__":
    import numpy as np

    from src.ExplainableTreeEnsemble import ExplainableTreeEnsemble


    from sklearn.datasets import fetch_covtype
    data = fetch_covtype(as_frame=False)
    X = data.data
    y = (data.target == 2).astype(int)

    # # Example: regression
    # data = load_boston()
    # X, y = data.data, data.target
    # data_type = "regression"

    # -------------------------
    # 2. Initialize ensemble
    # -------------------------
    ensemble = ExplainableTreeEnsemble(
        X=X,
        y=y,
    )

    # Train base trees
    ensemble.train_base_trees()
    print(f"Trained {len(ensemble.individual_trees)} base trees.")

    # -------------------------
    # 3. Generate predictions on validation set
    # -------------------------
    X_val = ensemble.X_eval_meta
    y_val = ensemble.y_eval_meta

    predictions_list = [t.predict(X_val) for t in ensemble.individual_trees]

    # -------------------------
    # 4. Apply Kappa pruning
    # -------------------------
    n_keep = 20  # number of diverse classifiers to keep
    pruner = KappaPruner(n_classifiers_to_keep=n_keep)
    selected_preds, selected_indices = pruner.prune(predictions_list, y_val)

    print(f"Selected {len(selected_indices)} most diverse classifiers: {selected_indices}")

    # -------------------------
    # 5. Keep only pruned trees in ensemble
    # -------------------------
    ensemble.individual_trees = [ensemble.individual_trees[i] for i in selected_indices]
    ensemble.n_trees = len(ensemble.individual_trees)

    # -------------------------
    # 6. Evaluate pruned ensemble
    # -------------------------
    results = ensemble._evaluate()
    print("Evaluation results of pruned ensemble:", results)
