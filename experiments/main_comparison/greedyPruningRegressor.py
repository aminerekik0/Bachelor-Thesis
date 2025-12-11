import numpy as np

class GreedyPruningRegressor:
    """
    A simple, standalone greedy pruner for regression.
    """
    def __init__(self, n_estimators=10, method="RE"):
        self.n_estimators = n_estimators
        self.method = method 
        self.selected_indices_ = []

    def select(self, preds, target):
        """
        Main function to select trees.
        preds:  (n_trees, n_samples) numpy array
        target: (n_samples,) numpy array
        """
        n_available = len(preds)

        # If we asked for more trees than available, return all
        if self.n_estimators >= n_available:
            return None, list(range(n_available))

        not_selected = list(range(n_available))
        selected = []

        
        for _ in range(self.n_estimators):
            best_score = float('inf')
            best_tree_idx = -1

            # Try adding each candidate tree to the current selection
            for candidate_idx in not_selected:
                score = self._calculate_score(candidate_idx, preds, selected, target)

                if score < best_score:
                    best_score = score
                    best_tree_idx = candidate_idx

            # Lock in the best tree
            if best_tree_idx != -1:
                selected.append(best_tree_idx)
                not_selected.remove(best_tree_idx)

        self.selected_indices_ = selected
        return None, selected

    def _calculate_score(self, candidate_idx, preds, selected_indices, target):
        candidate_pred = preds[candidate_idx]

        # --- METHOD 1: REDUCED ERROR (RE) ---
        # Pick tree that minimizes total Ensemble MSE
        if self.method == "RE":
            if len(selected_indices) == 0:
                new_ensemble_pred = candidate_pred
            else:
                # Average of existing trees + candidate
                current_sum = preds[selected_indices].sum(axis=0)
                new_ensemble_pred = (current_sum + candidate_pred) / (len(selected_indices) + 1)

            # Return MSE
            return ((new_ensemble_pred - target) ** 2).mean()

        # --- METHOD 2: INDIVIDUAL CONTRIBUTION (IC) ---
        # Pick tree that has low error where ensemble has high error
        elif self.method == "IC":
            if len(selected_indices) == 0:
                return np.abs(candidate_pred - target).mean()
            else:
                # Current ensemble error
                current_mean = preds[selected_indices].mean(axis=0)
                ensemble_error = np.abs(current_mean - target)

                # Candidate error
                candidate_error = np.abs(candidate_pred - target)

                # Minimize correlation of errors
                return np.mean(candidate_error * ensemble_error)

        return float('inf')



class REPruningRegressor(GreedyPruningRegressor):
    def __init__(self, n_estimators=10):
        super().__init__(n_estimators=n_estimators, method="RE")

class ICPruningRegressor(GreedyPruningRegressor):
    def __init__(self, n_estimators=10):
        super().__init__(n_estimators=n_estimators, method="IC")
