import numpy as np
from scipy.stats import mode
from sklearn.metrics import mean_squared_error, accuracy_score


class EarlyStopPruning:
    """
    Early-Stopping Ensemble Selection (Caruana-style)
    Works for regression and classification.
    
    Algorithm:
      1. Evaluate individual models.
      2. Sort by individual performance.
      3. Add models greedily.
      4. STOP immediately when adding a model does NOT improve validation score.
    """

    def __init__(self, data_type="regression"):
        self.data_type = data_type

    def _evaluate_ensemble(self, preds_list, y_true):
        """Evaluate current ensemble predictions."""
        preds = np.array(preds_list)

        if self.data_type == "regression":
            final_pred = np.mean(preds, axis=0)
            return -mean_squared_error(y_true, final_pred)  # larger = better

        # classification
        final_pred = mode(preds, axis=0, keepdims=False).mode
        return accuracy_score(y_true, final_pred)

    def select(self, predictions_list, y_true):
        """
        predictions_list: list of arrays, each model's predictions on validation set
        y_true: validation labels
        """
        predictions_list = [np.array(p) for p in predictions_list]
        n_models = len(predictions_list)

        # Step 1 — Evaluate individual model score
        individual_scores = [
            self._evaluate_ensemble([pred], y_true)
            for pred in predictions_list
        ]

        # Step 2 — Sort by best individual score
        sorted_idx = np.argsort(individual_scores)[::-1]  # descending

        selected_indices = []
        selected_preds = []

        # Step 3 — Add first model
        best_first = sorted_idx[0]
        selected_indices.append(best_first)
        selected_preds.append(predictions_list[best_first])
        best_score = individual_scores[best_first]

        # Step 4 — Greedy addition with EARLY STOP
        for idx in sorted_idx[1:]:
            trial_preds = selected_preds + [predictions_list[idx]]
            new_score = self._evaluate_ensemble(trial_preds, y_true)

            if new_score > best_score:
                # improvement → keep
                best_score = new_score
                selected_indices.append(idx)
                selected_preds.append(predictions_list[idx])
            else:
                # EARLY STOP
                break

        return selected_preds, selected_indices
