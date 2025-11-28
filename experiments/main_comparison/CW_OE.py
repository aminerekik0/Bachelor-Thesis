import numpy as np
from scipy.stats import mode
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from copy import deepcopy

class ConstructiveWithoutExploration:
    """
    Constructive Without Exploration (Cw/oE) ensemble selection.
    Works for classification and regression.
    """
    def __init__(self, data_type="regression", metric=None):
        """
        Args:
            data_type (str): "regression" or "classification"
            metric (callable): Optional performance metric function.
                               If None, defaults are used.
        """
        self.data_type = data_type
        self.metric = metric

    def _evaluate_ensemble(self, ensemble_preds, y_true):
        ensemble_preds = np.array(ensemble_preds)
        if self.data_type == "regression":
            mean_preds = np.mean(ensemble_preds, axis=0)
            return -mean_squared_error(y_true, mean_preds) if self.metric is None else self.metric(y_true, mean_preds)
        else:
            mode_res = mode(ensemble_preds, axis=0, keepdims=False)
            majority_vote = mode_res.mode
            return accuracy_score(y_true, majority_vote) if self.metric is None else self.metric(y_true, majority_vote)

    def select(self, predictions_list, y_true):
        predictions_list = [np.array(p) for p in predictions_list]
        n_models = len(predictions_list)

        # Step 1: Evaluate each individual model
        individual_scores = []
        for pred in predictions_list:
            individual_scores.append(self._evaluate_ensemble([pred], y_true))

        # Step 2: Sort models by individual performance (descending)
        sorted_indices = np.argsort(individual_scores)[::-1]
        selected_indices = []
        selected_preds = []

        # Step 3: Constructive selection
        for idx in sorted_indices:
            candidate_preds = predictions_list[idx]
            trial_preds = selected_preds + [candidate_preds]
            score = self._evaluate_ensemble(trial_preds, y_true)

            if not selected_preds:
                selected_preds.append(candidate_preds)
                selected_indices.append(idx)
            else:
                # Keep candidate only if ensemble improves
                prev_score = self._evaluate_ensemble(selected_preds, y_true)
                if score > prev_score:
                    selected_preds.append(candidate_preds)
                    selected_indices.append(idx)

        return selected_preds, selected_indices
