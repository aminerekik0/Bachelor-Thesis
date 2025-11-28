import numpy as np
from scipy.stats import mode
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from copy import deepcopy

class PruningWithoutExploration:
    """
    Pruning Without Exploration (Pw/oE) ensemble selection.
    Starts with all models, removes worst performer iteratively.
    """
    def __init__(self, data_type="regression", metric=None):
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
        selected_indices = list(range(n_models))
        selected_preds = predictions_list.copy()

        while len(selected_indices) > 1:
            # Evaluate individual performances
            individual_scores = [self._evaluate_ensemble([selected_preds[i]], y_true) for i in range(len(selected_preds))]
            worst_idx_in_selected = np.argmin(individual_scores)

            # Try removing worst performer
            trial_preds = [selected_preds[i] for i in range(len(selected_preds)) if i != worst_idx_in_selected]
            trial_score = self._evaluate_ensemble(trial_preds, y_true)
            current_score = self._evaluate_ensemble(selected_preds, y_true)

            if trial_score >= current_score:
                # Removal improves ensemble
                del selected_preds[worst_idx_in_selected]
                del selected_indices[worst_idx_in_selected]
            else:
                # Stop removing if no improvement
                break

        return selected_preds, selected_indices
