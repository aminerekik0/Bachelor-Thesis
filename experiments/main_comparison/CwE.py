import numpy as np
from scipy.stats import mode
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from copy import deepcopy
class ConstructiveWithExploration:
    """
    Constructive With Exploration (CwE) ensemble selection.

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


        individual_scores = [self._evaluate_ensemble([p], y_true) for p in predictions_list]
        sorted_indices = list(np.argsort(individual_scores)[::-1])

        selected_indices = []
        selected_preds = []

        while sorted_indices:
            best_idx = None
            best_score = -np.inf
            for idx in sorted_indices:
                trial_preds = selected_preds + [predictions_list[idx]]
                score = self._evaluate_ensemble(trial_preds, y_true)
                if score > best_score:
                    best_score = score
                    best_idx = idx


            prev_score = self._evaluate_ensemble(selected_preds, y_true) if selected_preds else -np.inf
            if best_score <= prev_score:
                break


            selected_preds.append(predictions_list[best_idx])
            selected_indices.append(best_idx)
            sorted_indices.remove(best_idx)

        return selected_preds, selected_indices
