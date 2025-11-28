import numpy as np
from scipy.stats import mode
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from copy import deepcopy

class PruningWithExploration:
    """
    Pruning With Exploration (PwE) ensemble selection.
    Evaluates removing all candidates (except best) at each step.
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
            best_score = self._evaluate_ensemble(selected_preds, y_true)

            # Exclude the best performer from removal
            individual_scores = [self._evaluate_ensemble([p], y_true) for p in selected_preds]
            best_idx = np.argmax(individual_scores)
            removable = [i for i in range(len(selected_preds)) if i != best_idx]

            # Check which removal improves ensemble most
            best_improve = -np.inf
            remove_idx = None
            for i in removable:
                trial_preds = [selected_preds[j] for j in range(len(selected_preds)) if j != i]
                trial_score = self._evaluate_ensemble(trial_preds, y_true)
                improvement = trial_score - best_score
                if improvement > best_improve:
                    best_improve = improvement
                    remove_idx = i

            if best_improve > 0 and remove_idx is not None:
                del selected_preds[remove_idx]
                del selected_indices[remove_idx]
            else:
                break

        return selected_preds, selected_indices
