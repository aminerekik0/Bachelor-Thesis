import numpy as np
from scipy.stats import mode
from sklearn.metrics import mean_squared_error, accuracy_score, cohen_kappa_score


class KTPruning:
    """
    Kuncheva & Whitaker (2003) Diversity-Based Pruning.
    
    KT-diversity measures pairwise disagreement:
        dis(i, j) = P(h_i != h_j)
    A tree with HIGH average disagreement is more diverse.
    
    We keep the top-k most diverse trees.
    """

    def __init__(self, data_type="classification"):
        self.data_type = data_type

    def disagreement(self, a, b):
        """P(h_i != h_j)."""
        return np.mean(a != b)

    def select(self, base_preds_val, y_val):
        n_trees = len(base_preds_val)
        if n_trees == 0:
            return None, []

        # Regression → KT diversity not meaningful
        if self.data_type == "regression":
            return None, list(range(n_trees))

        # Compute full disagreement matrix
        D = np.zeros((n_trees, n_trees))
        for i in range(n_trees):
            for j in range(i + 1, n_trees):
                d = self.disagreement(base_preds_val[i], base_preds_val[j])
                D[i, j] = d
                D[j, i] = d

        # Average diversity per tree
        avg_div = np.mean(D, axis=1)

        # Select the TOP-K most diverse trees
        k = max(3, n_trees // 4)  # e.g., keep 25% most diverse
        order = np.argsort(-avg_div)  # descending (high = diverse)

        selected = order[:k].tolist()
        return None, selected
# ===============================================================
#  1) REPruning — Reduced Ensemble (Backward elimination)
#     Based on Margineantu & Dietterich style pruning
# ===============================================================
class REPruning:
    """
    Reduced Ensemble Pruning (RE):
    - Start from FULL ensemble.
    - Iteratively remove the model whose removal
      hurts performance the least.
    - Stop when removal no longer improves performance.
    """

    def __init__(self, data_type="regression", min_trees=1):
        self.data_type = data_type
        self.min_trees = min_trees

    def _evaluate(self, preds_list, y_true):
        preds = np.array(preds_list)

        if self.data_type == "regression":
            final_pred = np.mean(preds, axis=0)
            # negative MSE so that "larger is better"
            return -mean_squared_error(y_true, final_pred)

        # classification
        final_pred = mode(preds, axis=0, keepdims=False).mode
        return accuracy_score(y_true, final_pred)

    def select(self, predictions_list, y_true):
        predictions_list = [np.array(p) for p in predictions_list]
        n_models = len(predictions_list)

        # Start with all models
        current_indices = list(range(n_models))
        current_preds = [predictions_list[i] for i in current_indices]
        best_score = self._evaluate(current_preds, y_true)

        improved = True

        while improved and len(current_indices) > self.min_trees:
            improved = False
            best_candidate_score = best_score
            best_remove_idx = None

            # Try removing each model once
            for idx in current_indices:
                trial_indices = [j for j in current_indices if j != idx]
                trial_preds = [predictions_list[j] for j in trial_indices]

                score = self._evaluate(trial_preds, y_true)

                # We want highest score (accuracy or -MSE)
                if score >= best_candidate_score:
                    best_candidate_score = score
                    best_remove_idx = idx

            # If we found a removal that does not worsen performance, apply it
            if best_remove_idx is not None and best_candidate_score >= best_score:
                current_indices.remove(best_remove_idx)
                best_score = best_candidate_score
                improved = True

        selected_preds = [predictions_list[i] for i in current_indices]
        return selected_preds, current_indices


# ===============================================================
#  2) KappaPruning — Kappa-based diversity pruning (classification)
# ===============================================================
class KappaPruning:
    """
    Kappa-based pruning for classification ensembles.

    - Works ONLY for classification (Cohen's kappa is classification-based).
    - Greedy forward selection:
        1. Start with the most accurate model.
        2. At each step, among remaining models, consider:
            - candidate has low average kappa with selected set (high diversity)
            - AND leads to best ensemble accuracy.
        3. Stop when no candidate improves ensemble accuracy.
    """

    def __init__(self, data_type="classification", max_kappa=0.9, min_trees=1):
        self.data_type = data_type
        self.max_kappa = max_kappa
        self.min_trees = min_trees

    def _evaluate(self, preds_list, y_true):
        preds = np.array(preds_list)
        final_pred = mode(preds, axis=0, keepdims=False).mode
        return accuracy_score(y_true, final_pred)

    def select(self, predictions_list, y_true):
        predictions_list = [np.array(p) for p in predictions_list]
        n_models = len(predictions_list)

        # If regression, just return all models (kappa is undefined)
        if self.data_type == "regression":
            print("[KappaPruning] Warning: regression task, returning all models unchanged.")
            selected_indices = list(range(n_models))
            selected_preds = [predictions_list[i] for i in selected_indices]
            return selected_preds, selected_indices

        # classification case
        # Step 1: accuracies of individual models
        indiv_acc = [accuracy_score(y_true, p) for p in predictions_list]
        sorted_idx = np.argsort(indiv_acc)[::-1]

        # Start with best individual
        selected_indices = [sorted_idx[0]]
        selected_preds = [predictions_list[sorted_idx[0]]]
        best_score = indiv_acc[sorted_idx[0]]

        remaining = [i for i in range(n_models) if i not in selected_indices]

        while remaining and len(selected_indices) < n_models:
            best_candidate = None
            best_candidate_score = best_score

            for idx in remaining:
                # compute average kappa between this candidate and current selected set
                kappas = []
                for sel_idx in selected_indices:
                    k = cohen_kappa_score(predictions_list[sel_idx], predictions_list[idx])
                    kappas.append(k)
                avg_kappa = np.mean(kappas)

                # enforce diversity constraint
                if avg_kappa > self.max_kappa:
                    continue

                # evaluate ensemble with this candidate
                trial_preds = selected_preds + [predictions_list[idx]]
                score = self._evaluate(trial_preds, y_true)

                # keep if we improve accuracy
                if score > best_candidate_score:
                    best_candidate_score = score
                    best_candidate = idx

            if best_candidate is not None:
                selected_indices.append(best_candidate)
                selected_preds.append(predictions_list[best_candidate])
                best_score = best_candidate_score
                remaining.remove(best_candidate)
            else:
                # no candidate improves performance under diversity constraint
                break

        # if we ended up with too few trees, we could relax constraints,
        # but for now we just return what we have
        if len(selected_indices) < self.min_trees:
            # simple fallback: fill with best remaining according to accuracy
            remaining_sorted = [i for i in sorted_idx if i not in selected_indices]
            to_add = remaining_sorted[: (self.min_trees - len(selected_indices))]
            for idx in to_add:
                selected_indices.append(idx)
                selected_preds.append(predictions_list[idx])

        return selected_preds, selected_indices