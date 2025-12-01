import numpy as np
from scipy.stats import mode
from sklearn.metrics import mean_squared_error, accuracy_score

class ADPPruner:
    """
    Accuracy-Diversity Pruning (ADP) algorithm.
    Reference: Bhatnagar et al. (2014). "Accuracy-diversity based pruning of classifier ensembles".

    Adapted Search Strategy for Scalability:
    1. Filter top individuals (capped at 20 to prevent explosion).
    2. Filter top diverse pairs (capped at 50 to prevent explosion).
    3. Iteratively grow ensembles by adding disjoint diverse pairs to best candidates.
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
        """Evaluates a list of predictions (ensemble) against y_true."""
        if len(ensemble_preds) == 0:
            return -float('inf') if self.data_type == "classification" else float('inf')

        # Stack predictions: (n_samples, n_members)
        preds_stack = np.column_stack(ensemble_preds)

        if self.data_type == "regression":
            # Averaging
            mean_preds = np.mean(preds_stack, axis=1)
            # Default: Minimize MSE (return negative MSE for maximization logic)
            if self.metric is None:
                return -mean_squared_error(y_true, mean_preds)
            else:
                return self.metric(y_true, mean_preds)
        else:
            # Majority Voting
            mode_res = mode(preds_stack, axis=1, keepdims=False)
            majority_vote = mode_res.mode.ravel()
            # Default: Maximize Accuracy
            if self.metric is None:
                return accuracy_score(y_true, majority_vote)
            else:
                return self.metric(y_true, majority_vote)

    def _calculate_diversity(self, pred_i, pred_j, y_true):
        """
        Calculates pairwise diversity.
        Regression: Correlation coefficient (Lower is more diverse).
        Classification: Q-statistic (Lower is more diverse).
        """
        if self.data_type == "regression":
            # Pearson correlation
            if np.std(pred_i) == 0 or np.std(pred_j) == 0:
                return 1.0 # Treat constant predictions as highly correlated (bad)
            corr = np.corrcoef(pred_i, pred_j)[0, 1]
            return corr
        else:
            # Q-statistic (Yule's Q)
            ci = (pred_i == y_true)
            cj = (pred_j == y_true)

            N11 = np.sum(ci & cj)       # Both correct
            N00 = np.sum(~ci & ~cj)     # Both incorrect
            N10 = np.sum(ci & ~cj)      # i correct, j incorrect
            N01 = np.sum(~ci & cj)      # i incorrect, j correct

            numerator = (N11 * N00) - (N10 * N01)
            denominator = (N11 * N00) + (N10 * N01)

            if denominator == 0:
                return 0.0
            return numerator / denominator

    def select(self, predictions_list, y_true):
        """
        Main ADP selection logic with optimization caps.
        """
        predictions_list = [np.array(p) for p in predictions_list]
        n_models = len(predictions_list)
        if n_models == 0:
            return [], []

        # --- Step 1: Filter Top Individuals ---
        individual_scores = []
        for i, pred in enumerate(predictions_list):
            score = self._evaluate_ensemble([pred], y_true)
            individual_scores.append((i, score))

        # Sort descending (Higher score is better)
        individual_scores.sort(key=lambda x: x[1], reverse=True)

        # OPTIMIZATION: Keep max 20 candidates to prevent O(N^4) explosion
        n_keep_ind = min(100, max(1, int(0.5 * n_models)))

        # Initial candidates are ensembles of size 1: [[idx1], [idx2], ...]
        C_candidates = [[x[0]] for x in individual_scores[:n_keep_ind]]

        # Best ensemble found so far (initialized with best single model)
        best_indices = C_candidates[0]
        best_score = individual_scores[0][1]

        # --- Step 2: Filter Top Diverse Pairs ---
        all_pairs = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                div_val = self._calculate_diversity(predictions_list[i], predictions_list[j], y_true)
                all_pairs.append(([i, j], div_val))

        # Sort by diversity ascending (Lower correlation/Q is better/more diverse)
        all_pairs.sort(key=lambda x: x[1])

        # OPTIMIZATION: Keep max 50 pairs to prevent explosion

        n_keep_pairs = min(500, max(1, int(0.5 * len(all_pairs))))
        DP_pairs = [p[0] for p in all_pairs[:n_keep_pairs]]

        # --- Step 3: Grow Loop ---
        # Loop l=1...k. We loop until improvement stops or max size reached.
        max_iters = min(10, n_models // 2)

        for _ in range(max_iters):
            S_next = [] # Candidate ensembles for this level

            # Combine every candidate with every disjoint diverse pair
            for cand_indices in C_candidates:
                cand_set = set(cand_indices)

                for pair in DP_pairs:
                    if pair[0] not in cand_set and pair[1] not in cand_set:
                        # Create new ensemble
                        new_indices = cand_indices + pair

                        # Evaluate
                        current_preds = [predictions_list[k] for k in new_indices]
                        score = self._evaluate_ensemble(current_preds, y_true)

                        S_next.append((new_indices, score))

            if not S_next:
                break # No valid extensions found

            # Sort S by accuracy (descending)
            S_next.sort(key=lambda x: x[1], reverse=True)

            best_in_S_indices = S_next[0][0]
            best_in_S_score = S_next[0][1]

            # --- Step 4: Check Improvement ---
            if best_in_S_score > best_score:
                best_score = best_in_S_score
                best_indices = best_in_S_indices

                # Prepare candidates for next level: Top 50% of S (capped at 20)

                n_keep_S = min(100, max(1, int(0.5 * len(S_next))))
                # Only keep the indices list for next iteration
                C_candidates = [item[0] for item in S_next[:n_keep_S]]
            else:
                # Stop if no improvement over previous best
                break

        # Return format matching ConstructiveWithoutExploration
        selected_preds = [predictions_list[i] for i in best_indices]
        return selected_preds, best_indices