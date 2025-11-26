import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import accuracy_score

class ST_DES(BaseEstimator, ClassifierMixin):
    """
    Similarity Tree - Dynamic Ensemble Selection (ST-DES).

    Corrected to handle large datasets using batch processing.
    """

    def __init__(self, n_estimators=100, sigma_st=0.5, theta_st=0.5, val_size=0.2, random_state=None):
        self.n_estimators = n_estimators
        self.sigma_st = sigma_st
        self.theta_st = theta_st
        self.val_size = val_size
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        # 1. Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.val_size, random_state=self.random_state
        )
        self.X_val_ = X_val
        self.y_val_ = y_val
        self.classes_ = np.unique(y)

        # 2. Train Random Forest
        self.rf_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
        self.rf_.fit(X_train, y_train)

        # 3. Pre-compute validation stats
        self.val_predictions_correctness_ = []
        self.val_paths_ = []

        for tree in self.rf_.estimators_:
            # [cite_start]Correctness: 1 if correct, -1 if wrong [cite: 212]
            preds = tree.predict(X_val)
            correctness = np.where(preds == y_val, 1, -1)
            self.val_predictions_correctness_.append(correctness)

            # Store validation paths (CSR Matrix)
            self.val_paths_.append(tree.decision_path(X_val))

        return self

    def _compute_similarity_batch(self, tree_idx, test_paths_batch):
        """
        Computes Similarity for a specific batch of test instances.
        """
        val_paths = self.val_paths_[tree_idx] # (n_val, n_nodes)

        # Dot product: (batch_size, n_nodes) @ (n_nodes, n_val) -> (batch_size, n_val)
        # This is where the memory explosion happened previously.
        # By passing a batch, the result is small enough for RAM.
        shared_nodes = test_paths_batch.dot(val_paths.T).toarray()

        # Path lengths for the current batch
        path_lengths = np.diff(test_paths_batch.indptr).reshape(-1, 1)

        # Eq 2: Raw Similarity
        similarity = shared_nodes / path_lengths

        # [cite_start]Eq 3: Thresholding [cite: 201]
        similarity[similarity < self.sigma_st] = 0

        return similarity

    def predict(self, X, batch_size=1000):
        """
        Predicts class labels using DES with batch processing to avoid memory errors.

        Args:
            X: Test data
            batch_size: Number of test samples to process at once.
                        1000 samples * 100k val samples ~= 800MB RAM.
        """
        check_is_fitted(self)
        X = check_array(X)
        n_test = X.shape[0]
        n_classes = len(self.classes_)

        # Initialize weighted votes accumulator
        final_votes = np.zeros((n_test, n_classes))

        # Pre-compute all test paths (Sparse matrix is small, so this is safe)
        # However, we iterate trees, so we do this inside the tree loop to save memory
        # or do it globally if RAM allows. Doing it per tree is safer.

        for i, tree in enumerate(self.rf_.estimators_):
            # Get all paths for this tree (returns sparse CSR)
            all_test_paths = tree.decision_path(X)

            # --- BATCH PROCESSING START ---
            for start_idx in range(0, n_test, batch_size):
                end_idx = min(start_idx + batch_size, n_test)

                # Slice the sparse matrix for the current batch
                batch_paths = all_test_paths[start_idx:end_idx]

                # 1. Compute Similarity for Batch
                sim_matrix = self._compute_similarity_batch(i, batch_paths)

                # 2. Compute Competence for Batch
                correctness = self.val_predictions_correctness_[i].reshape(1, -1)
                numerator = np.sum(sim_matrix * correctness, axis=1)
                denominator = np.sum(sim_matrix, axis=1)

                with np.errstate(divide='ignore', invalid='ignore'):
                    competence = numerator / denominator
                    competence = np.nan_to_num(competence)

                # 3. Selection
                selected_mask = competence > self.theta_st
                selected_indices_local = np.where(selected_mask)[0]

                if len(selected_indices_local) > 0:
                    # Predict only for the batch
                    # Note: tree.predict(X) is fast, but better to slice X if strictly needed.
                    # Since tree.predict is fast, we can slice predictions.
                    batch_preds = tree.predict(X[start_idx:end_idx])

                    for loc_idx in selected_indices_local:
                        pred_class = batch_preds[loc_idx]
                        class_idx = np.where(self.classes_ == pred_class)[0][0]

                        # Add vote to the global votes array
                        # Global index = start_idx + loc_idx
                        final_votes[start_idx + loc_idx, class_idx] += competence[loc_idx]
            # --- BATCH PROCESSING END ---

        # 4. Final Aggregation
        final_predictions = []
        rf_standard_preds = self.rf_.predict(X)

        for k in range(n_test):
            if np.sum(final_votes[k]) == 0:
                final_predictions.append(rf_standard_preds[k])
            else:
                final_predictions.append(self.classes_[np.argmax(final_votes[k])])

        return np.array(final_predictions)

if __name__ == "__main__":
    from sklearn.datasets import fetch_covtype




    from sklearn.datasets import fetch_covtype
    data = fetch_covtype(as_frame=False)
    X = data.data
    y = (data.target == 2).astype(int)

    # Split
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Train size: {X_train_full.shape[0]}, Test size: {X_test.shape[0]}")

    # Run ST-DES
    print("Training ST-DES...")
    st_des = ST_DES(n_estimators=50, sigma_st=0.5, theta_st=0.5, random_state=42) # Reduced estimators for speed demo
    st_des.fit(X_train_full, y_train_full)

    print("Predicting (with batching)...")
    y_pred = st_des.predict(X_test, batch_size=1000)

    acc = accuracy_score(y_test, y_pred)
    print(f"ST-DES Accuracy: {acc:.4f}")