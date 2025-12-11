from PyPruning.ProxPruningClassifier import ProxPruningClassifier
import numpy as np

class L1PruningClassifier:
    """
    Wrapper for the official L1 Pruning implementation from PyPruning.
    Handles both classification and regression.
    Corresponds to the 'L1' baseline in Buschjäger & Morik (2023).
    """
    def __init__(self, l_reg=0.7, step_size=0.01, epochs=200, task="regression"):
        """
        task: "regression" or "classification"
        """
        self.task = task.lower()
        if self.task not in ["regression", "classification"]:
            raise ValueError(f"Unsupported task: {task}. Use 'regression' or 'classification'.")


        loss = "mse" if self.task == "regression" else "cross-entropy"

        self.model = ProxPruningClassifier(
            loss=loss,
            ensemble_regularizer="L1",
            l_ensemble_reg=l_reg,
            step_size=step_size,
            epochs=epochs,
            regularizer=None,      
            normalize_weights=True
        )

    def select(self, predictions_list, y_true):
        """
        predictions_list: List of arrays from base trees
            - Regression: (n_samples,)
            - Classification: (n_samples, n_classes) or (n_samples,) for labels
        y_true: true targets
            - Regression: (n_samples,)
            - Classification: (n_samples,) integer labels
        Returns:
            selected_preds: List of selected tree predictions
            selected_indices: List of indices of selected trees
        """
        proba = np.array(predictions_list)

        if self.task == "regression":
            if proba.ndim == 2:
                proba = proba[:, :, np.newaxis] 
        else:
            if proba.ndim == 2:
                n_classes = int(np.max(proba) + 1)
                one_hot = np.zeros((proba.shape[0], proba.shape[1], n_classes))
                for i in range(proba.shape[0]):
                    one_hot[i, np.arange(proba.shape[1]), proba[i]] = 1
                proba = one_hot

        selected_indices, weights = self.model.prune_(proba, y_true)

        selected_indices = list(selected_indices)
        selected_preds = [predictions_list[i] for i in selected_indices]

        return selected_preds, selected_indices
