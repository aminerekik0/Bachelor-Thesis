import numpy as np
import shap
from sklearn.linear_model import LinearRegression, LogisticRegression
from src.BaseMetaModel import BaseMetaModel


class PrePruner(BaseMetaModel):
    def __init__(self, keep_ratio=0.25, data_type="regression", **kwargs):
        super().__init__(**kwargs)

        self.keep_ratio = keep_ratio
        self.meta_model = None
        self.shap_values = None
        self.pruned_trees = None
        self.data_type = data_type


    def train(self, *args, **kwargs):
        print(f"=== Stage 1: Training model and pruning by SHAP (keep top {self.keep_ratio*100}%) ===")

        X_meta_train = self._get_meta_features(self.workflow.X_train_meta, self.workflow.individual_trees)
        X_meta_eval = self._get_meta_features(self.workflow.X_eval_meta, self.workflow.individual_trees)
        y_train_meta = self.workflow.y_train_meta


        if self.data_type == "regression":
            self.meta_model = LinearRegression()
            self.meta_model.fit(X_meta_train, y_train_meta)
        else:
            self.meta_model = LogisticRegression(
                random_state=self.random_state,
                max_iter=2000,
                solver="lbfgs"
            )
            self.meta_model.fit(X_meta_train, y_train_meta)

        explainer = shap.Explainer(self.meta_model, X_meta_eval, algorithm="linear")
        shap_result = explainer(X_meta_eval)
        self.shap_values = np.array(shap_result.values)

        self._prune_trees_by_shap()

    def _get_meta_features(self, X, trees_list, use_proba=True):
        if use_proba and self.data_type == "classification":
            return np.column_stack([t.predict_proba(X)[:,1] for t in trees_list]).astype(np.float32)
        else:
            return np.column_stack([t.predict(X) for t in trees_list]).astype(np.float32)

    def _prune_trees_by_shap(self):

        shap_vals_for_importance = self.shap_values

        tree_importance = np.mean(np.abs(shap_vals_for_importance), axis=0)

        tree_importance_norm =   (tree_importance / (np.max(tree_importance) + 1e-12))

        all_trees = self.workflow.individual_trees

        n_trees = len(all_trees)

        k = max(5, int(n_trees * self.keep_ratio))

        top_indices = np.argsort(tree_importance_norm)[-k:][::-1]

        self.pruned_trees = [self.workflow.individual_trees[i] for i in top_indices]

        self.pruned_tree_weights = tree_importance_norm[top_indices]




