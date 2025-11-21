import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score ,accuracy_score , f1_score
import shap

import lightgbm as lgb
from sklearn.linear_model import LinearRegression, LogisticRegression

from BaseMetaModel import BaseMetaModel


class BasicMetaModel(BaseMetaModel):
    def __init__(self, n_estimators=50, max_depth=5, keep_ratio=0.3 , **kwargs):
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.keep_ratio = keep_ratio

        self.meta_model = None
        self.shap_values = None
        self.pruned_trees = None
        self.tree_importance = None
        self.main_loss = None
        self.total_loss = None
        self.full_metric = None

        self.pruned_ensemble_metric = None  # <- updated name
        self.mse = None
        self.rmse = None
        self.mae = None

        self.acc = None
        self.r2 =None
        self.f1 = None
        self.auc = None
        self.pruned_tree_weights = None

        self.shap_then_corr_trees = None
        self.shap_then_corr_weights = None

        self.corr_then_shap_trees = None
        self.corr_then_shap_mse = None

    def train(self, *args, **kwargs):
        """
        Runs the "SHAP-first" (Option B, Stage 1) pruning.
        """
        print(f"=== Stage 1: Training model and pruning by SHAP (keep top {self.keep_ratio*100}%) ===")

        X_meta_train = self._get_meta_features(self.workflow.X_train_meta, self.workflow.individual_trees)
        X_meta_eval = self._get_meta_features(self.workflow.X_eval_meta, self.workflow.individual_trees)
        y_train_meta = self.workflow.y_train_meta
        y_eval_meta = self.workflow.y_eval_meta


        if self.data_type == "regression":
            self.meta_model = LinearRegression()
            self.meta_model.fit(X_meta_train, y_train_meta)
        else:
            # binary logistic regression
            self.meta_model = LogisticRegression(
                random_state=self.random_state,
                max_iter=2000,
                solver="lbfgs"
            )
            self.meta_model.fit(X_meta_train, y_train_meta)

        y_pred = self.meta_model.predict(X_meta_eval)

        if self.data_type == "regression":
            self.mse = mean_squared_error(y_eval_meta, y_pred)
            self.rmse = np.sqrt(self.mse)
            self.mae = mean_absolute_error(y_eval_meta, y_pred)
            self.r2 = r2_score(y_eval_meta, y_pred)
        else:
            self.acc = accuracy_score(y_eval_meta, y_pred)

        # SHAP for binary classification
        explainer = shap.Explainer(self.meta_model, X_meta_eval, algorithm="linear")
        shap_result = explainer(X_meta_eval)
        self.shap_values = np.array(shap_result.values)

        self._prune_trees_by_shap()


    def _get_meta_features(self, X, trees_list):
        return np.column_stack([t.predict(X) for t in trees_list]).astype(np.float32)


    def _prune_trees_by_shap(self):
        """
        (Helper for train()) Prunes by SHAP and keep_ratio.
        """
        shap_vals_for_importance = self.shap_values

        tree_importance = np.mean(np.abs(shap_vals_for_importance), axis=0)
        normalized_shap = tree_importance / (np.max(tree_importance) + 1e-12)

        self.tree_importance = normalized_shap

        all_trees = self.workflow.individual_trees
        n_trees = len(all_trees)
        k = max(5, int(n_trees * self.keep_ratio))

        top_indices = np.argsort(tree_importance)[-k:][::-1]

        self.pruned_trees = [self.workflow.individual_trees[i] for i in top_indices]
        self.pruned_tree_weights = tree_importance[top_indices]


    def evaluate(self):
        """
        Evaluates the "SHAP-first" (Option B, Stage 1) ensemble.
        """
        X_test = self.workflow.X_test
        y_test = self.workflow.y_test

        if self.pruned_tree_weights is None or len(self.pruned_tree_weights) == 0:
            print("[WARN] No pruned tree weights found. Cannot evaluate weighted ensemble.")
            return None, self.main_loss

        pruned_weights = self.pruned_tree_weights
        normalized_weights = pruned_weights / (np.sum(pruned_weights) + 1e-12)


        if self.data_type == "regression":
            X_train_final = self._get_meta_features(self.workflow.X_train_meta, self.pruned_trees)
            y_train_final = self.workflow.y_train_meta
            final_eval_model = LinearRegression().fit(X_train_final, y_train_final)
            w_final = np.abs(final_eval_model.coef_)
            w_final /= np.sum(w_final)
        else:
            X_train_final = self._get_meta_features(self.workflow.X_train_meta, self.pruned_trees)
            y_train_final = self.workflow.y_train_meta

            final_eval_model = LogisticRegression(max_iter=2000).fit(X_train_final, y_train_final)
            w_final = np.abs(final_eval_model.coef_[0])
            w_final /= np.sum(w_final)


        if len(self.pruned_trees) == 1:
            pruned_preds_matrix = self._get_meta_features(X_test, self.pruned_trees)
            final_preds = pruned_preds_matrix.squeeze()
        else:
            if self.data_type == "regression":
                pruned_preds_matrix = self._get_meta_features(X_test, self.pruned_trees)
                final_preds = pruned_preds_matrix @ w_final
            else:
                tree_labels = np.vstack([t.predict(X_test) for t in self.pruned_trees])
                from scipy.stats import mode
                mode_result = mode(tree_labels, axis=0, keepdims=False)
                final_preds_class = mode_result.mode



        if self.data_type == "regression":
            self.pruned_ensemble_metric = mean_squared_error(y_test, final_preds)
            print("Pre-Pruned ensemble Metric (Weighted):", self.pruned_ensemble_metric)
        else:
            self.pruned_ensemble_metric = accuracy_score(y_test, final_preds_class)
            self.f1 = f1_score(y_test, final_preds_class, average='weighted')
            from sklearn.metrics import roc_auc_score
            self.auc = roc_auc_score(y_test, final_preds_class)
            print("Pruned Metric:", self.pruned_ensemble_metric)
            print("Pruned F1:", self.f1)
            print("Pruned AUC:", self.auc)

        return self.pruned_ensemble_metric, self.main_loss
