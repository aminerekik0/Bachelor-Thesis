import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, roc_auc_score
import shap

from sklearn.linear_model import LinearRegression, LogisticRegression

from src.BaseMetaModel import BaseMetaModel


class BasicMetaModel(BaseMetaModel):
    def __init__(self, n_estimators=50, max_depth=5, keep_ratio=0.25, **kwargs):
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

        self.pruned_ensemble_metric = None
        self.mse = None
        self.rmse = None
        self.mae = None
        self.r2 = None

        self.acc = None
        self.f1 = None
        self.auc = None
        self.pruned_tree_weights = None

        self.shap_then_corr_trees = None
        self.shap_then_corr_weights = None

        self.corr_then_shap_trees = None
        self.corr_then_shap_mse = None

    def train(self, *args, **kwargs):
        print(f"=== Stage 1: Training model and pruning by SHAP (keep top {self.keep_ratio*100}%) ===")

        X_meta_train = self._get_meta_features(self.workflow.X_train_meta, self.workflow.individual_trees)
        X_meta_eval = self._get_meta_features(self.workflow.X_eval_meta, self.workflow.individual_trees)
        y_train_meta = self.workflow.y_train_meta
        y_eval_meta = self.workflow.y_eval_meta

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

        y_pred = self.meta_model.predict(X_meta_eval)

        if self.data_type == "regression":
            self.mse = mean_squared_error(y_eval_meta, y_pred)
            self.rmse = np.sqrt(self.mse)
            self.mae = mean_absolute_error(y_eval_meta, y_pred)
            self.r2 = r2_score(y_eval_meta, y_pred)
        else:
            self.acc = accuracy_score(y_eval_meta, y_pred)

        explainer = shap.Explainer(self.meta_model, X_meta_eval, algorithm="linear")
        shap_result = explainer(X_meta_eval)
        self.shap_values = np.array(shap_result.values)

        self._prune_trees_by_shap()

    def _get_meta_features(self, X, trees_list, use_proba=True):
        if use_proba and self.data_type == "classification":
            # shape: (n_samples, n_trees * n_classes)
            return np.column_stack([t.predict_proba(X)[:,1] for t in trees_list]).astype(np.float32)
        else:
            return np.column_stack([t.predict(X) for t in trees_list]).astype(np.float32)

    def _prune_trees_by_shap(self):
        shap_vals_for_importance = self.shap_values

        tree_importance = np.mean(np.abs(shap_vals_for_importance), axis=0)

        X_meta_eval = self._get_meta_features(self.workflow.X_eval_meta, self.workflow.individual_trees)
        y_eval_meta = self.workflow.y_eval_meta

        rmse_scores = []
        for t in self.workflow.individual_trees:
            pred = t.predict(self.workflow.X_eval_meta)
            rmse_scores.append(np.sqrt(mean_squared_error(y_eval_meta, pred)))
        rmse_scores = np.array(rmse_scores)
        rmse_scores = 1 / (rmse_scores + 1e-12)

        combined =   (tree_importance / (np.max(tree_importance) + 1e-12))
        self.tree_importance = combined

        all_trees = self.workflow.individual_trees
        n_trees = len(all_trees)

        k = max(5, int(n_trees * self.keep_ratio))
        top_indices = np.argsort(combined)[-k:][::-1]

        self.pruned_trees = [self.workflow.individual_trees[i] for i in top_indices]
        self.pruned_tree_weights = combined[top_indices]

    def evaluate(self):
        X_test = self.workflow.X_test
        y_test = self.workflow.y_test

        if self.pruned_tree_weights is None or len(self.pruned_tree_weights) == 0:
            return None, self.main_loss

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
            preds_matrix = self._get_meta_features(X_test, self.pruned_trees)
            final_preds = preds_matrix.squeeze()
        else:
            if self.data_type == "regression":
                preds_matrix = self._get_meta_features(X_test, self.pruned_trees)
                final_preds = preds_matrix @ w_final
            else:
                tree_labels = np.vstack([t.predict(X_test) for t in self.pruned_trees])
                from scipy.stats import mode
                mode_result = mode(tree_labels, axis=0, keepdims=False)
                final_preds_class = mode_result.mode

        if self.data_type == "regression":
            rmse = np.sqrt(mean_squared_error(y_test, final_preds))
            r2 = r2_score(y_test, final_preds)

            self.pruned_ensemble_metric = rmse
            self.r2 = r2

            print(f"Pre-Pruned RMSE: {rmse:.4f} | R2: {r2:.4f}")
            return rmse, r2

        else:
            acc = accuracy_score(y_test, final_preds_class)
            f1 = f1_score(y_test, final_preds_class, average="weighted")
            auc = roc_auc_score(y_test, final_preds_class)

            self.acc = acc
            self.f1 = f1
            self.auc = auc

            print(f"Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

            return f1, auc
