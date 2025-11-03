import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score ,accuracy_score
import shap
from .BaseMetaModel import BaseMetaModel

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

        self.pruned_ensemble_mse = None
        self.mse = None
        self.rmse = None
        self.mae = None
        self.r2 =None

    def train(self):

        X_meta_train = self._get_meta_features(self.workflow.X_train_meta)
        X_meta_eval = self._get_meta_features(self.workflow.X_eval_meta)
        y_train_meta = self.workflow.y_train_meta
        y_eval_meta = self.workflow.y_eval_meta


        if self.data_type == "regression":
            self.meta_model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
        else:
            self.meta_model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
        self.meta_model.fit(X_meta_train, y_train_meta)

        y_pred = self.meta_model.predict(X_meta_eval)

        if self.data_type == "regression":
            self.mse = mean_squared_error(y_eval_meta, y_pred)
            self.rmse = np.sqrt(self.mse)
            self.mae = mean_absolute_error(y_eval_meta, y_pred)
            self.r2 = r2_score(y_eval_meta, y_pred)

        else:
            self.mse = accuracy_score(y_eval_meta, y_pred)

        explainer = shap.TreeExplainer(self.meta_model)
        self.shap_values = np.array(explainer.shap_values(X_meta_eval))

        self._prune_trees()

    def _get_meta_features(self, X):
        return np.column_stack([t.predict(X) for t in self.workflow.individual_trees]).astype(np.float32)

    def _prune_trees(self):
        """
        Prune least important trees based on SHAP values and keep_ratio.
        """
        tree_importance = np.mean(np.abs(self.shap_values), axis=0)
        sorted_indices = np.argsort(tree_importance)
        self.tree_importance = tree_importance

        total_trees = len(self.workflow.individual_trees)
        keep_count = int(self.keep_ratio * total_trees)

        trees_to_keep = sorted_indices[-keep_count:]
        self.pruned_trees = [self.workflow.individual_trees[i] for i in trees_to_keep]

        # Compute total loss (main + prune + diversity)
        L_prune = self._prune_loss(self.shap_values)
        L_div = self._diversity_loss(self.shap_values)
        self.total_loss = self.mse + L_prune + L_div

    def evaluate(self):
        """
        Evaluate full vs pruned ensemble on workflow test set.
        """

        X_test = self.workflow.X_test
        y_test = self.workflow.y_test

        # Pruned ensemble
        pruned_preds = np.mean(np.vstack([t.predict(X_test) for t in self.pruned_trees]), axis=0)
        self.pruned_ensemble_mse = mean_squared_error(y_test, pruned_preds) if self.data_type == "regression"\
            else accuracy_score(y_test, pruned_preds)

        print ("full_metric" , self.workflow.full_metric)
        print ("pruned_metric", self.pruned_ensemble_mse)

        return self.full_metric, self.main_loss
    @staticmethod
    def _prune_loss(shap_values):
        abs_shap = np.abs(shap_values)
        p_hat = abs_shap / (abs_shap.sum(axis=1, keepdims=True) + 1e-12)
        return -np.mean(np.sum(p_hat * np.log(p_hat + 1e-12), axis=1))

    @staticmethod
    def _diversity_loss(shap_values):
        phi_bar = np.mean(np.abs(shap_values), axis=0)
        p_tilde = phi_bar / (np.sum(phi_bar) + 1e-8)
        return float(np.sum(p_tilde ** 2))