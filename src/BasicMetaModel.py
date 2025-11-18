import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score ,accuracy_score , f1_score
import shap
from BaseMetaModel import BaseMetaModel
import lightgbm as lgb
from sklearn.linear_model import LinearRegression, LogisticRegression

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

        self.pruned_ensemble_mse = None
        self.mse = None
        self.rmse = None
        self.mae = None

        self.acc = None
        self.r2 =None
        self.f1 = None
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

            #TODO : check this model
            self.meta_model = LogisticRegression(random_state=self.random_state, max_iter=1000)
            self.meta_model.fit(X_meta_train, y_train_meta)

        y_pred = self.meta_model.predict(X_meta_eval)

        if self.data_type == "regression":
            self.mse = mean_squared_error(y_eval_meta, y_pred)
            self.rmse = np.sqrt(self.mse)
            self.mae = mean_absolute_error(y_eval_meta, y_pred)
            self.r2 = r2_score(y_eval_meta, y_pred)
        else:

            self.acc = accuracy_score(y_eval_meta, y_pred)


        explainer = shap.Explainer(self.meta_model, X_meta_eval)
        self.shap_values = np.array(explainer(X_meta_eval).values)


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


        X_train_final = self._get_meta_features(self.workflow.X_train_meta, self.pruned_trees)
        y_train_final = self.workflow.y_train_meta
        final_eval_model = LinearRegression().fit(X_train_final, y_train_final)
        w_final = np.abs(final_eval_model.coef_)
        w_final /= np.sum(w_final)


        if len(self.pruned_trees) == 1:

            pruned_preds_matrix = self._get_meta_features(X_test, self.pruned_trees)

            final_preds = pruned_preds_matrix.squeeze()

        else:

            if self.data_type == "regression":

                pruned_preds_matrix = self._get_meta_features(X_test, self.pruned_trees)

                final_preds = pruned_preds_matrix @ w_final

            else :
                #TODO : check this another time
                tree_preds = np.vstack([t.predict(X_test) for t in self.pruned_trees])
                final_preds = np.apply_along_axis(
                    lambda x: np.bincount(x.astype(int), weights=normalized_weights).argmax(),
                    axis=0,
                    arr=tree_preds
                )

        if self.data_type == "regression":

            self.pruned_ensemble_mse = mean_squared_error(y_test, final_preds)

            print("Pre-Pruned ensemble MSE (Weighted):", self.pruned_ensemble_mse)

        else:
            self.pruned_ensemble_mse = accuracy_score(y_test, final_preds)
            self.f1 = f1_score(y_test, final_preds, average='weighted')
            print ("pruned_metric (Weighted)", self.pruned_ensemble_mse)

        return self.pruned_ensemble_mse, self.main_loss




    def prune_by_correlation(self, corr_thresh=0.9):
        """
        (Method B, Stage 2)
        Prunes the self.pruned_trees list using a STATIC correlation filter.
        """
        if not self.pruned_trees:
            print("[ERROR] Must call train() and _prune_trees() first.")
            return

        print(f"=== [Method B] Stage 2: Pruning {len(self.pruned_trees)} trees by correlation (threshold={corr_thresh}) ===")

        X_meta_train_pruned = self._get_meta_features(self.workflow.X_train_meta, self.pruned_trees)

        if X_meta_train_pruned.ndim == 1 or X_meta_train_pruned.shape[1] <= 1:
            print("[INFO] [Method B] Skipped (<= 1 tree).")
            self.shap_then_corr_trees = self.pruned_trees
            self.shap_then_corr_weights = self.pruned_tree_weights
            return

        corr_matrix = np.corrcoef(X_meta_train_pruned.T)
        np.fill_diagonal(corr_matrix, 0)

        if corr_matrix.size == 0:
            print("[INFO] [Method B] No pairwise correlations to check.")
            self.shap_then_corr_trees = self.pruned_trees
            self.shap_then_corr_weights = self.pruned_tree_weights
            return

        dynamic_thresh = corr_thresh
        print(f"[INFO] [Method B] Using static threshold: {dynamic_thresh:.4f}")

        redundant_local_indices = set(np.unique(np.where(np.abs(corr_matrix) > dynamic_thresh)[0]))

        final_trees_list = []
        final_weights_list = []
        for i, (tree, weight) in enumerate(zip(self.pruned_trees, self.pruned_tree_weights)):
            if i not in redundant_local_indices:
                final_trees_list.append(tree)
                final_weights_list.append(weight)

        self.shap_then_corr_trees = final_trees_list
        self.shap_then_corr_weights = np.array(final_weights_list)

        print(f"[INFO] [Method B] Removed {len(redundant_local_indices)} trees. Final size: {len(self.shap_then_corr_trees)}")

    def evaluate_shap_then_corr(self):
        """
        (Method B) Evaluates the (SHAP -> Corr) pruned ensemble.
        """
        X_test = self.workflow.X_test
        y_test = self.workflow.y_test
        trees_to_evaluate = self.shap_then_corr_trees


        if self.shap_then_corr_weights is None or len(self.shap_then_corr_weights) == 0:
            print("[WARN] [Method B] No trees/weights found. Run prune_by_correlation() first.")
            return None, self.main_loss

        X_train_final = self._get_meta_features(self.workflow.X_train_meta, trees_to_evaluate)
        y_train_final = self.workflow.y_train_meta

        pruned_weights = self.shap_then_corr_weights
        normalized_weights = pruned_weights / (np.sum(pruned_weights) + 1e-12)

        if len(self.shap_then_corr_trees) == 1:
            pruned_preds_matrix = self._get_meta_features(X_test, self.shap_then_corr_trees)
            final_preds = pruned_preds_matrix.squeeze()
        else:
            if self.data_type == "regression":
                final_eval_model = LinearRegression().fit(X_train_final, y_train_final)
                w_abs = np.abs(final_eval_model.coef_)
                weights_to_use = w_abs / (np.sum(w_abs) + 1e-12)
                pruned_preds_matrix = self._get_meta_features(X_test, trees_to_evaluate)
                final_preds = pruned_preds_matrix @ weights_to_use
            else:
                tree_preds = np.vstack([t.predict(X_test) for t in self.shap_then_corr_trees])


        if self.data_type == "regression":
            mse = mean_squared_error(y_test, final_preds)
            print("[Method B] Final (SHAP->Corr) Pruned MSE (Weighted):", mse)
            return mse, self.main_loss
        else:
            acc = accuracy_score(y_test, final_preds)
            print ("[Method B] Final (SHAP->Corr) Pruned Metric (Weighted):", acc)
            return acc, self.main_loss


    def train_corr_first(self, corr_thresh=0.9, shap_cutoff_percentile=70):
        """
        (Method C) Implements the "Correlation-first, then SHAP" logic.
        """
        all_trees = self.workflow.individual_trees
        X_train_full = self._get_meta_features(self.workflow.X_train_meta, all_trees)
        y_train = self.workflow.y_train_meta
        X_val_full = self._get_meta_features(self.workflow.X_eval_meta, all_trees)
        n_trees = X_train_full.shape[1]

        # --- Stage 1: Redundancy filtering by correlation ---
        print(f"=== [Method C] Stage 1: Redundancy filtering (threshold={corr_thresh}) ===")
        corr_matrix = np.corrcoef(X_train_full, rowvar=False)
        np.fill_diagonal(corr_matrix, 0)

        dynamic_thresh = corr_thresh
        print(f"[INFO] [Method C] Using static threshold: {dynamic_thresh:.4f}")

        keep_mask = np.ones(n_trees, dtype=bool)
        for i in range(n_trees):
            for j in range(i + 1, n_trees):
                if keep_mask[j] and abs(corr_matrix[i, j]) > dynamic_thresh:
                    keep_mask[j] = False

        X_train_red = X_train_full[:, keep_mask]
        X_val_red = X_val_full[:, keep_mask]

        original_indices_stage1 = np.where(keep_mask)[0]
        trees_stage1 = [all_trees[i] for i in original_indices_stage1]

        kept_after_corr = len(trees_stage1)
        print(f"Kept {kept_after_corr}/{n_trees} trees after correlation filtering\n")

        if kept_after_corr == 0:
            print("[WARN] [Method C] Stage 1 removed all trees.")
            self.corr_first_pruned_trees = []
            return

        # --- Stage 2: SHAP-based pruning on reduced ensemble ---
        print(f"=== [Method C] Stage 2: SHAP pruning (Remove bottom {shap_cutoff_percentile}%) ===")

        lin_red = LinearRegression().fit(X_train_red, y_train)
        explainer = shap.Explainer(lin_red, X_val_red)
        shap_vals = explainer(X_val_red).values

        if shap_vals.ndim == 3:
            shap_vals_for_importance = np.mean(np.abs(shap_vals), axis=2)
        else:
            shap_vals_for_importance = shap_vals

        mean_shap = np.mean(np.abs(shap_vals_for_importance), axis=0)
        shap_threshold = np.percentile(mean_shap, shap_cutoff_percentile)

        keep_idx_shap_local = [i for i, shap_val in enumerate(mean_shap) if shap_val >= shap_threshold]
        final_keep_indices = [original_indices_stage1[i] for i in keep_idx_shap_local]

        self.corr_first_pruned_trees = [all_trees[i] for i in final_keep_indices]
        print(f"Kept {len(self.corr_first_pruned_trees)}/{kept_after_corr} trees after SHAP pruning.\n")

    def evaluate_corr_first(self):
        """
        (Method C) Evaluates the "Correlation-first" pruned ensemble.
        """
        X_test = self.workflow.X_test
        y_test = self.workflow.y_test

        if not self.corr_first_pruned_trees:
            print("[WARN] [Method C] No pruned trees to evaluate.")
            return None, None

        print(f"[INFO] [Method C]: Re-training final *LinearRegression* on {len(self.corr_first_pruned_trees)} pruned trees...")

        X_train_final = self._get_meta_features(self.workflow.X_train_meta, self.corr_first_pruned_trees)
        y_train_final = self.workflow.y_train_meta

        final_eval_model = LinearRegression().fit(X_train_final, y_train_final)

        if self.data_type == "regression":
            w_abs = np.abs(final_eval_model.coef_)
            normalized_weights = w_abs / (np.sum(w_abs) + 1e-12)

            pruned_preds_matrix = self._get_meta_features(X_test, self.corr_first_pruned_trees)

            if pruned_preds_matrix.ndim == 1:
                final_preds = pruned_preds_matrix.squeeze()
            else:
                final_preds = pruned_preds_matrix @ normalized_weights

            self.corr_first_pruned_mse = mean_squared_error(y_test, final_preds)
            print("[Method C] Final Pruned MSE (Weighted by Retrain):", self.corr_first_pruned_mse)

        else:
            pruned_preds_matrix = self._get_meta_features(X_test, self.corr_first_pruned_trees)
            final_preds = final_eval_model.predict(pruned_preds_matrix)
            self.corr_first_pruned_mse = accuracy_score(y_test, final_preds)
            print ("[Method C] Final Pruned Metric (Weighted by Retrain)", self.corr_first_pruned_mse)

        return self.corr_first_pruned_mse
