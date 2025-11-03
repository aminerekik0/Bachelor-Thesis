import shap
import lightgbm as lgb
import numpy as np
from .BaseMetaModel import BaseMetaModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score

class AdvancedMetaModel(BaseMetaModel):


    def __init__(self, num_iter=5, learning_rate=0.1, num_leaves=20,
                 lambda_prune_start=2.0, lambda_prune_decay=0.6,
                 lambda_div=0.15, **kwargs):

        super().__init__(**kwargs)
        self.num_iter = num_iter
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves

        # MODIFIED: Set new dynamic lambda parameters
        self.lambda_prune_start = lambda_prune_start
        self.lambda_prune_decay = lambda_prune_decay
        self.lambda_div = lambda_div

        self.loss_history = []
        self.meta_model = None
        self.prune_loss_final = None
        self.div_loss_final = None
        self.mse = None
        self.rmse = None
        self.mae = None
        self.r2 =None
        self.version = ""
        self.model = None
        self.normal_model_mse = None

    def train(self):
        num_iter = self.num_iter
        # MODIFIED: Get base lambda
        lambda_div = self.lambda_div
        learning_rate = self.learning_rate
        num_leaves = self.num_leaves

        n_trees = len(self.workflow.individual_trees)
        X_meta_train = self._get_meta_features(self.workflow.X_train_meta)
        X_meta_eval  = self._get_meta_features(self.workflow.X_eval_meta)
        lgb_train = lgb.Dataset(X_meta_train, label=self.workflow.y_train_meta )
        lgb_eval = lgb.Dataset(X_meta_eval, label=self.workflow.y_eval_meta, reference=lgb_train)

        if self.data_type == "regression":
            objective = "regression"
            metric = "rmse"
        else:
            objective = "binary"
            metric = "binary_error"

        params = {
            "objective" : objective ,
            "metric" : metric ,
            "learning_rate" : learning_rate ,
            "num_leaves" : num_leaves ,
            "verbose" : -1 ,
        }
        params_optimized = params.copy()

        self.model = lgb.train(params, lgb_train , num_boost_round=200)
        self.meta_model = self.model
        current_rewards_for_training = np.ones(n_trees)

        for iter in range(num_iter) :
            print(f"\n ======= Iteration {iter+1} / {num_iter} ======")


            current_lambda_prune = 1.0

            params_optimized["feature_contri"] = current_rewards_for_training.tolist()
            self.meta_model = lgb.train(params_optimized, lgb_train,
                                        num_boost_round=200)

            gains = self.meta_model.feature_importance(importance_type='gain')
            explainer = shap.TreeExplainer(self.meta_model)
            shap_values = np.array(explainer.shap_values(X_meta_eval))

            L_prune, reward_prune = self._prune_loss_entropy_based(shap_values)

            L_div, reward_div = self._diversity_loss_global(shap_values)

            tot_loss = current_lambda_prune * L_prune + lambda_div * L_div
            self.loss_history.append({
                "iter" : iter+1 , "L_prune" : L_prune ,
                "L_div" : L_div , "total_loss" : tot_loss ,
            })

            print(f"L_prune: {L_prune:.4f}, L_div: {L_div:.4f}, (lambda_prune: {current_lambda_prune:.3f}), Total Loss: {tot_loss:.4f}")

            self.prune_loss_final = L_prune
            self.div_loss_final = L_div

            # Calculate the new reward vector
            new_rewards_for_next_iter = (
                    current_lambda_prune * reward_prune
                    + lambda_div * reward_div
            )


            #new_rewards_for_next_iter = new_rewards_for_next_iter / (new_rewards_for_next_iter.mean() + 1e-12)


            new_rewards_for_next_iter = np.clip(new_rewards_for_next_iter, 0, 5)

            print(f"\n--- Analysis for Iteration {iter+1} (Sorted by Gain) ---")
            print(f"{'Tree':<6} | {'Reward (Input)':<15} | {'Gain (Result)':<15} | {'Reward (New Output)':<17}")
            print("-" * 59)

            sorted_indices = np.argsort(gains)[::-1]
            for k in sorted_indices[:200]:
                reward_in = current_rewards_for_training[k]
                gain_out = gains[k]
                reward_out = new_rewards_for_next_iter[k]
                print(f"T_{k:<4} | {reward_in:<15.4f} | {gain_out:<15.1f} | {reward_out:<17.4f}")

                # 5. --- Update rewards for the next loop ---
            current_rewards_for_training = new_rewards_for_next_iter

        print("loss history", self.loss_history)

    def _get_meta_features(self, X):
        return np.column_stack([t.predict(X) for t in self.workflow.individual_trees]).astype(np.float32)

    @staticmethod
    def _prune_loss_entropy_based(shap_values):
        s = np.mean(np.abs(shap_values), axis=0)

        s_norm = s / (np.sum(s) + 1e-8)
        abs_shap = np.abs(shap_values)
        p_hat = abs_shap / (abs_shap.sum(axis=1, keepdims=True) + 1e-12)
        reward = np.mean(p_hat, axis=0)
        entropy_per_sample = -np.sum(p_hat * np.log(p_hat + 1e-12), axis=1)
        loss = np.mean(entropy_per_sample)
        return loss, reward

    # MODIFIED: Corrected thresholding logic
    @staticmethod
    def _diversity_loss_global(shap_values):

        s = np.mean(np.abs(shap_values), axis=0)

        s_norm = s / (np.sum(s) + 1e-8)

        global_loss = float(np.sum(s_norm ** 2))

        low_thresh = np.percentile(s_norm, 20)

        reward = np.zeros_like(s_norm)

        useful_trees_mask = (s_norm > low_thresh)

        reward[useful_trees_mask] = 1.0 - s_norm[useful_trees_mask]**2

        return global_loss, reward

    # MODIFIED: Added thresholding logic as requested
    @staticmethod
    def _diversity_loss_corr_based(shap_values):
        # 1. Get average absolute SHAP (for importance)
        s = np.mean(np.abs(shap_values), axis=0)
        # 2. Normalize SHAP to get an importance score (sums to 1)
        s_norm = s / (np.sum(s) + 1e-8)

        # 3. Calculate the correlation matrix (for diversity)
        M = shap_values.shape[1]
        corr = np.corrcoef(shap_values.T)
        corr[np.isnan(corr)] = 0.0

        # 4. Calculate the base diversity reward (for *all* trees)
        diversity_factor = np.mean(np.abs(corr), axis=1)
        reward_base = 1.0 - (diversity_factor / (np.max(diversity_factor) + 1e-12))

        # --- 5. Apply the 30th Percentile Threshold ---
        low_thresh = np.percentile(s_norm, 30)
        reward = np.zeros_like(s_norm)
        useful_trees_mask = (s_norm > low_thresh)
        reward[useful_trees_mask] = reward_base[useful_trees_mask]

        # --- 6. Calculate the Loss Function ---
        loss = np.mean((corr - np.eye(M))**2)

        return loss, reward

    def evaluate(self):
        X_test = self.workflow.X_test
        y_test = self.workflow.y_test
        X_test_meta = self._get_meta_features(self.workflow.X_test)
        y_pred = self.meta_model.predict(X_test_meta)
        y_pred_normal_model = self.model.predict(X_test_meta)

        if self.data_type == "regression":
            self.normal_model_mse = mean_squared_error(y_test, y_pred_normal_model)
            self.mse = mean_squared_error(y_test, y_pred)
            self.rmse = np.sqrt(self.mse)
            self.mae = mean_absolute_error(y_test, y_pred)
            self.r2 = r2_score(y_test, y_pred)
        else:
            self.mse = accuracy_score(y_test, y_pred)

        print("full ensemble model main loss " , self.workflow.full_metric)
        print("meta model main loss " , self.mse)
        print("normal model" , self.normal_model_mse)
        print("prune loss " , self.prune_loss_final)
        print("div loss " , self.div_loss_final)

        return self.mse , self.rmse , self.mae , self.r2