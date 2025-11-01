import shap
import lightgbm as lgb
import numpy as np
from .BaseMetaModel import BaseMetaModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score
class AdvancedMetaModel(BaseMetaModel):
    def __init__(self, num_iter=10, learning_rate=0.05, num_leaves=16,
                 lambda_prune=0.6, lambda_div=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_iter = num_iter
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.lambda_prune = lambda_prune
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

    def train(self):

        num_iter = self.num_iter
        lambda_prune = self.lambda_prune
        lambda_div = self.lambda_div
        learning_rate = self.learning_rate
        num_leaves = self.num_leaves


        X_meta_train = self._get_meta_features(self.workflow.X_train_meta)
        X_meta_eval  = self._get_meta_features(self.workflow.X_eval_meta)

        lgb_train = lgb.Dataset(X_meta_train, label=self.workflow.y_train_meta)
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
        loss_history = []
        for iter in range(num_iter) :

            print(f"\n ======= {iter+1}======")
            # train the meta-model
            self.meta_model = lgb.train(params , lgb_train)

            # shap-values
            explainer = shap.TreeExplainer(self.meta_model)
            shap_values = np.array(explainer.shap_values(X_meta_eval))


            L_prune, reward_prune = self._prune_loss_entropy_based(shap_values)
            L_div, reward_div = self._diversity_loss_corr_based(shap_values)

            tot_loss = lambda_prune * L_prune + lambda_div * L_div

            self.loss_history.append({
                "iter" : iter+1 ,
                "L_prune" : L_prune ,
                "L_div" : L_div ,
                "total_loss" : tot_loss ,
            })

            print(f"L_prune : {L_prune} ,L_div : {L_div}  , total_loss : {tot_loss} ")

            self.prune_loss_final = L_prune
            self.div_loss_final = L_div


            combined_reward = (
                    lambda_prune * reward_prune
                    + lambda_div * reward_div
            )

            combined_reward = np.clip(combined_reward, 0, 2)

            params["feature_contri"] = combined_reward.tolist()



        print("loss history", self.loss_history)

    def _get_meta_features(self, X):
        return np.column_stack([t.predict(X) for t in self.workflow.individual_trees]).astype(np.float32)
    #basic
    @staticmethod
    def _prune_loss_entropy_based(shap_values):
        abs_shap = np.abs(shap_values)
        p_hat = abs_shap / (abs_shap.sum(axis=1, keepdims=True) + 1e-12)
        reward = np.mean(p_hat, axis=0)
        return -np.mean(np.sum(p_hat * np.log(p_hat + 1e-12), axis=1)) , reward
    @staticmethod
    def _prune_loss_weighted_L1(shap_values):
        s = np.mean(np.abs(shap_values), axis=0)
        s_norm = s / (np.sum(s) + 1e-8)
        reward = s_norm
        return np.mean(1.0 - s_norm) , reward
    @staticmethod
    def _prune_loss_weighted_L2(shap_values):
        s = np.mean(np.abs(shap_values), axis=0)
        s_norm = s / (np.sum(s) + 1e-8)
        reward = s_norm**2
        return np.mean((1.0 - s_norm)**2) , reward

    # basic
    @staticmethod
    def _diversity_loss_global(shap_values):
        s = np.mean(np.abs(shap_values), axis=0)
        s_norm = s / (np.sum(s) + 1e-8)
        global_loss = float(np.sum(s_norm ** 2))
        reward = 1 - (s_norm **2)
        return global_loss , reward

    @staticmethod
    def _diversity_loss_entropy_based(shap_values):
        abs_shap = np.abs(shap_values)
        p = abs_shap / (abs_shap.sum(axis=1, keepdims=True) + 1e-12)
        entropy_loss = -(p * np.log(p +1e-8)).sum(axis=1).mean()
        reward = 1 - np.mean(p * np.log(p + 1e-12), axis=0)
        return entropy_loss , reward
    @staticmethod
    def _diversity_loss_cov_based(shap_values):
        M = shap_values.shape[1]
        cov_matrix = np.cov(shap_values.T)
        cov_matrix[np.isnan(cov_matrix)] = 0.0
        off_diag_sq_sum = np.sum(np.triu(cov_matrix, k=1)**2) * 2
        mean_abs_cov_per_tree = np.mean(np.abs(cov_matrix), axis=1)
        max_mean_cov = np.max(mean_abs_cov_per_tree)
        reward = 1 -mean_abs_cov_per_tree / (max_mean_cov + 1e-12)
        return off_diag_sq_sum / (M**2 - M + 1e-12) , reward
    @staticmethod
    def _diversity_loss_corr_based(shap_values):
        s = np.mean(np.abs(shap_values), axis=0)
        corr = np.corrcoef(shap_values.T)
        corr[np.isnan(corr)] = 0.0
        mean_abs_corr = np.mean(np.abs(corr), axis=1)
        reward = 1.0 - mean_abs_corr / (np.max(mean_abs_corr) + 1e-12 )
        return np.mean((corr - np.eye(len(s)))**2) , reward



    def evaluate(self):

        X_test = self.workflow.X_test
        y_test = self.workflow.y_test

        X_test_meta = self._get_meta_features(self.workflow.X_test)
        y_pred = self.meta_model.predict(X_test_meta)

        if self.data_type == "regression":
            self.mse = mean_squared_error(y_test, y_pred)
            self.rmse = np.sqrt(self.mse)
            self.mae = mean_absolute_error(y_test, y_pred)
            self.r2 = r2_score(y_test, y_pred)
        else:
            self.mse = accuracy_score(y_test, y_pred)


        print("full ensemble model main loss " , self.workflow.full_metric)
        print("meta model main loss " , self.mse)


        print("prune loss " , self.prune_loss_final)
        print("div loss " , self.div_loss_final)
        return self.mse , self.rmse , self.mae , self.r2