import shap
import lightgbm as lgb
import numpy as np
from .BaseMetaModel import BaseMetaModel
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
class AdvancedMetaModel(BaseMetaModel):
    def __init__(self, num_iter=10, learning_rate=0.05, num_leaves=16,
                 lambda_prune=0.5, lambda_div=0.02, **kwargs):
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
        self.main_loss = None

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


            s = np.mean(np.abs(shap_values), axis=0)
            s_norm = s / (np.sum(s) + 1e-8)

            L_prune = self._prune_loss(shap_values)

            corr = np.corrcoef(shap_values.T)
            corr[np.isnan(corr)] = 0.0
            diversity_penalty = np.mean((corr - np.eye(len(s)))**2)

            L_div = self._diversity_loss(shap_values)

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


            diversity_factor = np.mean(np.abs(corr), axis=1)

            reward_importance = s_norm

            norm_div_factor = diversity_factor / (np.max(diversity_factor) + 1e-8)
            reward_diversity = 1.0 - norm_div_factor
            combined_reward = (
                    lambda_prune * reward_importance +
                    lambda_div * reward_diversity
            )

            combined_reward = np.clip(combined_reward, 0, 2)

            params["feature_contri"] = combined_reward



        print("loss history", self.loss_history)

    def _get_meta_features(self, X):
        return np.column_stack([t.predict(X) for t in self.workflow.individual_trees]).astype(np.float32)

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

    def evaluate(self):

        X_test = self.workflow.X_test
        y_test = self.workflow.y_test

        X_test_meta = self._get_meta_features(self.workflow.X_test)
        y_pred = self.meta_model.predict(X_test_meta)

        if self.data_type == "regression":
           self.main_loss = mean_squared_error(y_test, y_pred)
        else:
           self.main_loss = accuracy_score(y_test, y_pred)
           self.main_loss = accuracy_score(y_test, y_pred)

        print("full ensemble model main loss " , self.workflow.full_metric)
        print("meta model main loss " , self.main_loss)


        print("prune loss " , self.prune_loss_final)
        print("div loss " , self.div_loss_final)
        return self.main_loss
