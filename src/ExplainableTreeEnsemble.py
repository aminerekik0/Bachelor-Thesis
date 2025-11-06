import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier ,DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score , accuracy_score ,f1_score
import shap
from uci_datasets import Dataset
import pandas as pd
import os
from src.AdvancedMetaModel import AdvancedMetaModel
from src.BasicMetaModel import BasicMetaModel
from src.AdvancedMetaModelSecond import AdvancedMetaModelSecond



class ExplainableTreeEnsemble:
    """
    Explainability-Driven Tree Ensemble with SHAP-based pruning.

    Steps:
        1. Train M trees.
        2. Build meta-model using trees outputs as features.
        3. Compute SHAP values for explainability.
        4. Prune unimportant trees based on SHAP importance.
        5. test the remaining trees on the test data
    """
    def __init__(self, dataset_name, n_trees=200, max_depth=5,
                 meta_estimators=50, meta_depth=5,learning_rate = 0.05 ,
                 lambda_prune=0.5, lambda_div=0.02, random_state=42 , data_type = "regression"):
        self.dataset_name = dataset_name
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.meta_estimators = meta_estimators
        self.meta_depth = meta_depth
        self.LAMBDA1 = lambda_prune
        self.LAMBDA2 = lambda_div
        self.random_state = random_state
        self.learning_rate = learning_rate

        self.individual_trees = []
        self.individual_trees1 = []
        self.individual_trees2 = []
        self.meta_model = None
        self.shap_values = None
        self.tree_importance = None
        self.data_type = data_type
        self._prepare_data()
        self.n_features = None
        self.n_samples = None

        self.main_loss = None
        self.total_loss = None
        self.full_metric = None
        self.pruned_metric = None

        self.mse = None
        self.rmse = None
        self.mae = None
        self.r2 =None

        self.f1 = None
        self.acc = None

    def _prepare_data(self):

        if self.data_type == "classification" :
           from ucimlrepo import fetch_ucirepo

           from sklearn.datasets import fetch_covtype
           data = fetch_covtype(as_frame=False)
           X = data.data
           y = data.target


        else :
           data = Dataset(self.dataset_name)
           X, y = data.x.astype(np.float32), data.y.ravel()

        # splits
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=self.random_state
        )
        X_train_meta, X_eval_meta, y_train_meta, y_eval_meta = train_test_split(
            X_val, y_val, test_size=0.5, random_state=self.random_state
        )

        self.X_train = X_train.to_numpy() if hasattr(X_train, "to_numpy") else np.array(X_train)
        self.X_train_meta = X_train_meta.to_numpy() if hasattr(X_train_meta, "to_numpy") else np.array(X_train_meta)
        self.X_eval_meta = X_eval_meta.to_numpy() if hasattr(X_eval_meta, "to_numpy") else np.array(X_eval_meta)
        self.X_test = X_test.to_numpy() if hasattr(X_test, "to_numpy") else np.array(X_test)

        self.y_train = y_train.to_numpy() if hasattr(y_train, "to_numpy") else np.array(y_train)
        self.y_train_meta = y_train_meta.to_numpy() if hasattr(y_train_meta, "to_numpy") else np.array(y_train_meta)
        self.y_eval_meta = y_eval_meta.to_numpy() if hasattr(y_eval_meta, "to_numpy") else np.array(y_eval_meta)
        self.y_test = y_test.to_numpy() if hasattr(y_test, "to_numpy") else np.array(y_test)


    def train_base_trees(self):
        """Train M base trees using bootstrap sampling."""
        import numpy as np
        print("-------------creating the base Trees-------------- ")
        n_samples = self.X_train.shape[0]
        self.n_samples=n_samples
        self.n_features= self.X_train.shape[1]
        for i in range(self.n_trees):
            indices = np.random.choice(n_samples, int(0.7*n_samples), replace=True)
            X_sub, y_sub = self.X_train[indices], self.y_train[indices]
            n_features = X_sub.shape[1]
            n_features_subset = int(np.sqrt(n_features))
            if self.data_type == "regression":
                trees = DecisionTreeRegressor(
                max_depth=np.random.choice([2, 5 , 6, 9 , 10]),
                random_state=self.random_state+i ,
                max_features = n_features_subset ,
            )
            else :

                trees = DecisionTreeClassifier(
                    max_depth=np.random.choice([2, 5 , 6, 9 , 10]),
                    random_state=self.random_state+i ,
                    max_features = n_features_subset ,
                )

            trees.fit(X_sub, y_sub)
            self.individual_trees.append(trees)
        self._evaluate()


    def _evaluate(self):
        """Evaluate full ensemble using stored test set."""


        if self.data_type == "regression" :
           tree_preds = np.vstack([t.predict(self.X_test) for t in self.individual_trees])
           self.full_preds = np.mean(tree_preds, axis=0)
           self.mse = mean_squared_error(self.y_test, self.full_preds)
           self.rmse = np.sqrt(self.mse)
           self.mae = mean_absolute_error(self.y_test, self.full_preds)
           self.r2 = r2_score(self.y_test, self.full_preds)
        else :
           tree_preds = np.vstack([t.predict(self.X_test) for t in self.individual_trees])
           from scipy.stats import mode
           mode_result = stats.mode(tree_preds, axis=0, keepdims=False)
           majority_vote_preds = mode_result.mode
           print("preds" ,majority_vote_preds )
           print("true " ,self.y_test )
           self.acc = accuracy_score(self.y_test, majority_vote_preds)
           self.f1 = f1_score(self.y_test, majority_vote_preds, average='weighted')
        print("full ensemble acc" , self.acc)
        print("full ensemble f1 " , self.f1)
        return self.mse , self.acc



if __name__ == "__main__":
    import itertools
    import pandas as pd
    num_iter_values = [15]
    learning_rate_values = [0.05]
    lambda_prune_values = [0.7 , 0.1]
    lambda_div_values = [0.4]
    keep_ratio_values = [0.1]

    dataset_names = ["slice" ,"3droad" ,"bike"]


    results = []
    for dataset in dataset_names:
            workflow = ExplainableTreeEnsemble(data_type="regression", dataset_name=dataset)
            workflow.train_base_trees()
            workflow._evaluate()

            for (num_iter, lr, lambda_prune, lambda_div, keep_ratio) in itertools.product(
                    num_iter_values, learning_rate_values, lambda_prune_values, lambda_div_values,
                    keep_ratio_values
            ):
                print(f"\n=== Grid Search: num_iter={num_iter}, lr={lr}, λ_prune={lambda_prune}, "
                      f"λ_div={lambda_div}, keep_pct={keep_ratio}")

                meta_model = AdvancedMetaModel(
                num_iter=num_iter,
                learning_rate=lr,
                lambda_prune=lambda_prune,
                lambda_div=lambda_div,
                    keep_ratio=keep_ratio,

            )


                meta_model.attach_to(workflow)
                meta_model.train()
                mse_advanced, meta_model_mse, normal_model_mse, _, _ = meta_model.evaluate()
                meta_model.save_results()


                basic_meta_model = BasicMetaModel(keep_ratio = keep_ratio)
                basic_meta_model.attach_to(workflow)
                basic_meta_model.train()
                mse_basic, _ = basic_meta_model.evaluate()
                basic_meta_model.save_results()

                results.append({
                "dataset": dataset,
                "num_iter": num_iter,
                "learning_rate": lr,
                "lambda_prune": lambda_prune,
                "lambda_div": lambda_div,
                "keep_percentile": keep_ratio,
                "mse_advanced": mse_advanced,
                "mse_basic": mse_basic ,
                "meta_model_mse" : meta_model_mse ,
                "normal_model_mse" : normal_model_mse
            })

                df_results = pd.DataFrame(results)
                file_path ="grid_search_results2.csv"
                df_results.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)






    print("Grid search finished. Results saved to grid_search_results.csv")






  #workflow.train_meta_model_basic()
     #workflow.explain_and_prune_regression(keep_ratio=0.25)
     #workflow.evaluate()
     #workflow.save_results()
