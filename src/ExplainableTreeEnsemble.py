import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import ExtraTreeRegressor ,DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score , accuracy_score
import shap
from uci_datasets import Dataset
import pandas as pd
import os
from src.AdvancedMetaModel import AdvancedMetaModel
from src.BasicMetaModel import BasicMetaModel



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
    def __init__(self, dataset_name, n_trees=50, max_depth=5,
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

    def _prepare_data(self):
        """Load dataset and create train/validation/test splits."""

        #TODO : get the x and y from a classification dataset
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

        self.X_train, self.X_train_meta, self.X_eval_meta, self.X_test = X_train, X_train_meta, X_eval_meta, X_test

        self.y_train, self.y_train_meta, self.y_eval_meta, self.y_test =  y_train, y_train_meta, y_eval_meta, y_test


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
            #here I'm using random_state + i to get more randomness in choosing the features in each tree
            # this helped me to have more diverse trees as usuall .
            trees = DecisionTreeRegressor(
                max_depth=np.random.choice([2, 5 , 6, 9 , 10]),
                random_state=self.random_state+i ,
                max_features = n_features_subset ,
            )

            trees.fit(X_sub, y_sub)
            self.individual_trees.append(trees)
        self._evaluate()


        #TODO : DecisionTreeClassifier
    def _evaluate(self):
        """Evaluate full ensemble using stored test set."""
        tree_preds = np.vstack([t.predict(self.X_test) for t in self.individual_trees])
        self.full_preds = np.mean(tree_preds, axis=0)
        if self.data_type == "regression" :
           self.mse = mean_squared_error(self.y_test, self.full_preds)
           self.rmse = np.sqrt(self.mse)
           self.mae = mean_absolute_error(self.y_test, self.full_preds)
           self.r2 = r2_score(self.y_test, self.full_preds)
        else :
            self.mse = accuracy_score(self.y_test, self.full_preds)

        return self.mse



if __name__ == "__main__":
  dataset_names=["3droad" , "bike" , "slice" ,  "houseelectric" , "song"]
  for dataset in dataset_names :
     workflow = ExplainableTreeEnsemble( data_type = "regression" , dataset_name=dataset)
     workflow.train_base_trees()
     meta_model = BasicMetaModel()
     meta_model.attach_to(workflow)
     meta_model.train()
     meta_model.evaluate()
     meta_model.save_results()

     #workflow.train_meta_model_basic()
     #workflow.explain_and_prune_regression(keep_ratio=0.25)
     #workflow.evaluate()
     #workflow.save_results()
