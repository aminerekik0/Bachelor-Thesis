import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import ExtraTreeRegressor ,DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import shap
from uci_datasets import Dataset
import pandas as pd
import os

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
    def __init__(self, dataset_name="keggundirected", n_trees=100, max_depth=2,
                 meta_estimators=50, meta_depth=3,
                 lambda_prune=0.1, lambda_div=0.02, random_state=42 , data_type = "regression"):
        self.dataset_name = dataset_name
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.meta_estimators = meta_estimators
        self.meta_depth = meta_depth
        self.LAMBDA1 = lambda_prune
        self.LAMBDA2 = lambda_div
        self.random_state = random_state

        self.individual_trees = []
        self.meta_model = None
        self.shap_values = None
        self.tree_importance = None
        self.data_type = data_type
        self._prepare_data()

        self.main_loss = None
        self.total_loss = None
        self.full_metric = None
        self.pruned_metric = None

    def _prepare_data(self):
        """Load dataset and create train/validation/test splits."""


        #TODO : get the x and y from a classification dataset
        data = Dataset(self.dataset_name)
        X, y = data.x.astype(np.float32), data.y.ravel()

        # splits
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
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
        for i in range(self.n_trees):
            subsample_ratio = 0.7
            indices = np.random.choice(n_samples, int(subsample_ratio * n_samples), replace=True)
            X_sub, y_sub = self.X_train[indices], self.y_train[indices]
            n_features = X_sub.shape[1]
            n_features_subset = int(np.sqrt(n_features))
           #here I'm using random_state + i to get more randomness in choosing the features in each tree
           # this helped me to have more diverse trees as usuall .
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                random_state=self.random_state+i ,
                max_features = n_features_subset ,
                min_samples_split=np.random.randint(2, 10),
                min_samples_leaf=np.random.randint(1, 10)
            )


        #TODO : DecisionTreeClassifier

            tree.fit(X_sub, y_sub)

            self.individual_trees.append(tree)


        custom_preds = np.array([tree.predict(self.X_train) for tree in self.individual_trees])

        y_pred_custom = np.mean(custom_preds, axis=0)


        mse_custom = mean_squared_error(self.y_train, y_pred_custom)

        print(f"\n--- Performance Comparison ---")
        print(f"Custom Ensemble MSE: {mse_custom:.6f}")


        import numpy as np
        import seaborn as sns
        import matplotlib.pyplot as plt
        print ("--------------my ensemble--------------------")
        corr_matrix = np.corrcoef(custom_preds)
        sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
        plt.title("Pairwise Correlation of Custom Trees' Predictions")
        plt.show()
        print ("--------------random-Forest--------------------")



        print("----------base Trees created------------------- \n ")
        print("----------training the base Trees--------------")
        print(f"Created {len(self.individual_trees)} base trees.")

    @staticmethod
    def _get_meta_features(X, trees):
        """Generate meta-features (tree predictions)."""
        return np.column_stack([t.predict(X) for t in trees]).astype(np.float32)

    def train_meta_model(self):
        """Train meta-model using meta-features as input."""
        X_meta_train = self._get_meta_features(self.X_train_meta, self.individual_trees)
        X_meta_eval = self._get_meta_features(self.X_eval_meta, self.individual_trees)

        print("\n---------- Individual Tree MSEs on Meta Train Set -----------")
        for i in range(20) :
          preds_train = X_meta_train[:, i]
          mse_train = mean_squared_error(self.y_train_meta, preds_train)
          print(f"Tree_{i+1}: Train MSE = {mse_train}")

        if self.data_type == "regression" :
            meta_model = RandomForestRegressor(
            n_estimators=self.meta_estimators,
            max_depth=self.meta_depth,
            random_state=self.random_state,
            )


            meta_model.fit(X_meta_train, self.y_train_meta)

            y_meta_pred = meta_model.predict(X_meta_eval)

            mse = mean_squared_error(self.y_eval_meta, y_meta_pred)
            print(f"--------Meta-model MSE on Eval-meta ------- \n: {mse}")

            self.meta_model = meta_model
            self.main_loss = mse

            return mse
        else :

            from sklearn.ensemble import RandomForestClassifier
            meta_model = RandomForestClassifier(
            n_estimators=self.meta_estimators,
            max_depth=self.meta_depth,
            random_state=self.random_state,
            )
            meta_model.fit(X_meta_train, self.y_train_meta)

            y_meta_pred = meta_model.predict(X_meta_eval)
            from sklearn.metrics import accuracy_score
            acc = accuracy_score(self.y_eval_meta, y_meta_pred)
            print(f"Meta-model accuracy on evaluation set: {acc:.4f}")

            self.meta_model = meta_model
            self.main_loss = acc
            return acc


    @staticmethod
    def prune_loss(shap_values, eps=1e-12):
        abs_shap = np.abs(shap_values)
        # keepdims was necessary to-do the division with the same dimensions
        denom = abs_shap.sum(axis=1, keepdims=True) + eps
        p_hat = abs_shap / denom
        # We should add eps because same trees have a 0 shap-values
        entropy = -np.sum(p_hat * np.log(p_hat + eps), axis=1)
        return np.mean(entropy)

    @staticmethod
    def diversity_loss(shap_values, eps=1e-8):
        phi_bar = np.mean(np.abs(shap_values), axis=0)
        p_tilde = phi_bar / (np.sum(phi_bar) + eps)
        return float(np.sum(p_tilde ** 2))

    def explain_and_prune_regression(self, keep_ratio=0.25):
        """Compute SHAP values , losses and prune least important trees."""

        X_eval_meta_features = self._get_meta_features(self.X_eval_meta, self.individual_trees)
        explainer = shap.TreeExplainer(self.meta_model)
        shap_values = explainer.shap_values(X_eval_meta_features)


        self.shap_values = shap_values

        if self.data_type == "regression":
           main_loss = mean_squared_error(
           self.y_eval_meta,
           self.meta_model.predict(X_eval_meta_features)
        )

        else:
           from sklearn.metrics import accuracy_score
           main_loss = accuracy_score(
           self.y_eval_meta,
           self.meta_model.predict(X_eval_meta_features)
         )

        total_loss = (
                main_loss
                + self.LAMBDA1 * self.prune_loss(shap_values)
                + self.LAMBDA2 * self.diversity_loss(shap_values)
        )
        self.total_loss = total_loss

        print(f"Total Loss : {total_loss:.4f}")
        print(f"MSE        : {main_loss:.4f}")
        print(f"Loss       : {self.prune_loss(shap_values):.4f}")
        print(f"Div Loss   : {self.diversity_loss(shap_values):.4f}")

        # Compute importance per tree
        tree_importance = np.mean(np.abs(shap_values), axis=0)
        sorted_indices = np.argsort(tree_importance)
        self.tree_importance = tree_importance

        total_trees = len(self.individual_trees)
        keep_count = int(keep_ratio * total_trees)

        trees_to_keep = sorted_indices[-keep_count:]
        trees_to_prune = sorted_indices[:-keep_count]

        print(f"\n Keeping {keep_count}/{total_trees} trees.")
        print("Kept  :", [f"Tree_{i+1}" for i in trees_to_keep])
        print("Pruned:", [f"Tree_{i+1}" for i in trees_to_prune])

        # Keep selected trees
        self.pruned_trees = [self.individual_trees[i] for i in trees_to_keep]

    # TODO : explain_and_prune_classification

    def evaluate(self):
        """Evaluate full vs pruned ensemble."""

        if self.data_type == "regression":
          # Full ensemble
         tree_preds = np.vstack([t.predict(self.X_test) for t in self.individual_trees])
         full_preds = np.mean(tree_preds, axis=0)
         full_metric = mean_squared_error(self.y_test, full_preds)
         self.full_metric = full_metric
         # Pruned ensemble
         pruned_tree_preds = np.vstack([t.predict(self.X_test) for t in self.pruned_trees])
         pruned_preds = np.mean(pruned_tree_preds, axis=0)
         pruned_metric = mean_squared_error(self.y_test, pruned_preds)

         self.pruned_metric = pruned_metric

         print(f"\n Full Ensemble MSE  : {full_metric:.4f}")
         print(f" Pruned Ensemble MSE: {pruned_metric:.4f}")

        # TODO : else: for classification

        return full_metric, pruned_metric


    def save_results(self, filename="results_summary.csv"):
        """
        Save all the results , and compare between the full ensemble and the pruned one .
        """

        from datetime import datetime

        current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


        df = pd.DataFrame({
            "dataset": [self.dataset_name],
            "data_type": [self.data_type],
            "meta_main_loss": [self.main_loss],
            "meta_total_loss": [self.total_loss],
            "full_ensemble_loss": [self.full_metric],
            "pruned_ensemble_loss": [self.pruned_metric]
        })


        float_format = '%.4f'

        if os.path.exists(filename):
            df.to_csv(
                filename,
                mode='a',
                header=False,
                index=False,
                float_format=float_format
            )
        else:
            df.to_csv(
                filename,
                index=False,
                float_format=float_format
            )



if __name__ == "__main__":
    model = ExplainableTreeEnsemble( data_type = "regression")
    model.train_base_trees()
    model.train_meta_model()
    model.explain_and_prune_regression(keep_ratio=0.25)
    model.evaluate()
    model.save_results()