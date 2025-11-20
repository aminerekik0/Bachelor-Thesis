import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier ,DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score , accuracy_score ,f1_score
from uci_datasets import Dataset

from src.BasicMetaModel import BasicMetaModel
from src.LinearMetaModel import LinearMetaModel


class ExplainableTreeEnsemble:
    """
    Explainability-Driven Tree Ensemble with SHAP-based pruning.

    Steps:
        1. Build M trees using Boostrap Sampling
        2. Train each tree
        3. get the mean or the mode prediction based on the dataset type
        4. Evaluate this final prediction using multiple metric
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
        self.auc = None

    def _prepare_data(self):

        if self.data_type == "classification" :
            from ucimlrepo import fetch_ucirepo

            from sklearn.datasets import fetch_covtype
            data = fetch_covtype(as_frame=False)
            X = data.data
            y = data.target
            y = (y == 2).astype(int)

            # Install dependencies as needed:
# pip install kagglehub[pandas-datasets]

            import kagglehub
            from kagglehub import KaggleDatasetAdapter
            path = kagglehub.dataset_download("uciml/forest-cover-type-dataset")
            import os
            import pandas as pd
            csv_path = os.path.join(path, "covtype.csv")
            df = pd.read_csv(csv_path)

            print("path" , path)

            print("First 5 records:", df.head())

            y = (df['Cover_Type'] == 2).astype(int).values

# Features
            X = df.drop(columns=['Cover_Type'])

            from pmlb import fetch_data, classification_dataset_names
            import pandas as pd

# Liste aller Klassifikations-Datensätze
            print(classification_dataset_names)

# Einen bestimmten Datensatz laden, z.B. "adult"
            X, y = fetch_data('adult', return_X_y=True, local_cache_dir='./data')

# In DataFrame umwandeln
            df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
            df['target'] = y

            print(df.shape)
            print(df.head())

# Convert categorical features to numeric using one-hot encoding



            print("Feature matrix shape:", X.shape)
            print("Target vector shape:", y.shape)






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
                    max_depth=np.random.choice([10]),
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




    def _evaluate(self):
        """Evaluate full ensemble using stored test set."""


        if self.data_type == "regression" :

            tree_preds = np.vstack([t.predict(self.X_test) for t in self.individual_trees])

            self.full_preds = np.mean(tree_preds, axis=0)

            self.mse = mean_squared_error(self.y_test, self.full_preds)

            self.rmse = np.sqrt(self.mse)

            self.mae = mean_absolute_error(self.y_test, self.full_preds)

            self.r2 = r2_score(self.y_test, self.full_preds)

            print("full ensemble mse" , self.mse)

        else :

            tree_preds = np.vstack([t.predict(self.X_test) for t in self.individual_trees])

            print(tree_preds)

            from scipy.stats import mode

            mode_result = stats.mode(tree_preds, axis=0, keepdims=False)

            majority_vote_preds = mode_result.mode

            self.acc = accuracy_score(self.y_test, majority_vote_preds)

            self.f1 = f1_score(self.y_test, majority_vote_preds, average='weighted')
            from sklearn.metrics import roc_auc_score

            self.auc = roc_auc_score(self.y_test, majority_vote_preds)

            print("full ensemble acc" , self.acc)

            print("full ensemble f1 " , self.f1)

        return self.mse , self.rmse , self.mae , self.r2 , self.acc , self.f1



def run_classification_test():
    """Runs a classification test on the Covertype dataset."""

    workflow = ExplainableTreeEnsemble(
        dataset_name=None,
        data_type="classification",

    )

    print("\n===== Training Base Trees =====")
    workflow.train_base_trees()

    print("\n===== Evaluating Ensemble =====")
    mse, rmse, mae, r2, acc, f1 = workflow._evaluate()

    print("\n===== Classification Results =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Auc Score: {workflow.auc:.4f}")
    print("==================================")

    meta_model = BasicMetaModel(
        data_type="classification"
    )
    meta_model.attach_to(workflow)
    meta_model.train()
    metric, _ = meta_model.evaluate()
    print(f"\npre Pruned Ensemble Metric: {metric:.4f}")
    print(f"F1 Score: {meta_model.f1:.4f}")
    print(f"Auc Score: {meta_model.auc:.4f}")

    model = LinearMetaModel(data_type="classification" )
    model.attach_to(workflow)
    model.train(meta_model.pruned_trees)
    model.prune()
    metric , _ = model.evaluate()
    print(f"Auc Score: {model.auc:.4f}")
    print(f"\nFinal Pruned Ensemble Metric: {metric:.4f}")





if __name__ == "__main__":
    run_classification_test()