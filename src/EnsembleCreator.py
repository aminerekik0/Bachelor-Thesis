import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score



class EnsembleCreator:
    """
    Explainability-Driven Tree Ensemble.
    """
    def __init__(self, X=None, y=None,n_trees=200, max_depth="AUTO" , test_size=0.2, random_state=42, data_type="regression" ,r_bootstrap=0.7):

        self.X = X
        self.y = y

        self.n_trees = n_trees
        self.max_depth = max_depth
        self.test_size = test_size
        self.random_state = random_state
        self.r_bootstrap = r_bootstrap

        self.individual_trees = []
        self.data_type = data_type
        self._prepare_data()

        # Regression metrics
        self.mse = None
        self.rmse = None
        self.mae = None
        self.r2 = None

        # Classification metrics
        self.acc = None
        self.f1 = None
        self.auc = None

    def _prepare_data(self):

        X = self.X
        y = self.y


        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=self.random_state
        )
        X_train_meta, X_eval_meta, y_train_meta, y_eval_meta = train_test_split(
            X_val, y_val, test_size=0.5, random_state=self.random_state
        )

        self.X_train = np.array(X_train)
        self.X_train_meta = np.array(X_train_meta)
        self.X_eval_meta = np.array(X_eval_meta)
        self.X_test = np.array(X_test)

        self.y_train = np.array(y_train)
        self.y_train_meta = np.array(y_train_meta)
        self.y_eval_meta = np.array(y_eval_meta)
        self.y_test = np.array(y_test)

    def train_base_trees(self):
        """Train M base decision trees."""
        print("------------- Creating the Base Trees --------------")

        n_samples = self.X_train.shape[0]
        for i in range(self.n_trees):

            indices = np.random.choice(n_samples, int(self.r_bootstrap * n_samples), replace=True)
            X_sub, y_sub = self.X_train[indices], self.y_train[indices]

            n_features = X_sub.shape[1]
            n_features_subset = int(np.sqrt(n_features))

            if self.data_type == "regression":
                trees = DecisionTreeRegressor(
                    max_depth = self.max_depth if self.max_depth !="AUTO" else np.random.choice([2, 5, 6, 10, 15]),
                    random_state=self.random_state + i,
                    max_features=n_features_subset,
                )
            else:
                trees = DecisionTreeClassifier(
                    max_depth = self.max_depth if self.max_depth != "AUTO" else np.random.choice([2, 5, 6, 10, 15]),
                    random_state=self.random_state + i,
                    max_features=n_features_subset,
                )

            trees.fit(X_sub, y_sub)
            self.individual_trees.append(trees)

    def _evaluate(self):
        """Evaluate FULL ensemble on the test set."""

        if self.data_type == "regression":

            tree_preds = np.vstack([t.predict(self.X_test) for t in self.individual_trees])
            self.full_preds = np.mean(tree_preds, axis=0)

            self.mse = mean_squared_error(self.y_test, self.full_preds)
            self.rmse = np.sqrt(self.mse)
            self.mae = mean_absolute_error(self.y_test, self.full_preds)
            self.r2 = r2_score(self.y_test, self.full_preds)

            print("Full Ensemble RMSE:", self.rmse)
            print("Full Ensemble R2:", self.r2)




        else:
            tree_preds = np.vstack([t.predict(self.X_test) for t in self.individual_trees])

            from scipy.stats import mode
            mode_res = stats.mode(tree_preds, axis=0, keepdims=False)
            majority_vote_preds = mode_res.mode

            self.acc = accuracy_score(self.y_test, majority_vote_preds)
            self.f1 = f1_score(self.y_test, majority_vote_preds, average='weighted')

            from sklearn.metrics import roc_auc_score
            self.auc = roc_auc_score(self.y_test, majority_vote_preds)

            print("Full Ensemble ACC:", self.acc)
            print("Full Ensemble F1:", self.f1)
            print("Full Ensemble AUC:", self.auc)


        return self.mse, self.rmse, self.mae, self.r2 ,self.acc, self.f1

