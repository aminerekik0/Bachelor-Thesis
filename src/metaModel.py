from sklearn.datasets import fetch_california_housing
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


from ucimlrepo import fetch_ucirepo

from sklearn.datasets import fetch_covtype
import numpy as np

data = fetch_covtype()
X = data.data
y = data.target

# fetch dataset
predict_students_dropout_and_academic_success = fetch_ucirepo(id=697)

# data (as pandas dataframes)
X_clf = predict_students_dropout_and_academic_success.data.features
y_clf = predict_students_dropout_and_academic_success.data.targets

# metadata
print(predict_students_dropout_and_academic_success.metadata)

# variable information
print(predict_students_dropout_and_academic_success.variables)



print("before:", y_clf.shape)
print("before:", y_clf)


y_clf_encoded = LabelEncoder().fit_transform(y_clf.values.ravel())


print("after:", y_clf_encoded.shape)
print("after:", y_clf_encoded)




# Classification splits
X_train_clf, X_temp_clf, y_train_clf, y_temp_clf = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_val_clf, X_test_clf, y_val_clf, y_test_clf = train_test_split(
    X_temp_clf, y_temp_clf, test_size=0.5, random_state=42
)

X_train_meta, X_eval_meta, y_train_meta, y_eval_meta = train_test_split(
    X_val_clf, y_val_clf, test_size=0.5, random_state=42
)


print(f"Training: {X_train_clf.shape}, Validation: {X_val_clf.shape}, Test: {X_test_clf.shape}")




# Ensure X is numpy array
X_train_clf = np.array(X_train_clf)
X_train_meta = np.array(X_train_meta)
X_eval_meta = np.array(X_eval_meta)
X_test_clf = np.array(X_test_clf)

# ----- Build the RandomForest model -----
trees = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    bootstrap=True,
    random_state=42
)
# ----- train the model -----
trees.fit(X_train_clf, y_train_clf)

# get the prediction of each tree
for i, tree in enumerate(trees.estimators_):
    y_pred = tree.predict(X_train_meta)
    accuracy = accuracy_score(y_train_meta, y_pred)
    print(f"Tree {i+1} accuracy on train_meta: {accuracy:.4f}")



# --- Parameters ---
lambda1 = 0.5
lambda2 = 0.2
prune_every = 5
prune_percentile = 30
min_trees_to_keep = 5
epochs = 20
epsilon = 1e-8

# --- Helper: compute meta-features from base trees ---
def get_meta_features(X, base_trees):
    return np.column_stack([t.predict(X) for t in base_trees])

base_trees = trees.estimators_
X_train_meta_features = get_meta_features(X_train_meta, base_trees)
X_eval_meta_features = get_meta_features(X_eval_meta, base_trees)
X_test_meta_features = get_meta_features(X_test_clf, base_trees)

print(X_train_meta_features)


from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

meta_model_standard = RandomForestRegressor(
    n_estimators=100, max_depth=3, random_state=42
)

# 2. Train the Meta Model
print("Starting standard meta-model training...")
meta_model_standard.fit(X_train_meta_features, y_train_meta)
print("Standard meta-model training complete.")

# 3. Evaluate the Standard Model (Optional, but good practice)
y_train_meta_pred = meta_model_standard.predict(X_train_meta_features)
mse_train = mean_squared_error(y_train_meta, y_train_meta_pred)
print(f"Training MSE of standard meta-model: {mse_train:.4f}")

# --- Prepare Evaluation Data ---
X_eval_meta_features = get_meta_features(X_eval_meta, base_trees)

import shap
import numpy as np
import shap

# Use your trained ensemble meta-model
explainer = shap.TreeExplainer(meta_model_standard)   # e.g., RandomForestRegressor

# Compute SHAP values
shap_values = explainer.shap_values(X_eval_meta_features)

print(f"SHAP values shape: {shap_values}")


shap.summary_plot(shap_values, X_eval_meta_features)

import numpy as np

def prune_loss(shap_values, eps=1e-12):
    """
    Compute the pruning loss based on SHAP values.

    Parameters:
        shap_values: np.ndarray of shape (N, M)
            SHAP values for N samples and M trees/features.
        eps: float
            Small constant to avoid log(0).

    Returns:
        float: prune loss
    """
    # Step 1: absolute SHAP values
    abs_shap = np.abs(shap_values)  # (N, M)

    # Step 2: normalize across features/trees for each sample
    denom = abs_shap.sum(axis=1, keepdims=True) + eps
    p_hat = abs_shap / denom  # (N, M)
    # Step 3: compute entropy term per sample
    entropy = -np.sum(p_hat * np.log(p_hat + eps), axis=1)  # (N,)

    # Step 4: average across samples
    L_prune = np.mean(entropy)
    return L_prune

prune_loss(shap_values)

import numpy as np

def diversity_loss(shap_values, eps=1e-8):
    """
    Computes SHAP-based diversity loss.

    Parameters:
        shap_values: np.ndarray of shape (N, M)
            SHAP values where N = samples, M = trees.
        eps: small constant for numerical stability.

    Returns:
        float: diversity loss value
    """
    # Step 1: average attribution per tree (absolute SHAP values)
    phi_bar = np.mean(np.abs(shap_values), axis=0)  # shape (M,)

    # Step 2: normalize across trees
    p_tilde = phi_bar / (np.sum(phi_bar) + eps)

    # Step 3: compute diversity loss
    loss_div = np.sum(p_tilde ** 2)
    return float(loss_div)

# Example usage with your shap_values
loss_div = diversity_loss(shap_values)
print("Diversity Loss:", loss_div)

import numpy as np
from sklearn.metrics import mean_squared_error
import shap

# Hyperparameters
LAMBDA1 = 0.03
LAMBDA2 = 0.01
prune_threshold = 0.008
max_iter = 4


num_trees = X_train_meta_features.shape[1]
remaining_features = list(range(num_trees))

meta_model_standard.fit(X_train_meta_features, y_train_meta)


y_train_pred_full = meta_model_standard.predict(X_train_meta_features)
y_test_pred_full  = meta_model_standard.predict(X_test_meta_features)

mse_train_full = mean_squared_error(y_train_meta, y_train_pred_full)
mse_test_full  = mean_squared_error(y_test_clf, y_test_pred_full)

print("=== Baseline (no pruning) ===")
print("Train MSE:", mse_train_full)
print("Test MSE :", mse_test_full)


for iteration in range(max_iter):
    print(f"\nIteration {iteration+1}")

    # keep only current remaining trees
    X_meta_pruned = X_train_meta_features[:, remaining_features]
    X_eval_meta_pruned = X_eval_meta_features[:, remaining_features]

    # Step 2: Train meta-model
    meta_model_standard.fit(X_meta_pruned, y_train_meta)

    # Step 3: Compute training MSE
    y_train_meta_pred = meta_model_standard.predict(X_meta_pruned)
    mse_train = mean_squared_error(y_train_meta, y_train_meta_pred)

    # Step 4: Compute SHAP values on training features
    explainer = shap.TreeExplainer(meta_model_standard)
    shap_values = explainer.shap_values(X_meta_pruned ,  check_additivity=False)


    # Step 5: Compute total loss
    total_loss = mse_train + LAMBDA1 * prune_loss(shap_values) + LAMBDA2 * diversity_loss(shap_values)
    print(f"main: {mse_train:.4f}")
    print(f"prune loss: {prune_loss(shap_values):.4f}")
    print(f"div loss: {diversity_loss(shap_values):.4f}")
    print(f"Total loss: {total_loss:.4f}")

    # Step 6: Prune more useless trees
    abs_shap = np.abs(shap_values)
    mean_shap_per_tree = abs_shap.mean(axis=0)

    low_utility_trees = [i for i, shap_val in enumerate(mean_shap_per_tree) if shap_val < prune_threshold]
    if low_utility_trees:
        print("Low-utility trees pruned:", low_utility_trees)
        remaining_features = [f for j, f in enumerate(remaining_features) if j not in low_utility_trees]

    print("here")
    #  Promote diversity
   # if remaining_features:
    #    mean_shap_per_tree_pruned = np.abs(shap_values).mean(axis=0)
     #   threshold_90 = np.percentile(mean_shap_per_tree_pruned, 90)
      #  diverse_mask = mean_shap_per_tree_pruned < threshold_90
       # remaining_features = [f for f, keep in zip(remaining_features, diverse_mask) if keep]
        #print("Trees kept for diversity:", remaining_features)

    # Step 8: Stop early if no trees removed
    if not low_utility_trees :
        # and all(diverse_mask):
        print("here")
        print("No more trees removed. Stopping.")
        break




# Step 9: Final SHAP values and visualization
X_eval_meta_pruned = X_eval_meta_features[:, remaining_features]
explainer = shap.TreeExplainer(meta_model_standard)
shap_values = explainer.shap_values(X_eval_meta_pruned , check_additivity=False)

shap.summary_plot(shap_values, X_eval_meta_pruned, feature_names=[f"Tree {i}" for i in remaining_features])

X_train_pruned = X_train_meta_features[:, remaining_features]
X_test_pruned  = X_test_meta_features[:, remaining_features]

meta_model_standard.fit(X_train_pruned, y_train_meta)

y_train_pred_pruned = meta_model_standard.predict(X_train_pruned)
y_test_pred_pruned  = meta_model_standard.predict(X_test_pruned)

mse_train_pruned = mean_squared_error(y_train_meta, y_train_pred_pruned)
mse_test_pruned  = mean_squared_error(y_test_clf, y_test_pred_pruned)

print("\n=== After Pruning ===")
print("Train MSE:", mse_train_pruned)
print("Test MSE :", mse_test_pruned)

