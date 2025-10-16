import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor





from uci_datasets import Dataset
data = Dataset("song")

X = data.x
y = data.y

# splits
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

X_train_meta, X_eval_meta, y_train_meta, y_eval_meta = train_test_split(
    X_val, y_val, test_size=0.5, random_state=42
)

X_train = X_train.astype(np.float32)

X_train_meta = X_train_meta.astype(np.float32)

X_eval_meta = X_eval_meta.astype(np.float32)

X_test = X_test.astype(np.float32)



y_train = y_train.ravel()

y_train_meta = y_train_meta.ravel()

y_eval_meta = y_eval_meta.ravel()

y_test = y_test.ravel()

M = 200

n_samples = X_train.shape[0]

individual_trees = []

for i in range(M):

    indices = np.random.choice(n_samples, size=n_samples, replace=True)

    X_bootstrap, y_bootstrap = X_train[indices], y_train[indices]

    tree = DecisionTreeRegressor(max_depth=5, random_state=None)

    tree.fit(X_bootstrap, y_bootstrap)

    individual_trees.append(tree)


print(f"Created {len(individual_trees)} trees.")

from sklearn.metrics import mean_squared_error

# Prepare to store MSEs
tree_predictions = []

tree_mse = []

# Iterate over each individual tree
for i, tree in enumerate(individual_trees):
    y_pred_tree = tree.predict(X_train_meta)
    mse = mean_squared_error(y_train_meta, y_pred_tree)

    tree_predictions.append(y_pred_tree)
    tree_mse.append(mse)
    print(f"True label : {y_train_meta[:5]}")
    print(f"Tree {i+1} predictions: {y_pred_tree[:5]} ...")
    print(f"Tree {i+1} MSE: {mse:.4f}\n")

# Optionally, get average MSE across all trees
avg_mse = np.mean(tree_mse)
print(f"Average MSE of all trees: {avg_mse:.4f}")





def get_meta_features(X, base_trees):
    return np.column_stack([t.predict(X) for t in base_trees]).astype(np.float32)

X_train_meta_features = get_meta_features(X_train_meta, individual_trees)
X_eval_meta_features = get_meta_features(X_eval_meta, individual_trees)
X_test_meta_features = get_meta_features(X_test, individual_trees)

from sklearn.ensemble import RandomForestRegressor
meta_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)


# Train on meta-features
meta_model.fit(X_train_meta_features,  y_train_meta)

y_meta_eval_pred = meta_model.predict(X_eval_meta_features)
mse_eval = mean_squared_error(y_eval_meta, y_meta_eval_pred)
print(f"Meta-tree MSE on evaluation set: {mse_eval:.4f}")

import shap


def prune_loss(shap_values, eps=1e-12):
    abs_shap = np.abs(shap_values)
    denom = abs_shap.sum(axis=1, keepdims=True) + eps
    p_hat = abs_shap / denom
    entropy = -np.sum(p_hat * np.log(p_hat + eps), axis=1)
    return np.mean(entropy)

def diversity_loss(shap_values, eps=1e-8):
    phi_bar = np.mean(np.abs(shap_values), axis=0)
    p_tilde = phi_bar / (np.sum(phi_bar) + eps)
    return float(np.sum(p_tilde ** 2))

# Create TreeExplainer for the meta-tree
explainer = shap.TreeExplainer(meta_model)

# Compute SHAP values on the test set meta-features
shap_values = explainer.shap_values(X_eval_meta_features)

LAMBDA1 = 0.1
LAMBDA2 = 0.02

total_loss = mse_eval + LAMBDA1 * prune_loss(shap_values) + LAMBDA2 * diversity_loss(shap_values)

print(f"Total loss : {total_loss} , prune Loss : {prune_loss(shap_values)} , div loss : {diversity_loss(shap_values)}")


print("shap values mean : " , np.mean(np.abs(shap_values)))
# shap_values shape: (num_samples, num_trees)
tree_importance = np.mean(np.abs(shap_values), axis=0)

# Sort trees by importance
sorted_indices = np.argsort(tree_importance)  # ascending: least important first
print("Tree importance (least to most important):")
for i in sorted_indices:
    print(f"Tree_{i+1}: {tree_importance[i]:.4f}")


# --- Pruning setup ---
total_trees = len(individual_trees)
keep_count = int(0.25 * total_trees)

# Sort by importance (ascending) and select top keep_count
trees_to_keep = sorted_indices[-keep_count:]
trees_to_prune = sorted_indices[:-keep_count]

print(f"\nKeeping {len(trees_to_keep)} trees out of {total_trees}")
print("Trees kept   :", [f"Tree_{i+1}" for i in trees_to_keep])
print("Trees pruned :", [f"Tree_{i+1}" for i in trees_to_prune])

# --- Collect pruned trees ---
pruned_trees = [individual_trees[i] for i in trees_to_keep]

# --- Evaluate individual pruned trees ---
pruned_tree_mse = []
print("\nMSE of kept trees:")
for idx, (i, tree) in enumerate(zip(trees_to_keep, pruned_trees)):
    y_pred = tree.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    pruned_tree_mse.append(mse)
    print(f"Tree_{i+1:3d}) -> MSE: {mse:.4f}")

# --- Ensemble prediction (average of kept trees) ---
pruned_tree_predictions = np.column_stack([t.predict(X_test) for t in pruned_trees])
final_predictions = np.mean(pruned_tree_predictions, axis=1)

ensemble_mse = mean_squared_error(y_test, final_predictions)
print(f"\nFinal ensemble MSE (average of kept trees): {ensemble_mse:.4f}")
print("Final ensemble predictions (first 10):", final_predictions[:10])

full_preds = np.mean([t.predict(X_test) for t in individual_trees], axis=0)
full_mse = mean_squared_error(y_test, full_preds)
print("Full ensemble MSE:", full_mse)