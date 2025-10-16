import torch
import torch.nn as nn
import torch.optim as optim
import shap

X_train_meta_tensor = torch.tensor(X_train_meta_features, dtype=torch.float32)
y_train_meta_tensor = torch.tensor(y_train_meta, dtype=torch.float32).unsqueeze(1)
X_eval_meta_tensor = torch.tensor(X_eval_meta_features, dtype=torch.float32)
y_eval_meta_tensor = torch.tensor(y_eval_meta, dtype=torch.float32).unsqueeze(1)
X_test_meta_tensor = torch.tensor(X_test_meta_features, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_clf, dtype=torch.float32).unsqueeze(1)


class MetaModel(nn.Module):
    def __init__(self, n_features):
        super(MetaModel, self).__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        return self.linear(x)

# --- 3. Training Loop ---
meta_model = MetaModel(100) # n_trees is 10
main_loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(meta_model.parameters(), lr=0.01)

lambda1 = 0.5
lambda2 = 0.5
epsilon = 1e-6
epochs = 500

for epoch in range(epochs):
    # Forward pass on training data
    logits = meta_model(X_train_meta_tensor)
    loss_main = main_loss_fn(logits, y_train_meta_tensor)

    # Calculate SHAP-based penalty terms
    X_eval_meta_numpy = X_eval_meta_tensor.detach().cpu().numpy()

    explainer = shap.Explainer(meta_model.forward, X_eval_meta_numpy)
    shap_values = explainer(X_eval_meta_numpy).values
    shap_values_tensor = torch.tensor(shap_values, dtype=torch.float32)

    # ---- Pruning loss (average entropy across samples) ----
    # According to the thesis, p_hat is the normalized absolute SHAP value for each sample.
    abs_shap_values = torch.abs(shap_values_tensor)
    sum_abs_shap = torch.sum(abs_shap_values, dim=1, keepdim=True)
    p_hat = abs_shap_values / (sum_abs_shap + epsilon)

    entropy_per_sample = -torch.sum(p_hat * torch.log(p_hat + epsilon), dim=1)
    loss_prune = torch.mean(entropy_per_sample)

    # ---- Diversity loss ----
    # According to the thesis, diversity loss is based on the average attribution per tree, p_tilde.
    avg_abs_shap_per_tree = torch.mean(abs_shap_values, dim=0)
    sum_avg_abs_shap = torch.sum(avg_abs_shap_per_tree)
    p_tilde = avg_abs_shap_per_tree / (sum_avg_abs_shap + epsilon)
    loss_diversity = torch.sum(p_tilde ** 2)

    # ---- Total loss ----
    total_loss = loss_main + lambda1 * loss_prune + lambda2 * loss_diversity

    # Backpropagation
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Total Loss: {total_loss.item():.4f}, Main Loss: {loss_main.item():.4f}, Pruning Loss: {loss_prune.item():.4f}, Diversity Loss: {loss_diversity.item():.4f}')

