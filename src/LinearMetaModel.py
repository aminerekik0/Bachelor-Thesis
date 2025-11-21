import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from src.BaseMetaModel import BaseMetaModel

class _TorchModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_features, 1) * 0.1)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        return X @ self.w + self.b

class LinearMetaModel(BaseMetaModel):
    """
    Linear Meta-Model to combine pre-pruned trees.
    Supports regression and classification.
    """

    def __init__(self, λ_prune=0.5, λ_div=0.3, epochs=200, lr=1e-2, epsilon=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.λ_prune = λ_prune
        self.λ_div = λ_div
        self.epochs = epochs
        self.lr = lr
        self.epsilon = epsilon
        self.initial_pruned_trees = None

        self.model = None
        self.w_final = None
        self.pruned_trees = None
        self.pruned_ensemble_metric = None
        self.total_loss = None
        self.pruned_exp = False

        self.final_corr_matrix = None
        self.final_pruned_indices = None

        self.prune_loss = None
        self.div_loss = None
        self.initial_prune_loss = None
        self.initial_div_loss = None
        self.initial_total_loss = None

        self.auc = None

    def _get_meta_features(self, X, trees_list):
        if not trees_list:
            raise ValueError("trees_list cannot be empty.")
        return np.column_stack([t.predict(X) for t in trees_list]).astype(np.float32)

    @staticmethod
    def _loss_accuracy(shap_vals, y_true, y_pred):
        errors_t = (y_true - y_pred.reshape(y_pred.shape)).reshape(-1, 1)
        sign_of_neg_errors_t = torch.sign(-errors_t)
        return -torch.mean(torch.abs(shap_vals) * sign_of_neg_errors_t)

    @staticmethod
    def _loss_prune(shap_vals_t, epsilon=1e-8):
        abs_shap = torch.abs(shap_vals_t)
        sum_abs_shap_per_sample = torch.sum(abs_shap, dim=1, keepdim=True)
        p_hat = abs_shap / (sum_abs_shap_per_sample + epsilon)
        entropy_per_sample = -torch.sum(p_hat * torch.log(p_hat + epsilon), dim=1)
        return torch.mean(entropy_per_sample)

    @staticmethod
    def _loss_diversity(shap_vals_t, epsilon=1e-8):
        if shap_vals_t.shape[1] <= 1:
            return torch.tensor(0.0, device=shap_vals_t.device)
        phi_bar = torch.mean(torch.abs(shap_vals_t), dim=0)
        p_tilde = phi_bar / (torch.sum(phi_bar) + epsilon)
        return torch.sum(p_tilde ** 2)

    def train(self, pruned_trees_list):
        print(f"[INFO] Training LinearMetaModel on {len(pruned_trees_list)} pruned trees...")

        if not pruned_trees_list:
            print("[ERROR] Received empty pruned tree list. Aborting train.")
            return

        self.initial_pruned_trees = pruned_trees_list
        self.pruned_trees = pruned_trees_list

        X_train = self._get_meta_features(self.workflow.X_train_meta, self.pruned_trees)
        y_train = self.workflow.y_train_meta
        X_eval = self._get_meta_features(self.workflow.X_eval_meta, self.pruned_trees)
        y_eval = self.workflow.y_eval_meta

        n_features = X_train.shape[1]
        if n_features == 0:
            print("[ERROR] Meta-features have 0 features. Aborting.")
            return

        self.model = _TorchModel(n_features)
        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_eval_t = torch.tensor(X_eval, dtype=torch.float32)
        y_eval_t = torch.tensor(y_eval, dtype=torch.float32).view(-1, 1)
        X_baseline_t = torch.mean(X_eval_t, dim=0, keepdim=True)

        for epoch in range(self.epochs):
            self.model.train()
            opt.zero_grad()
            y_pred = self.model(X_t)
            loss_mse = nn.functional.mse_loss(y_pred, y_t)
            y_eval_pred = self.model(X_eval_t)
            X_baselined_t = X_eval_t - X_baseline_t
            shap_vals_t = X_baselined_t * self.model.w.T
            loss_acc = self._loss_accuracy(shap_vals_t, y_eval_t, y_eval_pred)
            loss_prune = self._loss_prune(shap_vals_t)
            loss_div = self._loss_diversity(shap_vals_t)

            loss_mse_norm = loss_mse / (torch.mean(torch.abs(y_t)) + self.epsilon)
            num_trees = shap_vals_t.shape[1]
            max_entropy = np.log(num_trees + self.epsilon)
            loss_prune_norm =  loss_prune / max_entropy
            loss_div_norm = loss_div

            if epoch == 0:


                self.lambda_prune = float((loss_prune / (loss_mse + self.epsilon)).item())
                self.lambda_div = self.λ_div * self.lambda_prune
                self.initial_prune_loss = loss_prune.item()
                self.initial_div_loss = loss_div.item()
                self.initial_total_loss = (loss_mse + self.lambda_prune * loss_prune + self.lambda_div * loss_div).item()
                print(f" Lambda prune: {self.lambda_prune:.4f} | Lambda div: {self.lambda_div:.4f}")

            loss_total = loss_mse_norm +  0.8 * loss_prune_norm + 0.5 *loss_div_norm
            self.prune_loss = loss_prune_norm.item()
            self.div_loss = loss_div_norm.item()

            if epoch % 20 == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch:4d} | Total Loss: {loss_total.item():.4f} | "
                      f"MSE: {loss_mse_norm.item():.4f} | Prune: {loss_prune_norm.item():.4f} | Div: {loss_div_norm.item():.6f}")

            loss_total.backward()
            opt.step()

            if epoch == self.epochs - 1:
                self.total_loss = loss_total.item()

        w_final_abs = np.abs(self.model.w.detach().numpy().squeeze())
        if np.sum(w_final_abs) > 1e-8:
            self.w_final = w_final_abs / np.sum(w_final_abs)
        else:
            self.w_final = np.zeros_like(w_final_abs)
        print(f"[INFO] LinearMetaModel training complete. Final loss: {self.total_loss:.4f}")

    def prune(self, prune_threshold=0.005, corr_thresh=0.95):
        if self.w_final is None:
            print("[ERROR] Call train() before prune()")
            return

        initial_tree_list = self.initial_pruned_trees
        w_max = np.max(self.w_final)
        actual_threshold = prune_threshold * w_max
        if w_max == 0:
            keep_idx_weights = []
        else:
            keep_idx_weights = np.where(self.w_final > actual_threshold)[0]

        if len(keep_idx_weights) > 1:
            trees_after_weights = [initial_tree_list[i] for i in keep_idx_weights]
            X_meta_train_pruned = self._get_meta_features(self.workflow.X_train_meta, trees_after_weights)
            corr_matrix = np.corrcoef(X_meta_train_pruned.T)
            np.fill_diagonal(corr_matrix, 0)
            redundant_local_indices = set(np.unique(np.where(np.abs(corr_matrix) > corr_thresh)[0]))
            final_keep_idx = [idx for i, idx in enumerate(keep_idx_weights) if i not in redundant_local_indices]
            keep_idx = final_keep_idx
        else:
            keep_idx = keep_idx_weights

        self.pruned_trees = [initial_tree_list[i] for i in keep_idx]
        self.pruned_exp = True
        print(f"[INFO] Final ensemble size after pruning: {len(self.pruned_trees)}")

    def evaluate(self):
        if self.model is None or self.pruned_trees is None:
            print("[ERROR] Model not trained or pruned_trees missing.")
            return None, None

        if not self.pruned_exp:
            trees_to_evaluate = self.initial_pruned_trees
            weights_to_use = self.w_final
        else:
            trees_to_evaluate = self.pruned_trees
            if len(trees_to_evaluate) == 0:
                print("[WARN] No trees left after pruning. Cannot evaluate.")
                return None, self.total_loss

            X_train_final = self._get_meta_features(self.workflow.X_train_meta, trees_to_evaluate)
            y_train_final = self.workflow.y_train_meta
            if self.data_type == "regression":
                final_eval_model = LinearRegression().fit(X_train_final, y_train_final)
                w_abs = np.abs(final_eval_model.coef_)
            else:
                final_eval_model = LogisticRegression(max_iter=2000).fit(X_train_final, y_train_final)
                w_abs = np.abs(final_eval_model.coef_[0])

            weights_to_use = w_abs / (np.sum(w_abs) + 1e-12)

        tree_preds = self._get_meta_features(self.workflow.X_test, trees_to_evaluate)
        if self.data_type == "regression":
            final_preds = tree_preds @ weights_to_use
            self.pruned_ensemble_metric = mean_squared_error(self.workflow.y_test, final_preds)
            print(f"[INFO] Final Pruned Ensemble MSE: {self.pruned_ensemble_metric:.4f}")
        else:
            tree_labels = np.vstack([t.predict(self.workflow.X_test) for t in trees_to_evaluate])
            from scipy.stats import mode
            mode_result = mode(tree_labels, axis=0, keepdims=False)
            final_preds = mode_result.mode

            self.pruned_ensemble_metric = accuracy_score(self.workflow.y_test, final_preds)
            from sklearn.metrics import roc_auc_score
            self.auc = roc_auc_score(self.workflow.y_test, final_preds)
            f1 = f1_score(self.workflow.y_test, final_preds, average="weighted")
            print(f"[INFO] Final Pruned Ensemble Accuracy: {self.pruned_ensemble_metric:.4f} | F1: {f1:.4f}")

        return self.pruned_ensemble_metric, self.total_loss
