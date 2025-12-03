import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, roc_auc_score, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from src.BaseMetaModel import BaseMetaModel

class _TorchModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        # Initialize weights slightly larger for L1 to have something to prune
        self.w = nn.Parameter(torch.randn(n_features, 1) * 0.05)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        return X @ self.w + self.b

class LinearMetaModel(BaseMetaModel):

    def __init__(self, λ_prune=1.0, λ_div=0.5, epochs=200, lr=1e-2, epsilon=1e-8, mode="SHAP", **kwargs):
        super().__init__(**kwargs)
        self.mode = mode  # "SHAP" or "L1"
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

        # Tracking losses
        self.initial_main_loss = None
        self.initial_prune_loss = None
        self.initial_div_loss = None
        self.final_total_loss = None

        self.kept_after_weight_pruning = None

        # Metrics
        self.auc = None
        self.f1 = None
        self.acc = None
        self.r2 = None

    def _get_meta_features(self, X, trees_list):
        if not trees_list:
            raise ValueError("trees_list cannot be empty.")
        return np.column_stack([t.predict(X) for t in trees_list]).astype(np.float32)

    @staticmethod
    def _loss_accuracy(shap_vals_t, y_true_t, y_pred_t):
        # Helper for classification alignment (not strictly needed for L1 but kept for compatibility)
        error = (y_true_t - y_pred_t).view(-1,1)
        sign_term = torch.tanh(-error / (error.abs().mean() + 1e-8))
        scaled_abs_shap = torch.abs(shap_vals_t)
        alignment = scaled_abs_shap * sign_term
        return -alignment.mean()

    @staticmethod
    def _loss_prune(shap_vals_t, y_true_t, y_pred_t, model_weights=None, mode="SHAP", epsilon=1e-8):
        """
        Calculates the pruning loss based on the selected mode.
        """
        # ============================
        # 1. L1 (Lasso) Mode
        # ============================
        if mode == "L1":
            # Simple L1 Norm: Sum of absolute values of weights
            # This pushes weights towards zero.
            if model_weights is None:
                return torch.tensor(0.0)
            return torch.norm(model_weights, p=1)

        # ============================
        # 2. SHAP Entropy Mode
        # ============================
        else:
            abs_shap = torch.abs(shap_vals_t)
            sum_abs_shap_per_sample = torch.sum(abs_shap, dim=1, keepdim=True)
            p_hat = abs_shap / (sum_abs_shap_per_sample + epsilon)
            entropy_per_sample = -torch.sum(p_hat * torch.log(p_hat + epsilon), dim=1)

            # Original logic: return mean entropy
            return torch.mean(entropy_per_sample)

    @staticmethod
    def _loss_diversity(shap_vals_t, epsilon=1e-8):
        if shap_vals_t.shape[1] <= 1:
            return torch.tensor(0.0, device=shap_vals_t.device)
        phi_bar = torch.mean(torch.abs(shap_vals_t), dim=0)
        p_tilde = phi_bar / (torch.sum(phi_bar) + epsilon)
        return torch.sum(p_tilde ** 2)


    def train(self, pruned_trees_list):
        print(f"[INFO] Training LinearMetaModel (Mode: {self.mode}) on {len(pruned_trees_list)} trees...")

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

        self.model = _TorchModel(n_features)
        opt = optim.Adam(self.model.parameters(), lr=self.lr)

        # Convert to tensors
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_eval_t = torch.tensor(X_eval, dtype=torch.float32)
        y_eval_t = torch.tensor(y_eval, dtype=torch.float32).view(-1, 1)

        # Baseline for SHAP calc (needed for diversity even in L1 mode)
        X_baseline_t = torch.mean(X_eval_t, dim=0, keepdim=True)

        for epoch in range(self.epochs):
            self.model.train()
            opt.zero_grad()

            # --- 1. Main Prediction Loss ---
            y_pred = self.model(X_t)
            if self.data_type == "classification":
                loss_main = nn.functional.binary_cross_entropy_with_logits(y_pred, y_t)
                # Normalize roughly to keeping scale consistent
                loss_main_norm = loss_main / (torch.mean(torch.abs(y_t)) + self.epsilon)
            else:
                loss_main = torch.sqrt(nn.functional.mse_loss(y_pred, y_t))
                loss_main_norm = loss_main / (torch.mean(torch.abs(y_t)) + self.epsilon)

            # --- 2. Pruning Loss ---
            # We need SHAP values even for L1 if we still want to calculate diversity based on SHAP
            # OR if we are in SHAP mode.
            y_eval_pred = self.model(X_eval_t)
            X_baselined_t = X_eval_t - X_baseline_t
            shap_vals_t = X_baselined_t * self.model.w.T

            loss_prune = self._loss_prune(
                shap_vals_t,
                y_eval_t,
                y_eval_pred,
                model_weights=self.model.w, # <--- PASSED WEIGHTS HERE
                mode=self.mode
            )

            # Normalize Pruning Loss based on mode
            if self.mode == "SHAP":
                num_trees = shap_vals_t.shape[1]
                max_entropy = np.log(num_trees + self.epsilon)
                loss_prune_norm = loss_prune / max_entropy
            else:
                # For L1, we usually don't normalize by log(N), but we might want
                # to scale it so it doesn't overpower MSE immediately.
                # Using the raw L1 sum is standard.
                loss_prune_norm = loss_prune

            # --- 3. Diversity Loss ---
            # Even with L1, we can keep diversity penalty to ensure we don't just pick one correlated tree
            loss_div = self._loss_diversity(shap_vals_t)
            loss_div_norm = loss_div

            # Total Loss
            loss_total = loss_main_norm + self.λ_prune * loss_prune_norm + self.λ_div * loss_div_norm

            # Logging
            if epoch == 0:
                self.initial_main_loss = float(loss_main_norm.item())
                self.initial_prune_loss = float(loss_prune_norm.item())
                self.initial_div_loss = float(loss_div_norm.item())
                self.initial_total_loss = float(loss_total.item())

            if epoch % 20 == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch:4d} | Total: {loss_total.item():.4f} | "
                      f"MSE: {loss_main_norm.item():.4f} | "
                      f"{self.mode} Loss: {loss_prune_norm.item():.4f} | Div: {loss_div_norm.item():.6f}")

            loss_total.backward()
            opt.step()

            # Tracking
            if epoch == self.epochs - 1:
                self.final_main_loss = float(loss_main_norm.item())
                self.final_prune_loss = float(loss_prune_norm.item())
                self.final_div_loss = float(loss_div_norm.item())
                self.final_total_loss = float(loss_total.item())

        # Store final weights
        w_final_abs = np.abs(self.model.w.detach().numpy().squeeze())

        # Normalize for inspection (softmax/sum style)
        if np.sum(w_final_abs) > 1e-8:
            self.w_final = w_final_abs # Keep absolute values for L1 pruning thresholding
        else:
            self.w_final = np.zeros_like(w_final_abs)

        print(f"[INFO] LinearMetaModel training complete. Final loss: {self.final_total_loss:.4f}")

    def prune(self, prune_threshold=0.01, corr_thresh=0.95):
        if self.w_final is None:
            print("[ERROR] Call train() before prune()")
            return

        initial_tree_list = self.initial_pruned_trees

        # For L1, the logic is simple: if weight is close to 0, prune it.
        # We use a relative threshold here to be safe against floating point noise.
        w_max = np.max(self.w_final)
        actual_threshold = prune_threshold * w_max

        print(f"\n==================== {self.mode} WEIGHT PRUNING ====================")
        # print("Tree | Weight")
        # for i, w in enumerate(self.w_final):
        #     print(i, "|", float(w))

        if w_max == 0:
            keep_idx_weights = []
        else:
            # Standard magnitude pruning
            keep_idx_weights = np.where(self.w_final > actual_threshold)[0]

        self.kept_after_weight_pruning = list(keep_idx_weights)

        print("Kept count:", len(keep_idx_weights))

        # Proceed with Correlation Pruning (Same for both L1 and SHAP)
        if len(keep_idx_weights) > 1:
            trees_after_weights = [initial_tree_list[i] for i in keep_idx_weights]
            X_meta_eval_pruned = self._get_meta_features(self.workflow.X_eval_meta, trees_after_weights)
            corr_matrix = np.corrcoef(X_meta_eval_pruned.T)
            np.fill_diagonal(corr_matrix, 0)

            redundant_local_indices = set(np.unique(np.where(np.abs(corr_matrix) > corr_thresh)[0]))
            redundant_global_indices = {keep_idx_weights[i] for i in redundant_local_indices}

            keep_idx = [idx for idx in keep_idx_weights if idx not in redundant_global_indices]
        else:
            keep_idx = keep_idx_weights

        self.pruned_trees = [initial_tree_list[i] for i in keep_idx]
        self.pruned_exp = True

    def evaluate(self):
        # (Keep exactly the same as your previous code)
        if self.model is None or self.pruned_trees is None:
            print("[ERROR] Model not trained or pruned_trees missing.")
            return None, None

        if not self.pruned_exp:
            trees_to_evaluate = self.initial_pruned_trees
        else:
            trees_to_evaluate = self.pruned_trees

        if len(trees_to_evaluate) == 0:
            print("[WARN] No trees left after pruning. Cannot evaluate.")
            return None, self.total_loss

        # Re-calculate weights for the final subset using standard Regression/Logistic
        # This "Refits" the ensemble, which is standard practice after L1 selection
        X_train_final = self._get_meta_features(self.workflow.X_train_meta, trees_to_evaluate)
        y_train_final = self.workflow.y_train_meta

        if self.data_type == "regression":
            final_eval_model = LinearRegression().fit(X_train_final, y_train_final)
            weights_to_use = final_eval_model.coef_ # Use raw coefficients
        else:
            final_eval_model = LogisticRegression(max_iter=2000).fit(X_train_final, y_train_final)
            weights_to_use = final_eval_model.coef_[0]

        tree_preds = self._get_meta_features(self.workflow.X_test, trees_to_evaluate)

        # =============== REGRESSION METRICS ===============
        if self.data_type == "regression":
            # Simple weighted sum
            final_preds = tree_preds @ weights_to_use + final_eval_model.intercept_
            rmse = np.sqrt(mean_squared_error(self.workflow.y_test, final_preds))
            r2 = r2_score(self.workflow.y_test, final_preds)
            self.pruned_ensemble_metric = rmse
            self.r2 = r2
            print(f"[INFO] Final Pruned Ensemble RMSE: {rmse:.4f} | R2: {r2:.4f}")
            return rmse, r2

        # =============== CLASSIFICATION METRICS ===============
        else:
            # For classification, we usually use Voting or recalculate probabilities
            # But to keep it consistent with your code:
            tree_labels = np.vstack([t.predict(self.workflow.X_test) for t in trees_to_evaluate])
            from scipy.stats import mode
            mode_result = mode(tree_labels, axis=0, keepdims=False)
            final_preds = mode_result.mode

            acc = accuracy_score(self.workflow.y_test, final_preds)
            f1 = f1_score(self.workflow.y_test, final_preds, average="weighted")
            auc = roc_auc_score(self.workflow.y_test, final_preds) # Note: check if y_test is one-hot or Label enc.

            self.acc = acc
            self.f1 = f1
            self.auc = auc
            print(f"[INFO] Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
            return f1, auc