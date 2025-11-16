
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from BaseMetaModel import BaseMetaModel
from sklearn.linear_model import LinearRegression # Import this


class _TorchModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_features, 1) * 0.1)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        return X @ self.w + self.b

class LinearMetaModel(BaseMetaModel):
    """
    A Linear Meta-Model that learns to combine a
    pre-pruned list of trees.
    """

    def __init__(self, λ_prune=1.0, λ_div=0.5, epochs=200, lr=1e-2, epsilon=1e-8, **kwargs):
        """
        Initializes the LinearMetaModel.
        """
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
        self.pruned_ensemble_mse = None
        self.total_loss = None
        self.pruned_exp = False

    def _get_meta_features(self, X, trees_list):
        """
        Helper function to get predictions from a specific list of trees.
        """
        if not trees_list:
            raise ValueError("trees_list provided to _get_meta_features cannot be empty.")

        return np.column_stack([t.predict(X) for t in trees_list]).astype(np.float32)


    @staticmethod
    def _loss_accuracy(shap_vals, y_true, y_pred):
        errors_t = (y_true - y_pred.reshape(y_pred.shape)).reshape(-1, 1)
        sign_of_neg_errors_t = torch.sign(-errors_t)
        return -torch.mean(torch.abs(shap_vals) * sign_of_neg_errors_t)

    @staticmethod
    def _loss_prune(shap_vals_t, epsilon=1e-8):
        """
        Calculates L_prune (Entropy) Using torch calculations
        """
        abs_shap = torch.abs(shap_vals_t)
        sum_abs_shap_per_sample = torch.sum(abs_shap, dim=1, keepdim=True)
        p_hat = abs_shap / (sum_abs_shap_per_sample + epsilon)

        entropy_per_sample = -torch.sum(p_hat * torch.log(p_hat + epsilon), dim=1)

        loss_prune = torch.mean(entropy_per_sample)
        return loss_prune

    @staticmethod
    def _loss_diversity_corr(shap_vals_t, epsilon=1e-8):
        """
        Calculates L_div (Mean Absolute Correlation) in pure PyTorch.
        """
        N, M = shap_vals_t.shape
        if M <= 1:
            return torch.tensor(0.0, device=shap_vals_t.device)

        mean = torch.mean(shap_vals_t, dim=0, keepdim=True)
        centered_data = shap_vals_t - mean

        cov_matrix = (centered_data.T @ centered_data) / (N - 1)

        std_devs = torch.std(shap_vals_t, dim=0) + epsilon

        std_dev_matrix = std_devs.unsqueeze(1) @ std_devs.unsqueeze(0)

        corr = cov_matrix / (std_dev_matrix + epsilon)

        corr.fill_diagonal_(0)

        return torch.mean(torch.abs(corr))

    @staticmethod
    def _loss_diversity(shap_vals_t, epsilon=1e-8):
        """
        Calculates L_div (Global SHAP Concentration) from the exposé [cite: 157-161].
        This penalizes the model if a few trees dominate the
        global average attributions.
        """
        if shap_vals_t.shape[1] <= 1: # Cannot be non-diverse with 1 tree
            return torch.tensor(0.0, device=shap_vals_t.device)

        # 1. Calculate average attribution per tree:
        #    phi_bar_k = (1/N) * sum_i |phi_ik|
        phi_bar = torch.mean(torch.abs(shap_vals_t), dim=0)

        # 2. Calculate total sum of average attributions:
        #    sum_j phi_bar_j
        sum_phi_bar = torch.sum(phi_bar)

        # 3. Normalize to get p_tilde_k:
        #    p_tilde_k = phi_bar_k / (sum_j phi_bar_j + epsilon)
        p_tilde = phi_bar / (sum_phi_bar + epsilon)

        # 4. Calculate L_diversity: sum_k (p_tilde_k^2)
        loss_div = torch.sum(p_tilde ** 2)

        return loss_div

    def train(self, pruned_trees_list):
        """
        Train the Linear Meta-Model on the predictions of the pruned trees.
        """
        print(f"[INFO] Training LinearMetaModel on {len(pruned_trees_list)} pruned trees...")

        if not pruned_trees_list:
            print("[ERROR] LinearMetaModel received an empty list of pruned trees. Aborting train.")
            return

        self.initial_pruned_trees = pruned_trees_list
        self.pruned_trees = pruned_trees_list

        X_train = self._get_meta_features(self.workflow.X_train_meta, self.pruned_trees)
        y_train = self.workflow.y_train_meta
        X_eval = self._get_meta_features(self.workflow.X_eval_meta, self.pruned_trees)

        n_features = X_train.shape[1]
        if n_features == 0:
            print("[ERROR] Meta-features for LinearMetaModel have 0 features. Aborting.")
            return

        self.model = _TorchModel(n_features)
        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_eval_t = torch.tensor(X_eval, dtype=torch.float32)
        y_eval_t = torch.tensor(self.workflow.y_eval_meta, dtype=torch.float32).view(-1, 1)
        X_baseline_t = torch.mean(X_eval_t, dim=0, keepdim=True)

        for epoch in range(self.epochs):
            self.model.train()
            opt.zero_grad()

            y_pred = self.model(X_t)
            loss_mse = nn.functional.mse_loss(y_pred, y_t)
            y_eval_pred = self.model(X_eval_t)

            X_baselined_t = X_eval_t - X_baseline_t
            shap_vals_t = X_baselined_t * self.model.w.T

            loss_acc = self._loss_accuracy(shap_vals_t ,y_eval_t,y_eval_pred )
            loss_prune = self._loss_prune(shap_vals_t, self.epsilon)
            loss_div = self._loss_diversity(shap_vals_t, self.epsilon)
            if epoch == 0:
                 self.lambda_prune = self.λ_prune * float((loss_mse / (loss_prune + self.epsilon)).item())
                 self.lambda_div   = self.λ_div  * self.lambda_prune
                 print(" Lambda prune : " , self.lambda_prune )
                 print(" Lambda div : " , self.lambda_div ) 
           
            

            loss_total = loss_mse + self.lambda_prune * loss_prune + self.lambda_div * loss_div

            if epoch % 20 == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch:4d} | "
                      f"Total Loss: {loss_total.item():.4f} | "
                      f"MSE Loss: {loss_mse.item():.4f} | "
                      f"Prune Loss: {loss_prune.item():.4f} | "
                      f"Div Loss: {loss_div.item():.6f}")

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

      

    def prune(self , prune_threshold = 0.009 , corr_thresh = 0.95):
        """
        prune features (Trees) based on their final weights
        """
        if self.w_final is None :
            print("[ERROR] call train() before prune() ")
            return

        initial_tree_list = self.initial_pruned_trees
        print(f"[INFO] Pruning {len(initial_tree_list)} trees...")


        w_max = np.max(self.w_final)
        if w_max == 0 :
            print("[WARN] All weights are zeros . No feature to prune ")
            keep_idx_weights = []
        else :
            keep_idx_weights = np.where(self.w_final > prune_threshold * w_max)[0]

        print(f"[INFO] Stage 1 (Weight): Kept {len(keep_idx_weights)} / {len(self.w_final)} trees.")


        if len(keep_idx_weights) > 1:
            print(f"[INFO] Stage 2 (Correlation): Checking {len(keep_idx_weights)} trees for correlation > {corr_thresh}...")

            trees_after_weights = [initial_tree_list[i] for i in keep_idx_weights]
            X_meta_train_pruned = self._get_meta_features(self.workflow.X_train_meta, trees_after_weights)

            corr_matrix = np.corrcoef(X_meta_train_pruned.T)
            np.fill_diagonal(corr_matrix, 0)

            redundant_local_indices = set(np.unique(np.where(np.abs(corr_matrix) > corr_thresh)[0]))

            final_keep_idx = []
            for i, original_index in enumerate(keep_idx_weights):
                if i not in redundant_local_indices:
                    final_keep_idx.append(original_index)

            print(f"[INFO] Stage 2 (Correlation): Removed {len(redundant_local_indices)} redundant trees.")
            keep_idx = final_keep_idx

        else:
            print("[INFO] Stage 2 (Correlation): Skipped (<= 1 tree).")
            keep_idx = keep_idx_weights

        self.pruned_trees = [initial_tree_list[i] for i in keep_idx]
        self.pruned_exp = True
        print(f"[INFO] Final ensemble size: {len(self.pruned_trees)}")



    def evaluate(self):
        """
        Evaluate the trained linear model on the test set using a weighted average.
        """
        if self.model is None or self.pruned_trees is None:
            print("[ERROR] Model is not trained or pruned_trees list is not set. Call train() first.")
            return None, None

        if self.pruned_exp == False :
            print("[WARN] prune() was not called. Evaluating on the full list from BasicModel.")

            trees_to_evaluate = self.initial_pruned_trees
            weights_to_use = self.w_final
            if weights_to_use is None:
                print("[ERROR] train() did not produce weights.")
                return None, None
        else:

            print("[INFO] Re-training final model on pruned set to get evaluation weights...")
            trees_to_evaluate = self.pruned_trees

            if len(trees_to_evaluate) == 0:
                print("[WARN] No trees left after pruning. Cannot evaluate.")
                return None, self.total_loss


            X_train_final = self._get_meta_features(self.workflow.X_train_meta, trees_to_evaluate)
            y_train_final = self.workflow.y_train_meta


            final_eval_model = LinearRegression().fit(X_train_final, y_train_final)

            w_abs = np.abs(final_eval_model.coef_)
            weights_to_use = w_abs / (np.sum(w_abs) + 1e-12)


        tree_preds = self._get_meta_features(self.workflow.X_test, trees_to_evaluate)


        final_preds = tree_preds @ weights_to_use
        y_test = self.workflow.y_test

        self.pruned_ensemble_mse = mean_squared_error(y_test, final_preds)
        print(f"[INFO] Final Ensemble Pruned MSE (Weighted): {self.pruned_ensemble_mse:.4f}")

        return self.pruned_ensemble_mse, self.total_loss