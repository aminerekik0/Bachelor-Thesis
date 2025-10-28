from abc import ABC, abstractmethod
import pandas as pd
import os
class BaseMetaModel(ABC):
    """Base class for all meta-models."""

    def __init__(self, data_type="regression", random_state=42):
        self.data_type = data_type
        self.random_state = random_state
        self.workflow = None

    def attach_to(self, workflow):
        """Attach the meta model to the main workflow"""
        self.workflow = workflow

    @abstractmethod
    def train(self):
        """Train the meta-model."""
        pass

    @abstractmethod
    def evaluate(self):
        """Evaluate performance"""
        pass

    def save_results(self, filename=None):
        """
        Save the results of this model, including meta-model losses and full ensemble metrics.
        Each model will have its own file if filename is not explicitly provided.
        """

        from datetime import datetime
        import os
        import pandas as pd

    # Default filename uses model class name + dataset
        if filename is None:
           filename = f"{self.__class__.__name__}_results.csv"

        current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        df = pd.DataFrame({
        "dataset": [self.workflow.dataset_name],
        "data_type": [self.data_type],
        "n_samples": [self.workflow.n_samples],
        "n_features": [self.workflow.n_features],
        "full_ensemble_loss": [getattr(self.workflow, 'full_metric', None)],
        "meta_main_loss": [self.main_loss],
        "meta_prune_loss_final": [getattr(self, 'prune_loss_final', None)],
        "meta_div_loss_final": [getattr(self, 'div_loss_final', None)],
    })

    # Save to CSV (append if exists)
        if os.path.exists(filename):
           df.to_csv(filename, mode='a', header=False, index=False, float_format='%.4f')
        else:
           df.to_csv(filename, index=False, float_format='%.4f')

        print(f"[INFO] Results saved to {filename}")

