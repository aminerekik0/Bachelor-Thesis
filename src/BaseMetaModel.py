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
    def train(self , *args , **kwargs):
        """Train the meta-model."""
        pass




