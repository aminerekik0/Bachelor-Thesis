# Explainability-Driven Tree Ensemble Pruning Using Meta-Level Optimization (X-MOP)

**Author:** Mohamed Amine Rekik 


---

## 1. Abstract
X-MOP is a Python framework for pruning tree ensembles by explicitly optimizing for explainability, sparsity, and diversity. Unlike traditional pruning strategies that focus solely on predictive accuracy, X-MOP leverages SHAP values and meta-level optimization to select trees that provide meaningful contributions, resulting in compact, interpretable, and diverse ensembles suitable for both regression and classification tasks.



## 2. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/aminerekik0/Bachelor-Thesis.git
cd X-MOP
pip install -r requirements.txt
```


## 3. Simple Example Usage with X-MOP

```python
import numpy as np
from sklearn.datasets import load_iris
from src.EnsembleCreator import EnsembleCreator
from src.PrePruner import PrePruner
from src.MetaOptimizer import MetaOptimizer

# Load Iris dataset
data = load_iris()
X = data.data
y = data.target

# 1. Create Base Trees
ensemble = EnsembleCreator(X, y, n_trees=200, data_type="classification")
ensemble.train_base_trees()

# 2. Pre-Pruning using SHAP (keep top 70%)
prepruner = PrePruner(keep_ratio=0.3, data_type="classification")
prepruner.attach_to(ensemble)
prepruner.train()
pruned_trees = prepruner.pruned_trees

# 3. Meta-Optimization for pruning & diversity
meta_opt = MetaOptimizer(λ_prune=1.0, λ_div=0.5, data_type="classification")
meta_opt.attach_to(ensemble)
meta_opt.train(pruned_trees)
meta_opt.prune()

```
