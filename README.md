[![arXiv](https://img.shields.io/badge/arXiv-2105.09821-b31b1b.svg)](https://arxiv.org/abs/2404.16795)

# In-context Freeze-Thaw Bayesian Optimization for Hyperparameter Optimization

This repository contains the official code for our [ICML 2024 paper](https://openreview.net/forum?id=VyoY3Wh9Wd). This `main` branch provides the `Freeze-Thaw PFN surrogate (FT-PFN)` surrogate model as a drop-in surrogate for multi-fidelity Bayesian Optimization loops. Along with the synthetic prior generation and training code. To reproduce experiments from the above paper version, please refer to the branch [`icml-2024`](https://github.com/automl/ifBO/tree/icml-2024).

To use the `ifBO` algorithm in practice, please refer to [NePS](https://automl.github.io/neps/latest/), a package for hyperparameter optimization that maintains the latest, improved `ifBO` version (TBA, TODO).

# Setup

```bash
conda create -n ifBO-env python=3.10 setuptools
conda activate ifBO-env
pip install -e .
```


# Surrogate versions

| Version | Identifier | Notes |
| -------- | -------- | -------- |
| 0.0.1 | ICML '24 submission | FT-PFN from ifBO, trained on LCNet curves, DPL power law, broke scaling law |


# Surrogate usage API

```python
# To initialize the surrogate and load pretrained weights
from ifbo.surrogate import FTPFN

model = FTPFN()
```

*NOTE*: This creates a `.model/` directory in the current working directory for the surrogate model. To have control over this specify a `target_path: Path` when initializing.

# To cite:

If using our surrogate, code, experiment setup, kindly cite using:
```bibtex
@inproceedings{
  rakotoarison-icml24,
  title={In-Context Freeze-Thaw Bayesian Optimization for Hyperparameter Optimization},
  author={H. Rakotoarison and S. Adriaensen and N. Mallik and S. Garibov and E. Bergman and F. Hutter},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024},
  url={https://openreview.net/forum?id=VyoY3Wh9Wd}
}
```
