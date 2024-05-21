# Overview

This repository contains code for the ICML 2024 submission: [In-context Freeze-Thaw Bayesian Optimization for Hyperparameter Optimization](https://openreview.net/forum?id=VyoY3Wh9Wd).

This `main` branch provides the `Freeze-Thaw PFN surrogate (FT-PFN)` surrogate model as a drop-in surrogate for multi-fidelity Bayesian Optimization loops. Along with the synthetic prior generation and training code.

To reproduce experiments from the above paper version, please refer to the branch [`icml-2024`](https://github.com/automl/ifBO/tree/icml-2024).

To use the `ifBO` algorithm in practice, please refer to [NePS](https://automl.github.io/neps/latest/), a package for hyperparameter optimization that maintains the latest, improved `ifBO` version (TBA, TODO).


# Surrogate versions

| Version | Identifier | Notes |
| -------- | -------- | -------- |
| 0.1 | ICML '24 submission | FT-PFN from ifBO, trained on LCNet curves, DPL power law, broke scaling law |
| TBA | TBA  | TBA |


# Usage

TBA.

# To cite:

If using our surrogate, code, experiment setup, kindly cite using:
```bibtex
@inproceedings{
  rakotoarison-icml24,
  title={In-Context Freeze-Thaw Bayesian Optimization for Hyperparameter Optimization},
  author={Herilalaina Rakotoarison and Steven Adriaensen and Neeratyoy Mallik and Samir Garibov and Eddie Bergman and Frank Hutter},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024},
  url={https://openreview.net/forum?id=VyoY3Wh9Wd}
}
```
