[![arXiv](https://img.shields.io/badge/arXiv-2105.09821-b31b1b.svg)](https://arxiv.org/abs/2404.16795)

# Overview

This repository contains code for the ICML 2024 submission: [In-context Freeze-Thaw Bayesian Optimization for Hyperparameter Optimization](https://openreview.net/forum?id=VyoY3Wh9Wd).


To use the `ifBO` algorithm in practice, please refer to [NePS](https://automl.github.io/neps/latest/), a package for hyperparameter optimization that maintains the latest, improved `ifBO` version (TBA, TODO).

This `main` branch provides the `Freeze-Thaw PFN surrogate (FT-PFN)` surrogate model as a drop-in surrogate for multi-fidelity Bayesian Optimization loops. Along with the synthetic prior generation and training code.

# Install dependencies

Install `Python 3.10.12` and `pip`.

Run:
```bash
pip install -r requirements.txt
```

# Setup Benchmarks

```bash
python -m mfpbench download --benchmark pd1
python -m mfpbench download --benchmark yahpo
python -m mfpbench download --benchmark jahs
python -m mfpbench download --benchmark pd1-tabular
python -m mfpbench download --benchmark lcbench-tabular
python -m mfpbench download --benchmark taskset-tabular
```

# Training PFNs

```bash
TODO
```


# How to run an optimizer on a benchmark

```bash
TODO
```


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
