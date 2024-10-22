[![arXiv](https://img.shields.io/badge/arXiv-2105.09821-b31b1b.svg)](https://arxiv.org/abs/2404.16795)

# `ifBO`: In-context Freeze-Thaw Bayesian Optimization for Hyperparameter Optimization

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/herilalaina/ifbo)

This repository contains the official code for our [ICML 2024 paper](https://openreview.net/forum?id=VyoY3Wh9Wd). `ifBO` is an efficient Bayesian Optimization algorithm that dynamically selects and incrementally evaluates candidates during the optimization process. It uses a model called the `Freeze-Thaw surrogate (FT-PFN)` to predict the performance of candidate configurations as more resources are allocated. The `main` branch includes the necessary API to use `FT-PFN`. Refer to the following sections:
- [Surrogate API](#surrogate-api): to learn how to initialize and use the surrogate model.
- [Bayesian Optimization with ifBO](#bayesian-optimization-with-ifbo): to understand how to use `ifBO` for Hyperparameter Optimization.


> To reproduce experiments from the above paper version, please refer to the branch [`icml-2024`](https://github.com/automl/ifBO/tree/icml-2024).

# Installation

Requires Python 3.11.

```bash
pip install -U ifBO
```

# Usage

## Surrogate API

Checkout out this [notebook](https://github.com/automl/ifBO/blob/main/examples/Getting%20started%20with%20ifBO.ipynb).

**Initializing the model**

```python
from ifbo.surrogate import FTPFN
from ifbo import Curve, PredictionResult

model = FTPFN(version="0.0.1")
```

This creates a ``.model/`` directory in the current working directory for the surrogate model. To have control over this, specify a ``target_path: Path`` when initializing. 

Supported versions:

| Version | Identifier       | Notes                                                                 |
| ------- | ---------------- | --------------------------------------------------------------------- |
| 0.0.1   | ICML '24 submission | Supports up to ``1000`` unique configurations in the context, with each configuration having a maximum of ``10`` dimensions. |

**Creating context and query points**

The code snippet below demonstrates how to create instances of learning curves using `ifbo.Curve` class. Each curve represents the performance over time of a configuration (vector of hyperparameter values). These instances are used to form the context and query points for the model:

- `context`: known data points with both time (`t`) and observed values (`y`).
- `query`: points where predictions are needed, with only time (`t`) provided.

> __Note__: All values (hyperparameters, performances, and times) must be normalized to the range $[0, 1]$.

```python
import torch

context = [
  Curve(
    hyperparameters=torch.tensor([0.2, 0.1, 0.5]), 
    t=torch.tensor([0.1, 0.2, 0.3]), 
    y=torch.tensor([0.1, 0.15, 0.3])
  ),
  Curve(
    hyperparameters=torch.tensor([0.2, 0.3, 0.25]), 
    t=torch.tensor([0.1, 0.2, 0.3, 0.4]), 
    y=torch.tensor([0.2, 0.5, 0.6, 0.75])
  ),
]
query = [
  Curve(
    hyperparameters=torch.tensor([0.2, 0.1, 0.5]), 
    t=torch.tensor([0.3, 0.4, 0.5, 0.6, 0.7, 0.9])
  ),
  Curve(
    hyperparameters=torch.tensor([0.2, 0.3, 0.25]), 
    t=torch.tensor([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
  ),
]
```

**Making predictions** 

Use the model to predict performances at the ``query`` points.

```python
predictions: list[PredictionResult] = model.predict(context=context, query=query)

# Get predictions for the first curve
prediction: PredictionResult = predictions[0]

# Print the 5% and 95% percentiles of the predictive posterior distribution
print(prediction.quantile(0.05), prediction.quantile(0.95))
```

Following the PFN approach, the FT-PFN model outputs the Predictive Posterior Distribution (PPD) of the performances for each query point. Each PPD is encapsulated in an `ifbo.PredictionResult` object, which provides an interface to compute various quantities from the distribution, including:

* ``likelihood(y_test: torch.Tensor)``: Computes the negative log-likelihood of the test targets (``y_test``).
* ``ucb()``: Computes the upper confidence bound.
* ``ei(y_best: torch.Tensor)``: Computes the expected improvement over ``y_best``.
* ``pi(y_best: torch.Tensor)``: Computes the probability of improvement over ``y_best``.
* `quantile(q: float)`: Computes the value at the specified quantile level ``q``.


## Bayesian Optimization with ifBO

To use the `ifBO` algorithm in practice, refer to [NePS](https://automl.github.io/neps/latest/), a package for hyperparameter optimization that includes the latest and improved version of `ifBO`. Below is a template example of how to use `ifBO` with NePS. For a complete Python script, see the [full example](https://github.com/automl/neps/blob/master/neps_examples/efficiency/freeze_thaw.py).

```python
import neps

def training_pipeline(
    num_layers,
    num_neurons,
    epochs,
    learning_rate,
    weight_decay
):
    # Training logic and checkpoint loading here
    pass

pipeline_space = {
    "learning_rate": neps.Float(1e-5, 1e-1, log=True),
    "num_layers": neps.Integer(1, 5),
    "num_neurons": neps.Integer(64, 128),
    "weight_decay": neps.Float(1e-5, 0.1, log=True),
    "epochs": neps.Integer(1, 10, is_fidelity=True),
}

neps.run(
    pipeline_space=pipeline_space,
    run_pipeline=training_pipeline,
    searcher="ifbo",
    max_evaluations_total=50,
    step_size=1,
    surrogate_model_args=dict(
        version="0.0.1",
        target_path=None,
    ),
)
```



# Citation

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
