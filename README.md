[![arXiv](https://img.shields.io/badge/arXiv-2105.09821-b31b1b.svg)](https://arxiv.org/abs/2404.16795)

# Overview

This repository contains code for the ICML 2024 submission: [In-context Freeze-Thaw Bayesian Optimization for Hyperparameter Optimization](https://openreview.net/forum?id=VyoY3Wh9Wd).


To use the `ifBO` algorithm in practice, please refer to [NePS](https://automl.github.io/neps/latest/), a package for hyperparameter optimization that maintains the latest, improved `ifBO` version (TBA, TODO).

This `main` branch provides the `Freeze-Thaw PFN surrogate (FT-PFN)` surrogate model as a drop-in surrogate for multi-fidelity Bayesian Optimization loops. Along with the synthetic prior generation and training code.

# Setup

All the following commands assume the working directory to be the root level of the `ifBO/` repo.

## Install dependencies

Install `Python 3.10.12` and `pip`.

Run:
```bash
pip install -r core_requirements.txt
```

*NOTE*:
* `requirements.txt`: contains the pip-freeze version of our final environment
* `nvidia_requirements.txt`: system-dependent base packages omitted from `requirements.txt`
* `extra_requirements.txt`: packages omitted from `requirements.txt` that has issues installing on Mac OS 14.2 when installing in bulk

## Install JUST

To install our command runner just you can check the [just documentation](https://github.com/casey/just#installation), or run the below command

```bash
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to $HOME/.just
```

Also, to make the `just` command available you should add

```bash
export PATH="$HOME/.just:$PATH"
```

to your `.zshrc` / `.bashrc` or alternatively simply run the export manually.

## Install dependent code in editable format

```bash
bash install.sh
```

*NOTE*: Depending on the operating system and permissions available, this might require manual intervention

# Setup Benchmarks

```bash
python -m mfpbench download --benchmark pd1-tabular
python -m mfpbench download --benchmark lcbench-tabular
python -m mfpbench download --benchmark taskset-tabular
```


# How to run an optimizer on a benchmark

From the root directory of `ifBO/`:

```bash
just run asha pd1-tabular-cifar100_wideresnet_2048 test 123
```

This runs the *optimizer* `asha` on the benchmark instance `pd1-tabular-cifar100_wideresnet_2048`, storing it in an experiment folder `test` (by default stored as `./results/test`), using the seed `123`.

The format for the `just run` command is defined in `justfile` and can be adapted for custom use.

## Optimizers

To view all available optimizer/algorithm configurations:
```bash
just algorithms  # | grep "pfn"
```

The baselines from the paper:

| configuration keys | names from plots in paper |
|--|--|
| `random_search` | Random Search |
| `random_search` | HyperBand |
| `random_search` | ASHA |
| `random_search` | Freeze-Thaw with GPs |
| `random_search` | DyHPO |
| `random_search` | DPL |
| `random_search` | ifBO |

## Benchmarks

To view all available benchmark configurations:
```bash
just benchmarks  # | grep "pd1"
```

The benchmarks used in the paper:

| configuration keys | benchmark family + task ID/name |
|--|--|
| `random_search` | Random Search |
TODO.

## Running a batch of experiments

Batch files of multiple such `just run` commands can define the entire suite of experiments. Such runs can be programmatically defined as strings. Moreover, configuration files for benchmarks and algorithms can be modified, added programmatically too for scaling.

```bash
#!/bin/bash/

# example script running 4 experiments in parallel
just run random_search pd1-tabular-cifar100_wideresnet_2048 test 123 &
just run asha pd1-tabular-cifar100_wideresnet_2048 test 123 &
just run random_search lcbench-tabular-kr test 123 &
just run asha lcbench-tabular-kr test 123;

```

## Running on distributed clusters

The atomic `just run` command allows the construction of pipelines and workflows that sweep over all required runs for experiments. Generating a file with all required run commands followed by distributing each line in the file as a job submitted to a resource is the simple core flow that can be used. 

Given that every distributed cluster has their own specific quirks, we focus on how a single run can be collected and expect the user to design the wrapper that distributes the required set of runs.

For reference, `./src/pfns_hpo/pfns_hpo/submit.py` is the Python wrapper we created that would allow us to use `just submit` to create 1-line strings that could specify all the experiment runs as a large array job for the custom-SLURM scheduler we interfaced.

```bash
just submit random_search,asha lcbench-tabular-kr,pd1-tabular-cifar100_wideresnet_2048 range(10) exp_name job_name partition_name max_tasks time_limit memory
```

Here, `range(10)` would collect runs for seeds `{0, ..., 9}`.
In all, this results into a Cartesian product of runs from (algorithms) x (benchmarks) x (seeds): `{random_search, asha}` x `{lcbench-tabular-kr,pd1-tabular-cifar100_wideresnet_2048}` x `{0, ..., 9}`.


# To cite:

If using our surrogate, code, experiment setup, kindly cite using:
```bibtex
@inproceedings{
  rakotoarison-icml24,
  title={In-Context Freeze-Thaw Bayesian Optimization for Hyperparameter Optimization},
  author={Herilalaina Rakotoarison and Steven Adriaensen and Neeratyoy Mallik and Samir Garibov and Edward Bergman and Frank Hutter},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024},
  url={https://openreview.net/forum?id=VyoY3Wh9Wd}
}
```
