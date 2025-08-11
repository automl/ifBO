[![arXiv](https://img.shields.io/badge/arXiv-2105.09821-b31b1b.svg)](https://arxiv.org/abs/2404.16795)

# Overview

This repository contains code for the ICML 2024 paper: [In-context Freeze-Thaw Bayesian Optimization for Hyperparameter Optimization](https://openreview.net/forum?id=VyoY3Wh9Wd).


To use the `ifBO` for hyperparameter tuning in practice, please refer to [NePS](https://automl.github.io/neps/latest/), a package for hyperparameter optimization that maintains the latest, improved `ifBO` version (TBA).

The `main` branch provides the `Freeze-Thaw PFN surrogate (FT-PFN)` surrogate model as a drop-in surrogate for multi-fidelity Bayesian Optimization loops. Along with the synthetic prior generation and training code.

### Table of Contents
1. [Setup](#setup)
2. [How to run an optimizer on a benchmark](#how-to-run-an-optimizer-on-a-benchmark)
3. [Plot results](#plot-results)


# Setup

All the following commands assume the working directory to be the root level of the `ifBO/` repo.

### Install dependencies

Install `Python 3.10.12` and `pip`.

Run:
```bash
pip install -r core_requirements.txt
```

*NOTE*:
* `requirements.txt`: contains the pip-freeze version of our final environment
* `nvidia_requirements.txt`: system-dependent base packages omitted from `requirements.txt`
* `extra_requirements.txt`: packages omitted from `requirements.txt` that has issues installing on Mac OS 14.2 when installing in bulk

### Install JUST

(To know why you need this is recommended, scroll below!)

To install our command runner just you can check the [just documentation](https://github.com/casey/just#installation), or run the below command

```bash
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to $HOME/.just
```

Also, to make the `just` command available you should add,

```bash
export PATH="$HOME/.just:$PATH"
```

to your `.zshrc` / `.bashrc` or alternatively simply run the export manually.

### Install dependent code in editable format

<!--
```bash
bash install.sh
```
-->

```bash
make install
```

*NOTE*: Depending on the operating system and permissions available, this might require manual intervention

### Setup Benchmarks

```bash
python -m mfpbench download --benchmark pd1-tabular
python -m mfpbench download --benchmark lcbench-tabular
python -m mfpbench download --benchmark taskset-tabular
```
These should create a `data/` folder at the ifBO root.

### PFN surrogate

To download the surrogate used for the experiments in the paper:
```bash
just download_pfn
# or
just download_pfn 0.0.1
```

For subsequent updates to the surrogates, check for new versions [here](src/pfns_hpo/pfns_hpo/download.py), and run:
```bash
just download_pfn 0.0.2
```

Refer to [src/PFNs4HPO/](https://github.com/automl/ifBO/tree/icml-2024/src/PFNs4HPO), if you are interested in retraining FT-PFN.


# How to run an optimizer on a benchmark

From the root directory of `ifbo/`:

```bash
just run asha pd1-tabular-cifar100_wideresnet_2048 test 123
```

This runs the *optimizer* `asha` on the benchmark instance `pd1-tabular-cifar100_wideresnet_2048`, storing it in an *experiment group* `test` (by default stored as `./results/test`), using the seed `123`.

The format for the `just run` command is defined in `justfile` and can be adapted for custom use.


### Running ifBO on a benchmark

Similar format as above:
```bash
just run ifbo pd1-tabular-cifar100_wideresnet_2048 test 123 1
```
Note that `ifbo` configuration or its hyper-hyperparameters are specified in [`ifbo.yaml`](src/pfns_hpo/pfns_hpo/configs/algorithm/ifbo.yaml).

If a new surrogate is to be used or the acquisition needs to be updated, this *yaml* file needs to be updated.

### Optimizers

To view all available optimizer/algorithm configurations:
```bash
just algorithms  # | grep "pfn"
```

The baselines from the paper:

| configuration keys | names from plots in paper |
|--|--|
| `random_search` | Random Search |
| `hyperband` | HyperBand |
| `asha` | ASHA |
| `mf_ei_bo` | Freeze-Thaw with GPs |
| `dyhpo-neps-v2` | DyHPO |
| `dpl-neps-max` | DPL |
| `ifbo` | ifBO |

### Benchmarks

To view all available benchmark configurations:
```bash
just benchmarks  # | grep "pd1"
```

The benchmark keys used in the paper can be found [in this file](benchmarks.md).


### Running a batch of experiments

Batch files of multiple such `just run` commands can define the entire suite of experiments. Such runs can be programmatically defined as strings. Moreover, configuration files for benchmarks and algorithms can be modified, added programmatically too for scaling.

```bash
#!/bin/bash/

# example script running 4 experiments in parallel
just run random_search pd1-tabular-cifar100_wideresnet_2048 test 123 &
just run asha pd1-tabular-cifar100_wideresnet_2048 test 123 &
just run random_search lcbench-tabular-kr test 123 &
just run asha lcbench-tabular-kr test 123;

```

### Running on distributed clusters

The atomic `just run` command allows the construction of pipelines and workflows that sweep over all required runs for experiments. Generating a file with all required run commands followed by distributing each line in the file as a job submitted to a resource is the simple core flow that can be used.

Given that every distributed cluster has their own specific quirks, we focus on how a single run can be collected and expect the user to design the wrapper that distributes the required set of runs.

For reference, `./src/pfns_hpo/pfns_hpo/submit.py` is the Python wrapper we created that would allow us to use `just submit` to create 1-line strings that could specify all the experiment runs as a large array job for the custom-SLURM scheduler we interfaced.

```bash
just submit random_search,asha lcbench-tabular-kr,pd1-tabular-cifar100_wideresnet_2048 range(10) exp_name job_name partition_name max_tasks time_limit memory
```

Here, `range(10)` would collect runs for seeds `{0, ..., 9}`.
In all, this results into a Cartesian product of runs from (algorithms) x (benchmarks) x (seeds): `{random_search, asha}` x `{lcbench-tabular-kr,pd1-tabular-cifar100_wideresnet_2048}` x `{0, ..., 9}`.

### Runtime info

Here we list some possible keypoints regarding our code framework that can assist in debugging if any errors show up:

* Each optimizer run will be stored in a hierarchy of `results/<experiment_group_name>/<benchmark=...>/<algorithm=...>/neps_root_directory/`
* The main run function is in [pfns_hpo/run.py](src/pfns_hpo/pfns_hpo/run.py)
  * For quick debugging runs, control HPO budgets from L37-40
  * To overwrite existing directory and restart an optimization run, set L33 to `True`
* For any cryptic error that prevents any optimizer run from starting, first debugging step could be delete the output folder if it exists already for this run
* One shortcut to check if a single HPO run is complete, is to check if `summary_csv` exists for that run inside `neps_root_directory/`
  * The trajectory summary will be saved in `summary_csv/config_data.csv`
* To not have filesystems be too fragmented when storing multiple runs, we trigger a zipping of all output files per configuration of an HPO run into a `results.zip`

**NOTE**: All optimizers will continue its optimization if running on the same output directory and HPO budget is remaining. To start a new run, set `overwrite=True` in `pfns_hpo/run.py` at a global scope or set the flag `overwrite_working_directory` in `neps.run()` in `pfns_hpo/run.py`

**NOTE, again**: Launching multiple `just run ...` pointing to the same output directory (i.e., same algorithm-benchmark-seed-exp_group) will trigger a multi-worker parallel HPO. All experiments here are assumed to be single worker runs for fair comparison.

# Plot results

To regenerate Figure 3 in the main paper, first download data ([bounds](http://ml.informatik.uni-freiburg.de/research-artifacts/ifbo/bounds.hkl) and [results](http://ml.informatik.uni-freiburg.de/research-artifacts/ifbo/allresults.hkl) files), then run ``bash plot.sh``.


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
