# Reproducing the experiments in Section 5.1

The results and datasets used in the experiments can be downloaded with the following commands:

```bash
wget http://ml.informatik.uni-freiburg.de/research-artifacts/ifbo/dataSection5.1.zip
unzip dataSection5.1.zip
```

You can generate Table 1 from the paper by running the notebook `generate_table_1.ipynb`.

## Generate Tasks

To generate tasks from a benchmark (``lcbench_tabular``, ``pd1_tabular`` or ``taskset_tabular``), use the command below:

```bash
python generate_tasks.py --ntasks_per_dataset 100 --benchmark <BENCHMARK> --seed 42 --data_path ../../data
```

## Evaluate (LogLikelihood, MSE, Runtime)

To compute the LogLikelihood, MSE, and runtime of predictions for a model (``bopfn_broken_no_hps_1000curves_10params_2M`` for no HPs FT-PFN or ``bopfn_broken_unisep_1000curves_10params_2M``), run the following commands:

```bash
python evaluate_pfn.py --model <MODEL> --benchmark lcbench_tabular --data_path ../../data/  
```

**Note:** 
* Command to download no HPs FT-PFN with: ``just download_pfn 0.0.1.nohps``
* Evaluating the DyHPO and DPL models requires installation from their original repositories.
