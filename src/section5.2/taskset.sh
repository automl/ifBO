#!/bin/bash
#SBATCH --time 0-23:00
#SBATCH --array 0-1679%400
#SBATCH --error .slurm_logs/%N_%A_%x_%a.oe
#SBATCH --output .slurm_logs/%N_%A_%x_%a.oe
#SBATCH -c 1
#SBATCH --mem 8000

EXP_GROUP="taskset_experiment"
ALGORITHMS=("random_search" "ifbo" "dyhpo-neps-v2" "dpl-neps-max" "asha" "hyperband" "mf_ei_bo")
# 24 tasks
BENCHMARKS=( 
    "taskset-tabular-normalized-nlp-10-4p"
    "taskset-tabular-normalized-nlp-10-8p"
    "taskset-tabular-normalized-nlp-11-4p"
    "taskset-tabular-normalized-nlp-11-8p"
    "taskset-tabular-normalized-nlp-12-4p"
    "taskset-tabular-normalized-nlp-12-8p"
    "taskset-tabular-normalized-nlp-1-4p"
    "taskset-tabular-normalized-nlp-1-8p"
    "taskset-tabular-normalized-nlp-2-4p"
    "taskset-tabular-normalized-nlp-2-8p"
    "taskset-tabular-normalized-nlp-3-4p"
    "taskset-tabular-normalized-nlp-3-8p"
    "taskset-tabular-normalized-nlp-4-4p"
    "taskset-tabular-normalized-nlp-4-8p"
    "taskset-tabular-normalized-nlp-5-4p"
    "taskset-tabular-normalized-nlp-5-8p"
    "taskset-tabular-normalized-nlp-6-4p"
    "taskset-tabular-normalized-nlp-6-8p"
    "taskset-tabular-normalized-nlp-7-4p"
    "taskset-tabular-normalized-nlp-7-8p"
    "taskset-tabular-normalized-nlp-8-4p"
    "taskset-tabular-normalized-nlp-8-8p"
    "taskset-tabular-normalized-nlp-9-4p"
    "taskset-tabular-normalized-nlp-9-8p"
)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

TOTAL_ALGOS=${#ALGORITHMS[@]}
TOTAL_BENCHMARKS=${#BENCHMARKS[@]}
TOTAL_SEEDS=${#SEEDS[@]}
COMBOS_PER_ALGO=$((TOTAL_BENCHMARKS * TOTAL_SEEDS))

# Map array task ID to parameters
ALGO_INDEX=$((SLURM_ARRAY_TASK_ID / COMBOS_PER_ALGO))
REMAINING=$((SLURM_ARRAY_TASK_ID % COMBOS_PER_ALGO))
BENCHMARK_INDEX=$((REMAINING / TOTAL_SEEDS))
SEED_INDEX=$((REMAINING % TOTAL_SEEDS))

# Get actual parameter values
ALGORITHM=${ALGORITHMS[$ALGO_INDEX]}
BENCHMARK=${BENCHMARKS[$BENCHMARK_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}

# Run the experiment
cd $PWD/../pfns_hpo/
python -m pfns_hpo.run experiment_group=$EXP_GROUP \
    algorithm=$ALGORITHM \
    benchmark=$BENCHMARK \
    n_workers=1 \
    seed=$SEED \
    hydra/job_logging=full