#!/bin/bash

filename="figure_3"
max_seeds=10
algos=(
     "random_search"
     "hyperband"
     "asha"
     "mf_ei_bo"
     "dyhpo-neps-v2"
     "dpl-neps-max"
     "pfn-bopfn-broken-unisep-pi-random"
)


time python ./src/pfns_hpo/pfns_hpo/plot_results.py \
    --algorithms ${algos[@]} \
    --seeds $max_seeds \
    --filename $filename