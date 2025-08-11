#!/bin/bash

#############################
# EXPERIMENT SPECIFICATIONS #
#############################

expgroup="taskset_experiment"
bench="taskset"
max_seeds=10
algos=(
    "random_search"
    "mf_ei_bo"
    "hyperband"
    "asha"
    "dyhpo-neps-v2"
    "dpl-neps-max"
    "ifbo"
)


#####################
# PLOTTING RUN CALL #
#####################

if [[ $bench == "lcbench" ]]; then 
    benchmarks=(
        "lcbench-tabular-adult-balanced"
        "lcbench-tabular-airlines-balanced"
        "lcbench-tabular-albert-balanced"
        "lcbench-tabular-Amazon-balanced"
        "lcbench-tabular-APSFailure-balanced"
        "lcbench-tabular-Australian-balanced"
        "lcbench-tabular-bank-balanced"
        "lcbench-tabular-blood-balanced"
        "lcbench-tabular-car-balanced"
        "lcbench-tabular-christine-balanced"
        "lcbench-tabular-cnae-balanced"
        "lcbench-tabular-connect-balanced"
        "lcbench-tabular-covertype-balanced"
        "lcbench-tabular-credit-balanced"
        "lcbench-tabular-dionis-balanced"
        "lcbench-tabular-fabert-balanced"
        "lcbench-tabular-Fashion-balanced"
        "lcbench-tabular-helena-balanced"
        "lcbench-tabular-higgs-balanced"
        "lcbench-tabular-jannis-balanced"
        "lcbench-tabular-jasmine-balanced"
        "lcbench-tabular-jungle-balanced"
        "lcbench-tabular-kc1-balanced"
        "lcbench-tabular-KDDCup09-balanced"
        "lcbench-tabular-kr-balanced"
        "lcbench-tabular-mfeat-balanced"
        "lcbench-tabular-MiniBooNE-balanced"
        "lcbench-tabular-nomao-balanced"
        "lcbench-tabular-numerai-balanced"
        "lcbench-tabular-phoneme-balanced"
        "lcbench-tabular-segment-balanced"
        "lcbench-tabular-shuttle-balanced"
        "lcbench-tabular-sylvine-balanced"
        "lcbench-tabular-vehicle-balanced"
        "lcbench-tabular-volkert-balanced"
    )
elif [[ $bench == "pd1" ]]; then 
    tabular_pd1_benchmarks=(
        "pd1-tabular-cifar100_wideresnet_2048"
        "pd1-tabular-cifar100_wideresnet_256"
        "pd1-tabular-cifar10_wideresnet_2048"
        "pd1-tabular-cifar10_wideresnet_256"
        "pd1-tabular-fashion_simplecnn_2048"
        "pd1-tabular-fashion_simplecnn_256"
        "pd1-tabular-imagenet_resnet_1024-10"
        "pd1-tabular-imagenet_resnet_1024-1"
        "pd1-tabular-imagenet_resnet_1024-2"
        "pd1-tabular-imagenet_resnet_1024-5"
        "pd1-tabular-imagenet_resnet_256-10"
        "pd1-tabular-imagenet_resnet_256-1"
        "pd1-tabular-imagenet_resnet_256-2"
        "pd1-tabular-imagenet_resnet_256-5"
        "pd1-tabular-imagenet_resnet_512-10"
        "pd1-tabular-imagenet_resnet_512-1"
        "pd1-tabular-imagenet_resnet_512-2"
        "pd1-tabular-imagenet_resnet_512-5"
        "pd1-tabular-lm1b_transformer_2048"
        "pd1-tabular-mnist_simplecnn_2048"
        "pd1-tabular-mnist_simplecnn_256"
        "pd1-tabular-svhn_wideresnet_1024"
        "pd1-tabular-svhn_wideresnet_256"
        "pd1-tabular-translate_xformertranslate_64-10"
        "pd1-tabular-translate_xformertranslate_64-1"
        "pd1-tabular-translate_xformertranslate_64-2"
        "pd1-tabular-translate_xformertranslate_64-5"
        "pd1-tabular-uniref50_transformer_128-1"
    )
elif [[ $bench == "taskset" ]]; then 
    benchmarks=(
        "lcbench-tabular-adult-balanced"
        "lcbench-tabular-airlines-balanced"
        "lcbench-tabular-albert-balanced"
        "lcbench-tabular-Amazon-balanced"
        "lcbench-tabular-APSFailure-balanced"
        "lcbench-tabular-Australian-balanced"
        "lcbench-tabular-bank-balanced"
        "lcbench-tabular-blood-balanced"
        "lcbench-tabular-car-balanced"
        "lcbench-tabular-christine-balanced"
        "lcbench-tabular-cnae-balanced"
        "lcbench-tabular-connect-balanced"
        "lcbench-tabular-covertype-balanced"
        "lcbench-tabular-credit-balanced"
        "lcbench-tabular-dionis-balanced"
        "lcbench-tabular-fabert-balanced"
        "lcbench-tabular-Fashion-balanced"
        "lcbench-tabular-helena-balanced"
        "lcbench-tabular-higgs-balanced"
        "lcbench-tabular-jannis-balanced"
        "lcbench-tabular-jasmine-balanced"
        "lcbench-tabular-jungle-balanced"
        "lcbench-tabular-kc1-balanced"
        "lcbench-tabular-KDDCup09-balanced"
        "lcbench-tabular-kr-balanced"
        "lcbench-tabular-mfeat-balanced"
        "lcbench-tabular-MiniBooNE-balanced"
        "lcbench-tabular-nomao-balanced"
        "lcbench-tabular-numerai-balanced"
        "lcbench-tabular-phoneme-balanced"
        "lcbench-tabular-segment-balanced"
        "lcbench-tabular-shuttle-balanced"
        "lcbench-tabular-sylvine-balanced"
        "lcbench-tabular-vehicle-balanced"
        "lcbench-tabular-volkert-balanced"
    )
else
    echo "Invalid benchmarks ${benchmarks}" && exit
fi

echo "Experiment group: $expgroup"
echo "Selected algos: "${algos[@]}
echo "Selected tasks: "${benchmarks[@]}

python ../pfns_hpo/pfns_hpo/regret_plot.py \
    --expgroup $expgroup \
    --continuations \
    --algorithms ${algos[@]} \
    --benchmarks ${benchmarks[@]} \
    --seeds $max_seeds \
    --x_range 0 1000 \
    --log_y \
    --plot_all \
    --filename $expgroup  \
    --strict

# end of script