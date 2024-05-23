from mfpbench.pd1.benchmark import PD1Benchmark
from mfpbench.pd1.benchmarks import (
    PD1cifar100_wideresnet_2048,
    PD1imagenet_resnet_512,
    PD1lm1b_transformer_2048,
    PD1translatewmt_xformer_64,
    PD1uniref50_transformer_128,
)

__all__ = [
    "PD1lm1b_transformer_2048",
    "PD1uniref50_transformer_128",
    "PD1Benchmark",
    "PD1translatewmt_xformer_64",
    "PD1cifar100_wideresnet_2048",
    "PD1imagenet_resnet_512",
]
