from mfpbench.pd1.benchmarks.cifar100 import PD1cifar100_wideresnet_2048
from mfpbench.pd1.benchmarks.imagenet import PD1imagenet_resnet_512
from mfpbench.pd1.benchmarks.lm1b import PD1lm1b_transformer_2048
from mfpbench.pd1.benchmarks.translate_wmt import PD1translatewmt_xformer_64
from mfpbench.pd1.benchmarks.uniref50 import PD1uniref50_transformer_128

__all__ = [
    "PD1lm1b_transformer_2048",
    "PD1uniref50_transformer_128",
    "PD1translatewmt_xformer_64",
    "PD1cifar100_wideresnet_2048",
    "PD1imagenet_resnet_512",
]
