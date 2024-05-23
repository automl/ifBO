from __future__ import annotations

from mfpbench.benchmark import Benchmark
from mfpbench.config import Config, TabularConfig
from mfpbench.get import _mapping, get
from mfpbench.jahs import JAHSBenchmark
from mfpbench.lcbench_tabular import (
    LCBenchTabularBenchmark,
    LCBenchTabularConfig,
    LCBenchTabularResult,
)
from mfpbench.metric import Metric
from mfpbench.pd1 import (
    PD1Benchmark,
    PD1cifar100_wideresnet_2048,
    PD1imagenet_resnet_512,
    PD1lm1b_transformer_2048,
    PD1translatewmt_xformer_64,
    PD1uniref50_transformer_128,
)
from mfpbench.result import Result
from mfpbench.synthetic.hartmann import (
    MFHartmann3Benchmark,
    MFHartmann3BenchmarkBad,
    MFHartmann3BenchmarkGood,
    MFHartmann3BenchmarkModerate,
    MFHartmann3BenchmarkTerrible,
    MFHartmann6Benchmark,
    MFHartmann6BenchmarkBad,
    MFHartmann6BenchmarkGood,
    MFHartmann6BenchmarkModerate,
    MFHartmann6BenchmarkTerrible,
    MFHartmannBenchmark,
)
from mfpbench.tabular import TabularBenchmark
from mfpbench.yahpo import (
    IAMLglmnetBenchmark,
    IAMLrangerBenchmark,
    IAMLrpartBenchmark,
    IAMLSuperBenchmark,
    IAMLxgboostBenchmark,
    LCBenchBenchmark,
    NB301Benchmark,
    RBV2aknnBenchmark,
    RBV2glmnetBenchmark,
    RBV2rangerBenchmark,
    RBV2rpartBenchmark,
    RBV2SuperBenchmark,
    RBV2svmBenchmark,
    RBV2xgboostBenchmark,
    YAHPOBenchmark,
)

__all__ = [
    "Benchmark",
    "Result",
    "get",
    "JAHSBenchmark",
    "YAHPOBenchmark",
    "PD1Benchmark",
    "TabularBenchmark",
    "Config",
    "TabularConfig",
    "MFHartmannBenchmark",
    "MFHartmann3Benchmark",
    "MFHartmann6Benchmark",
    "LCBenchTabularBenchmark",
    "LCBenchTabularConfig",
    "LCBenchTabularResult",
    "IAMLglmnetBenchmark",
    "IAMLrangerBenchmark",
    "IAMLrpartBenchmark",
    "IAMLSuperBenchmark",
    "IAMLxgboostBenchmark",
    "LCBenchBenchmark",
    "NB301Benchmark",
    "RBV2aknnBenchmark",
    "RBV2glmnetBenchmark",
    "RBV2rangerBenchmark",
    "RBV2rpartBenchmark",
    "RBV2SuperBenchmark",
    "RBV2svmBenchmark",
    "RBV2xgboostBenchmark",
    "MFHartmann3BenchmarkBad",
    "MFHartmann3BenchmarkGood",
    "MFHartmann3BenchmarkModerate",
    "MFHartmann3BenchmarkTerrible",
    "MFHartmann6BenchmarkBad",
    "MFHartmann6BenchmarkGood",
    "MFHartmann6BenchmarkModerate",
    "MFHartmann6BenchmarkTerrible",
    "JAHSBenchmark",
    "PD1cifar100_wideresnet_2048",
    "PD1imagenet_resnet_512",
    "PD1lm1b_transformer_2048",
    "PD1translatewmt_xformer_64",
    "PD1uniref50_transformer_128",
    "Metric",
    "_mapping",
]
