| key | Surrogate | PFN Prior | Acquisition | Continuous | Notes |
| -------- | -------- | -------- | --- | -------- | --- |
| `mf_ei_bo`    | GP   | - | MFEI (DyHPO)     |  ✅     | Write something...     |
| `dyhpo-arlind`    | GP   | - | MFEI (DyHPO)     |  :x:   | Arlind's __original__ code     |
| `dyhpo-neps`   | DeepGP  | -  | MFEI (DyHPO)     |  ✅     | DEPRECATED NePS version of Arlind's DyHPO |
| `dyhpo-neps-v2`   | DeepGP  | -  | MFEI (DyHPO)     |  ✅     | FINAL NePS version of Arlind's DyHPO |
| `pfn_bnn3-ei`   | PFNs | BNN Priors (*bnn3_1000curves_10params_2M*) | MF-EI (DyHPO)     |  ✅     | Our method |
| `pfn-lcnet-ei`   | PFNs  | LCNet Prior (*lcnet_prior_10feat_1M_24M*) | MF-EI (DyHPO)     |  ✅     | Our method |
| `dyhpo-neps-max`  | DeepGP  | - | MFEI-max    |  ✅     | Calculates EI at max against the global incumbent |
| `dyhpo-neps-ucb`  | DeepGP  | - | MF-UCB    |  ✅     | Calculates EUCB with DyHPO like acquisition design |
| `dyhpo-neps-ucb-max`  | DeepGP  | - | MF-UCB-max    |  ✅     | Calculates UCB at max against the global incumbent |
| `dyhpo-neps-ucb-dyna`  | DeepGP  | - | MF-UCB-dyna    |  ✅     | Calculates UCB at a dynamic z_max decided per config in the candidate set |
| `dyhpo-neps-mf-2step`  | DeepGP  | - | MF-2step    |  ✅     | Filters new and partial candidates using a dynamic-MF-EI and then uses dynamic-MF-UCB |
