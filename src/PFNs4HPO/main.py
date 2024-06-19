import argparse
import json
import os

import torch
from pfns4hpo import bar_distribution, encoders, priors, train, utils


def main(configs):
    if configs["seq_len"] is None:
        seq_len = configs["ncurves_per_example"] * configs["max_epochs_per_curve"]
        configs["bptt"] = (
            configs["ncurves_per_example"] * configs["max_epochs_per_curve"]
        )
        hps = {
            "ncurves": configs["ncurves_per_example"],
            "nepochs": configs["max_epochs_per_curve"],
            **args.prior_hps,
        }
    else:
        seq_len = configs["seq_len"]
        configs["bptt"] = configs["seq_len"]
        hps = {**args.prior_hps}

    prior = getattr(priors.hpo, configs["prior"], None)

    if configs["load_path"] is None:
        get_batch_func = prior.get_batch
    else:
        assert (
            configs["batch_size"] == 25
        )  # priors are assumed to be stored with batch size 25
        if configs["num_gpus"] == 1:
            prior_data = priors.utils.PriorDataLoader(
                configs["load_path"],
                subsample=configs["subsample"],
                n_chunks=configs["n_chunks"],
            )
        else:
            prior_data = priors.utils.DistributedPriorDataLoader(
                configs["load_path"],
                subsample=configs["subsample"],
                n_chunks=configs["n_chunks"],
                n_gpus=configs["num_gpus"],
            )
        get_batch_func = lambda *args, **kwargs: prior_data.get_batch(
            kwargs.get("device", args[4] if len(args) >= 5 else "cpu")
        )

    num_features = configs["num_features"]

    if configs["linspace_borders"]:
        bucket_limits = torch.linspace(0.0, 1.0, 1001)
    else:
        ys_bucket = torch.zeros((seq_len, configs["border_batch_size"]))
        offset = 0
        while offset < configs["border_batch_size"]:
            print(offset)
            ys = get_batch_func(
                configs["batch_size"],
                configs["bptt"],
                num_features,
                hyperparameters=hps,
                single_eval_pos=configs["bptt"],
            )
            _, eff_batch_size = ys.target_y.shape
            ys_bucket[
                :, offset : min(offset + eff_batch_size, configs["border_batch_size"])
            ] = ys.target_y[
                :, : min(eff_batch_size, configs["border_batch_size"] - offset)
            ]
            offset += eff_batch_size

        bucket_limits = bar_distribution.get_bucket_limits(
            configs["num_borders"], ys=ys_bucket
        )

    # Discretization of the predictive distributions
    if configs["full_support"]:
        criterion = bar_distribution.FullSupportBarDistribution(bucket_limits)
    else:
        print("running without full support")
        print(f"support [{bucket_limits[0]},{bucket_limits[-1]}]")
        if bucket_limits[0] > 0.0:
            print("warning: strict positive lower bound set to 0")
            bucket_limits[0] = 0.0
        if bucket_limits[-1] < 1.0:
            print("warning: < 1.0 upper bound set to 1.0")
            bucket_limits[-1] = 1.0
        criterion = bar_distribution.BarDistribution(bucket_limits)

    configs_train = {
        _: configs[_]
        for _ in [
            "nlayers",
            "emsize",
            "epochs",
            "lr",
            "nhead",
            "aggregate_k_gradients",
            "bptt",
            "steps_per_epoch",
            "train_mixed_precision",
            "batch_size",
        ]
    }

    dataloader = priors.get_batch_to_dataloader(
        priors.get_batch_sequence(
            get_batch_func, priors.utils.sample_num_feaetures_get_batch
        )
    )

    configs_train["nhid"] = configs["emsize"] * 2
    configs_train["warmup_epochs"] = (
        configs["epochs"] // 4
        if configs["warmup_epochs"] == -1
        else configs["warmup_epochs"]
    )
    if configs["load_path"] is None:
        single_eval_pos_gen = utils.get_weighted_single_eval_pos_sampler(
            max_len=configs["bptt"],
            min_len=0,
            p=configs["power_single_eval_pos_sampler"],
        )
    else:
        single_eval_pos_gen = lambda *args, **kwargs: prior_data.get_single_eval_pos()

    configs_train.update(
        dict(
            priordataloader_class=priors.get_batch_to_dataloader(get_batch_func),
            criterion=criterion,
            encoder_generator=prior.get_encoder(),
            y_encoder_generator=encoders.get_normalized_uniform_encoder(
                encoders.Linear
            ),
            scheduler=utils.get_cosine_schedule_with_warmup,
            extra_prior_kwargs_dict={
                # "num_workers": 10,
                "num_features": num_features,
                "hyperparameters": {
                    **hps,
                },
            },
            single_eval_pos_gen=single_eval_pos_gen,
            **configs["model_extra_args"],
        )
    )

    _, _, model, _ = train.train(**configs_train)
    torch.save(model, os.path.join("final_models", configs["output_file"]))


if __name__ == "__main__":
    print("here")
    import os

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    print("WARNING: DEBUGGING")

    parser = argparse.ArgumentParser(description="LC-PFN foorrr HPO")

    parser.add_argument(
        "--warmup_epochs",
        type=int,
        required=False,
        default=-1,
        help="Number of Warmup Epochs",
    )

    parser.add_argument("--nlayers", type=int, help="Number of layers", default=12)
    parser.add_argument("--emsize", type=int, default=512, help="Size of Embeddings")
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch Size for Training"
    )
    parser.add_argument(
        "--epochs", type=int, required=True, help="Number of Training Epochs"
    )
    parser.add_argument(
        "--num_borders",
        type=int,
        default=10000,
        help="Number of borders considered in Bar distribution",
    )

    parser.add_argument(
        "--ncurves_per_example",
        type=int,
        default=50,
        help="Number of curves in one example",
    )
    parser.add_argument(
        "--max_epochs_per_curve",
        type=int,
        default=50,
        help="Maximum number of epochs for each curve in one example",
    )

    parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate")
    parser.add_argument(
        "--seq_len", type=int, required=False, default=None, help="Sequence Length"
    )
    parser.add_argument(
        "--aggregate_k_gradients", type=int, default=1, help="Step Every k Gradients"
    )
    parser.add_argument(
        "--steps_per_epoch", type=int, default=100, help="Number of Steps per Epoch"
    )
    parser.add_argument(
        "--train_mixed_precision",
        action="store_true",
        help="Enable Mixed Precision Training",
    )
    parser.add_argument(
        "--prior",
        type=str,
        required=True,
        help="Name of prior module. It has to be referenced in priors/hpo.py",
    )

    parser.add_argument(
        "--run_on_submitit",
        action="store_true",
        help="Train on compute node using slurm",
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument(
        "--time",
        type=int,
        default=23 * 60 + 55,
        help="Time budget (min) for the training",
    )
    parser.add_argument(
        "--partition", type=str, default="alldlc_gpu-rtx2080", help="Slurm partition"
    )
    parser.add_argument("--output_file", type=str, help="Trained model")

    # Added
    parser.add_argument(
        "--num_features",
        type=int,
        required=False,
        help="The total number of features for each datapoint in an example.",
        default=20,
    )
    parser.add_argument(
        "--border_batch_size",
        type=int,
        required=False,
        help="The size of the batch size used to determine the borders of the Riemann distribution.",
        default=10,
    )
    parser.add_argument(
        "--prior_hps",
        type=str,
        required=False,
        help="Hyperparameters of the prior, specified as a JSON string that will be parsed as a dict.",
        default="{}",
    )
    parser.add_argument(
        "--load_path",
        type=str,
        required=False,
        help="The path where the pre-generated prior data is stored (only works with batch_size 25 for now).",
        default=None,
    )
    parser.add_argument(
        "--power_single_eval_pos_sampler",
        type=int,
        required=False,
        help="Power of an exponential distribution to weight sampling of single eval pos.",
        default=-2,
    )
    parser.add_argument(
        "--model_extra_args",
        type=str,
        required=False,
        help="Hyperparameters of the Transformer, specified as a JSON string that will be parsed as a dict.",
        default="{}",
    )
    parser.add_argument("--nhead", type=int, default=4, help="Number of heads")
    parser.add_argument(
        "--subsample",
        type=int,
        default=1,
        help="How many times to subsample stored batch sizes, resulting in a lower effective batch size, only relevant if load_path is specified",
    )
    parser.add_argument(
        "--n_chunks",
        type=int,
        default=2_000,
        help="After this many chunks of pre-generated prior data are used, training will wrap to use the first chunk (start a new epoch), rather than the next chunk (or throwing an exception if the next chunk does not exist).",
    )
    parser.add_argument("--full_support", action="store_true")
    parser.add_argument("--no-full_support", dest="full_support", action="store_false")
    parser.set_defaults(full_support=True)
    parser.add_argument("--linspace_borders", action="store_true")
    parser.add_argument(
        "--no-linspace_borders", dest="linspace_borders", action="store_false"
    )
    parser.set_defaults(linspace_borders=False)

    args = parser.parse_args()

    args.prior_hps = json.loads(args.prior_hps)
    args.model_extra_args = json.loads(args.model_extra_args)

    main(vars(args))
