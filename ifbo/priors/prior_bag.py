from __future__ import annotations

from typing import Any

import torch

from ifbo.priors.prior import Batch
from ifbo.priors.utils import get_batch_to_dataloader
from ifbo.utils import default_device


def get_batch(
    batch_size: int,
    seq_len: int,
    num_features: int,
    hyperparameters: dict[str, Any],
    device: torch.device = default_device,
    batch_size_per_gp_sample: int | None = None,
    **kwargs: Any,
) -> Batch:
    batch_size_per_gp_sample = batch_size_per_gp_sample or (min(64, batch_size))
    num_models = batch_size // batch_size_per_gp_sample
    assert (
        num_models * batch_size_per_gp_sample == batch_size
    ), f"Batch size ({batch_size}) not divisible by batch_size_per_gp_sample ({batch_size_per_gp_sample})"

    args = {
        "device": device,
        "seq_len": seq_len,
        "num_features": num_features,
        "batch_size": batch_size_per_gp_sample,
    }

    prior_bag_priors_get_batch = hyperparameters["prior_bag_get_batch"]
    prior_bag_priors_p = [1.0] + [
        hyperparameters[f"prior_bag_exp_weights_{i}"]
        for i in range(1, len(prior_bag_priors_get_batch))
    ]

    weights = torch.tensor(prior_bag_priors_p, dtype=torch.float)  # create a tensor of weights
    batch_assignments = torch.multinomial(
        torch.softmax(weights, 0), num_models, replacement=True
    ).numpy()

    if "verbose" in hyperparameters and hyperparameters["verbose"]:
        print(
            "PRIOR_BAG:",
            weights,
            batch_assignments,
            num_models,
            batch_size_per_gp_sample,
            batch_size,
        )
        sample: list[Batch] = [
            prior_bag_priors_get_batch[int(prior_idx)](
                hyperparameters=hyperparameters, **args, **kwargs
            )
            for prior_idx in batch_assignments
        ]

    def merge(sample: list[Batch], k: str) -> Any:
        x = [getattr(x_, k) for x_ in sample]
        if torch.is_tensor(x[0]):
            return torch.cat(x, 1).detach()
        else:
            return [*x]

    merged_sample = {k: merge(sample, k) for k in sample[0].other_filled_attributes(set())}
    if hyperparameters.get("verbose"):
        print({k: v.shape for k, v in merged_sample.items()})

    return Batch(**merged_sample)


if __name__ == "__main__":
    DataLoader = get_batch_to_dataloader(get_batch)
