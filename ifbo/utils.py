import argparse
from dataclasses import dataclass
import datetime
import itertools
import math
import numpy as np
import os
import random

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from typing import Tuple, List, Dict, Set, Optional

from ifbo.priors.prior import Batch
from ifbo.bar_distribution import BarDistribution


@dataclass
class Curve:
    hyperparameters: torch.Tensor
    t: torch.Tensor
    y: Optional[torch.Tensor] = None


@dataclass(unsafe_hash=True)
class PredictionResult:
    logits: torch.Tensor
    criterion: BarDistribution

    @torch.no_grad()
    def likelihood(self, y_test):
        return -self.criterion(self.logits, y_test).squeeze(1)

    @torch.no_grad()
    def ucb(self):
        return self.criterion.ucb(self.logits, best_f=None).squeeze(1)

    @torch.no_grad()
    def ei(self, y_best):
        return self.criterion.ei(self.logits, f_best=y_best).squeeze(1)

    @torch.no_grad()
    def pi(self, y_best):
        return self.criterion.pi(self.logits, f_best=y_best).squeeze(1)
    
    @torch.no_grad()
    def quantile(self, q):
        return self.criterion.icdf(self.logits, q).squeeze(1)


# copied from huggingface
def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1
):
    """Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# copied from huggingface
def get_restarting_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    steps_per_restart,
    num_cycles=0.5,
    last_epoch=-1,
):
    assert num_training_steps % steps_per_restart == 0

    def inner_lr_lambda(current_step, num_warmup_steps, num_training_steps):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    def lr_lambda(current_step):
        inner_step = current_step % steps_per_restart
        return inner_lr_lambda(
            inner_step,
            num_warmup_steps if current_step < steps_per_restart else 0,
            steps_per_restart,
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# copied from huggingface
def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_openai_lr(transformer_model):
    num_params = sum(p.numel() for p in transformer_model.parameters())
    return 0.003239 - 0.0001395 * math.log(num_params)


def get_weighted_single_eval_pos_sampler(max_len, min_len=0, p=1.0):
    """
    This gives a sampler that can be used for `single_eval_pos` which yields good performance for all positions p,
    where p <= `max_len`. At most `max_len` - 1 examples are shown to the Transformer.
    :return: Sampler that can be fed to `train()` as `single_eval_pos_gen`.
    """
    return lambda: random.choices(
        range(min_len, max_len),
        [1 / math.pow(((max_len - min_len) - i), p) for i in range(max_len - min_len)],
    )[0]


def get_uniform_single_eval_pos_sampler(max_len, min_len=0):
    """
    Just sample any evaluation position with the same weight
    :return: Sampler that can be fed to `train()` as `single_eval_pos_gen`.
    """
    return lambda: random.choices(range(min_len, max_len))[0]


class SeqBN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_model)
        self.d_model = d_model

    def forward(self, x):
        assert self.d_model == x.shape[-1]
        flat_x = x.view(-1, self.d_model)
        flat_x = self.bn(flat_x)
        return flat_x.view(*x.shape)


def set_locals_in_self(locals):
    """
    Call this function like `set_locals_in_self(locals())` to set all local variables as object variables.
    Especially useful right at the beginning of `__init__`.
    :param locals: `locals()`
    """
    self = locals["self"]
    for var_name, val in locals.items():
        if var_name != "self":
            setattr(self, var_name, val)


default_device = "cuda:0" if torch.cuda.is_available() else "cpu:0"


# Copied from StackOverflow, but we do an eval on the values additionally
class StoreDictKeyPair(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDictKeyPair, self).__init__(
            option_strings, dest, nargs=nargs, **kwargs
        )

    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values:
            k, v = kv.split("=")
            try:
                my_dict[k] = eval(v)
            except NameError:
                my_dict[k] = v
        setattr(namespace, self.dest, my_dict)
        print("dict values: {}".format(my_dict))


def get_nan_value(v, set_value_to_nan=1.0):
    if random.random() < set_value_to_nan:
        return v
    else:
        return random.choice([-999, 0, 1, 999])


def to_ranking(data):
    x = data >= data.unsqueeze(-3)
    x = x.sum(0)
    return x


# TODO: Is there a better way to do this?
#   1. Cmparing to unique elements: When all values are different we still get quadratic blowup
#   2. Argsort(Argsort()) returns ranking, but with duplicate values there is an ordering which is problematic
#   3. Argsort(Argsort(Unique))->Scatter seems a bit complicated, doesn't have quadratic blowup, but how fast?
def to_ranking_low_mem(data):
    x = torch.zeros_like(data)
    for col in range(data.shape[-1]):
        x_ = data[:, :, col] >= data[:, :, col].unsqueeze(-2)
        x_ = x_.sum(0)
        x[:, :, col] = x_
    return x


def nan_handling_missing_for_unknown_reason_value(nan_prob=1.0):
    return get_nan_value(float("nan"), nan_prob)


def nan_handling_missing_for_no_reason_value(nan_prob=1.0):
    return get_nan_value(float("-inf"), nan_prob)


def nan_handling_missing_for_a_reason_value(nan_prob=1.0):
    return get_nan_value(float("inf"), nan_prob)


def torch_nanmean(x, axis=0, return_nanshare=False):
    num = torch.where(torch.isnan(x), torch.full_like(x, 0), torch.full_like(x, 1)).sum(
        axis=axis
    )
    value = torch.where(torch.isnan(x), torch.full_like(x, 0), x).sum(axis=axis)
    if return_nanshare:
        return value / num, 1.0 - num / x.shape[axis]
    return value / num


def torch_nanstd(x, axis=0):
    num = torch.where(torch.isnan(x), torch.full_like(x, 0), torch.full_like(x, 1)).sum(
        axis=axis
    )
    value = torch.where(torch.isnan(x), torch.full_like(x, 0), x).sum(axis=axis)
    mean = value / num
    mean_broadcast = torch.repeat_interleave(
        mean.unsqueeze(axis), x.shape[axis], dim=axis
    )
    return torch.sqrt(
        torch.nansum(torch.square(mean_broadcast - x), axis=axis) / (num - 1)
    )


def normalize_data(data, normalize_positions=-1, return_scaling=False):
    if normalize_positions > 0:
        mean = torch_nanmean(data[:normalize_positions], axis=0)
        std = torch_nanstd(data[:normalize_positions], axis=0) + 0.000001
    else:
        mean = torch_nanmean(data, axis=0)
        std = torch_nanstd(data, axis=0) + 0.000001
    data = (data - mean) / std
    data = torch.clip(data, min=-100, max=100)

    if return_scaling:
        return data, (mean, std)
    return data


def remove_outliers(X, n_sigma=4, normalize_positions=-1):
    # Expects T, B, H
    assert len(X.shape) == 3, "X must be T,B,H"
    # for b in range(X.shape[1]):
    # for col in range(X.shape[2]):
    data = X if normalize_positions == -1 else X[:normalize_positions]
    data_clean = data[:].clone()
    data_mean, data_std = torch_nanmean(data, axis=0), torch_nanstd(data, axis=0)
    cut_off = data_std * n_sigma
    lower, upper = data_mean - cut_off, data_mean + cut_off

    data_clean[torch.logical_or(data_clean > upper, data_clean < lower)] = np.nan
    data_mean, data_std = (
        torch_nanmean(data_clean, axis=0),
        torch_nanstd(data_clean, axis=0),
    )
    cut_off = data_std * n_sigma
    lower, upper = data_mean - cut_off, data_mean + cut_off

    X = torch.maximum(-torch.log(1 + torch.abs(X)) + lower, X)
    X = torch.minimum(torch.log(1 + torch.abs(X)) + upper, X)
    # print(ds[1][data < lower, col], ds[1][data > upper, col], ds[1][~np.isnan(data), col].shape, data_mean, data_std)
    return X


def bool_mask_to_att_mask(mask):
    return (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )


def print_on_master_only(is_master):
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_dist(device):
    print("init dist")
    if "LOCAL_RANK" in os.environ:
        # launched with torch.distributed.launch
        rank = int(os.environ["LOCAL_RANK"])
        print("torch.distributed.launch and my rank is", rank)
        torch.cuda.set_device(rank)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(seconds=20),
            world_size=torch.cuda.device_count(),
            rank=rank,
        )
        torch.distributed.barrier()
        print_on_master_only(rank == 0)
        print(
            f"Distributed training on {torch.cuda.device_count()} GPUs, this is rank {rank}, "
            "only I can print, but when using print(..., force=True) it will print on all ranks."
        )
        return True, rank, f"cuda:{rank}"
    elif "SLURM_PROCID" in os.environ and torch.cuda.device_count() > 1:
        # this is for multi gpu when starting with submitit
        assert device != "cpu:0"
        rank = int(os.environ["SLURM_PROCID"])
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        torch.cuda.set_device(rank)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
        print("distributed submitit launch and my rank is", rank)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(seconds=20),
            world_size=torch.cuda.device_count(),
            rank=rank,
        )
        torch.distributed.barrier()
        print_on_master_only(rank == 0)
        print(
            f"Distributed training on {torch.cuda.device_count()} GPUs, this is rank {rank}, "
            "only I can print, but when using print(..., force=True) it will print on all ranks."
        )

        return True, rank, f"cuda:{rank}"
    else:
        print("Not using distributed")
        # will not change any of the behavior of print, but allows putting the force=True in the print calls
        print_on_master_only(True)
        return False, 0, device


# NOP decorator for python with statements (x = NOP(); with x:)
class NOP:
    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        pass


def check_compatibility(dl):
    if hasattr(dl, "num_outputs"):
        print(
            "`num_outputs` for the DataLoader is deprecated. It is assumed to be 1 from now on."
        )
        assert dl.num_outputs != 1, (
            "We assume num_outputs to be 1. Instead of the num_ouputs change your loss."
            "We specify the number of classes in the CE loss."
        )


def product_dict(dic):
    keys = dic.keys()
    vals = dic.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def to_tensor(x, device=None):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return torch.tensor(x, device=device)


printed_already = set()


def print_once(*msgs: str):
    msg = " ".join([repr(m) for m in msgs])
    if msg not in printed_already:
        print(msg)
        printed_already.add(msg)


def tokenize(
    context: List[Curve], query: List[Curve], device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # takes as input a list of curves and query points (does not have y values)
    # returns the tokenized representation of
    #   - context curves: ([id curve, t value, hyperparameters]) and the corresponding y values.
    #   - query points: ([id curve, t value, hyperparameters])
    # The id curve is a unique identifier for each curve in the context.

    config_to_id: Dict[torch.Tensor, int] = {}
    context_tokens = []
    context_y_values = []
    query_tokens = []
    current_id = 1

    def get_curve_id(hyperparameters: torch.Tensor) -> int:
        nonlocal current_id
        for config, cid in config_to_id.items():
            if torch.equal(config, hyperparameters):
                return cid
        config_to_id[hyperparameters] = current_id
        current_id += 1
        return config_to_id[hyperparameters]

    for curve in context:
        curve_id = get_curve_id(curve.hyperparameters)
        num_points = curve.t.size(0)
        for i in range(num_points):
            context_tokens.append(
                torch.cat(
                    (torch.tensor([curve_id, curve.t[i].item()]), curve.hyperparameters.cpu())
                )
            )
            context_y_values.append(curve.y[i])

    for curve in query:
        curve_id = get_curve_id(curve.hyperparameters)
        num_points = curve.t.size(0)
        for i in range(num_points):
            query_tokens.append(
                torch.cat(
                    (torch.tensor([curve_id, curve.t[i].item()]), curve.hyperparameters.cpu())
                )
            )

    # Convert lists to tensors
    context_tokens_tensor = torch.stack(context_tokens, dim=0).to(device)
    context_y_values_tensor = torch.stack(context_y_values, dim=0).to(device)
    query_tokens_tensor = torch.stack(query_tokens, dim=0).to(device)

    return context_tokens_tensor, context_y_values_tensor, query_tokens_tensor


def detokenize(batch: Batch, context_size: int, device: Optional[torch.device] = None) -> Tuple[List[Curve], List[Curve]]:
    (
        context_tokens_tensor,
        context_y_values_tensor,
        query_tokens_tensor,
        query_y_values_tensor,
    ) = (
        batch.x.squeeze(1)[:context_size, ...],
        batch.y.squeeze(1)[:context_size, ...],
        batch.x.squeeze(1)[context_size:, ...],
        batch.y.squeeze(1)[context_size:, ...],
    )
    id_to_config: Dict[int, torch.Tensor] = {}
    context_curves: Dict[int, List[Tuple[float, float]]] = {}
    query_curves: Dict[int, List[float]] = {}
    used_ids: Set[int] = set()
    all_possible_ids: Set[int] = set(range(1, 1001))

    def get_curve_config(curve_id: int) -> torch.Tensor:
        if curve_id in id_to_config:
            return id_to_config[curve_id]
        else:
            raise KeyError(f"Curve ID {curve_id} not found in id_to_config")

    # Process context tokens and y values
    for i in range(context_tokens_tensor.size(0)):
        token = context_tokens_tensor[i]
        y_value = context_y_values_tensor[i]

        curve_id = int(token[0].item())
        x_value = token[1].item()
        configuration = token[2:]

        if curve_id not in id_to_config:
            id_to_config[curve_id] = configuration
            used_ids.add(curve_id)

        if curve_id not in context_curves:
            context_curves[curve_id] = []

        context_curves[curve_id].append((x_value, y_value.item()))

    unused_ids = all_possible_ids - used_ids

    # Process query tokens
    for i in range(query_tokens_tensor.size(0)):
        token = query_tokens_tensor[i]
        y_value = (
            query_y_values_tensor[i]
            if query_y_values_tensor is not None
            else torch.tensor([0.0]) * float("nan")
        )

        curve_id = int(token[0].item())
        x_value = token[1].item()
        configuration = token[2:]

        # Assign a new unique ID for configurations with curve_id 0 not in context tokens
        if curve_id not in used_ids:
            found = False
            for existing_id, config in id_to_config.items():
                if torch.equal(config, configuration):
                    curve_id = existing_id
                    found = True
                    break
            if not found:
                if not unused_ids:
                    raise ValueError("No unused IDs available")
                curve_id = unused_ids.pop()
                id_to_config[curve_id] = configuration

        if curve_id not in query_curves:
            query_curves[curve_id] = []

        query_curves[curve_id].append([x_value, y_value.item()])

    # Convert the context curves dictionary to list of Curve objects
    context_list = []
    for curve_id, points in context_curves.items():
        x_values = torch.tensor([p[0] for p in points]).to(device)
        y_values = torch.tensor([p[1] for p in points]).to(device)
        configuration = get_curve_config(curve_id)
        context_list.append(
            Curve(t=x_values, y=y_values, hyperparameters=configuration)
        )

    # Convert the query curves dictionary to list of Curve objects
    query_list = []
    for curve_id, points in query_curves.items():
        x_values = torch.tensor([p[0] for p in points]).to(device)
        if query_y_values_tensor is not None:
            y_values = torch.tensor([p[1] for p in points]).to(device)
        configuration = get_curve_config(curve_id)
        query_list.append(
            Curve(
                t=x_values,
                hyperparameters=configuration,
                y=y_values if query_y_values_tensor is not None else None,
            )
        )

    return context_list, query_list
