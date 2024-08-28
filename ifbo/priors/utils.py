from __future__ import annotations

from collections.abc import Callable
from collections.abc import Generator
from functools import partial
import inspect
import math
import os
import random
import sys
import time
import types
from typing import Any

import cloudpickle
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import submitit
import torch
from torch import nn

from ifbo.priors.prior import Batch
from ifbo.utils import normalize_data
from ifbo.utils import set_locals_in_self


def get_uniform_sampler(min_eval_pos: int, seq_len: int) -> Callable[[], int]:
    print(f"Using this sampler single_eval_pos = {min_eval_pos} is equally likely as {seq_len-1}")

    def foo() -> int:
        return np.random.randint(min_eval_pos, seq_len)

    return foo


def get_expon_sep_sampler(base: int, min_eval_pos: int, seq_len: int) -> Callable[[], int]:
    p_levels = np.array([np.power(base, i) for i in range(seq_len - min_eval_pos)])
    p_levels /= p_levels.sum()
    print(
        f"Using this sampler single_eval_pos = {min_eval_pos} is {p_levels[0]/p_levels[-1]} times more likely than {seq_len-1}"
    )

    def foo() -> int:
        return np.random.choice(seq_len - min_eval_pos, p=p_levels) + min_eval_pos

    return foo


class PriorDataLoader:
    def _load_chunk(self, chunk_id: int) -> None:
        if self.partition:
            partition_id = chunk_id // 1000
            chunk_file = os.path.join(
                self.path, f"partition_{partition_id}", f"chunk_{chunk_id}.pkl"
            )
        else:
            chunk_file = os.path.join(self.path, f"chunk_{chunk_id}.pkl")
        with open(chunk_file, "rb") as f:
            self.loaded_chunk = cloudpickle.load(f)
        self.loaded_chunk_id = chunk_id
        self.batch_counter = 0
        self.subsample_counter = 0

    def __init__(
        self,
        load_path: str,
        n_chunks: int = 2_000,
        store: bool = False,
        subsample: int = 1,
        partition: bool | None = None,
    ) -> None:
        self.path = load_path
        if store:
            self.partition = True  # new collections are partitioned
        elif partition is None:
            # check whether the current directory contains a directory partition_0
            self.partition = os.path.isdir(os.path.join(self.path, "partition_0"))
        else:
            self.partition = partition
        if not store:
            self._load_chunk(0)

        self.n_chunks = n_chunks
        self.subsample = subsample

    def get_batch(self, device: torch.device | None) -> Batch:
        if self.subsample == 1:
            _, batch_data = self.loaded_chunk[self.batch_counter]
            batch_data.x = batch_data.x.to(device)
            batch_data.y = batch_data.y.to(device)
            batch_data.target_y = batch_data.target_y.to(device)
            self.batch_counter += 1
            if self.batch_counter >= len(self.loaded_chunk):
                self._load_chunk((self.loaded_chunk_id + 1) % self.n_chunks)
        else:
            _, full_batch_data = self.loaded_chunk[self.batch_counter]
            seq_len, batch_size = full_batch_data.y.shape
            subsample_size = batch_size // self.subsample
            if self.subsample_counter < self.subsample - 1:
                low = subsample_size * self.subsample_counter
                high = subsample_size * (self.subsample_counter + 1)
                batch_data = Batch(
                    full_batch_data.x[:, low:high, :].to(device),
                    full_batch_data.y[:, low:high].to(device),
                    full_batch_data.target_y[:, low:high].to(device),
                )
                self.subsample_counter += 1
            else:
                low = subsample_size * self.subsample_counter
                batch_data = Batch(
                    full_batch_data.x[:, low:, :].to(device),
                    full_batch_data.y[:, low:].to(device),
                    full_batch_data.target_y[:, low:].to(device),
                )
                self.subsample_counter = 0
                self.batch_counter += 1
                if self.batch_counter >= len(self.loaded_chunk):
                    self._load_chunk((self.loaded_chunk_id + 1) % self.n_chunks)

        return batch_data

    def get_single_eval_pos(self) -> int:
        single_eval_pos, _ = self.loaded_chunk[self.batch_counter]
        if single_eval_pos == 1000:
            print(
                "WARNING: as a TEMP hack single eval pos = 1000 is manually corrected to 999",
                file=sys.stderr,
            )
            single_eval_pos = 999
        return single_eval_pos

    def store_prior(
        self,
        prior: Any,
        local: bool = False,
        chunk_size: int = 1_000,
        batch_size: int = 25,
        seq_len: int = 1_000,
        n_features: int = 12,
        prior_hyperparameters: dict[str, Any] = {},
        partition: str = "gki_cpu-cascadelake",
        eval_pos_sampler: Callable[[], int] | None = None,
    ) -> None:
        # generate batches in parallel and store them for efficient training
        assert chunk_size % batch_size == 0

        def store_batch(
            path: str,
            chunk_id: int,
            chunk_size: int,
            batch_size: int,
            seq_len: int,
            n_features: int,
            partition: bool,
            prior_hyperparameters: dict[str, Any],
        ) -> None:
            if partition:
                partition_id = chunk_id // 1000
                chunk_dir = os.path.join(self.path, f"partition_{partition_id}")
                chunk_file = os.path.join(chunk_dir, f"chunk_{chunk_id}.pkl")
            else:
                chunk_file = os.path.join(self.path, f"chunk_{chunk_id}.pkl")
            if not os.path.exists(chunk_file):
                np.random.seed((os.getpid() * int(time.time())) % 123456789)
                chunk_data = []
                for bid in range(chunk_size // batch_size):
                    if eval_pos_sampler is None:
                        # sample single eval pos log-uniformly ({1, ..., seq_len} log-uniformly - 1)
                        single_eval_pos = int(
                            np.floor(np.exp(np.random.uniform(0, np.log(seq_len + 1)))) - 1
                        )
                    else:
                        single_eval_pos = eval_pos_sampler()
                    assert single_eval_pos < seq_len
                    b = prior.get_batch(  # type: ignore
                        batch_size=batch_size,
                        single_eval_pos=single_eval_pos,
                        seq_len=seq_len,
                        num_features=n_features,
                        hyperparameters=prior_hyperparameters,
                    )
                    chunk_data.append((single_eval_pos, b))
                with open(chunk_file, "wb") as file:
                    cloudpickle.dump(chunk_data, file)
            else:
                print("Already done.")

        if partition:
            for partition_id in range(self.n_chunks // 1000):
                chunk_dir = os.path.join(self.path, f"partition_{partition_id}")
                if not os.path.exists(chunk_dir):
                    print(f"Creating directory: {chunk_dir}")
                    os.makedirs(chunk_dir)

        kwargss = [
            {
                "path": self.path,
                "chunk_id": i,
                "chunk_size": chunk_size,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "n_features": n_features,
                "partition": self.partition,
                "prior_hyperparameters": prior_hyperparameters,
            }
            for i in range(0, self.n_chunks)
        ]

        # check how long a task takes & if too long run them on submitit
        runonsubmitit = False
        done = 0
        for kwargs in kwargss:
            print(f"Calculating the {kwargs['chunk_id']}th chunk...")
            before = time.time()
            store_batch(**kwargs)  # type: ignore
            duration = time.time() - before
            print(f"Done, took {duration}s")
            done += 1
            if not local and duration > 5:
                # run stuff on submitit
                runonsubmitit = True
                # allocate 3x more (+ 3min) time to avoid timeouts
                tlimit = int(3 * duration / 60.0 + 1)
                print(f"Generating remaining chunks using submitit with time limit {tlimit}min")
                break

        if runonsubmitit:
            executor = submitit.get_executor(folder="/tmp/")
            executor.update_parameters(
                time=tlimit,
                # partition="alldlc_gpu-rtx2080",
                partition=partition,
                cpus_per_task=1,
                slurm_gres="gpu:0",
            )

            kwargss = kwargss[done:]
            job_name = os.path.basename(self.path)
            print(job_name, len(kwargss))
            job_group = executor.submit_group(job_name, store_batch, kwargss)
            print(job_group)


def get_rank() -> int:
    if "LOCAL_RANK" in os.environ:
        # launched with torch.distributed.launch
        rank = int(os.environ["LOCAL_RANK"])
        return rank
    elif "SLURM_PROCID" in os.environ and torch.cuda.device_count() > 1:
        # this is for multi gpu when starting with submitit
        rank = int(os.environ["SLURM_PROCID"])
        return rank
    else:
        # not using a distributed setting, assume rank 0
        return 0


class DistributedPriorDataLoader(PriorDataLoader):
    def __init__(
        self,
        load_path: str,
        n_gpus: int = 1,
        n_chunks: int = 2_000,
        store: bool = False,
        subsample: int = 1,
        partition: bool | None = None,
    ):
        self.path = load_path
        if store:
            self.partition = True  # new collections are partitioned
        elif partition is None:
            # check whether the current directory contains a directory partition_0
            self.partition = os.path.isdir(os.path.join(self.path, "partition_0"))
        else:
            self.partition = partition
        if not store:
            self.n_gpus = n_gpus
            self.loaded_chunk = (
                None  # lazy load, as the rank of the process may be unknown on initialization
            )

        self.n_chunks = n_chunks
        self.subsample = subsample
        self.rank: int

    def data_sync(self) -> None:
        # lazy loading or reloading in case this object is used by multiple processes (should be avoided)
        if self.loaded_chunk is None or self.rank != get_rank():
            self.rank = get_rank()
            offset = self.rank * self.n_chunks // self.n_gpus
            self._load_chunk(offset)

    def get_batch(self, device: torch.device | None) -> Batch:
        self.data_sync()

        if self.subsample == 1:
            _, batch_data = self.loaded_chunk[self.batch_counter]
            batch_data.x = batch_data.x.to(device)
            batch_data.y = batch_data.y.to(device)
            batch_data.target_y = batch_data.target_y.to(device)
            self.batch_counter += 1
            if self.batch_counter >= len(self.loaded_chunk):
                self._load_chunk((self.loaded_chunk_id + 1) % self.n_chunks)
        else:
            _, full_batch_data = self.loaded_chunk[self.batch_counter]
            seq_len, batch_size = full_batch_data.y.shape
            subsample_size = batch_size // self.subsample
            if self.subsample_counter < self.subsample - 1:
                low = subsample_size * self.subsample_counter
                high = subsample_size * (self.subsample_counter + 1)
                batch_data = Batch(
                    full_batch_data.x[:, low:high, :].to(device),
                    full_batch_data.y[:, low:high].to(device),
                    full_batch_data.target_y[:, low:high].to(device),
                )
                self.subsample_counter += 1
            else:
                low = subsample_size * self.subsample_counter
                batch_data = Batch(
                    full_batch_data.x[:, low:, :].to(device),
                    full_batch_data.y[:, low:].to(device),
                    full_batch_data.target_y[:, low:].to(device),
                )
                self.subsample_counter = 0
                self.batch_counter += 1
                if self.batch_counter >= len(self.loaded_chunk):
                    self._load_chunk((self.loaded_chunk_id + 1) % self.n_chunks)

        return batch_data

    def get_single_eval_pos(self) -> int:
        self.data_sync()

        single_eval_pos, _ = self.loaded_chunk[self.batch_counter]
        if single_eval_pos == 1000:
            print(
                "WARNING: as a TEMP hack single eval pos = 1000 is manually corrected to 999",
                file=sys.stderr,
            )
            single_eval_pos = 999
        return single_eval_pos


def get_batch_to_dataloader(get_batch_method_: Callable) -> PriorDataLoader:
    # DL = partial(DL, get_batch_method=get_batch_method_)
    class DL(PriorDataLoader):
        get_batch_method = get_batch_method_

        # Caution, you might need to set self.num_features manually if it is not part of the args.
        def __init__(self, num_steps: int, **get_batch_kwargs: Any) -> None:
            set_locals_in_self(locals())

            # The stuff outside the or is set as class attribute before instantiation.
            self.num_features: int = get_batch_kwargs.get("num_features") or self.num_features
            self.epoch_count = 0
            print("DataLoader.__dict__", self.__dict__)
            self.num_steps: int
            self.get_batch_kwargs: dict[str, Any]

        @staticmethod
        def gbm(
            *args: Any, eval_pos_seq_len_sampler: Callable[[], tuple[int, int]], **kwargs: Any
        ) -> Batch:
            kwargs["single_eval_pos"], kwargs["seq_len"] = eval_pos_seq_len_sampler()
            # Scales the batch size dynamically with the power of 'dynamic_batch_size'.
            # A transformer with quadratic memory usage in the seq len would need a power of 2 to keep memory constant.
            if (
                "dynamic_batch_size" in kwargs
                and kwargs["dynamic_batch_size"] > 0
                and kwargs["dynamic_batch_size"] is not None
            ):
                kwargs["batch_size"] = kwargs["batch_size"] * math.floor(
                    math.pow(kwargs["seq_len_maximum"], kwargs["dynamic_batch_size"])
                    / math.pow(kwargs["seq_len"], kwargs["dynamic_batch_size"])
                )
            batch: Batch = get_batch_method_(*args, **kwargs)
            if batch.single_eval_pos is None:
                batch.single_eval_pos = kwargs["single_eval_pos"]
            return batch

        def __len__(self) -> int:
            return self.num_steps

        def get_test_batch(self, **kwargs: Any) -> Batch:  # does not increase epoch_count
            return self.gbm(
                **self.get_batch_kwargs,
                epoch=self.epoch_count,
                model=self.model if hasattr(self, "model") else None,
                **kwargs,
            )

        def __iter__(self) -> Generator:
            assert hasattr(
                self, "model"
            ), "Please assign model with `dl.model = ...` before training."
            self.epoch_count += 1
            return iter(
                self.gbm(**self.get_batch_kwargs, epoch=self.epoch_count - 1, model=self.model)
                for _ in range(self.num_steps)
            )

    return DL  # type: ignore


def plot_features(
    data: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    fig: plt.Figure | None = None,
    categorical: bool = True,
    plot_diagonal: bool = True,
) -> None:
    import seaborn as sns

    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

    fig2 = fig if fig else plt.figure(figsize=(8, 8))
    spec2 = gridspec.GridSpec(ncols=data.shape[1], nrows=data.shape[1], figure=fig2)
    for d in range(0, data.shape[1]):
        for d2 in range(0, data.shape[1]):
            if d > d2:
                continue
            sub_ax = fig2.add_subplot(spec2[d, d2])
            sub_ax.set_xticks([])
            sub_ax.set_yticks([])
            if d == d2:
                if plot_diagonal:
                    if categorical:
                        sns.histplot(
                            data[:, d], hue=targets[:], ax=sub_ax, legend=False, palette="deep"
                        )
                    else:
                        sns.histplot(data[:, d], ax=sub_ax, legend=False)
                sub_ax.set(ylabel=None)
            else:
                if categorical:
                    sns.scatterplot(
                        x=data[:, d], y=data[:, d2], hue=targets[:], legend=False, palette="deep"
                    )
                else:
                    sns.scatterplot(x=data[:, d], y=data[:, d2], hue=targets[:], legend=False)
    fig2.show()


def trunc_norm_sampler_f(mu: float, sigma: float) -> Callable[[], float]:
    return lambda: stats.truncnorm(
        (0 - mu) / sigma, (1000000 - mu) / sigma, loc=mu, scale=sigma
    ).rvs(1)[0]


def beta_sampler_f(a: float, b: float) -> Callable[[], float]:
    return lambda: np.random.beta(a, b)


def gamma_sampler_f(a: float, b: float) -> Callable[[], float]:
    return lambda: np.random.gamma(a, b)


def uniform_sampler_f(a: float, b: float) -> Callable[[], float]:
    return lambda: np.random.uniform(a, b)


def uniform_int_sampler_f(a: float, b: float) -> Callable[[], float]:
    return lambda: round(np.random.uniform(a, b))


def zipf_sampler_f(a: float, b: float, c: float) -> Callable[[], int]:
    x = np.arange(b, c)
    weights = x ** (-a)
    weights /= weights.sum()
    return lambda: stats.rv_discrete(name="bounded_zipf", values=(x, weights)).rvs(1)


def scaled_beta_sampler_f(a: float, b: float, scale: float, minimum: float) -> Callable[[], float]:
    return lambda: minimum + round(beta_sampler_f(a, b)() * (scale - minimum))


def normalize_by_used_features_f(
    x: torch.Tensor, num_features_used: int, num_features: int, normalize_with_sqrt: bool = False
) -> torch.Tensor:
    if normalize_with_sqrt:
        return x / (num_features_used / num_features) ** (1 / 2)
    return x / (num_features_used / num_features)


def order_by_y(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    order = torch.argsort(y if random.randint(0, 1) else -y, dim=0)[:, 0, 0]
    order = order.reshape(2, -1).transpose(0, 1).reshape(-1)  # .reshape(seq_len)
    x = x[order]  # .reshape(2, -1).transpose(0, 1).reshape(-1).flip([0]).reshape(seq_len, 1, -1)
    y = y[order]  # .reshape(2, -1).transpose(0, 1).reshape(-1).reshape(seq_len, 1, -1)

    return x, y


def randomize_classes(x: torch.Tensor, num_classes: int) -> torch.Tensor:
    classes = torch.arange(0, num_classes, device=x.device)
    random_classes = torch.randperm(num_classes, device=x.device).type(x.type())
    x = ((x.unsqueeze(-1) == classes) * random_classes).sum(-1)
    return x


@torch.no_grad()
def sample_num_feaetures_get_batch(
    batch_size: int,
    seq_len: int,
    num_features: int,
    hyperparameters: dict[str, Any],
    get_batch: Callable,
    **kwargs: Any,
) -> Batch:
    if (
        hyperparameters.get("sample_num_features", True) and kwargs["epoch"] > 0
    ):  # don't sample on test batch
        num_features = random.randint(3, num_features)
    return get_batch(batch_size, seq_len, num_features, hyperparameters=hyperparameters, **kwargs)


class CategoricalActivation(nn.Module):
    def __init__(
        self,
        categorical_p: float = 0.1,
        ordered_p: float = 0.7,
        keep_activation_size: bool = False,
        num_classes_sampler: Callable[[], int] = zipf_sampler_f(0.8, 1, 10),
    ) -> None:
        self.categorical_p = categorical_p
        self.ordered_p = ordered_p
        self.keep_activation_size = keep_activation_size
        self.num_classes_sampler = num_classes_sampler

        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: T, B, H

        x = nn.Softsign()(x)

        num_classes = self.num_classes_sampler()
        hid_strength = torch.abs(x).mean(0).unsqueeze(0) if self.keep_activation_size else None

        categorical_classes = torch.rand((x.shape[1], x.shape[2])) < self.categorical_p
        class_boundaries = torch.zeros(
            (num_classes - 1, x.shape[1], x.shape[2]), device=x.device, dtype=x.dtype
        )
        # Sample a different index for each hidden dimension, but shared for all batches
        for b in range(x.shape[1]):
            for h in range(x.shape[2]):
                ind = torch.randint(0, x.shape[0], (num_classes - 1,))
                class_boundaries[:, b, h] = x[ind, b, h]

        for b in range(x.shape[1]):
            x_rel = x[:, b, categorical_classes[b]]
            boundaries_rel = class_boundaries[:, b, categorical_classes[b]].unsqueeze(1)
            x[:, b, categorical_classes[b]] = (x_rel > boundaries_rel).sum(
                dim=0
            ).float() - num_classes / 2

        ordered_classes = torch.rand((x.shape[1], x.shape[2])) < self.ordered_p
        ordered_classes = torch.logical_and(ordered_classes, categorical_classes)
        x[:, ordered_classes] = randomize_classes(x[:, ordered_classes], num_classes)

        x = x * hid_strength if self.keep_activation_size else x

        return x


class QuantizationActivation(torch.nn.Module):
    def __init__(self, n_thresholds: int, reorder_p: float = 0.5) -> None:
        super().__init__()
        self.n_thresholds = n_thresholds
        self.reorder_p = reorder_p
        self.thresholds = torch.nn.Parameter(torch.randn(self.n_thresholds))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = normalize_data(x)
        assert isinstance(x, torch.Tensor)
        x = x.unsqueeze(-1)
        x = (x > self.thresholds).sum(-1)

        if random.random() < self.reorder_p:
            x = randomize_classes(x.unsqueeze(-1), self.n_thresholds).squeeze(-1)
        # x = ((x.float() - self.n_thresholds/2) / self.n_thresholds)# * data_std + data_mean
        x = normalize_data(x)
        return x


class NormalizationActivation(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = normalize_data(x)
        return x


class PowerActivation(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.exp = torch.nn.Parameter(0.5 * torch.ones(1))
        self.shared_exp_strength = 0.5
        # TODO: Somehow this is only initialized once, so it's the same for all runs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(torch.nn.functional.softplus(x), self.exp)
        shared_exp = torch.randn(1)
        exp = torch.nn.Parameter(
            (
                shared_exp * self.shared_exp_strength
                + shared_exp * torch.randn(x.shape[-1]) * (1 - self.shared_exp_strength)
            )
            * 2
            + 0.5
        ).to(x.device)
        x_ = torch.pow(torch.nn.functional.softplus(x) + 0.001, exp)
        if False:
            print(
                x[0:3, 0, 0].cpu().numpy(),
                torch.nn.functional.softplus(x[0:3, 0, 0]).cpu().numpy(),
                x_[0:3, 0, 0].cpu().numpy(),
                normalize_data(x_)[0:3, 0, 0].cpu().numpy(),
                self.exp.cpu().numpy(),
            )
        return x_


def lambda_time(f: Callable, name: str = "", enabled: bool = True) -> Any:
    if not enabled:
        return f()
    start = time.time()
    r = f()
    print("Timing", name, time.time() - start)
    return r


def pretty_get_batch(get_batch: Callable) -> str:
    """
    Genereate string representation of get_batch function
    :param get_batch:
    :return:
    """
    if isinstance(get_batch, types.FunctionType):
        return f"<{get_batch.__module__}.{get_batch.__name__} {inspect.signature(get_batch)}"
    else:
        return repr(get_batch)


class get_batch_sequence(list):
    """
    This will call the get_batch_methods in order from the back and pass the previous as `get_batch` kwarg.
    For example for `get_batch_methods=[get_batch_1, get_batch_2, get_batch_3]` this will produce a call
    equivalent to `get_batch_3(*args,get_batch=partial(partial(get_batch_2),get_batch=get_batch_1,**kwargs))`.
    get_batch_methods: all priors, but the first, muste have a `get_batch` argument
    """

    def __init__(self, *get_batch_methods: Any) -> None:
        if len(get_batch_methods) == 0:
            raise ValueError("Must have at least one get_batch method")
        super().__init__(get_batch_methods)

    def __repr__(self) -> str:
        s = ",\n\t".join([f"{pretty_get_batch(get_batch)}" for get_batch in self])
        return f"get_batch_sequence(\n\t{s}\n)"

    def __call__(self, *args: Any, **kwargs: Any) -> Batch:
        """

        Standard kwargs are: batch_size, seq_len, num_features
        This returns a priors.Batch object.
        """
        final_get_batch = self[0]
        for get_batch in self[1:]:
            final_get_batch = partial(get_batch, get_batch=final_get_batch)
        return final_get_batch(*args, **kwargs)
