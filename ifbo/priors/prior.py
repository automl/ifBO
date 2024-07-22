from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import fields
from typing import Any

import torch
from torch.utils.data import DataLoader


@dataclass
class Batch:
    """
    A batch of data, with non-optional x, y, and target_y attributes.
    All other attributes are optional.

    If you want to add an attribute for testing only, you can just assign it after creation like:
    ```
        batch = Batch(x=x, y=y, target_y=target_y)
        batch.test_attribute = test_attribute
    ```
    """

    # Required entries
    x: torch.Tensor
    y: torch.Tensor
    target_y: torch.Tensor

    # Optional Batch Entries
    style: torch.Tensor | None = None
    style_hyperparameter_values: torch.Tensor | None = None
    single_eval_pos: torch.Tensor | None = None
    causal_model_dag: object | None = None
    mean_prediction: bool | None = (
        None  # this controls whether to do mean prediction in bar_distribution for nonmyopic BO
    )

    def other_filled_attributes(
        self, set_of_attributes: set[str] = set(("x", "y", "target_y"))
    ) -> list[str]:
        return [
            f.name
            for f in fields(self)
            if f.name not in set_of_attributes and getattr(self, f.name) is not None
        ]


def safe_merge_batches_in_batch_dim(*batches: Any, ignore_attributes: list[str] = []) -> Batch:
    """
    Merge all supported non-None fields in a pre-specified (general) way,
    e.g. mutliple batch.x are concatenated in the batch dimension.
    :param ignore_attributes: attributes to remove from the merged batch, treated as if they were None.
    :return:
    """
    not_none_fields = [
        f.name
        for f in fields(batches[0])
        if f.name not in ignore_attributes and getattr(batches[0], f.name) is not None
    ]
    assert all(
        [
            set(not_none_fields)
            == set(
                [
                    f.name
                    for f in fields(b)
                    if f.name not in ignore_attributes and getattr(b, f.name) is not None
                ]
            )
            for b in batches
        ]
    ), "All batches must have the same fields!"
    merge_funcs = {
        "x": lambda xs: torch.cat(xs, 1),
        "y": lambda ys: torch.cat(ys, 1),
        "target_y": lambda target_ys: torch.cat(target_ys, 1),
        "style": lambda styles: torch.cat(styles, 0),
    }
    assert all(
        f in merge_funcs for f in not_none_fields
    ), "Unknown fields encountered in `safe_merge_batches_in_batch_dim`."
    return Batch(
        **{f: merge_funcs[f]([getattr(batch, f) for batch in batches]) for f in not_none_fields}  # type: ignore
    )


def merge_batches(*batches: Any, ignore_attributes: list[str] = []) -> Batch:
    assert False, "TODO: isn't this broken!? because catting in dim 0 seems wrong!?"

    def merge_attribute(attr_name: str, batch_sizes: list[int]) -> Any:
        attr = [getattr(batch, attr_name) for batch in batches]
        if isinstance(attr[0], list):

            def make_list(sublist, i):
                if sublist is None:
                    return [None for _ in range(batch_sizes[i])]
                return sublist

            return sum([make_list(sublist, i) for i, sublist in enumerate(attr)], [])
        elif type(attr[0]) is torch.Tensor:
            return torch.cat(attr, 0)
        else:
            assert all(a is None for a in attr), (
                f"Unknown type encountered in `merge_batches`."
                f"To ignore this, please add `{attr}` to the `ignore_attributes`."
                f"The following values are the problem: {attr_name}."
            )
            return None

    batch_sizes = [batch.x.shape[0] for batch in batches]
    return Batch(
        **{
            f.name: merge_attribute(f.name, batch_sizes)
            for f in fields(batches[0])
            if f.name not in ignore_attributes
        }
    )


class PriorDataLoader(DataLoader, metaclass=ABCMeta):
    @abstractmethod
    def __init__(
        self,
        num_steps: int,
        batch_size: int,
        eval_pos_seq_len_sampler: Callable,
        seq_len_maximum: int,
        device: torch.device | None,
        **kwargs: Any,
    ) -> None:
        """

        :param num_steps: int, first argument, the number of steps to take per epoch, i.e. iteration of the DataLoader
        :param batch_size: int, number of datasets per batch
        :param eval_pos_seq_len_sampler: callable, it takes no arguments and returns a tuple (single eval pos, bptt)
        :param kwargs: for future compatibility it is good to have a final all catch, as new kwargs might be introduced
        """
        pass

    # A class or object variable `num_features`: int
    # Optional: `validate` function that accepts a transformer model

    # The DataLoader iter should return batches of the form ([style], x, y), target_y, single_eval_pos
    # We follow sequence len (s) first, batch size (b) second. So x: (s,b,num_features), y,target_y: (s,b)
    # and style: Optional[(b,num_style_params)], style can be omitted or set to None, if it is not intended to be used.

    # For more references, see `priors/utils.py` for a pretty general implementation of a DataLoader
    # and `train.py` for the only call of it.
