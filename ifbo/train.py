from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
import itertools
import time
from typing import Any

import argparse
import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from ifbo import positional_encodings
from ifbo import utils, encoders, bar_distribution
from ifbo.bar_distribution import BarDistribution
from ifbo.bar_distribution import get_custom_bar_dist
from ifbo.priors import prior, ftpfn_prior
from ifbo.priors.utils import get_batch_to_dataloader
from ifbo.transformer import TransformerModel
from ifbo.utils import get_cosine_schedule_with_warmup
from ifbo.utils import get_openai_lr
from ifbo.utils import init_dist

from ifbo.utils import default_device


class Losses:
    def get_cross_entropy_loss(self, num_classes: int) -> nn.CrossEntropyLoss:
        return nn.CrossEntropyLoss(reduction="none", weight=torch.ones(num_classes))

    gaussian = nn.GaussianNLLLoss(full=True, reduction="none")
    mse = nn.MSELoss(reduction="none")
    ce = get_cross_entropy_loss
    bce = nn.BCEWithLogitsLoss(reduction="none")
    get_BarDistribution = BarDistribution


def train(
    priordataloader_class: prior.PriorDataLoader,
    criterion: nn.Module | BarDistribution,
    encoder_generator: Callable[[int, int], nn.Module],
    style_encoder_generator: Callable[[int, int], nn.Module],
    y_encoder_generator: Callable[[int, int], nn.Module],
    emsize: int = 200,
    nhid: int = 200,
    nlayers: int = 6,
    nhead: int = 2,
    dropout: float = 0.0,
    epochs: int = 10,
    steps_per_epoch: int = 100,
    batch_size: int = 200,
    bptt: int = 10,
    lr: float | None = None,
    weight_decay: float = 0.0,
    warmup_epochs: int = 10,
    input_normalization: bool = False,
    pos_encoder_generator: Callable[[int, int], nn.Module] | None = None,
    decoder_dict: dict[str, Any] = {},
    extra_prior_kwargs_dict: dict[str, Any] = {},
    scheduler_generator: Callable = get_cosine_schedule_with_warmup,
    load_weights_from_this_state_dict: dict[str, Any] | None = None,
    validation_period: int = 10,
    single_eval_pos_gen: Callable[[], int] | int | None = None,
    bptt_extra_samples: int | None = None,
    gpu_device: torch.device = "cuda:0",
    aggregate_k_gradients: int = 1,
    verbose: bool = True,
    epoch_callback: Callable | None = None,
    step_callback: Callable | None = None,
    continue_model: nn.Module = None,
    initializer: Callable | None = None,
    initialize_with_model: TransformerModel | None = None,
    train_mixed_precision: bool = False,
    efficient_eval_masking: bool = True,
    border_decoder: nn.Module | None = None,
    num_global_att_tokens: int = 0,
    progress_bar: bool = False,
    **model_extra_args: Any,
) -> tuple[float, float, TransformerModel, prior.PriorDataLoader] | None:
    device = gpu_device if torch.cuda.is_available() else "cpu:0"
    print(f"Using {device} device")
    using_dist, rank, device = init_dist(device)

    def eval_pos_seq_len_sampler() -> tuple[int | None, int]:
        if isinstance(single_eval_pos_gen, int):
            single_eval_pos = single_eval_pos_gen
        elif callable(single_eval_pos_gen):
            single_eval_pos = single_eval_pos_gen()
        else:
            single_eval_pos = None
        if bptt_extra_samples and False:  # TODO: Currently disabled
            return single_eval_pos, single_eval_pos + bptt_extra_samples
        else:
            return single_eval_pos, bptt

    dl = priordataloader_class(
        num_steps=steps_per_epoch,
        batch_size=batch_size,
        eval_pos_seq_len_sampler=eval_pos_seq_len_sampler,
        seq_len_maximum=bptt,  # +(bptt_extra_samples if bptt_extra_samples else 0) # TODO: Currently disabled
        device=device,
        **extra_prior_kwargs_dict,
    )

    test_batch: prior.Batch = dl.get_test_batch()
    style_def = test_batch.style
    print(
        f"Style definition of first 3 examples: {style_def[:3] if style_def is not None else None}"
    )
    style_encoder = (
        style_encoder_generator(style_def.shape[1], emsize) if (style_def is not None) else None
    )
    pos_encoder = (pos_encoder_generator or positional_encodings.NoPositionalEncoding)(
        emsize, bptt * 2
    )
    if isinstance(criterion, nn.GaussianNLLLoss):
        n_out = 2
    elif (
        isinstance(criterion, BarDistribution) or "BarDistribution" in criterion.__class__.__name__
    ):  # TODO remove this fix (only for dev)
        n_out = criterion.num_bars
    elif isinstance(criterion, nn.CrossEntropyLoss):
        n_out = criterion.weight.shape[0]
    else:
        n_out = 1

    # border_decoder = None if border_decoder is None else border_decoder(emsize, criterion.num_bars + 1).to(device)

    if continue_model:
        model = continue_model
    else:
        decoder_dict = decoder_dict if decoder_dict else {"standard": (None, n_out)}

        decoder_once_dict = {}
        if test_batch.mean_prediction is not None:
            decoder_once_dict["mean_prediction"] = decoder_dict["standard"]

        encoder = encoder_generator(dl.num_features, emsize)
        model = TransformerModel(
            encoder=encoder,
            nhead=nhead,
            ninp=emsize,
            nhid=nhid,
            nlayers=nlayers,
            dropout=dropout,
            style_encoder=style_encoder,
            y_encoder=y_encoder_generator(1, emsize),
            input_normalization=input_normalization,
            pos_encoder=pos_encoder,
            decoder_dict=decoder_dict,
            init_method=initializer,
            efficient_eval_masking=efficient_eval_masking,
            decoder_once_dict=decoder_once_dict,
            num_global_att_tokens=num_global_att_tokens,
            **model_extra_args,
        )
    model.criterion = criterion
    if load_weights_from_this_state_dict is not None:
        model.load_state_dict(load_weights_from_this_state_dict)
    if initialize_with_model is not None:
        model.init_from_small_model(initialize_with_model)

    print(
        f"Using a Transformer with {sum(p.numel() for p in model.parameters())/1000/1000:.{2}f} M parameters"
    )

    try:
        for (k, v), (k2, v2) in zip(
            model.state_dict().items(),
            initialize_with_model.state_dict().items(),  # type: ignore
        ):
            print(k, ((v - v2) / v).abs().mean(), v.shape)
    except Exception:
        pass

    model.to(device)
    if using_dist:
        print("Distributed training")
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
            broadcast_buffers=False,
            find_unused_parameters=test_batch.mean_prediction is not None,
        )
        dl.model = model.module  # use local model, should not use multi-gpu functionality..
    else:
        dl.model = model

    # learning rate
    if lr is None:
        lr = get_openai_lr(model)
        print(f"Using OpenAI max lr of {lr}.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = scheduler_generator(
        optimizer, warmup_epochs, epochs if epochs is not None else 100
    )  # when training for fixed time lr schedule takes 100 steps

    scaler = GradScaler() if train_mixed_precision else None

    # check that everything uses up-to-date APIs
    utils.check_compatibility(dl)

    def train_epoch() -> tuple[float, list[float], float, float, float, float, float]:
        model.train()  # Turn on the train mode
        total_loss = 0.0
        total_positional_losses = torch.zeros(bptt)
        total_positional_losses_recorded = torch.zeros(bptt)
        nan_steps = torch.zeros(1).to(device)
        ignore_steps = torch.zeros(1).to(device)
        before_get_batch = time.time()
        assert (
            len(dl) % aggregate_k_gradients == 0
        ), "Please set the number of steps per epoch s.t. `aggregate_k_gradients` divides it."
        tqdm_iter = (
            tqdm(range(len(dl)), desc="Training Epoch") if rank == 0 and progress_bar else None
        )  # , disable=not verbose

        for batch, full_data in enumerate(dl):
            data, targets, single_eval_pos = (
                (full_data.style, full_data.x, full_data.y),
                full_data.target_y,
                full_data.single_eval_pos,
            )

            def get_metrics() -> tuple[float, list[float], float, float, float, float, float]:
                return (
                    total_loss / steps_per_epoch,
                    (total_positional_losses / total_positional_losses_recorded).tolist(),
                    time_to_get_batch,
                    forward_time,
                    step_time,
                    nan_steps.cpu().item() / (batch + 1),
                    ignore_steps.cpu().item() / (batch + 1),
                )

            tqdm_iter.update() if tqdm_iter is not None else None
            if using_dist and not (batch % aggregate_k_gradients == aggregate_k_gradients - 1):
                cm = model.no_sync()
            else:
                cm = nullcontext()
            with cm:
                time_to_get_batch = time.time() - before_get_batch
                before_forward = time.time()
                # TODO: This disables the bptt_extra_samples functionality but otherwise single eval pos is overwritten
                # if bptt_extra_samples is None:
                #    single_eval_pos = single_eval_pos_gen() if callable(single_eval_pos_gen) else single_eval_pos_gen
                # else:
                #    single_eval_pos = targets.shape[0] - bptt_extra_samples
                try:
                    metrics_to_log: dict[str, Any] = {}
                    with autocast(enabled=scaler is not None):
                        # If style is set to None, it should not be transferred to device
                        out = model(
                            tuple(e.to(device) if torch.is_tensor(e) else e for e in data),
                            single_eval_pos=single_eval_pos,
                            only_return_standard_out=False,
                        )

                        # this handling is for training old models only, this can be deleted soon(ish)
                        # to only support models that return a tuple of dicts
                        out, output_once = out if isinstance(out, tuple) else (out, None)
                        output = out["standard"] if isinstance(out, dict) else out

                        forward_time = time.time() - before_forward

                        if single_eval_pos is not None:
                            targets = targets[single_eval_pos:]

                        if len(targets.shape) == len(output.shape):
                            # this implies the prior uses a trailing 1 dimesnion
                            # below we assume this not to be the case
                            targets = targets.squeeze(-1)
                        assert targets.shape == output.shape[:-1], (
                            f"Target shape {targets.shape} "
                            "does not match output shape {output.shape}"
                        )
                        if isinstance(criterion, nn.GaussianNLLLoss):
                            assert (
                                output.shape[-1] == 2
                            ), "need to write a little bit of code to handle multiple regression targets at once"

                            mean_pred = output[..., 0]
                            var_pred = output[..., 1].abs()
                            losses = criterion(
                                mean_pred.flatten(), targets.flatten(), var=var_pred.flatten()
                            )
                        elif isinstance(criterion, (nn.MSELoss, nn.BCEWithLogitsLoss)):
                            targets[torch.isnan(targets)] = -100
                            losses = criterion(output.flatten(), targets.flatten())
                        elif isinstance(criterion, nn.CrossEntropyLoss):
                            targets[torch.isnan(targets)] = -100
                            print(f"{targets.min()=}, {targets.max()=}")
                            losses = criterion(output.reshape(-1, n_out), targets.long().flatten())
                        elif border_decoder is not None:

                            def apply_batch_wise_criterion(i: int) -> torch.Tensor:
                                output_, targets_, borders_ = (
                                    output_adaptive[:, i],
                                    targets[:, i],
                                    borders[i],
                                )
                                criterion_ = get_custom_bar_dist(borders_, criterion).to(device)
                                return criterion_(output_, targets_)

                            output_adaptive, borders = out["adaptive_bar"], output_once["borders"]
                            losses_adaptive_bar = torch.stack(
                                [
                                    apply_batch_wise_criterion(i)
                                    for i in range(output_adaptive.shape[1])
                                ],
                                1,
                            )
                            losses_fixed_bar = criterion(output, targets)
                            losses = (losses_adaptive_bar + losses_fixed_bar) / 2

                            metrics_to_log = {
                                **metrics_to_log,
                                **{
                                    "loss_fixed_bar": losses_fixed_bar.mean()
                                    .cpu()
                                    .detach()
                                    .item(),
                                    "loss_adaptive_bar": losses_adaptive_bar.mean()
                                    .cpu()
                                    .detach()
                                    .item(),
                                },
                            }
                        elif isinstance(criterion, BarDistribution) and full_data.mean_prediction:
                            assert "mean_prediction" in output_once
                            utils.print_once("Using mean prediction for loss")
                            losses = criterion(
                                output,
                                targets,
                                mean_prediction_logits=output_once["mean_prediction"],
                            )
                            # the mean pred loss appears as the last per sequence
                        else:
                            losses = criterion(output, targets)
                        losses = losses.view(
                            -1, output.shape[1]
                        )  # sometimes the seq length can be one off
                        # that is because bar dist appends the mean
                        loss, nan_share = utils.torch_nanmean(losses.mean(0), return_nanshare=True)
                        loss_scaled = loss / aggregate_k_gradients

                    if scaler:
                        loss_scaled = scaler.scale(loss_scaled)
                    loss_scaled.backward()

                    if batch % aggregate_k_gradients == aggregate_k_gradients - 1:
                        if scaler:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        if scaler:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad()

                    step_time = time.time() - before_forward

                    if not torch.isnan(loss):
                        total_loss += loss.cpu().detach().item()
                        total_positional_losses += (
                            losses.mean(1).cpu().detach()
                            if single_eval_pos is None
                            else nn.functional.one_hot(torch.tensor(single_eval_pos), bptt)
                            * utils.torch_nanmean(losses[: bptt - single_eval_pos].mean(0))
                            .cpu()
                            .detach()
                        )

                        total_positional_losses_recorded += (
                            torch.ones(bptt)
                            if single_eval_pos is None
                            else nn.functional.one_hot(torch.tensor(single_eval_pos), bptt)
                        )

                        metrics_to_log = {
                            **metrics_to_log,
                            **{"loss": loss, "single_eval_pos": single_eval_pos},
                        }
                        if step_callback is not None and rank == 0:
                            step_callback(metrics_to_log)
                        nan_steps += nan_share.detach()
                        ignore_steps += (targets == -100).float().mean()
                except Exception as e:
                    print("Invalid step encountered, skipping...")
                    print(e)
                    raise (e)

            # total_loss, total_positional_losses, time_to_get_batch, forward_time, step_time, nan_share, ignore_share = get_metrics()
            if tqdm_iter:
                tqdm_iter.set_postfix(
                    {
                        "data_time": time_to_get_batch,
                        "step_time": step_time,
                        "mean_loss": total_loss / (batch + 1),
                    }
                )

            before_get_batch = time.time()
        return get_metrics()

    total_loss = float("inf") * torch.ones(1)
    total_positional_losses = float("inf") * torch.ones(bptt)
    try:
        # Initially test the epoch callback function
        if epoch_callback is not None and rank == 0:
            epoch_callback(model, 1, data_loader=dl, scheduler=scheduler)
        for epoch in range(1, epochs + 1) if epochs is not None else itertools.count(1):
            epoch_start_time = time.time()
            try:
                (
                    total_loss,
                    total_positional_losses,
                    time_to_get_batch,
                    forward_time,
                    step_time,
                    nan_share,
                    ignore_share,
                ) = train_epoch()
            except Exception as e:
                print("Invalid epoch encountered, skipping...")
                print(e)
                raise (e)
            if hasattr(dl, "validate") and epoch % validation_period == 0:
                with torch.no_grad():
                    val_score = dl.validate(model)

            else:
                val_score = None

            if verbose:
                print("-" * 89)
                print(
                    f'| end of epoch {epoch:3d} | time: {(time.time() - epoch_start_time):5.2f}s | mean loss {total_loss:5.2f} | '
                    f"pos losses {','.join([f'{loss:5.2f}' for loss in total_positional_losses])}, lr {scheduler.get_last_lr()[0]}"
                    f' data time {time_to_get_batch:5.2f} step time {step_time:5.2f}'
                    f' forward time {forward_time:5.2f}'
                    f' nan share {nan_share:5.2f} ignore share (for classification tasks) {ignore_share:5.4f}'
                    + (f"val score {val_score}" if val_score is not None else "")
                )
                print("-" * 89)

            # stepping with wallclock time based scheduler
            if epoch_callback is not None and rank == 0:
                epoch_callback(model, epoch, data_loader=dl, scheduler=scheduler)
            scheduler.step()
    except KeyboardInterrupt:
        pass

    if rank == 0:  # trivially true for non-parallel training
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
            dl = None
        return total_loss, total_positional_losses, model.to("cpu"), dl

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an ifBO model")

    # transformer model parameters
    parser.add_argument("--nlayers", type=int, help="Number of layers", default=6)
    parser.add_argument("--emsize", type=int, default=512, help="Size of Embeddings")
    parser.add_argument("--nhead", type=int, default=4, help="Number of heads")

    # PFN parameters
    parser.add_argument(
        "--num_borders",
        type=int,
        default=1000,
        help="Number of borders considered in Bar distribution",
    )

    # Prior parameters
    parser.add_argument("--seq_len", type=int, required=True, help="Maximum sequence length")
    parser.add_argument(
        "--num_features",
        type=int,
        required=False,
        help="The total number of features for each datapoint in an example.",
        default=12, # has to be at least 3
    )
    parser.add_argument(
        "--power_single_eval_pos_sampler",
        type=int,
        required=False,
        help="Power of an exponential distribution to weight sampling of single eval pos.",
        default=-2,
    )

    # training parameters
    parser.add_argument("--epochs", type=int, required=True, help="Number of Training Epochs")
    parser.add_argument("--batch_size", type=int, default=25, help="Batch Size for Training")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate")
    parser.add_argument("--steps_per_epoch", type=int, default=100, help="Number of Steps per Epoch")
    parser.add_argument(
        "--train_mixed_precision",
        action="store_true",
        help="Enable Mixed Precision Training",
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")

    # other parameters
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the model")

    args = parser.parse_args()

    seq_len = args.seq_len

    bucket_limits = torch.linspace(0.0, 1.0, args.num_borders).to(default_device)
    criterion = bar_distribution.BarDistribution(bucket_limits)

    single_eval_pos_gen = utils.get_weighted_single_eval_pos_sampler(
        max_len=seq_len,
        min_len=0,
        p=args.power_single_eval_pos_sampler,
    )

    configs_train = {
        "nlayers": args.nlayers,
        "emsize": args.emsize,
        "epochs": args.epochs,
        "lr": args.lr,
        "nhead": args.nhead,
        "bptt": seq_len,  
        "steps_per_epoch": args.steps_per_epoch,
        "train_mixed_precision": args.train_mixed_precision,
        "batch_size": args.batch_size,
    }
    configs_train["bptt"] = seq_len
    configs_train["nhid"] = args.emsize * 2
    configs_train["warmup_epochs"] = args.epochs // 4
    configs_train.update(
        dict(
            priordataloader_class=get_batch_to_dataloader(ftpfn_prior.get_batch),
            criterion=criterion,
            encoder_generator=ftpfn_prior.get_encoder(seq_len),
            y_encoder_generator=encoders.get_normalized_uniform_encoder(
                encoders.Linear
            ),
            extra_prior_kwargs_dict={
                "num_features": args.num_features,
            },
            single_eval_pos_gen=single_eval_pos_gen,
            style_encoder_generator=None
        )
    )

    total_loss, total_positional_losses, model, dl = train(
        **configs_train
    )
    print(f"Total loss: {total_loss}, Total positional losses: {total_positional_losses}")
    torch.save(
        model,
        args.output_path,
    )
    print(f"Model saved to {args.output_path}")

