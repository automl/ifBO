from collections.abc import Callable
import math
import os
from typing import Any
import warnings

import numpy as np
from scipy.stats import beta
from scipy.stats import expon
from scipy.stats import gamma
from scipy.stats import norm
import torch

from ifbo import encoders
from ifbo.encoders import Normalize
from ifbo.priors.prior import Batch
from ifbo.utils import default_device


OUTPUT_SORTED = np.load(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_sorted.npy")
)


def progress_noise(X: np.ndarray, sigma: float, L: float) -> np.ndarray:
    EPS = 10**-9
    N = len(X)

    Z = np.random.normal(0, sigma, size=(N,))

    SIGMA = np.exp(-(np.subtract.outer(X, X) ** 2) / L)

    SIGMA += EPS * np.eye(N)  # to guarantee SPD

    C = np.linalg.cholesky(SIGMA)

    return C @ Z


def add_noise_and_break(
    x: np.ndarray, x_noise: None, Xsat: np.ndarray, Rpsat: np.ndarray
) -> np.ndarray:
    x = np.where(
        x < Xsat, x, Rpsat * (x - Xsat) + Xsat
    )  # add a breaking point when saturation is reached
    return x
    # noisy_x = x + x_noise
    # add the exponential tails to avoid negative x
    # TODO: actually make curve go to 0 in the negative range (would allow divergence beyond Y0)
    # noisy_x = np.where(noisy_x > 1/1000, noisy_x, np.exp(noisy_x-1/1000+np.log(1/1000)))
    # return noisy_x


def comb(
    x: np.ndarray,
    Y0: float = 0.2,
    Yinf: float = 0.8,
    sigma: float | None = 0.01,
    L: float | None = 0.0001,
    PREC: list[int] = [100] * 4,
    Xsat: list[float] = [1.0] * 4,
    alpha: list[float] = [np.exp(1), np.exp(-1), 1 + np.exp(-4), np.exp(0)],
    Rpsat: list[float] = [1.0] * 4,
    w: list[float] = [1 / 4] * 4,
) -> float:
    # x_noise = progress_noise(x,sigma,L)
    x_noise = None
    EPS = 10**-9
    x_eps = np.array([EPS, 2 * EPS])

    # POW4 with exponential tail
    x_pow = add_noise_and_break(x, x_noise, Xsat[0], Rpsat[0])
    pow_eps = (
        Yinf - (Yinf - Y0) * (((PREC[0]) ** (1 / alpha[0]) - 1) / Xsat[0] * x_eps + 1) ** -alpha[0]
    )
    pow_grad = (pow_eps[1] - pow_eps[0]) / EPS
    pow_y = np.where(
        x_pow > 0,
        Yinf
        - (Yinf - Y0) * (((PREC[0]) ** (1 / alpha[0]) - 1) / Xsat[0] * x_pow + 1) ** -alpha[0],
        Y0 * np.exp(x_pow * (pow_grad + EPS) / Y0),
    )

    x_exp = add_noise_and_break(x, x_noise, Xsat[1], Rpsat[1])
    exp_eps = Yinf - (Yinf - Y0) * PREC[1] ** (-((x_eps / Xsat[1]) ** alpha[1]))
    exp_grad = (exp_eps[1] - exp_eps[0]) / EPS
    exp_y = np.where(
        x_exp > 0,
        Yinf - (Yinf - Y0) * PREC[1] ** (-((x_exp / Xsat[1]) ** alpha[1])),
        Y0 * np.exp(x_exp * (exp_grad + EPS) / Y0),
    )

    x_log = add_noise_and_break(x, x_noise, Xsat[2], Rpsat[2])
    log_eps = Yinf - (Yinf - Y0) * np.log(alpha[2]) / (
        np.log((alpha[2] ** PREC[2] - alpha[2]) * x_eps / Xsat[2] + alpha[2])
    )
    log_grad = (log_eps[1] - log_eps[0]) / EPS
    log_y = np.where(
        x_log > 0,
        Yinf
        - (Yinf - Y0)
        * np.log(alpha[2])
        / (np.log((alpha[2] ** PREC[2] - alpha[2]) * x_log / Xsat[2] + alpha[2])),
        Y0 * np.exp(x_log * (log_grad + EPS) / Y0),
    )

    x_hill = add_noise_and_break(x, x_noise, Xsat[3], Rpsat[3])
    hill_eps = Yinf - (Yinf - Y0) / ((x_eps / Xsat[3]) ** alpha[3] * (PREC[3] - 1) + 1)
    hill_grad = (hill_eps[1] - hill_eps[0]) / EPS
    hill_y = np.where(
        x_hill > 0,
        Yinf - (Yinf - Y0) / ((x_hill / Xsat[3]) ** alpha[3] * (PREC[3] - 1) + 1),
        Y0 * np.exp(x_hill * (hill_grad + EPS) / Y0),
    )

    return w[0] * pow_y + w[1] * exp_y + w[2] * log_y + w[3] * hill_y


class MLP(torch.nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        super(MLP, self).__init__()

        num_layers = np.random.randint(8, 16)
        num_hidden = np.random.randint(36, 150)
        self.init_std = np.random.uniform(0.089, 0.193)
        self.sparseness = 0.145
        self.preactivation_noise_std = np.random.uniform(
            0.0003, 0.0014
        )  # TODO: check value for this!
        self.output_noise = np.random.uniform(0.0004, 0.0013)
        activation = "tanh"

        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(num_inputs, num_hidden)]
            + [torch.nn.Linear(num_hidden, num_hidden) for _ in range(num_layers - 2)]
            + [torch.nn.Linear(num_hidden, num_outputs)]
        )

        self.reset_parameters()

        self.activation = {
            "tanh": torch.nn.Tanh(),
            "relu": torch.nn.ReLU(),
            "elu": torch.nn.ELU(),
            "identity": torch.nn.Identity(),
        }[activation]

    def reset_parameters(
        self, init_std: float | None = None, sparseness: float | None = None
    ) -> None:
        init_std = init_std if init_std is not None else self.init_std
        sparseness = sparseness if sparseness is not None else self.sparseness
        for linear in self.linears:
            linear.reset_parameters()

        with torch.no_grad():
            if init_std is not None:
                for linear in self.linears:
                    linear.weight.normal_(0, init_std)
                    linear.bias.normal_(0, init_std)

            if sparseness > 0.0:
                for linear in self.linears[1:-1]:
                    linear.weight /= (1.0 - sparseness) ** (1 / 2)
                    linear.weight *= torch.bernoulli(
                        torch.ones_like(linear.weight) * (1.0 - sparseness)
                    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for linear in self.linears[:-1]:
            x = linear(x)
            x = x + torch.randn_like(x) * self.preactivation_noise_std
            x = torch.tanh(x)
        x = self.linears[-1](x)
        return x + torch.randn_like(x) * self.output_noise


class DatasetPrior:
    def _get_model(self) -> torch.nn.Module:
        return MLP(self.num_inputs, self.num_outputs).to("cpu")

    def _output_for(self, input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # normalize the inputs
            input = self.normalizer(input)
            # reweight the inputs for parameter importance
            # input = input*self.input_weights  # TODO: consider adding this again
            # apply the model produce the output
            output = self.model(input.float())
            # rescale and shift outputs to account for parameter sensitivity
            # This output scaling causes issues with
            # output = output * self.output_sensitivity + self.output_offset
            return output

    def __init__(self, num_params: int, num_outputs: int) -> None:
        self.num_features = num_params
        self.num_outputs = num_outputs
        self.num_inputs = num_params

        self.normalizer = Normalize(0.5, math.sqrt(1 / 12))

        self.new_dataset()

    def new_dataset(self) -> None:
        # reinitialize all dataset specific random variables
        # reinit the parameters of the BNN
        self.model = self._get_model()
        # initial performance (after init) & max performance
        u1 = np.random.uniform()
        u2 = np.random.uniform()
        self.y0 = min(u1, u2)
        self.ymax = max(u1, u2) if np.random.uniform() < 0.25 else 1.0
        # TODO: this is not standard BOPFN BNN, but consider adding this
        # the input weights (parameter importance & magnitude of aleatoric uncertainty on the curve)
        # param_importance = np.random.dirichlet([1]*(self.num_inputs-1) + [0.1]) # relative parameter importance
        # lscale = np.exp(np.random.normal(2, 0.5)) # length scale ~ complexity of the landscape
        # self.input_weights = np.concatenate((param_importance*lscale*self.num_inputs, np.full((1,),lscale)), axis=0)
        # the output weights (curve property sensitivity)
        # self.output_sensitivity = np.random.uniform(size=(self.num_outputs,))
        # self.output_offset = np.random.uniform((self.output_sensitivity-1)/2,(1-self.output_sensitivity)/2)

    def curves_for_configs(
        self, configs: np.ndarray, noise: bool = True
    ) -> Callable[[np.ndarray, int], np.ndarray]:
        # more efficient batch-wise
        ncurves = 4
        bnn_outputs = self.output_for_config(configs, noise=noise)

        indices = np.searchsorted(OUTPUT_SORTED, bnn_outputs, side="left")

        rng4config = MyRNG(indices)

        Y0 = self.y0

        # sample Yinf (shared by all components)
        Yinf = rng4config.uniform(a=Y0, b=self.ymax)  # 0
        assert isinstance(Yinf, np.ndarray)

        # sample weights for basis curves (dirichlet)
        w = np.stack([rng4config.gamma(a=1) for i in range(ncurves)]).T  # 1, 2, 3, 4
        w = w / w.sum(axis=1, keepdims=1)

        # sample shape/skew parameter for each basis curve
        alpha = np.stack(
            [
                np.exp(rng4config.normal(1, 1)),  # 5
                np.exp(rng4config.normal(0, 1)),  # 6
                1.0 + np.exp(rng4config.normal(-4, 1)),  # 7
                np.exp(rng4config.normal(0.5, 0.5)),
            ]
        ).T  # 8

        # sample saturation x for each basis curve
        Xsat_max = 10 ** rng4config.normal(0, 1)  # max saturation # 9
        assert isinstance(Xsat_max, np.ndarray)

        Xsat_rel = np.stack(
            [rng4config.gamma(a=1) for i in range(ncurves)]
        ).T  # relative saturation points # 10, 11, 12, 13

        Xsat = ((Xsat_max.T * Xsat_rel.T) / np.max(Xsat_rel, axis=1)).T

        # sample relative saturation y (PREC) for each basis curve
        PREC = np.stack(
            [1.0 / 10 ** rng4config.uniform(-3, 0) for i in range(ncurves)]
        ).T  # 14, 15, 16, 17

        # post saturation convergence/divergence rate for each basis curve
        Rpsat = np.stack(
            [1.0 - rng4config.exponential(scale=1) for i in range(ncurves)]
        ).T  # 18, 19, 20, 21

        # sample noise parameters
        sigma = np.exp(rng4config.normal(loc=-5, scale=1))
        # sigma_x = np.exp(rng4config.normal(-4,0.5)) # STD of the xGP 22
        # print("warning")
        # sigma_y_scaler = np.exp(rng4config.uniform(-5,0.0)) # STD of the yGP 23
        # L = 10**rng4config.normal(-5,1) # Length-scale of the xyGP 24

        def foo(x_: np.ndarray, cid: int = 0) -> np.ndarray:
            warnings.filterwarnings("ignore")
            y_ = comb(
                x_,
                Y0=Y0,
                Yinf=Yinf[cid],
                sigma=None,
                L=None,
                Xsat=Xsat[cid],
                alpha=alpha[cid],
                Rpsat=Rpsat[cid],
                w=w[cid],
                PREC=PREC[cid],
            )
            # y_ = comb(x_, Y0=Y0, Yinf=Yinf[cid], sigma=sigma_x[cid], L=L[cid], Xsat=Xsat[cid], alpha=alpha[cid], Rpsat=Rpsat[cid], w=w[cid], PREC=PREC[cid])
            y_noise = np.random.normal(size=x_.shape, scale=sigma[cid])
            # y_noise = progress_noise(x_,1,L)
            # y_noise *= np.minimum(y_,1.0-y_)/4*sigma_y_scaler[cid]
            return np.clip(y_ + y_noise, 0.0, 1.0)

        return foo

    def output_for_config(self, config: np.ndarray, noise: bool = True) -> np.ndarray:
        # add aleatoric noise & bias
        output = self._output_for(torch.from_numpy(config))
        return output.numpy()

    def uniform(self, bnn_output: np.ndarray, a: float = 0.0, b: float = 1.0) -> np.ndarray:
        indices = np.searchsorted(OUTPUT_SORTED, bnn_output, side="left")
        return (b - a) * indices / len(OUTPUT_SORTED) + a

    def normal(self, bnn_output: np.ndarray, loc: float = 0, scale: float = 1) -> np.ndarray:
        eps = 0.5 / len(OUTPUT_SORTED)  # to avoid infinite samples
        u = self.uniform(bnn_output, a=eps, b=1 - eps)
        return norm.ppf(u, loc=loc, scale=scale)

    def beta(
        self, bnn_output: np.ndarray, a: float = 1, b: float = 1, loc: float = 0, scale: float = 1
    ) -> np.ndarray:
        eps = 0.5 / len(OUTPUT_SORTED)  # to avoid infinite samples
        u = self.uniform(bnn_output, a=eps, b=1 - eps)
        return beta.ppf(u, a=a, b=b, loc=loc, scale=scale)

    def gamma(
        self, bnn_output: np.ndarray, a: float = 1, loc: float = 0, scale: float = 1
    ) -> np.ndarray:
        eps = 0.5 / len(OUTPUT_SORTED)  # to avoid infinite samples
        u = self.uniform(bnn_output, a=eps, b=1 - eps)
        return gamma.ppf(u, a=a, loc=loc, scale=scale)

    def exponential(self, bnn_output: np.ndarray, scale: float = 1) -> np.ndarray:
        eps = 0.5 / len(OUTPUT_SORTED)  # to avoid infinite samples
        u = self.uniform(bnn_output, a=eps, b=1 - eps)
        return expon.ppf(u, scale=scale)


class MyRNG:
    def __init__(self, indices: np.ndarray) -> None:
        self.indices = indices.T
        self.reset()

    def reset(self) -> None:
        self.counter = 0

    def uniform(self, a: float = 0.0, b: float = 1.0) -> float | np.ndarray:
        u = (b - a) * self.indices[self.counter] / len(OUTPUT_SORTED) + a
        self.counter += 1
        return u

    def normal(self, loc: float = 0, scale: float = 1) -> float | np.ndarray:
        eps = 0.5 / len(OUTPUT_SORTED)  # to avoid infinite samples
        u = self.uniform(a=eps, b=1 - eps)
        return norm.ppf(u, loc=loc, scale=scale)

    def beta(
        self, a: float = 1, b: float = 1, loc: float = 0, scale: float = 1
    ) -> float | np.ndarray:
        eps = 0.5 / len(OUTPUT_SORTED)  # to avoid infinite samples
        u = self.uniform(a=eps, b=1 - eps)
        return beta.ppf(u, a=a, b=b, loc=loc, scale=scale)

    def gamma(self, a: float = 1, loc: float = 0, scale: float = 1) -> float | np.ndarray:
        eps = 0.5 / len(OUTPUT_SORTED)  # to avoid infinite samples
        u = self.uniform(a=eps, b=1 - eps)
        return gamma.ppf(u, a=a, loc=loc, scale=scale)

    def exponential(self, scale: float = 1) -> float | np.ndarray:
        eps = 0.5 / len(OUTPUT_SORTED)  # to avoid infinite samples
        u = self.uniform(a=eps, b=1 - eps)
        return expon.ppf(u, scale=scale)


def curve_prior(
    dataset: DatasetPrior, config: np.ndarray
) -> Callable[[np.ndarray, int], np.ndarray]:
    # calls the more efficient batch-wise method
    return dataset.curves_for_configs(np.array([config]))


# function producing batches for PFN training
@torch.no_grad()
def get_batch(
    batch_size: int,
    seq_len: int,
    num_features: int,
    single_eval_pos: int,
    device: torch.device = default_device,
    hyperparameters: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Batch:
    # assert num_features == 2
    assert num_features >= 2
    EPS = 10**-9

    if hyperparameters is not None and "hp_dim" in hyperparameters:
        num_params = hyperparameters["hp_dim"]
    else:
        num_params = np.random.randint(1, num_features - 1)  # beware upper bound is exclusive!

    dataset_prior = DatasetPrior(num_params, 23)

    x = []
    y = []

    for i in range(batch_size):
        epoch = torch.zeros(seq_len)
        id_curve = torch.zeros(seq_len)
        curve_val = torch.zeros(seq_len)
        config = torch.zeros(seq_len, num_params)

        # determine the number of fidelity levels (ranging from 1: BB, up to seq_len)
        n_levels = int(np.round(10 ** np.random.uniform(0, 3)))
        # print(f"n_levels: {n_levels}")

        # determine # observations/queries per curve
        # TODO: also make this a dirichlet thing
        alpha = 10 ** np.random.uniform(-4, -1)
        # print(f"alpha: {alpha}")
        weights = np.random.gamma(alpha, alpha, seq_len) + EPS
        p = weights / np.sum(weights)
        ids = np.arange(seq_len)
        all_levels = np.repeat(ids, n_levels)
        all_p = np.repeat(p, n_levels) / n_levels
        ordering = np.random.choice(all_levels, p=all_p, size=seq_len, replace=False)

        # calculate the cutoff/samples for each curve
        cutoff_per_curve = np.zeros((seq_len,), dtype=int)
        epochs_per_curve = np.zeros((seq_len,), dtype=int)
        for i in range(seq_len):  # loop over every pos
            cid = ordering[i]
            epochs_per_curve[cid] += 1
            if i < single_eval_pos:
                cutoff_per_curve[cid] += 1

        # fix dataset specific random variables
        dataset_prior.new_dataset()

        # determine config, x, y for every curve
        curve_configs = np.random.uniform(size=(seq_len, num_params))
        curves = dataset_prior.curves_for_configs(curve_configs)
        curve_xs = []
        curve_ys = []
        for cid in range(seq_len):  # loop over every curve
            if epochs_per_curve[cid] > 0:
                # determine x (observations + query)
                x_ = np.zeros((epochs_per_curve[cid],))
                if cutoff_per_curve[cid] > 0:  # observations (if any)
                    x_[: cutoff_per_curve[cid]] = (
                        np.arange(1, cutoff_per_curve[cid] + 1) / n_levels
                    )
                if cutoff_per_curve[cid] < epochs_per_curve[cid]:  # queries (if any)
                    x_[cutoff_per_curve[cid] :] = (
                        np.random.choice(
                            np.arange(cutoff_per_curve[cid] + 1, n_levels + 1),
                            size=epochs_per_curve[cid] - cutoff_per_curve[cid],
                            replace=False,
                        )
                        / n_levels
                    )
                curve_xs.append(x_)
                # determine y's
                y_ = curves(x_, cid)
                curve_ys.append(y_)
            else:
                curve_xs.append(None)
                curve_ys.append(None)

        # construct the batch data element
        curve_counters = torch.zeros(seq_len).type(torch.int64)
        for i in range(seq_len):
            cid = ordering[i]
            if i < single_eval_pos or curve_counters[cid] > 0:
                id_curve[i] = cid + 1  # reserve ID 0 for queries
            else:
                id_curve[i] = 0  # queries for unseen curves always have ID 0
            epoch[i] = curve_xs[cid][curve_counters[cid]]
            config[i] = torch.from_numpy(curve_configs[cid])
            curve_val[i] = curve_ys[cid][curve_counters[cid]]
            curve_counters[cid] += 1

        x.append(torch.cat([torch.stack([id_curve, epoch], dim=1), config], dim=1))
        y.append(curve_val)

    x = torch.stack(x, dim=1).to(device).float()
    y = torch.stack(y, dim=1).to(device).float()

    return Batch(x=x, y=y, target_y=y)


class MultiCurvesEncoder(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        seq_len = 1000
        self.normalizer = torch.nn.Sequential(
            encoders.Normalize(0.5, math.sqrt(1 / 12)),
        )
        self.epoch_enc = torch.nn.Linear(1, out_dim, bias=False)
        self.idcurve_enc = torch.nn.Embedding(seq_len + 1, out_dim)
        self.configuration_enc = encoders.get_variable_num_features_encoder(encoders.Linear)(
            in_dim - 2, out_dim
        )

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        x = torch.cat(x, dim=-1)
        out = (
            self.epoch_enc(self.normalizer(x[..., 1:2]))
            + self.idcurve_enc(x[..., :1].int()).squeeze(2)
            + self.configuration_enc(x[..., 2:])
        )
        return out


def get_encoder() -> Callable[[int, int], torch.nn.Module]:
    return lambda num_features, emsize: MultiCurvesEncoder(num_features, emsize)


def sample_curves(
    num_hyperparameters: int = 1000, curve_length: int = 100, hyperparameter_dimensions: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    dataset_prior = DatasetPrior(hyperparameter_dimensions, 23)
    hyperparameters = np.random.uniform(size=(num_hyperparameters, hyperparameter_dimensions))
    dataset_prior.new_dataset()
    curve_sampler = dataset_prior.curves_for_configs(hyperparameters)
    curves = np.array(
        [curve_sampler(np.linspace(0, 1, curve_length), cid) for cid in range(num_hyperparameters)]
    )
    return hyperparameters, curves
