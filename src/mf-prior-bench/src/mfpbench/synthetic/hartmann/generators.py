"""Extends Hartmann functions to incorporate fidelities."""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import numpy as np


class MFHartmannGenerator(ABC):
    """A multifidelity version of the Hartmann3 function.

    Carried a bias term, which flattens the objective, and a noise term.
    The impact of both terms decrease with increasing fidelity, meaning that
    ``num_fidelities`` is the best fidelity. This fidelity level also constitutes
    a noiseless, true evaluation of the Hartmann function.
    """

    optimum: tuple[float, ...]

    # We need some kind of seed so that using the same config twice at the same fidelity
    # will have the same noise
    _default_seed: int = 1337

    # The dimensions to the hartmann generator
    dims: int

    def __init__(
        self,
        n_fidelities: int,
        fidelity_bias: float,
        fidelity_noise: float,
        seed: int | None = None,
    ):
        """Initialize the generator.

        Args:
            n_fidelities: The fidelity at which the function is evalated.
            fidelity_bias: Amount of bias, realized as a flattening of the objective.
            fidelity_noise: Amount of noise, decreasing linearly (in st.dev.) with
                fidelity.
            seed: The seed to use for the noise.
        """
        self.z_min, self.z_max = (1, n_fidelities)
        self.seed = seed if seed else self._default_seed
        self.bias = fidelity_bias
        self.noise = fidelity_noise
        self.random_state = np.random.default_rng(seed)

    @abstractmethod
    def __call__(self, z: int, Xs: tuple[float, ...]) -> float:
        """Evaluate the function at the given fidelity and points.

        Args:
            z: The fidelity at which to query.
            Xs: The Xs as input to the function, in the correct order

        Returns:
            Value at that position
        """
        ...


class MFHartmann3(MFHartmannGenerator):
    optimum = (0.114614, 0.555649, 0.852547)
    dims = 3

    def __call__(self, z: int, Xs: tuple[float, ...]) -> float:
        """Evaluate the function at the given fidelity and points.

        Args:
            z: The fidelity.
            Xs: Parameters of the function.

        Returns:
            The function value
        """
        assert len(Xs) == self.dims
        X_0, X_1, X_2 = Xs

        log_z = np.log(z)
        log_lb, log_ub = np.log(self.z_min), np.log(self.z_max)
        log_z_scaled = (log_z - log_lb) / (log_ub - log_lb)

        # Highest fidelity (1) accounts for the regular Hartmann
        X = np.array([X_0, X_1, X_2]).reshape(1, -1)
        alpha = np.array([1.0, 1.2, 3.0, 3.2])

        alpha_prime = alpha - self.bias * np.power(1 - log_z_scaled, 1)
        A: np.ndarray = np.array(
            [[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]],
            dtype=float,
        )
        P: np.ndarray = np.array(
            [
                [3689, 1170, 2673],
                [4699, 4387, 7470],
                [1091, 8732, 5547],
                [381, 5743, 8828],
            ],
            dtype=float,
        )

        inner_sum = np.sum(A * (X[:, np.newaxis, :] - 0.0001 * P) ** 2, axis=-1)
        H = -(np.sum(alpha_prime * np.exp(-inner_sum), axis=-1))

        # TODO: Didn't seem used
        # H_true = -(np.sum(alpha * np.exp(-inner_sum), axis=-1))

        # and add some noise
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Seed below will overflow
            rng = np.random.default_rng(seed=abs(self.seed * z * hash(Xs)))

        noise = np.abs(rng.normal(size=H.size)) * self.noise * (1 - log_z_scaled)

        return float((H + noise)[0])


class MFHartmann6(MFHartmannGenerator):
    optimum = (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)
    dims = 6

    def __call__(self, z: int, Xs: tuple[float, ...]) -> float:
        """Evaluate the function at the given fidelity and points.

        Args:
            z: The fidelity it's evaluated at.
            Xs: Parameters of the function

        Returns:
            The function value
        """
        assert len(Xs) == self.dims
        X_0, X_1, X_2, X_3, X_4, X_5 = Xs

        # Change by Carl - z now comes in normalized
        log_z = np.log(z)
        log_lb, log_ub = np.log(self.z_min), np.log(self.z_max)
        log_z_scaled = (log_z - log_lb) / (log_ub - log_lb)

        # Highest fidelity (1) accounts for the regular Hartmann
        X = np.array([X_0, X_1, X_2, X_3, X_4, X_5]).reshape(1, -1)
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        alpha_prime = alpha - self.bias * np.power(1 - log_z_scaled, 1)
        A: np.ndarray = np.array(
            [
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ],
            dtype=float,
        )
        P: np.ndarray = np.array(
            [
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ],
            dtype=float,
        )

        inner_sum = np.sum(A * (X[:, np.newaxis, :] - 0.0001 * P) ** 2, axis=-1)
        H = -(np.sum(alpha_prime * np.exp(-inner_sum), axis=-1))

        # TODO: Doesn't seem to be used?
        # H_true = -(np.sum(alpha * np.exp(-inner_sum), axis=-1))

        # and add some noise
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Seed below will overflow
            rng = np.random.default_rng(seed=abs(self.seed * z * hash(Xs)))

        noise = np.abs(rng.normal(size=H.size)) * self.noise * (1 - log_z_scaled)
        return float((H + noise)[0])
