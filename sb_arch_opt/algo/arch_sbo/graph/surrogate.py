"""
MIT License

Copyright: (c) 2026, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from pymoo.util.normalization import Normalization
from smt.sampling_methods import LHS
from smt.surrogate_models.krg_based import KrgBased
from smt.surrogate_models.krg_based.hyperparam_optim import CobylaOptimizer
from smt.utils.kriging import cross_distances

from .kernels import GraphKernel, ThetaParameter
from .representation import GraphDecoder, GraphRepresentation


__all__ = ["GraphKriging"]


class GraphKriging(KrgBased):
    """SMT kriging model whose correlation is supplied by a graph kernel."""

    name = "GraphKriging"

    def __init__(
        self,
        decoder: GraphDecoder,
        kernel: GraphKernel,
        normalization: Optional[Normalization] = None,
        **kwargs,
    ):
        kwargs.setdefault("hyper_opt", "Cobyla")
        super().__init__(**kwargs)
        self.decoder = decoder
        self.kernel = kernel
        self.normalization = normalization

        self._representations_by_x: Dict[Tuple[float, ...], GraphRepresentation] = {}
        self._train_kernel_matrix: Optional[np.ndarray] = None
        self._train_kernel_theta: Optional[Tuple[float, ...]] = None
        self._train_pairs: Optional[np.ndarray] = None
        self._theta_parameters: Sequence[ThetaParameter] = ()

    def _decode(self, x: np.ndarray):
        x = np.asarray(x, dtype=float)
        missing_keys = []
        missing_rows = []
        for row in x:
            key = tuple(row)
            if key not in self._representations_by_x:
                missing_keys.append(key)
                missing_rows.append(row)

        if missing_rows:
            raw = np.asarray(missing_rows)
            if self.normalization is not None:
                raw = self.normalization.backward(raw)
            for key, representation in zip(missing_keys, self.decoder.decode(raw)):
                self._representations_by_x[key] = representation

        return tuple(self._representations_by_x[tuple(row)] for row in x)

    def _check_param(self):
        requested_theta0 = np.asarray(self.options["theta0"], dtype=float).ravel()
        self.options["theta0"] = np.array([float(requested_theta0[0])])
        super()._check_param()

        parameters = tuple(self.kernel.get_theta_parameters())
        if not parameters:
            parameters = (ThetaParameter(0.01, 100.0, "linear", 1.0),)
        theta0 = np.asarray([parameter.initial for parameter in parameters], dtype=float)
        if requested_theta0.size > 1:
            if requested_theta0.size != theta0.size:
                raise ValueError(
                    f"GraphKriging expects theta0 with 1 or {theta0.size} values, "
                    f"got {requested_theta0.size}"
                )
            theta0 = requested_theta0

        self._theta_parameters = parameters
        self.options["theta0"] = theta0
        self._theta0 = list(theta0)
        self.n_param = theta0.size

    def _new_train(self):
        self._train_kernel_matrix = None
        self._train_kernel_theta = None
        self._train_pairs = None
        super()._new_train()

    def _to_optimizer_space(self, theta):
        theta = np.asarray(theta, dtype=float).copy()
        for index, parameter in enumerate(self._theta_parameters):
            if parameter.scale == "log":
                theta[index] = np.log10(
                    np.clip(theta[index], parameter.lower, parameter.upper)
                )
        return theta

    def _from_optimizer_space(self, theta):
        theta = np.asarray(theta, dtype=float).copy()
        for index, parameter in enumerate(self._theta_parameters):
            if parameter.scale == "log":
                theta[index] = 10.0 ** theta[index]
        return theta

    def _optimizer_bounds(self):
        bounds = []
        for parameter in self._theta_parameters:
            if parameter.scale == "log":
                bounds.append([
                    np.log10(parameter.lower),
                    np.log10(parameter.upper),
                ])
            else:
                bounds.append([parameter.lower, parameter.upper])
        return np.asarray(bounds)

    def _optimizer_constraints(self):
        constraints = []
        for index, parameter in enumerate(self._theta_parameters):
            lower = (
                np.log10(parameter.lower)
                if parameter.scale == "log"
                else parameter.lower
            )
            upper = (
                np.log10(parameter.upper)
                if parameter.scale == "log"
                else parameter.upper
            )
            constraints.append(lambda theta, i=index, bound=lower: theta[i] - bound)
            constraints.append(lambda theta, i=index, bound=upper: bound - theta[i])
        return constraints

    def _optimize_hyperparam(self, D, use_multistart=True, limit=None):
        del use_multistart, limit
        self.D = D

        if self.options["hyper_opt"] not in {"Cobyla", "NoOp"}:
            raise ValueError("GraphKriging supports hyper_opt='Cobyla' or 'NoOp'")
        if self._eval_noise:
            raise NotImplementedError("GraphKriging does not optimize the noise level")
        no_optimization = self.options["hyper_opt"] == "NoOp"
        self.noise0 = np.array(self.options["noise0"] if no_optimization else self._noise0)
        theta0 = np.asarray(self.options["theta0"], dtype=float).ravel()

        if no_optimization:
            optimal_theta = theta0
        else:
            bounds = self._optimizer_bounds()
            for index, parameter in enumerate(self._theta_parameters):
                theta0[index] = np.clip(theta0[index], parameter.lower, parameter.upper)
            theta0_optimizer = self._to_optimizer_space(theta0)

            def objective(theta):
                native_theta = self._from_optimizer_space(theta)
                return -self._reduced_likelihood_function(theta=native_theta)[0]

            random_start = bounds[:, 0] + self.rng.random(len(theta0)) * (
                bounds[:, 1] - bounds[:, 0]
            )
            starts = [theta0_optimizer, random_start]
            if self.options["n_start"] > 1:
                sampler = LHS(xlimits=bounds, criterion="maximin", seed=self.rng)
                starts.extend(sampler(self.options["n_start"]))

            result = CobylaOptimizer().optimize(
                objective=objective,
                theta_starts=np.vstack(starts),
                constraints=self._optimizer_constraints(),
                limit=max(12 * len(theta0), 50),
            )
            optimal_theta = (
                theta0
                if result is None or "x" not in result
                else self._from_optimizer_space(result["x"])
            )

        optimal_value, optimal_parameters = self._reduced_likelihood_function(
            theta=optimal_theta
        )
        return optimal_value, optimal_parameters, optimal_theta

    def _matrix_data_corr(self, theta, *, x=None, kplsk_second_loop=False, **_):
        if kplsk_second_loop:
            raise NotImplementedError("GraphKriging does not support KPLSK")

        theta = np.asarray(theta, dtype=float).ravel()
        if x is None:
            x_train = self.training_points[None][0][0]
            if self._train_pairs is None:
                _, self._train_pairs = cross_distances(x_train)
            theta_key = tuple(theta)
            if self._train_kernel_theta != theta_key:
                self._train_kernel_matrix = np.asarray(
                    self.kernel.fit_transform(self._decode(x_train), theta),
                    dtype=float,
                )
                self._train_kernel_theta = theta_key
            return self._train_kernel_matrix[
                self._train_pairs[:, 0], self._train_pairs[:, 1]
            ].reshape(-1, 1)

        kernel_matrix = self.kernel.transform(self._decode(np.asarray(x)), theta)
        return np.asarray(kernel_matrix, dtype=float).reshape(-1, 1)
