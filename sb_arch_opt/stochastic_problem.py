"""
MIT License

Copyright: (c) 2024, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
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
import numpy as np
from typing import Union, List, Optional
from pymoo.core.variable import Variable
from sb_arch_opt.design_space import ArchDesignSpace
from sb_arch_opt.problem import ArchOptProblemBase

try:
    from sb_arch_opt.uncertainty import StochasticParameterSpace, UQMethod, Scalarization, Mean
    HAS_UNCERTAINTY = True

except ImportError:
    HAS_UNCERTAINTY = False

__all__ = ['StochasticArchOptProblem', 'HAS_UNCERTAINTY', 'check_dependency']


def check_dependency():
    if not HAS_UNCERTAINTY:
        raise ImportError('Looks like SBArchOpt uncertainty package is not installed! Run: pip install sb-arch-opt[uncertainty]')


class StochasticArchOptProblem(ArchOptProblemBase):
    """
    Base class for a stochastic (robust) architecture optimization problem. It extends the ArchOptProblemBase class
    with support for stochastic optimization problems.

    Stochastic problem evaluates design vector for many realizations of the uncertain parameters and
    reduces the resulting distribution of each response to the single value the optimizer sees.
    Three things are therefore needed on top of the design variables:

    - `param_space`: a `StochasticParameterSpace`, the joint distribution of the uncertain parameters. These are
     quantities that influence the evaluation but are not chosen by the optimizer, so they are NOT design
     variables.
    - `uq_method`: a `UQMethod` (`MonteCarlo`, `PolynomialChaos`) that decides which realizations to evaluate and
     how to turn the responses into statistics. It is given a budget of `n_evaluations` expensive evaluations per
     design point.
    - a `Scalarization` per response, below.

    Requires specifying a type of optimization problem for each objective and constraint by providing a child
    instance of Scalarization object:

    - Mean: Minimize the expectation of the objective or constraint function
     --> for example min(E[F(x)])
    - Margin: Gaussian output distribution expected. Minimize for the objective or constraint function for a given
     confidence interval.
     --> min(E[F(x)] + k*sigma[F(x)]); The default k=1.645 is the one-sided 95% interval of a normal distribution.
    - Quantile: Makes no distributional assumption. Minimize the q quantile of the objective or constraint
     function --> min(F_q(x)), with q=0.95 by default.

    When type of optimization problem is not provided, the response defaults to `Mean()`.

    Implement `_arch_evaluate_sample`: it is called once per realization. All design points in a batch see the same
    realizations (common random numbers), so they stay comparable to each other and to a surrogate fitted through
    them.

    A failed evaluation (NaN) in any realization fails the whole design point: `scalarize` returns NaN,
    which is how SBArchOpt treats hidden-constraint violations elsewhere.

    After each evaluation the fitted response distributions are stored in the pymoo output dictionary as
    `StochasticOutput` objects, in `out['f_stochastic']`, `out['g_stochastic']` and `out['h_stochastic'].
    For polynomial chaos each output also carries the fitted expansion (`method_results`), from which Sobol
    indices can be obtained. If the output turns out to be deterministic after evaluation, a float is stored instead of a
    'StochasticOutput' object.
    """

    def __init__(self, des_vars: Union[List[Variable], ArchDesignSpace],
                 param_space: StochasticParameterSpace,
                 uq_method: UQMethod, n_obj=1, n_ieq_constr=0, n_eq_constr=0,
                 obj_scalar: Optional[List[Scalarization]] = None,
                 ieq_constr_scalar: Optional[List[Scalarization]] = None,
                 eq_constr_scalar: Optional[List[Scalarization]] = None,
                 **kwargs):

        check_dependency()
        self.obj_scalar = self.check_scalars(obj_scalar, n_obj)
        self.ieq_constr_scalar = self.check_scalars(ieq_constr_scalar, n_ieq_constr)
        self.eq_constr_scalar = self.check_scalars(eq_constr_scalar, n_eq_constr)
        self.scalars = self.obj_scalar + self.ieq_constr_scalar + self.eq_constr_scalar

        if uq_method is None:
            raise ValueError('A UQ method must be specified, e.g. uq_method=MonteCarlo(n_evaluations=100, seed=42)')
        if not isinstance(uq_method, UQMethod):
            raise ValueError(f'uq_method should be a UQMethod instance, got: {uq_method!r}')

        if param_space is None or param_space.n_parameters == 0:
            raise ValueError('param_space must contain valid parameter space')

        self.param_space = param_space
        self.uq_method = uq_method

        super().__init__(des_vars, n_obj=n_obj, n_ieq_constr=n_ieq_constr, n_eq_constr=n_eq_constr, **kwargs)

    @staticmethod
    def check_scalars(scalars: Optional[List[Scalarization]], n: int) -> List[Scalarization]:
        """Default unspecified responses to the expected value, and check the count"""
        if scalars is None:
            # Mean is the default robust scalar type
            return [Mean() for _ in range(n)]

        if len(scalars) != n:
            raise ValueError(f'scalars should have {n} Scalarization entries, got {len(scalars)}')
        for scalar in scalars:
            if not isinstance(scalar, Scalarization):
                raise ValueError(f'scalars should contain Scalarization instances, got: {scalar!r}')
        return list(scalars)

    def _print_extra_stats(self):
        print(f'stochastic           : True')
        print(f'n_stochastic_params  : {self.param_space.n_parameters}')
        print(f'uq_method            : {self.uq_method}')
        print(f'n_uq_samples         : {self.uq_method.n_evaluations}')
        print(f'obj_scalarize        : {self.obj_scalar}')
        print(f'ieq_constr_scalarize : {self.ieq_constr_scalar}')
        print(f'eq_constr_scalarize  : {self.eq_constr_scalar}')


    def _evaluate(self, x, out, *args, **kwargs):
        n = x.shape[0]
        f_stoch = np.empty((n, self.n_obj), dtype=object)
        g_stoch = np.empty((n, self.n_ieq_constr), dtype=object)
        h_stoch = np.empty((n, self.n_eq_constr), dtype=object)
        kwargs.update(f_stoch_out=f_stoch, g_stoch_out=g_stoch, h_stoch_out=h_stoch)

        super()._evaluate(x, out, *args, **kwargs)
        out['f_stochastic'] = f_stoch
        out['g_stochastic'] = g_stoch
        out['h_stochastic'] = h_stoch


    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, f_stoch_out: np.ndarray=None, g_stoch_out: np.ndarray=None, h_stoch_out: np.ndarray=None, **kwargs):
        """
        Evaluate architecture for the provided design vectors and samples.
        Implement _arch_evaluate_sample to evaluate architecture for single samples realizations.
        """

        if not self.design_space.is_explicit():
            self._correct_x_impute(x, is_active_out)

        # Get samples and include deterministic parameter values for evaluation
        samples = self.uq_method.get_samples(self.param_space)

        n_x, n_s = x.shape[0], samples.shape[0]

        f_s = np.zeros((n_x, n_s, self.n_obj))*np.nan
        g_s = np.zeros((n_x, n_s, self.n_ieq_constr))*np.nan
        h_s = np.zeros((n_x, n_s, self.n_eq_constr))*np.nan

        # Evaluate all design vectors for each realization of the uncertain parameters
        for i in range(n_s):
            self._arch_evaluate_sample(x, is_active_out, f_s[:, i, :], g_s[:, i, :], h_s[:, i, :], samples[i, :],*args, **kwargs)

        # Evaluate the stochastic result for all the evaluated design vectors and samples
        for x_i in range(n_x):
            outputs = self.uq_method.process_results(np.concatenate([f_s[x_i], g_s[x_i], h_s[x_i]], axis=1), self.param_space)

            # Reduce the sampled responses of each design point to the values the optimizer sees
            n_f, n_g = self.n_obj, self.n_ieq_constr
            for f_i, output in enumerate(outputs[:n_f]):
                f_stoch_out[x_i, f_i] = output
                obj_scalar = self.obj_scalar[f_i]
                f_out[x_i, f_i] = obj_scalar.scalarize(output) if not isinstance(output, float) else output
            for g_i, output in enumerate(outputs[n_f:n_f+n_g]):
                g_stoch_out[x_i, g_i] = output
                ieq_constr_scalar = self.ieq_constr_scalar[g_i]
                g_out[x_i, g_i] = ieq_constr_scalar.scalarize(output) if not isinstance(output, float) else output
            for h_i, output in enumerate(outputs[n_f+n_g:]):
                h_stoch_out[x_i, h_i] = output
                eq_constr_scalar = self.eq_constr_scalar[h_i]
                h_out[x_i, h_i] = eq_constr_scalar.scalarize(output) if not isinstance(output, float) else output

    def _arch_evaluate_sample(self, x: np.ndarray, is_active: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                              h_out: np.ndarray, parameters: np.ndarray, *args, **kwargs):
        """
        Implement evaluation and write results in the provided output matrices:
        - x (design vectors): discrete variables have integer values, imputed design vectors can be output here (except
                              if using an explicit design space definition)
        - is_active (activeness): vector specifying for each design variable whether it was active or not
        - f (objectives): written as a minimization
        - g (inequality constraints): written as "<= 0"
        - h (equality constraints): written as "= 0"
        - sample: realization of single sample
        """

        raise NotImplementedError