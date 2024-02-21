"""
MIT License

Copyright: (c) 2023, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
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
import logging
import numpy as np
from typing import Optional
from sb_arch_opt.problem import *
from sb_arch_opt.util import capture_log
from sb_arch_opt.algo.pymoo_interface.api import ResultsStorageCallback, ArchOptEvaluator
from sb_arch_opt.algo.pymoo_interface.storage_restart import initialize_from_previous_results
from ConfigSpace import ConfigurationSpace, Float, Integer, Categorical

import pymoo.core.variable as var
from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population
from pymoo.util.optimum import filter_optimum
from pymoo.core.initialization import Initialization
from pymoo.util.display.single import SingleObjectiveOutput

try:
    from tpe.optimizer import TPEOptimizer
    HAS_TPE = True
except ImportError:
    HAS_TPE = False

__all__ = ['HAS_TPE', 'ArchTPEInterface', 'TPEAlgorithm', 'initialize_from_previous_results']

log = logging.getLogger('sb_arch_opt.tpe')


def check_dependencies():
    if not HAS_TPE:
        raise RuntimeError('TPE dependencies not installed: pip install -e .[tpe]')


class ArchTPEInterface:
    """
    Class for interfacing the Tree-structured Parzen Estimator (TPE) optimization algorithm. For more info, see:

    Bergstra et al., "Algorithms for Hyper-Parameter Optimization", 2011, available at:
    https://papers.nips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf

    Currently only supports single-objective unconstrained problems.
    """

    def __init__(self, problem: ArchOptProblemBase):
        check_dependencies()
        capture_log()

        if problem.n_obj != 1:
            raise ValueError('Currently only single-objective problems are supported!')
        if problem.n_ieq_constr != 0 or problem.n_eq_constr != 0:
            raise ValueError('Currently only unconstrained problems are supported!')

        self._problem = problem
        self._optimizer: Optional['TPEOptimizer'] = None

    def initialize(self):
        self._optimizer = self._get_optimizer()

    def ask_init(self):
        if self._optimizer is None:
            self.initialize()
        return self._convert_to_x(self._optimizer.initial_sample())

    def ask(self):
        if self._optimizer is None:
            self.initialize()
        return self._convert_to_x(self._optimizer.sample())

    def _convert_to_x(self, config):
        is_cat_mask = self._problem.is_cat_mask

        x = []
        for ix in range(self._problem.n_var):
            key = f'x{ix}'
            x.append(int(config[key]) if is_cat_mask[ix] else config[key])
        return np.array([x])

    def tell(self, x: np.ndarray, f: float):
        assert x.shape == (self._problem.n_var,)
        assert self._optimizer is not None

        out_config = {}
        is_cat_mask = self._problem.is_cat_mask
        for ix in range(self._problem.n_var):
            key = f'x{ix}'
            out_config[key] = str(int(x[ix])) if is_cat_mask[ix] else x[ix]

        # Report outputs
        results = {'f': f}
        self._optimizer.update(out_config, results, runtime=0.)

    def optimize(self, n_init: int, n_infill: int):
        self.initialize()

        x_results, f_results = [], []
        for i_iter in range(n_init+n_infill):
            is_init = i_iter < n_init
            log.info(f'Iteration {i_iter+1}/{n_init+n_infill} ({"init" if is_init else "infill"})')

            # Get next point to evaluate
            x_eval = self.ask_init() if is_init else self.ask()

            # Evaluate
            out = self._problem.evaluate(x_eval, return_as_dictionary=True)

            x_out = out['X'][0, :]
            f = out['F'][0, 0]
            self.tell(x_out, f)

            log.info(f'Evaluated: {f:.3g} @ {x_out}')
            x_results.append(x_out)
            f_results.append(f)

        x_results, f_results = np.array(x_results), np.array(f_results)
        return x_results, f_results

    def _get_optimizer(self):
        return TPEOptimizer(
            obj_func=lambda *args, **kwargs: None,  # We're using the ask-tell interface
            config_space=self._get_config_space(),
            metric_name='f',
            result_keys=['f'],
        )

    def _get_config_space(self):
        params = {}
        for i, dv in enumerate(self._problem.des_vars):
            name = f'x{i}'
            if isinstance(dv, var.Real):
                params[name] = Float(name, bounds=dv.bounds)
            elif isinstance(dv, var.Integer):
                params[name] = Integer(name, bounds=dv.bounds)
            elif isinstance(dv, var.Binary):
                params[name] = Integer(name, bounds=(0, 1))
            elif isinstance(dv, var.Choice):
                params[name] = Categorical(name, items=[str(i) for i in range(len(dv.options))])
            else:
                raise ValueError(f'Unknown variable type: {dv!r}')

        return ConfigurationSpace(space=params)


class TPEInitialization(Initialization):

    def __init__(self):
        self.interface: Optional[ArchTPEInterface] = None
        super().__init__(sampling=None)

    def do(self, problem, n_samples, **kwargs):
        x_init = np.row_stack([self.interface.ask_init() for _ in range(n_samples)])
        return Population.new(X=x_init)


class TPEAlgorithm(Algorithm):
    """
    The Tree-structured Parzen Estimator (TPE) optimization algorithm implemented as a pymoo Algorithm.

    Note that through pymoo itself you can also access Optuna's TPE algorithm, however that one does not support design
    space hierarchy like SBArchOpt supports it.
    """

    def __init__(self, n_init: int, results_folder=None, output=SingleObjectiveOutput(), **kwargs):
        self._interface: Optional[ArchTPEInterface] = None
        self.n_init = n_init
        self.initialization = TPEInitialization()

        evaluator = ArchOptEvaluator(results_folder=results_folder)
        callback = ResultsStorageCallback(results_folder) if results_folder is not None else None

        super().__init__(evaluator=evaluator, callback=callback, output=output, **kwargs)

    def _setup(self, problem, **kwargs):
        if not isinstance(problem, ArchOptProblemBase):
            raise RuntimeError('The TPE algorithm only works with SBArchOpt problem definitions!')

        self._interface = interface = ArchTPEInterface(problem)
        interface.initialize()

        if isinstance(self.initialization, TPEInitialization):
            self.initialization.interface = self._interface

    def _initialize_infill(self):
        return self.initialization.do(self.problem, self.n_init)

    def _infill(self):
        return Population.new(X=self._interface.ask())

    def _initialize_advance(self, infills=None, **kwargs):
        self._advance(infills, is_init=True, **kwargs)

    def _advance(self, infills=None, is_init=False, **kwargs):
        if not is_init:
            self.pop = Population.merge(self.pop, infills)
        x, f = infills.get('X'), infills.get('F')
        for i in range(len(infills)):
            self._interface.tell(x[i, :], f[i, 0])

    def _set_optimum(self):
        pop = self.pop
        if self.opt is not None:
            pop = Population.merge(self.opt, pop)
        self.opt = filter_optimum(pop, least_infeasible=True)
