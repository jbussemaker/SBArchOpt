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
import os
import logging
import numpy as np
from typing import Tuple, Optional
from sb_arch_opt.sampling import *
from sb_arch_opt.algo.arch_sbo.models import *

from sb_arch_opt.util import capture_log
from pymoo.core.population import Population
from sb_arch_opt.problem import ArchOptProblemBase
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from sb_arch_opt.algo.arch_sbo.models import ModelFactory

if HAS_SMT:
    from smt.surrogate_models.krg_based import MixIntKernelType
    
try:
    from segomoe.sego import Sego
    from segomoe.constraint import Constraint
    from segomoe.sego_defs import get_sego_file_map, ExitStatus

    HAS_SEGOMOE = True
except ImportError:
    HAS_SEGOMOE = False

__all__ = ['HAS_SEGOMOE', 'HAS_SMT', 'check_dependencies', 'SEGOMOEInterface']

log = logging.getLogger('sb_arch_opt.segomoe')


def check_dependencies():
    if not HAS_SEGOMOE:
        raise ImportError('SEGOMOE not installed!')


class SEGOMOEInterface:
    """
    Class for interfacing with SEGOMOE
    """

    def __init__(self, problem: ArchOptProblemBase, results_folder: str, n_init: int, n_infill: int, use_moe=True,
                 sego_options=None, model_options=None, verbose=True):
        check_dependencies()
        self._problem = problem
        self._results_folder = results_folder
        self.n_init = n_init
        self.n_infill = n_infill
        self.use_moe = use_moe
        self.sego_options = sego_options or {}
        self.model_options = model_options or {}
        self.verbose = verbose

        self._x = None
        self._x_failed = None
        self._y = None

    @property
    def x(self) -> np.ndarray:
        """Design vectors"""
        if self._x is None:
            return np.zeros((0, self._problem.n_var))
        return self._x

    @property
    def n(self) -> int:
        """Number of available successfully evaluated points"""
        return 0 if self._x is None else self._x.shape[0]

    @property
    def x_failed(self) -> np.ndarray:
        """Design vectors"""
        if self._x_failed is None:
            return np.zeros((0, self._problem.n_var))
        return self._x_failed

    @property
    def n_failed(self) -> int:
        """Number of failed points"""
        return 0 if self._x_failed is None else self._x_failed.shape[0]

    @property
    def n_tried(self):
        """Number of points that were tried to be evaluated (n + n_failed)"""
        return self.n + self.n_failed

    @property
    def y(self) -> np.ndarray:
        """All outputs: f, g, h"""
        if self._y is None:
            p = self._problem
            return np.zeros((0, p.n_obj + p.n_ieq_constr + p.n_eq_constr))
        return self._y

    @property
    def f(self) -> np.ndarray:
        """Objective values"""
        f, _, _ = self._split_y(self.y)
        return f

    @property
    def g(self) -> np.ndarray:
        """Inequality constraints"""
        _, g, _ = self._split_y(self.y)
        return g

    @property
    def h(self) -> np.ndarray:
        """Equality constraints"""
        _, _, h = self._split_y(self.y)
        return h

    @property
    def pop(self) -> Population:
        """Population of all evaluated points"""
        return self.get_population(self.x, self.y, self.x_failed)

    @property
    def opt(self) -> Population:
        """Optimal points (Pareto front if multi-objective)"""
        return self._get_pareto_front(self.pop)

    @property
    def results_folder(self):
        return self._results_folder

    def initialize_from_previous(self, results_folder: str = None):
        capture_log()
        if results_folder is None:
            results_folder = self._results_folder

        # Load from problem state
        population = self._problem.load_previous_results(results_folder)
        if population is not None:
            self._x, self._x_failed, self._y = self._get_xy(population)
            log.info(f'Previous results loaded from problem results: {len(population)} design points '
                     f'({self.n} ok, {self.n_failed} failed)')
            return

        # Load from optimizer state
        x_path, y_path, x_failed_path = self._get_doe_paths()
        if os.path.exists(x_path) and os.path.exists(y_path):
            self._x = np.load(x_path)

            if os.path.exists(x_failed_path):
                self._x_failed = np.load(x_failed_path)
            else:
                self._x_failed = np.zeros((0, self._problem.n_var))

            # Flip inequality constraints, as the problem defines satisfaction G <= 0 but SEGOMOE saves it as opposite
            self._y = self._flip_g(np.load(y_path))

            log.info(f'Previous results loaded from optimizer state: {self._x.shape[0]} design points '
                     f'({self.n} ok, {self.n_failed} failed)')
            return

        log.info('No previous results found')

    def set_pop(self, pop: Population = None):
        if pop is None:
            self._x = self._x_failed = self._y = None
        else:
            self._x, self._x_failed, self._y = self._get_xy(pop)

    def run_optimization(self):
        capture_log()

        n_doe, n_infills = self._optimization_step()
        if n_doe is not None:
            log.info(f'Running DOE of {n_doe} points ({self.n_init} total)')
            self.run_doe(n_doe)

        if n_infills is not None:
            log.info(f'Running optimization: {n_infills} infill points (ok DOE points: {self.n})')
            self.run_infills(n_infills)

        # Save final results and return Pareto front
        self._save_results()
        return self.opt

    def _optimization_step(self):
        # Automatically initialize from previous results if reusing the same storage folder
        if self._x is None:
            self.initialize_from_previous()

        # Run DOE if needed
        n_available = self.n_tried
        n_doe = None
        if n_available < self.n_init:
            n_doe = self.n_init-n_available

        # Run optimization (infills)
        n_available = self.n_tried + (n_doe or 0)
        n_infills = None
        if n_available < self.n_init+self.n_infill:
            n_infills = self.n_infill - (n_available-self.n_init)

        return n_doe, n_infills

    def optimization_has_ask(self):
        n_doe, n_infills = self._optimization_step()
        return n_doe is not None or n_infills is not None

    def optimization_ask(self) -> Optional[np.ndarray]:
        n_doe, n_infills = self._optimization_step()

        if n_doe is not None:
            return self._sample_doe(n_doe)

        if n_infills is not None:
            return np.array([self._ask_infill()])

    def optimization_tell_pop(self, pop: Population):
        self._tell_infill(*self._get_xy(pop))

    def optimization_tell(self, x, x_failed, y):
        self._tell_infill(x, x_failed, y)

    def run_doe(self, n: int = None):
        if n is None:
            n = self.n_init

        x_doe = self._sample_doe(n)
        self._x, self._x_failed, self._y = self._get_xy(self._evaluate(x_doe))

        if self._x.shape[0] < 2:
            log.info(f'Not enough points sampled ({self._x.shape[0]} success, {self._x_failed.shape[0]} failed),'
                     f'problems with model fitting can be expected')

        self._save_results()

    def _sample_doe(self, n: int) -> np.ndarray:
        return HierarchicalSampling().do(self._problem, n).get('X')
    
    def run_infills(self, n_infills: int = None):
        if n_infills is None:
            n_infills = self.n_infill
        i_eval = 0
    
        def _grouped_eval(x):
             nonlocal i_eval
             i_eval += 1
             log.info(f"Evaluating: {i_eval}/{n_infills}")
    
             x, x_failed, y = self._get_xy(self._evaluate(np.array([x])))
    
             self._tell_infill(x, x_failed, y)
    
             if len(x_failed) > 0:
                 return [], True
             return y[0, :], False
    
        log.info(
            f"Running SEGO for {n_infills} infills ({self._x.shape[0]} points in database)"
        )
        sego = self._get_sego(_grouped_eval)
        sego.run_optim(n_iter=n_infills)   
        
    def run_infills_ask_tell(self, n_infills: int = None):
        if n_infills is None:
            n_infills = self.n_infill

        for i in range(n_infills):
            # Ask for a new infill point
            log.info(f'Getting new infill point {i+1}/{n_infills} (point {self._x.shape[0]+1} overall)')
            x = self._ask_infill()

            # Evaluate and impute
            log.info(f'Evaluating point {i+1}/{n_infills} (point {self._x.shape[0]+1} overall)')
            x, x_failed, y = self._get_xy(self._evaluate(np.array([x])))

            # Update and save DOE
            self._tell_infill(x, x_failed, y)

    def _ask_infill(self) -> np.ndarray:
        """
        Ask for one infill point, we do this in order to support imputation of the design vector.
        Implementation inspired by:
        https://github.com/OneraHub/WhatsOpt/blob/master/services/whatsopt_server/optimizer_store/segomoe_optimizer.py
        https://github.com/OneraHub/WhatsOpt/blob/master/services/whatsopt_server/optimizer_store/segmoomoe_optimizer.py
        """

        def _dummy_f_grouped(_):
            return np.max(self._y, axis=1), False

        sego = self._get_sego(_dummy_f_grouped)
        res = sego.run_optim(n_iter=1)
        if res is not None and res[0] == ExitStatus.runtime_error[0]:
            raise RuntimeError(f'Error during SEGOMOE infill search: {res[0]}')

        # Return latest point as suggested infill point
        return sego.get_x(i=-1)

    def _tell_infill(self, x, x_failed, y):
        self._x = np.row_stack([self._x, x]) if self._x is not None else x
        self._y = np.row_stack([self._y, y]) if self._y is not None else y
        self._x_failed = np.row_stack([self._x_failed, x_failed]) if self._x_failed is not None else x_failed
        self._save_results()

    def _get_sego(self, f_grouped):
        design_space_spec = self._get_design_space()

        if design_space_spec.is_mixed_discrete:
            model_type = {
                "type": "MIXED",
                "name": "KRG",
                "regr": "constant",
                "corr": "squar_exp",
                "design_space": design_space_spec.design_space,
                "categorical_kernel": MixIntKernelType.GOWER,
                "theta0": [1e-3],
                "thetaL": [1e-6],
                "thetaU": [10.0],
                "normalize": True,
                **self.model_options,
            }
        else:
            model_type = {
                "name": "KRG",
                "regr": "constant",
                "corr": "squar_exp",
                "design_space": design_space_spec.design_space,
                "categorical_kernel": MixIntKernelType.GOWER,
                "theta0": [1e-3],
                "thetaL": [1e-6],
                "thetaU": [10.0],
                "normalize": True,
                **self.model_options,
            }
       
        optim_settings = {
            'grouped_eval': True,
            'n_obj': self._problem.n_obj,
            'model_type': {'obj': model_type, 'con': model_type},
            'n_clusters': 0 if self.use_moe else 1,
            'optimizer': 'slsqp',
            'analytical_diff': False,
            'profiling': False,
            'verbose': self.verbose,
            'cst_crit': 'MC',
            **self.sego_options,
        }

        return Sego(
            fun=f_grouped,
            var=design_space_spec.var_defs,
            const=self._get_constraints(),
            optim_settings=optim_settings,
            path_hs=self._results_folder,
            comm=None,
        )

    def _get_design_space(self):
        return ModelFactory(self._problem).get_smt_design_space_spec()

    def _get_constraints(self):
        constraints = []
        for i in range(self._problem.n_ieq_constr):
            constraints.append(Constraint(con_type='<', bound=0., name=f'g{i}'))
        for i in range(self._problem.n_eq_constr):
            constraints.append(Constraint(con_type='=', bound=0., name=f'h{i}'))
        return constraints

    def _save_results(self):
        x_path, y_path, x_failed_path = self._get_doe_paths()
        if self._x is not None:
            np.save(x_path, self._x)
        if self._y is not None:
            # Flip inequality constraints, as SEGOMOE stores them as G >= 0, however the problem defines it as opposite
            np.save(y_path, self._flip_g(self._y))

        if self._x_failed is not None and self._x_failed.shape[0] > 0:
            np.save(x_failed_path, self._x_failed)
        elif os.path.exists(x_failed_path):
            os.remove(x_failed_path)

        self._problem.store_results(self._results_folder)

    def _get_doe_paths(self):
        return self._get_sego_file_path('x'), self._get_sego_file_path('y'), self._get_sego_file_path('x_fails')

    def _get_sego_file_path(self, key):
        return os.path.join(self._results_folder, get_sego_file_map()[key])

    def _evaluate(self, x: np.ndarray) -> Population:
        """
        Evaluates a list of design points (x is a matrix of size n x nx). A population is returned with matrices:
        - X: imputed design vectors
        - is_active: activeness vectors (booleans defining which design variable is active in each design vector)
        - F: objective values
        - G: inequality constraints (None if there are no inequality constraints)
        - H: equality constraints (None if there are no equality constraints)
        """
        out = self._problem.evaluate(x, return_as_dictionary=True)
        return Population.new(**out)

    def _get_xy(self, population: Population) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Concatenate evaluation outputs (F, G, H) and split x into evaluated and failed points.
        Returns: x, x_failed, y"""

        # Concatenate outputs
        outputs = [population.get('F')]
        if self._problem.n_ieq_constr > 0:
            outputs.append(population.get('G'))
        if self._problem.n_eq_constr > 0:
            outputs.append(population.get('H'))
        y = np.column_stack(outputs)

        # Split x into ok and failed points
        x = population.get('X')
        is_failed = self._problem.get_failed_points(population)
        x_failed = x[is_failed, :]
        x = x[~is_failed, :]
        y = y[~is_failed, :]

        return x, x_failed, y

    def _split_y(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split outputs (y) into F, G, H"""
        f, y = np.split(y, [self._problem.n_obj], axis=1)

        if self._problem.n_ieq_constr > 0:
            g, y = np.split(y, [self._problem.n_ieq_constr], axis=1)
        else:
            g = np.zeros((y.shape[0], 0))

        if self._problem.n_eq_constr > 0:
            h = y[:, :self._problem.n_eq_constr]
        else:
            h = np.zeros((y.shape[0], 0))

        return f, g, h

    def _flip_g(self, y: np.ndarray):
        f, g, h = self._split_y(y)
        g = -g
        return np.column_stack([f, g, h])

    def get_population(self, x: np.ndarray, y: np.ndarray, x_failed: np.ndarray = None) -> Population:
        # Inequality constraint values are flipped to correctly calculate constraint violation values in pymoo
        f, g, h = self._split_y(y)

        if x_failed is not None and len(x_failed) > 0:
            x = np.row_stack([x, x_failed])
            f = np.row_stack([f, np.zeros((x_failed.shape[0], f.shape[1]))*np.inf])
            g = np.row_stack([g, np.zeros((x_failed.shape[0], g.shape[1]))*np.inf])
            h = np.row_stack([h, np.zeros((x_failed.shape[0], h.shape[1]))*np.inf])

        kwargs = {'X': x, 'F': f, 'G': g, 'H': h}
        pop = Population.new(**kwargs)
        return pop

    @staticmethod
    def _get_pareto_front(population: Population) -> Population:
        f = population.get('F')
        if f.shape[0] == 0:
            return population.copy()

        f = f[population.get('feas')]
        if f.shape[0] == 0:
            return population.copy()

        i_nds = NonDominatedSorting().do(f, only_non_dominated_front=True)
        return population[i_nds]
