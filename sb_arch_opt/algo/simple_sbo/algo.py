"""
Licensed under the GNU General Public License, Version 3.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.gnu.org/licenses/gpl-3.0.html.en

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Copyright: (c) 2023, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de
"""
import copy
import timeit
import logging
import numpy as np
from typing import *
from sb_arch_opt.sampling import *
from sb_arch_opt.util import patch_ftol_bug
from sb_arch_opt.algo.pymoo_interface import *
from sb_arch_opt.problem import ArchOptProblemBase

from pymoo.core.repair import Repair
from pymoo.core.result import Result
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.core.sampling import Sampling
from pymoo.core.survival import Survival
from pymoo.core.algorithm import Algorithm
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population
from pymoo.core.infill import InfillCriterion
from pymoo.core.termination import Termination
from pymoo.core.initialization import Initialization
from pymoo.core.duplicate import DuplicateElimination
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.termination.default import DefaultMultiObjectiveTermination, DefaultSingleObjectiveTermination
from pymoo.optimize import minimize

try:
    from smt.surrogate_models.surrogate_model import SurrogateModel
    from sb_arch_opt.algo.simple_sbo.infill import *
except ImportError:
    pass

__all__ = ['InfillAlgorithm', 'SBOInfill', 'SurrogateInfillCallback', 'SurrogateInfillOptimizationProblem',
           'NormalizedRepair']

log = logging.getLogger('sb_arch_opt.sbo')


class InfillAlgorithm(Algorithm):
    """
    Algorithm that simpy uses some InfillCriterion to get infill points.
    The algorithm is compatible with the ask-tell interface.
    """

    def __init__(self, infill: InfillCriterion, infill_size=None, init_sampling: Sampling = None, init_size=100,
                 survival: Survival = None, **kwargs):
        super(InfillAlgorithm, self).__init__(**kwargs)

        self.init_size = init_size
        self.infill_size = infill_size or self.init_size
        self.infill_obj = infill

        if init_sampling is None:
            init_sampling = LatinHypercubeSampling()
        self.initialization = Initialization(
            init_sampling, repair=infill.repair, eliminate_duplicates=infill.eliminate_duplicates)
        self.survival = survival

        if self.output is None:
            from sb_arch_opt.algo.simple_sbo.metrics import SBOMultiObjectiveOutput
            self.output = SBOMultiObjectiveOutput()

    def _initialize(self):
        super(InfillAlgorithm, self)._initialize()

        self.pop = pop = self.initialization.do(self.problem, self.init_size, algorithm=self)
        self.evaluator.eval(self.problem, pop, algorithm=self)
        if self.survival is not None:
            self.pop = self.survival.do(self.problem, pop, len(pop), algorithm=self)
        self.is_initialized = True

    def _infill(self):
        off = self.infill_obj.do(self.problem, self.pop, self.infill_size, algorithm=self)

        # Stop if no new offspring is generated
        if len(off) == 0:
            self.termination.force_termination = True

        return off

    def _advance(self, infills=None, **kwargs):
        self.pop = Population.merge(self.pop, infills)
        if self.survival is not None:
            self.pop = self.survival.do(self.problem, self.pop, self.init_size, algorithm=self)

    def store_intermediate_results(self, results_folder: str):
        """Enable intermediate results storage to support restarting"""
        self.callback = ResultsStorageCallback(results_folder, callback=self.callback)

    def initialize_from_previous_results(self, problem: ArchOptProblemBase, result_folder: str) -> bool:
        """Initialize the SBO algorithm from previously stored results"""
        return initialize_from_previous_results(self, problem, result_folder)


class SBOInfill(InfillCriterion):
    """The main implementation of the SBO infill search"""

    _exclude = ['_surrogate_model', 'opt_results']

    def __init__(self, surrogate_model: 'SurrogateModel', infill: SurrogateInfill, pop_size=None,
                 termination: Union[Termination, int] = None, verbose=False, repair: Repair = None,
                 eliminate_duplicates: DuplicateElimination = None, force_new_points: bool = True, **kwargs):

        if eliminate_duplicates is None:
            eliminate_duplicates = LargeDuplicateElimination()
        super(SBOInfill, self).__init__(repair=repair, eliminate_duplicates=eliminate_duplicates, **kwargs)

        self._is_init = None
        self.problem: Optional[Problem] = None
        self.total_pop: Optional[Population] = None
        self._algorithm: Optional[Algorithm] = None

        self._surrogate_model_base = surrogate_model
        self._surrogate_model = None
        self.infill = infill

        self.x_train = None
        self.y_train = None
        self.y_train_min = None
        self.y_train_max = None
        self.y_train_centered = None
        self.n_train = 0
        self.time_train = None
        self.pf_estimate = None

        self.pop_size = pop_size or 100
        self.termination = termination
        self.verbose = verbose
        self.force_new_points = force_new_points

        self.opt_results: Optional[List[Result]] = None

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in self._exclude:
            state[key] = None
        return state

    def algorithm(self, infill_size=None, init_sampling: Sampling = None, init_size=100, survival: Survival = None,
                  **kwargs) -> InfillAlgorithm:
        if init_sampling is None and self.repair is not None:
            init_sampling = RepairedLatinHypercubeSampling(self.repair)
        return InfillAlgorithm(self, infill_size=infill_size, init_sampling=init_sampling, init_size=init_size,
                               survival=survival, **kwargs)

    def do(self, problem, pop, n_offsprings, **kwargs):
        self._algorithm = kwargs.pop('algorithm', None)

        # Check if we need to initialize
        if self._is_init is None:
            self.problem = problem
            self._is_init = problem

            self._initialize()

        elif self._is_init is not problem:
            raise RuntimeError('An instance of a ModelBasedInfillCriterion can only be used with one Problem!')

        # (Re-)build the surrogate model
        if self.total_pop is None:
            self.total_pop = pop
            new_population = pop
        else:
            new_population = self.eliminate_duplicates.do(pop, self.total_pop)
            self.total_pop = Population.merge(self.total_pop, new_population)

        self._build_model(new_population)

        # Search the surrogate model for infill points
        off = self._generate_infill_points(n_offsprings)

        off = self.repair.do(problem, off, **kwargs)
        off = self.eliminate_duplicates.do(off, pop)

        if self.verbose:
            n_eval_outer = self._algorithm.evaluator.n_eval if self._algorithm is not None else -1
            log.info(f'Infill: {len(off)} new (eval {len(self.total_pop)} real unique, {n_eval_outer} eval)')

        return off

    def _do(self, *args, **kwargs):
        raise RuntimeError

    @property
    def name(self):
        return f'{self._surrogate_model_base.__class__.__name__} / {self.infill.__class__.__name__}'

    @property
    def surrogate_model(self) -> 'SurrogateModel':
        if self._surrogate_model is None:
            self._surrogate_model = copy.deepcopy(self._surrogate_model_base)

            if self.infill.needs_variance and not self.supports_variances:
                raise ValueError(
                    f'Provided surrogate infill ({self.infill.__class__.__name__}) needs variances, but these are not '
                    f'supported by the underlying surrogate model ({self.surrogate_model.__class__.__name__})!')

        return self._surrogate_model

    @property
    def supports_variances(self):
        return self._surrogate_model_base.supports['variances']

    def _initialize(self):
        self.infill.initialize(self.problem, self.surrogate_model)

    def _build_model(self, _: Population):
        """Update the underlying model. New population is given, total population is available from self.total_pop"""

        # Get input training points (x) and normalize
        x = self.total_pop.get('X')
        x_norm = self._normalize(x)

        # Get output training points (y, from f and g) and normalize
        f_real = self.total_pop.get('F')
        f_is_invalid = ~np.isfinite(f_real)
        f_real[f_is_invalid] = np.nan

        f, self.y_train_min, self.y_train_max = self._normalize_y(f_real)
        self.y_train_centered = [False]*f.shape[1]

        g = np.zeros((x.shape[0], 0))
        if self.problem.n_constr > 0:
            g_real = self.total_pop.get('G')
            g_is_invalid = ~np.isfinite(g_real)
            g_real[g_is_invalid] = np.nan

            g, g_min, g_max = self._normalize_y(g_real, keep_centered=True)

            self.y_train_min = np.append(self.y_train_min, g_min)
            self.y_train_max = np.append(self.y_train_max, g_max)
            self.y_train_centered += [True]*g.shape[1]

        # Replace invalid points with the current maximum (i.e. worst) values known
        f_is_nan = np.any(np.isnan(f), axis=1)
        f[f_is_nan] = np.nanmax(f, axis=0)

        g_is_nan = np.any(np.isnan(g), axis=1)
        g[g_is_nan] = 1.

        y = np.append(f, g, axis=1)

        # Train the model
        self.x_train = x_norm
        self.y_train = y

        self.pf_estimate = None
        self._train_model()

    def _train_model(self):
        s = timeit.default_timer()
        self.surrogate_model.set_training_values(self.x_train, self.y_train)
        self.infill.set_samples(self.x_train, self.y_train)

        self.surrogate_model.train()
        self.n_train += 1
        self.time_train = timeit.default_timer()-s

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return normalize(x, self.problem.xl, self.problem.xu)

    def _denormalize(self, x_norm: np.ndarray) -> np.ndarray:
        return denormalize(x_norm, self.problem.xl, self.problem.xu)

    @staticmethod
    def _normalize_y(y: np.ndarray, keep_centered=False, y_min=None, y_max=None):
        if y_min is None:
            y_min = np.nanmin(y, axis=0)
        if y_max is None:
            y_max = np.nanmax(y, axis=0)

        norm = y_max-y_min
        norm[norm < 1e-6] = 1e-6

        if keep_centered:
            return y/norm, y_min, y_max
        return (y-y_min)/norm, y_min, y_max

    @staticmethod
    def _denormalize_y(y_norm: np.ndarray, keep_centered=False, y_min=None, y_max=None):
        norm = y_max-y_min
        norm[norm < 1e-6] = 1e-6

        if keep_centered:
            return y_norm*norm
        return (y_norm*norm) + y_min

    def _generate_infill_points(self, n_infill: int) -> Population:
        # Create infill problem and algorithm
        problem = self._get_infill_problem()
        algorithm = self._get_infill_algorithm()
        termination = self._get_termination(n_obj=problem.n_obj)

        n_callback = 20
        if isinstance(termination, MaximumGenerationTermination):
            n_callback = int(termination.n_max_gen/5)

        # Run infill problem
        n_eval_outer = self._algorithm.evaluator.n_eval if self._algorithm is not None else -1
        result = minimize(
            problem, algorithm,
            termination=termination,
            callback=SurrogateInfillCallback(n_gen_report=n_callback, verbose=self.verbose,
                                             n_points_outer=len(self.total_pop), n_eval_outer=n_eval_outer),
            copy_termination=False,  # Needed for maintaining the patch
            # verbose=True, progress=True,
        )
        if self.opt_results is None:
            self.opt_results = []
        self.opt_results.append(result)

        # Select infill points and denormalize the design vectors
        selected_pop = self.infill.select_infill_solutions(result.pop, problem, n_infill)
        result.opt = selected_pop

        x = self._denormalize(selected_pop.get('X'))
        return Population.new(X=x)

    def get_pf_estimate(self) -> Optional[Population]:
        """Estimate the location of the Pareto front as predicted by the surrogate model"""

        if self.problem is None or self.n_train == 0:
            return
        if self.pf_estimate is not None:
            return self.pf_estimate

        infill = FunctionEstimateInfill()
        infill.initialize(self.problem, self.surrogate_model)
        infill.set_samples(self.x_train, self.y_train)

        problem = self._get_infill_problem(infill, force_new_points=False)
        algorithm = self._get_infill_algorithm()
        termination = self._get_termination(n_obj=problem.n_obj)

        result = minimize(problem, algorithm, termination=termination, copy_termination=False)

        selected_pop = infill.select_infill_solutions(result.pop, problem, 100)

        y_min, y_max = self.y_train_min, self.y_train_max
        f_min, f_max = y_min[:self.problem.n_obj], y_max[:self.problem.n_obj]
        self.pf_estimate = self._denormalize_y(selected_pop.get('F'), y_min=f_min, y_max=f_max)
        return self.pf_estimate

    def _get_infill_problem(self, infill: SurrogateInfill = None, force_new_points=None):
        if infill is None:
            infill = self.infill
        if force_new_points is None:
            force_new_points = self.force_new_points

        x_exist_norm = self._normalize(self.total_pop.get('X')) if force_new_points else None
        return SurrogateInfillOptimizationProblem(infill, self.problem, x_exist_norm=x_exist_norm)

    def _get_termination(self, n_obj):
        termination = self.termination
        if termination is None or not isinstance(termination, Termination):
            # return MaximumGenerationTermination(n_max_gen=termination or 100)
            robust_period = 5
            n_max_gen = termination or 100
            n_max_eval = n_max_gen*self.pop_size
            if n_obj > 1:
                termination = DefaultMultiObjectiveTermination(
                    xtol=5e-4, cvtol=1e-8, ftol=5e-3, n_skip=5, period=robust_period, n_max_gen=n_max_gen,
                    n_max_evals=n_max_eval)
            else:
                termination = DefaultSingleObjectiveTermination(
                    xtol=1e-8, cvtol=1e-8, ftol=1e-6, period=robust_period, n_max_gen=n_max_gen, n_max_evals=n_max_eval)

        patch_ftol_bug(termination)
        return termination

    def _get_infill_algorithm(self):
        repair = self._get_infill_repair()
        return NSGA2(pop_size=self.pop_size, sampling=RepairedLatinHypercubeSampling(repair), repair=repair)

    def _get_infill_repair(self):
        if self.repair is None:
            return None
        return NormalizedRepair(self.problem, self.repair)


class SurrogateInfillCallback(Callback):
    """Callback for printing infill optimization progress."""

    def __init__(self, n_gen_report=20, verbose=False, n_points_outer=0, n_eval_outer=0):
        super(SurrogateInfillCallback, self).__init__()
        self.n_gen_report = n_gen_report
        self.verbose = verbose
        self.n_points_outer = n_points_outer
        self.n_eval_outer = n_eval_outer

    def notify(self, algorithm: Algorithm, **kwargs):
        if self.verbose and algorithm.n_gen % self.n_gen_report == 0:
            log.info(f'Surrogate infill gen {algorithm.n_gen} @ {algorithm.evaluator.n_eval} points evaluated '
                     f'({self.n_points_outer} real unique, {self.n_eval_outer} eval)')


class SurrogateInfillOptimizationProblem(Problem):
    """Problem class representing a surrogate infill problem given a SurrogateInfill instance."""

    def __init__(self, infill: SurrogateInfill, problem: Problem, x_exist_norm: np.ndarray = None):
        n_var = problem.n_var
        xl, xu = np.zeros(n_var), np.ones(n_var)

        n_obj = infill.get_n_infill_objectives()
        n_constr = infill.get_n_infill_constraints()

        self.pop_exist_norm = Population.new(X=x_exist_norm) if x_exist_norm is not None else None
        self.force_new_points = x_exist_norm is not None
        if self.force_new_points:
            n_constr += 1
        self.eliminate_duplicates = LargeDuplicateElimination()

        super(SurrogateInfillOptimizationProblem, self).__init__(
            n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)

        self.infill = infill

    def _evaluate(self, x, out, *args, **kwargs):
        # Get infill search objectives and constraints
        f, g = self.infill.evaluate(x)

        if f.shape != (x.shape[0], self.n_obj):
            raise RuntimeError(f'Wrong objective results shape: {f.shape!r} != {(x.shape[0], self.n_obj)!r}')
        out['F'] = f

        if g is None and self.n_constr > 0:
            g = np.zeros((x.shape[0], 0))

        # Add additional constraint to force the selection of new (discrete) points
        if self.force_new_points:
            g_force_new = np.zeros((x.shape[0],))
            _, _, is_dup = self.eliminate_duplicates.do(Population.new(X=x), self.pop_exist_norm, return_indices=True)
            g_force_new[is_dup] = 1.

            g = np.column_stack([g, g_force_new])

        if g.shape != (x.shape[0], self.n_constr):
            raise RuntimeError(f'Wrong constraint results shape: {g.shape!r} != {(x.shape[0], self.n_constr)!r}')
        out['G'] = g


class NormalizedRepair(Repair):
    """Repair to be used during infill search: the infill search space is normalized compared to the original problem"""

    def __init__(self, problem: Problem, repair: Repair):
        super().__init__()
        self._problem = problem
        self._repair = repair

    def _do(self, problem, pop, **kwargs):
        is_array = not isinstance(pop, Population)
        x = pop if is_array else pop.get("X")

        x_underlying = denormalize(x, self._problem.xl, self._problem.xu)
        x_underlying = self._repair.do(self._problem, Population.new(X=x_underlying), **kwargs).get("X")
        x = normalize(x_underlying, self._problem.xl, self._problem.xu)

        if is_array:
            return x
        pop.set("X", x)
        return pop
