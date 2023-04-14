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
from sb_arch_opt.problem import *
from sb_arch_opt.sampling import *
from scipy.spatial import distance
from sb_arch_opt.algo.pymoo_interface import *

from pymoo.core.repair import Repair
from pymoo.core.result import Result
from pymoo.core.variable import Real
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.core.sampling import Sampling
from pymoo.core.survival import Survival
from pymoo.core.algorithm import Algorithm
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population
from pymoo.core.infill import InfillCriterion
from pymoo.util.optimum import filter_optimum
from pymoo.core.termination import Termination
from pymoo.util.normalization import Normalization
from pymoo.core.initialization import Initialization
from pymoo.core.duplicate import DuplicateElimination
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.termination.default import DefaultMultiObjectiveTermination, DefaultSingleObjectiveTermination
from pymoo.optimize import minimize

try:
    from smt.surrogate_models.surrogate_model import SurrogateModel
    from sb_arch_opt.algo.simple_sbo.infill import *
    from sb_arch_opt.algo.simple_sbo.models import *
except ImportError:
    pass

__all__ = ['InfillAlgorithm', 'SBOInfill', 'SurrogateInfillCallback', 'SurrogateInfillOptimizationProblem']

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
            init_sampling = HierarchicalRandomSampling()
        self.initialization = Initialization(
            init_sampling, repair=infill.repair, eliminate_duplicates=infill.eliminate_duplicates)
        self.survival = survival

        if self.output is None:
            from sb_arch_opt.algo.simple_sbo.metrics import SBOMultiObjectiveOutput
            self.output = SBOMultiObjectiveOutput()

    def _initialize_infill(self):
        return self.initialization.do(self.problem, self.init_size, algorithm=self)

    def _initialize_advance(self, infills=None, **kwargs):
        if self.survival is not None:
            self.pop = self.survival.do(self.problem, infills, len(infills), algorithm=self)

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

    def _set_optimum(self):
        pop = self.pop
        i_failed = ArchOptProblemBase.get_failed_points(pop)
        valid_pop = pop[~i_failed]
        if len(valid_pop) == 0:
            self.opt = Population.new(X=[None])
        else:
            self.opt = filter_optimum(valid_pop, least_infeasible=True)

    def store_intermediate_results(self, results_folder: str):
        """Enable intermediate results storage to support restarting"""
        self.evaluator = ArchOptEvaluator(extreme_barrier=False, results_folder=results_folder)
        self.callback = ResultsStorageCallback(results_folder, callback=self.callback)

    def initialize_from_previous_results(self, problem: ArchOptProblemBase, result_folder: str) -> bool:
        """Initialize the SBO algorithm from previously stored results"""
        return initialize_from_previous_results(self, problem, result_folder)


class SBOInfill(InfillCriterion):
    """The main implementation of the SBO infill search"""

    _exclude = ['_surrogate_model', 'opt_results']

    def __init__(self, surrogate_model: 'SurrogateModel', infill: SurrogateInfill, pop_size=None,
                 termination: Union[Termination, int] = None, normalization: Normalization = None, verbose=False,
                 repair: Repair = None, eliminate_duplicates: DuplicateElimination = None,
                 force_new_points: bool = True, **kwargs):

        if eliminate_duplicates is None:
            eliminate_duplicates = LargeDuplicateElimination()
        super(SBOInfill, self).__init__(repair=repair, eliminate_duplicates=eliminate_duplicates, **kwargs)

        self._is_init = None
        self.problem: Optional[Problem] = None
        self.total_pop: Optional[Population] = None
        self._algorithm: Optional[Algorithm] = None
        self._normalization: Optional[Normalization] = normalization

        self._surrogate_model_base = surrogate_model
        self._surrogate_model = None
        self.infill = infill

        self.x_train = None
        self.y_train = None
        self.y_train_min = None
        self.y_train_max = None
        self.y_train_centered = None
        self.n_train = 0
        self.was_trained = False
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
            init_sampling = HierarchicalRandomSampling(self.repair)
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
        else:
            new_population = self.eliminate_duplicates.do(pop, self.total_pop)
            self.total_pop = Population.merge(self.total_pop, new_population)

        self._build_model()

        # Search the surrogate model for infill points
        off = self._generate_infill_points(n_offsprings)

        if self.repair is not None:
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
    def normalization(self) -> Normalization:
        if self._normalization is None:
            if self.problem is None:
                raise RuntimeError('Problem not set or not an architecture optimization problem!')
            self._normalization = ModelFactory.get_continuous_normalization(self.problem)
        return self._normalization

    @property
    def supports_variances(self):
        return self._surrogate_model_base.supports['variances']

    def _initialize(self):
        self.infill.initialize(self.problem, self.surrogate_model, self.normalization)

    def _build_model(self):
        """Update the underlying model. New population is given, total population is available from self.total_pop"""

        # Get input and output training points
        x = self.total_pop.get('X')
        y = self.total_pop.get('F')
        if self.problem.n_ieq_constr > 0:
            y = np.column_stack([y, self.total_pop.get('G')])

        # Select training values
        x, y = self._get_xy_train(x, y)

        # Normalize training output
        n_obj = self.problem.n_obj
        f_norm, self.y_train_min, self.y_train_max = self._normalize_y(y[:, :n_obj])
        self.y_train_centered = [False]*f_norm.shape[1]
        y_norm = f_norm

        if self.problem.n_ieq_constr > 0:
            g_norm, g_min, g_max = self._normalize_y(y[:, n_obj:], keep_centered=True)
            y_norm = np.column_stack([y_norm, g_norm])

            self.y_train_min = np.append(self.y_train_min, g_min)
            self.y_train_max = np.append(self.y_train_max, g_max)
            self.y_train_centered += [True]*g_norm.shape[1]

        # Train the model
        self.x_train = x
        self.y_train = y_norm

        self.pf_estimate = None
        self._train_model()

    def _get_xy_train(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Replace failed points with current worst values"""
        is_failed = np.any(~np.isfinite(y), axis=1)
        if ~np.any(is_failed):
            return x, y

        x = x.copy()
        y = y.copy()

        n_obj = self.problem.n_obj
        y[:, :n_obj] = np.nanmax(y[:, :n_obj], axis=0)  # f
        y[:, n_obj:] = 1.  # g

        y[np.isnan(y)] = 1.

        return x, y

    def _train_model(self):
        s = timeit.default_timer()
        self.surrogate_model.set_training_values(self.normalization.forward(self.x_train), self.y_train)
        self.infill.set_samples(self.x_train, self.y_train)

        if self.x_train.shape[0] > 1:
            self.was_trained = True
            self.surrogate_model.train()
        else:
            self.was_trained = False
        self.n_train += 1
        self.time_train = timeit.default_timer()-s

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return self.normalization.forward(x)

    @staticmethod
    def _normalize_y(y: np.ndarray, keep_centered=False, y_min=None, y_max=None):
        if y.shape[0] == 0:
            y_min, y_max = np.zeros((y.shape[1],)), np.ones((y.shape[1],))
            return y, y_min, y_max

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
        # Check if there are any valid points available
        if not self.was_trained:
            if self.verbose:
                log.info(f'Generating {n_infill} random point(s), because there were not enough prior valid points '
                         f'on a total of {len(self.total_pop)} points')
            return self._get_random_infill_points(n_infill)

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
            copy_termination=False,
            # verbose=True, progress=True,
        )
        if self.opt_results is None:
            self.opt_results = []
        self.opt_results.append(result)

        # Select infill points and denormalize the design vectors
        selected_pop = self.infill.select_infill_solutions(result.pop, problem, n_infill)
        result.opt = selected_pop

        x = selected_pop.get('X')
        return Population.new(X=x)

    def _get_random_infill_points(self, n_infill: int) -> Population:
        """Generate random infill points in case there were no valid points to train the surrogate with"""

        # Randomly sample some points
        pop_infill = HierarchicalRandomSampling().do(self.problem, max(100, n_infill))
        if len(pop_infill) <= n_infill:
            return pop_infill

        # Randomly select if there is no existing population to compare against
        if self.total_pop is None or len(self.total_pop) == 0:
            i_select = np.random.choice(len(pop_infill), n_infill, replace=False)
            return pop_infill[i_select]

        # Downselect by maximizing minimum distance from existing points
        normalization = ModelFactory.get_continuous_normalization(self.problem)
        x_dist = distance.cdist(normalization.forward(pop_infill.get('X')),
                                normalization.forward(self.total_pop.get('X')), metric='cityblock')
        min_x_dist = np.min(x_dist, axis=1)
        i_max_dist = np.argsort(min_x_dist)[-n_infill:]
        return pop_infill[i_max_dist]

    def get_pf_estimate(self) -> Optional[Population]:
        """Estimate the location of the Pareto front as predicted by the surrogate model"""

        if self.problem is None or self.n_train == 0 or not self.was_trained:
            return
        if self.pf_estimate is not None:
            return self.pf_estimate

        infill = FunctionEstimateInfill()
        infill.initialize(self.problem, self.surrogate_model, self.normalization)
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

        x_exist = self.total_pop.get('X') if force_new_points else None
        return SurrogateInfillOptimizationProblem(infill, self.problem, x_exist=x_exist)

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

        return termination

    def _get_infill_algorithm(self):
        repair = self._get_infill_repair()
        return NSGA2(pop_size=self.pop_size, sampling=HierarchicalRandomSampling(repair), repair=repair)

    def _get_infill_repair(self):
        if self.repair is None:
            return ArchOptRepair()
        return self.repair


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


class SurrogateInfillOptimizationProblem(ArchOptProblemBase):
    """Problem class representing a surrogate infill problem given a SurrogateInfill instance."""

    def __init__(self, infill: SurrogateInfill, problem: Problem, x_exist: np.ndarray = None):
        n_obj = infill.get_n_infill_objectives()
        n_ieq_constr = infill.get_n_infill_constraints()

        self.pop_exist = Population.new(X=x_exist) if x_exist is not None else None
        self.force_new_points = x_exist is not None
        if self.force_new_points:
            n_ieq_constr += 1
        self.eliminate_duplicates = LargeDuplicateElimination()

        if isinstance(problem, ArchOptProblemBase):
            des_vars = problem.des_vars
        elif problem.vars is not None:
            des_vars = list(problem.vars.values())
        else:
            des_vars = [Real(bounds=(problem.xl[i], problem.xu[i])) for i in range(problem.n_var)]

        super().__init__(des_vars=des_vars, n_obj=n_obj, n_ieq_constr=n_ieq_constr)

        self.infill = infill
        self._problem: Problem = problem

    def _get_n_valid_discrete(self) -> int:
        if isinstance(self._problem, ArchOptProblemBase):
            return self._problem.get_n_valid_discrete()

    def _gen_all_discrete_x(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if isinstance(self._problem, ArchOptProblemBase):
            return self._problem.all_discrete_x

    def might_have_hidden_constraints(self):
        return False

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        self._correct_x_impute(x, is_active_out)

        # Get infill search objectives and constraints
        f, g = self.infill.evaluate(x)

        if f.shape != (x.shape[0], self.n_obj):
            raise RuntimeError(f'Wrong objective results shape: {f.shape!r} != {(x.shape[0], self.n_obj)!r}')
        f_out[:, :] = f

        if g is None and self.n_constr > 0:
            g = np.zeros((x.shape[0], 0))

        # Add additional constraint to force the selection of new (discrete) points
        if self.force_new_points:
            g_force_new = np.zeros((x.shape[0],))
            _, _, is_dup = self.eliminate_duplicates.do(Population.new(X=x), self.pop_exist, return_indices=True)
            g_force_new[is_dup] = 1.

            g = np.column_stack([g, g_force_new])

        if g.shape != (x.shape[0], self.n_constr):
            raise RuntimeError(f'Wrong constraint results shape: {g.shape!r} != {(x.shape[0], self.n_constr)!r}')
        g_out[:, :] = g

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        if isinstance(self._problem, ArchOptProblemBase):
            x[:, :], is_active[:, :] = self._problem.correct_x(x)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.infill!r}, {self._problem!r})'
