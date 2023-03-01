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
from scipy.stats import norm
from sb_arch_opt.sampling import *
from sb_arch_opt.util import capture_log
from sb_arch_opt.problem import ArchOptRepair

from pymoo.core.repair import Repair
from pymoo.core.result import Result
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.core.sampling import Sampling
from pymoo.core.survival import Survival
from pymoo.core.population import Population
from pymoo.core.infill import InfillCriterion
from pymoo.core.termination import Termination
from pymoo.core.initialization import Initialization
from pymoo.core.algorithm import Algorithm, filter_optimum
from pymoo.core.duplicate import DuplicateElimination, DefaultDuplicateElimination
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.optimize import minimize

try:
    from smt.surrogate_models.rbf import RBF
    from smt.surrogate_models.krg import KRG
    from smt.surrogate_models.surrogate_model import SurrogateModel
    HAS_SIMPLE_SBO = True
except ImportError:
    HAS_SIMPLE_SBO = False

__all__ = ['get_simple_sbo_rbf', 'get_simple_sbo_krg', 'HAS_SIMPLE_SBO']

log = logging.getLogger('sb_arch_opt.sbo')


def _check_dependencies():
    if not HAS_SIMPLE_SBO:
        raise ImportError(f'simple_sbo dependencies not installed: python setup.py install[simple_sbo]')


def get_simple_sbo_rbf(init_size: int = 100, **kwargs):
    """
    Get a simple SBO algorithm using an RBF model as its surrogate model.
    """
    _check_dependencies()
    sm = RBF(
        print_global=False,
        d0=1.,
        poly_degree=-1,
        reg=1e-10,
    )
    return _get_sbo(sm, FunctionEstimateInfill(), init_size=init_size, **kwargs)


def get_simple_sbo_krg(init_size: int = 100, use_mvpf=True, use_ei=False, min_pof=.95, **kwargs):
    """
    Get a simple SBO algorithm using a Kriging model as its surrogate model.
    It can use one of the following infill strategies:
    - Expected improvement (multi-objectified)
    - Minimum Variance of the Pareto Front (MVPF)
    - Directly optimizing on the mean prediction
    All strategies support constraints.
    """
    _check_dependencies()
    sm = KRG(print_global=False)
    if use_ei:
        infill = ExpectedImprovementInfill(min_pof=min_pof)  # For single objective
    else:
        infill = MinVariancePFInfill(min_pof=min_pof) if use_mvpf else FunctionEstimatePoFInfill(min_pof=min_pof)
    return _get_sbo(sm, infill, init_size=init_size, **kwargs)


def _get_sbo(sm: 'SurrogateModel', infill: 'SurrogateInfill', infill_size: int = 1, init_size: int = 100,
             infill_pop_size: int = 100, infill_gens: int = 100, repair=None, **kwargs):
    capture_log()

    if repair is None:
        repair = ArchOptRepair()

    return SBOInfill(sm, infill, pop_size=infill_pop_size, termination=infill_gens, repair=repair, verbose=True)\
        .algorithm(infill_size=infill_size, init_size=init_size, **kwargs)


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


class SurrogateInfill:
    """Base class for surrogate infill criteria"""

    _exclude = ['surrogate_model']

    def __init__(self):
        self.problem: Optional[Problem] = None
        self.surrogate_model: Optional['SurrogateModel'] = None
        self.n_obj = 0
        self.n_constr = 0
        self.n_f_ic = None

        self.x_train = None
        self.y_train = None

        self.f_infill_log = []
        self.g_infill_log = []
        self.n_eval_infill = 0
        self.time_eval_infill = 0.

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in self._exclude:
            state[key] = None
        return state

    @property
    def needs_variance(self):
        return False

    def set_samples(self, x_train: np.ndarray, y_train: np.ndarray):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        try:
            y = self.surrogate_model.predict_values(x)
        except FloatingPointError:
            y = np.zeros((x.shape[0], self.surrogate_model.ny))*np.nan

        return self._split_f_g(y)

    def predict_variance(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        try:
            y_var = np.zeros((x.shape[0], self.surrogate_model.ny))
            for i in range(x.shape[0]):
                y_var[i, :] = self.surrogate_model.predict_variances(x[[i], :])

        except FloatingPointError:
            y_var = np.zeros((x.shape[0], self.surrogate_model.ny))*np.nan

        return self._split_f_g(y_var)

    def _normalize(self, x) -> np.ndarray:
        return normalize(x, self.problem.xl, self.problem.xu)

    def _denormalize(self, x_norm) -> np.ndarray:
        return denormalize(x_norm, self.problem.xl, self.problem.xu)

    def _split_f_g(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.n_constr > 0:
            return y[:, :self.n_obj], y[:, self.n_obj:self.n_obj+self.n_constr]
        return y[:, :self.n_obj], np.zeros((y.shape[0], 0))

    def initialize(self, problem: Problem, surrogate_model: 'SurrogateModel'):
        self.problem = problem
        self.n_obj = problem.n_obj
        self.n_constr = problem.n_constr

        self.surrogate_model = surrogate_model

        self._initialize()

        self.n_f_ic = self.get_n_infill_objectives()

    def select_infill_solutions(self, population: Population, infill_problem: Problem, n_infill) -> Population:
        """Select infill solutions from resulting population using rank and crowding selection (from NSGA2) algorithm.
        This method can be overwritten to implement a custom selection strategy."""

        # If there is only one objective, select the best point to prevent selecting duplicate points
        if self.n_f_ic == 1:
            return filter_optimum(population, least_infeasible=True)

        survival = RankAndCrowdingSurvival()
        return survival.do(infill_problem, population, n_survive=n_infill)

    @staticmethod
    def get_pareto_front(f: np.ndarray) -> np.ndarray:
        """Get the non-dominated set of objective values (the Pareto front)."""
        i_non_dom = NonDominatedSorting().do(f, only_non_dominated_front=True)
        return np.copy(f[i_non_dom, :])

    def reset_infill_log(self):
        self.f_infill_log = []
        self.g_infill_log = []
        self.n_eval_infill = 0
        self.time_eval_infill = 0.

    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Evaluate the surrogate infill objectives (and optionally constraints). Use the predict and predict_variance
        methods to query the surrogate model on its objectives and constraints."""

        s = timeit.default_timer()
        f_infill, g_infill = self._evaluate(x)
        self.time_eval_infill += timeit.default_timer()-s

        self.f_infill_log.append(f_infill)
        self.g_infill_log.append(g_infill)
        self.n_eval_infill += x.shape[0]
        return f_infill, g_infill

    def _initialize(self):
        pass

    def get_n_infill_objectives(self) -> int:
        raise NotImplementedError

    def get_n_infill_constraints(self) -> int:
        raise NotImplementedError

    def _evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Evaluate the surrogate infill objectives (and optionally constraints). Use the predict and predict_variance
        methods to query the surrogate model on its objectives and constraints."""
        raise NotImplementedError


class FunctionEstimateInfill(SurrogateInfill):
    """Infill that directly uses the underlying surrogate model prediction."""

    def get_n_infill_objectives(self) -> int:
        return self.problem.n_obj

    def get_n_infill_constraints(self) -> int:
        return self.problem.n_constr

    def _evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        f, g = self.predict(x)
        return f, g


class PoFInfill(SurrogateInfill):
    """Probability of Feasibility infill criterion base, for handling constraints using the PoF criterion"""

    def __init__(self, min_pof: float = .95):
        self.min_pof = min_pof
        super(PoFInfill, self).__init__()

    @property
    def needs_variance(self):
        return True

    def get_n_infill_constraints(self) -> int:
        return self.problem.n_constr

    def _evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        f, g = self.predict(x)
        f_var, g_var = self.predict_variance(x)

        # Calculate Probability of Feasibility and transform to constraint (g < 0 --> PoF(g) > PoF_min)
        g_pof = g
        if self.n_constr > 0:
            g_pof = self.min_pof-self._pof(g, g_var)

        f_infill = self._evaluate_f(f, f_var)
        return f_infill, g_pof

    @staticmethod
    def _pof(g: np.ndarray, g_var: np.ndarray) -> np.ndarray:
        pof = norm.cdf(-g/np.sqrt(g_var))
        is_nan_mask = np.isnan(pof)
        pof[is_nan_mask & (g <= 0.)] = 1.
        pof[is_nan_mask & (g > 0.)] = 0.
        return pof

    def get_n_infill_objectives(self) -> int:
        raise NotImplementedError

    def _evaluate_f(self, f_predict: np.ndarray, f_var_predict: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class FunctionEstimatePoFInfill(PoFInfill):
    """Probability of Feasibility combined with direct function estimate for the objectives."""

    def get_n_infill_objectives(self) -> int:
        return self.problem.n_obj

    def _evaluate_f(self, f_predict: np.ndarray, f_var_predict: np.ndarray) -> np.ndarray:
        return f_predict


class ExpectedImprovementInfill(PoFInfill):
    """
    The Expected Improvement (EI) naturally balances exploitation and exploration by representing the expected amount
    of improvement at some point taking into accounts its probability of improvement.

    EI(x) = (f_min-y(x)) * Phi((f_min - y(x))/s(x)) + s(x) * phi((f_min - y(x)) / s(x))
    where
    - f_min is the current best point (real)
    - y(x) the surrogate model estimate
    - s(x) the surrogate model variance estimate
    - Phi is the cumulative distribution function of the normal distribution
    - phi is the probability density function of the normal distribution

    Implementation based on:
    Jones, D.R., "Efficient Global Optimization of Expensive Black-Box Functions", 1998, 10.1023/A:1008306431147
    """

    def get_n_infill_objectives(self) -> int:
        return self.problem.n_obj

    def _evaluate_f(self, f_predict: np.ndarray, f_var_predict: np.ndarray) -> np.ndarray:
        return self._evaluate_f_ei(f_predict, f_var_predict, self.y_train[:, :f_predict.shape[1]])

    @classmethod
    def _evaluate_f_ei(cls, f: np.ndarray, f_var: np.ndarray, f_current: np.ndarray) -> np.ndarray:
        # Normalize current and predicted objectives
        f_pareto = cls.get_pareto_front(f_current)
        nadir_point, ideal_point = np.max(f_pareto, axis=0), np.min(f_pareto, axis=0)
        f_pareto_norm = normalize(f_pareto, xu=nadir_point, xl=ideal_point)
        f_norm, f_var_norm = cls._normalize_f_var(f, f_var, nadir_point, ideal_point)

        # Get EI for each point using closest point in the Pareto front
        f_ei = np.empty(f.shape)
        for i in range(f.shape[0]):
            i_par_closest = np.argmin(np.sum((f_pareto_norm-f_norm[i, :])**2, axis=1))
            f_par_min = f_pareto_norm[i_par_closest, :]
            ei = cls._ei(f_par_min, f_norm[i, :], f_var_norm[i, :])
            ei[ei < 0.] = 0.
            f_ei[i, :] = 1.-ei

        return f_ei

    @staticmethod
    def _normalize_f_var(f: np.ndarray, f_var: np.ndarray, nadir_point, ideal_point):
        f_norm = normalize(f, xu=nadir_point, xl=ideal_point)
        f_var_norm = f_var/((nadir_point-ideal_point+1e-30)**2)
        return f_norm, f_var_norm

    @staticmethod
    def _ei(f_min: np.ndarray, f: np.ndarray, f_var: np.ndarray) -> np.ndarray:
        dy = f_min-f
        ei = dy*norm.cdf(dy/np.sqrt(f_var)) + f_var*norm.pdf(dy/np.sqrt(f_var))

        is_nan_mask = np.isnan(ei)
        ei[is_nan_mask & (dy > 0.)] = 1.
        ei[is_nan_mask & (dy <= 0.)] = 0.

        return ei


class MinVariancePFInfill(FunctionEstimatePoFInfill):
    """
    Minimization of the Variance of Kriging-Predicted Front (MVPF).

    This works by first finding a new Pareto front directly using the predicted function value, and then selecting
    solutions with the highest variance for infill. This way, search is performed near the Pareto front, but with a
    high potential for exploration.

    Implementation based on:
    dos Passos, A.G., "Multi-Objective Optimization with Kriging Surrogates Using 'moko'", 2018, 10.1590/1679-78254324
    """

    def select_infill_solutions(self, population: Population, infill_problem: Problem, n_infill) -> Population:
        # Get Pareto front and associated design points
        f = population.get('F')
        i_pf = self.get_i_pareto_front(f)
        pop_pf = population[i_pf]
        x_pf = pop_pf.get('X')
        g_pf = pop_pf.get('G')

        # Get variances
        f_var, _ = self.predict_variance(x_pf)

        # Select points with highest variances
        f_std_obj = 1.-np.sqrt(f_var)
        survival = RankAndCrowdingSurvival()
        pop_var = Population.new(X=x_pf, F=f_std_obj, G=g_pf)
        i_select = survival.do(infill_problem, pop_var, n_survive=n_infill, return_indices=True)

        return pop_pf[i_select]

    @staticmethod
    def get_i_pareto_front(f: np.ndarray) -> np.ndarray:
        """Get the non-dominated set of objective values (the Pareto front)."""
        return NonDominatedSorting().do(f, only_non_dominated_front=True)


def normalize(x: np.ndarray, xl, xu) -> np.ndarray:
    return (x-xl)/(xu-xl)


def denormalize(x_norm: np.ndarray, xl, xu) -> np.ndarray:
    return x_norm*(xu-xl)+xl


class SBOInfill(InfillCriterion):
    """The main implementation of the SBO infill search"""

    _exclude = ['_surrogate_model', 'opt_results']

    def __init__(self, surrogate_model: 'SurrogateModel', infill: SurrogateInfill, pop_size=None,
                 termination: Union[Termination, int] = None, verbose=False, repair: Repair = None,
                 eliminate_duplicates: DuplicateElimination = None, force_new_points: bool = True, **kwargs):

        if eliminate_duplicates is None:
            eliminate_duplicates = DefaultDuplicateElimination()
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
        f_is_invalid = np.bitwise_or(np.isnan(f_real), np.isinf(f_real))
        f_real[f_is_invalid] = np.nan

        f, self.y_train_min, self.y_train_max = self._normalize_y(f_real)
        self.y_train_centered = [False]*f.shape[1]

        g = np.zeros((x.shape[0], 0))
        if self.problem.n_constr > 0:
            g_real = self.total_pop.get('G')
            g_is_invalid = np.bitwise_or(np.isnan(g_real), np.isinf(g_real))
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

    def _get_infill_problem(self):
        x_exist_norm = self._normalize(self.total_pop.get('X')) if self.force_new_points else None
        return SurrogateInfillOptimizationProblem(self.infill, self.problem, x_exist_norm=x_exist_norm)

    def _get_termination(self, n_obj):
        termination = self.termination
        if termination is None or not isinstance(termination, Termination):
            # return MaximumGenerationTermination(n_max_gen=termination or 100)
            from pymoo.termination.default import DefaultMultiObjectiveTermination, DefaultSingleObjectiveTermination
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


def patch_ftol_bug(term):  # Already fixed in upcoming release: https://github.com/anyoptimization/pymoo/issues/325
    from pymoo.termination.default import DefaultMultiObjectiveTermination
    from pymoo.termination.ftol import MultiObjectiveSpaceTermination
    data_func = None

    def _wrap_data(algorithm):
        data = data_func(algorithm)
        if data['ideal'] is None:
            data['feas'] = False
        return data

    if isinstance(term, DefaultMultiObjectiveTermination):
        ftol_term = term.criteria[2].termination
        if isinstance(ftol_term, MultiObjectiveSpaceTermination):
            data_func = ftol_term._data
            ftol_term._data = _wrap_data


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

        self.pop_exist_norm = Population.new(X=x_exist_norm)
        self.force_new_points = x_exist_norm is not None
        if self.force_new_points:
            n_constr += 1
        self.eliminate_duplicates = DefaultDuplicateElimination()

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
