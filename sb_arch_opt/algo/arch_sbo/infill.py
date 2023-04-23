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
import timeit
import numpy as np
from typing import *
from scipy.stats import norm

from pymoo.core.problem import Problem
from pymoo.core.population import Population
from pymoo.core.algorithm import filter_optimum
from pymoo.util.normalization import Normalization
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

__all__ = ['SurrogateInfill', 'FunctionEstimateInfill', 'ConstrainedInfill', 'FunctionEstimateConstrainedInfill',
           'ExpectedImprovementInfill', 'MinVariancePFInfill', 'ConstraintStrategy', 'MeanConstraintPrediction',
           'ProbabilityOfFeasibility']

try:
    from smt.surrogate_models.surrogate_model import SurrogateModel
except ImportError:
    pass


class SurrogateInfill:
    """Base class for surrogate infill criteria"""

    _exclude = ['surrogate_model']

    def __init__(self):
        self.problem: Optional[Problem] = None
        self.surrogate_model: Optional['SurrogateModel'] = None
        self.normalization: Optional[Normalization] = None
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
            y = self.surrogate_model.predict_values(self.normalization.forward(x))
        except FloatingPointError:
            y = np.zeros((x.shape[0], self.surrogate_model.ny))*np.nan

        return self._split_f_g(y)

    def predict_variance(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        try:
            y_var = np.zeros((x.shape[0], self.surrogate_model.ny))
            for i in range(x.shape[0]):
                y_var[i, :] = self.surrogate_model.predict_variances(self.normalization.forward(x[[i], :]))

        except FloatingPointError:
            y_var = np.zeros((x.shape[0], self.surrogate_model.ny))*np.nan

        return self._split_f_g(y_var)

    def _split_f_g(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.n_constr > 0:
            return y[:, :self.n_obj], y[:, self.n_obj:self.n_obj+self.n_constr]
        return y[:, :self.n_obj], np.zeros((y.shape[0], 0))

    def initialize(self, problem: Problem, surrogate_model: 'SurrogateModel', normalization: Normalization):
        self.problem = problem
        self.n_obj = problem.n_obj
        self.n_constr = problem.n_constr

        self.surrogate_model = surrogate_model
        self.normalization = normalization

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


class ConstraintStrategy:
    """Base class for a strategy for dealing with design constraints"""

    def __init__(self):
        self.problem: Optional[Problem] = None

    def initialize(self, problem: Problem):
        self.problem = problem

    def get_n_infill_constraints(self) -> int:
        raise NotImplementedError

    def evaluate(self, x: np.ndarray, g: np.ndarray, g_var: np.ndarray) -> np.ndarray:
        """Evaluate the infill constraint function(s) given x and predicted g and g_var"""
        raise NotImplementedError


class MeanConstraintPrediction(ConstraintStrategy):
    """Simple use the mean prediction of the constraint functions as the infill constraint"""

    def get_n_infill_constraints(self) -> int:
        return self.problem.n_ieq_constr

    def evaluate(self, x: np.ndarray, g: np.ndarray, g_var: np.ndarray) -> np.ndarray:
        return g


class ProbabilityOfFeasibility(ConstraintStrategy):
    """
    Uses a lower limit on the Probability of Feasibility (PoF) as the infill constraint.

    PoF(x) = Phi(-y(x)/sqrt(s(x)))
    where
    - Phi is the cumulative distribution function of the normal distribution
    - y(x) the surrogate model estimate
    - s(x) the surrogate model variance estimate

    Implementation based on discussion in:
    Schonlau, M., "Global Versus Local Search in Constrained Optimization of Computer Models", 1998,
        10.1214/lnms/1215456182
    """

    def __init__(self, min_pof: float = None):
        if min_pof is None:
            min_pof = .5
        self.min_pof = min_pof
        super().__init__()

    def get_n_infill_constraints(self) -> int:
        return self.problem.n_ieq_constr

    def evaluate(self, x: np.ndarray, g: np.ndarray, g_var: np.ndarray) -> np.ndarray:
        pof = self._pof(g, g_var)
        return self.min_pof - pof

    @staticmethod
    def _pof(g: np.ndarray, g_var: np.ndarray) -> np.ndarray:
        pof = norm.cdf(-g/np.sqrt(g_var))
        is_nan_mask = np.isnan(pof)
        pof[is_nan_mask & (g <= 0.)] = 1.
        pof[is_nan_mask & (g > 0.)] = 0.
        return pof


class ConstrainedInfill(SurrogateInfill):
    """Base class for an infill criterion with a constraint handling strategy"""

    def __init__(self, constraint_strategy: ConstraintStrategy = None, min_pof: float = None):
        if constraint_strategy is None:
            if min_pof is not None:
                constraint_strategy = ProbabilityOfFeasibility(min_pof=min_pof)
            else:
                constraint_strategy = MeanConstraintPrediction()

        self.constraint_strategy = constraint_strategy
        super(ConstrainedInfill, self).__init__()

    def _initialize(self):
        self.constraint_strategy.initialize(self.problem)

    @property
    def needs_variance(self):
        return True

    def get_n_infill_constraints(self) -> int:
        return self.constraint_strategy.get_n_infill_constraints()

    def _evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        f, g = self.predict(x)
        f_var, g_var = self.predict_variance(x)

        # Apply constraint handling strategy
        g_infill = g
        if self.n_constr > 0:
            g_infill = self.constraint_strategy.evaluate(x, g, g_var)

        f_infill = self._evaluate_f(f, f_var)
        return f_infill, g_infill

    def get_n_infill_objectives(self) -> int:
        raise NotImplementedError

    def _evaluate_f(self, f_predict: np.ndarray, f_var_predict: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class FunctionEstimateConstrainedInfill(ConstrainedInfill):
    """Probability of Feasibility combined with direct function estimate for the objectives."""

    def get_n_infill_objectives(self) -> int:
        return self.problem.n_obj

    def _evaluate_f(self, f_predict: np.ndarray, f_var_predict: np.ndarray) -> np.ndarray:
        return f_predict


class ExpectedImprovementInfill(ConstrainedInfill):
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
        nadir_point[nadir_point == ideal_point] = 1.
        f_pareto_norm = (f_pareto-ideal_point)/(nadir_point-ideal_point)
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
        f_norm = (f-ideal_point)/(nadir_point-ideal_point)
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


class MinVariancePFInfill(FunctionEstimateConstrainedInfill):
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
