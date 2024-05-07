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
import timeit
import numpy as np
from typing import *
from enum import Enum
from scipy.stats import norm
from scipy.special import ndtr
from scipy.optimize import minimize
from sb_arch_opt.problem import ArchOptProblemBase
from sb_arch_opt.algo.arch_sbo.hc_strategy import HiddenConstraintStrategy
from sb_arch_opt.algo.arch_sbo.models import *

from pymoo.core.problem import Problem
from pymoo.core.population import Population
from pymoo.core.algorithm import filter_optimum
from pymoo.util.normalization import Normalization
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival

try:
    # pymoo < 0.6.1
    from pymoo.algorithms.moo.nsga2 import calc_crowding_distance
except ImportError:
    # pymoo >= 0.6.1
    from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance

__all__ = ['SurrogateInfill', 'FunctionEstimateInfill', 'ConstrainedInfill', 'FunctionEstimateConstrainedInfill',
           'ExpectedImprovementInfill', 'MinVariancePFInfill', 'ConstraintStrategy', 'MeanConstraintPrediction',
           'ProbabilityOfFeasibility', 'ProbabilityOfImprovementInfill', 'LowerConfidenceBoundInfill',
           'MinimumPoIInfill', 'EnsembleInfill', 'IgnoreConstraints', 'get_default_infill', 'HCInfill',
           'ConstraintAggregation']

if HAS_SMT:
    from smt.surrogate_models.surrogate_model import SurrogateModel
    from smt.surrogate_models.krg_based import KrgBased


def get_default_infill(problem: ArchOptProblemBase, n_parallel: int = None, min_pof: float = None,
                       g_aggregation: 'ConstraintAggregation' = None) -> Tuple['ConstrainedInfill', int]:
    """
    Get the default infill criterion according to the following logic:
    - Single-objective: Ensemble of EI, LCB, PoI with n_batch = n_parallel
    - Multi-objective:  Ensemble of MPoI, MEPoI  with n_batch = n_parallel
    - n_parallel = 1 if parallelization is not possible
    - Set Probability of Feasibility as constraint handling technique if min_pof != .5, otherwise use g-mean prediction

    Returns the infill and recommended infill batch size.
    """

    # Determine number of evaluations that can be run in parallel
    if n_parallel is None:
        n_parallel = problem.get_n_batch_evaluate()
        if n_parallel is None:
            n_parallel = 1

    so_ensemble = [ExpectedImprovementInfill(), LowerConfidenceBoundInfill(), ProbabilityOfImprovementInfill()]
    mo_ensemble = [MinimumPoIInfill(), MinimumPoIInfill(euclidean=True)]

    def _get_infill():
        # Single-objective ensemble infill
        if problem.n_obj == 1:
            return EnsembleInfill(so_ensemble), n_parallel

        # Multi-objective ensemble infill
        return EnsembleInfill(mo_ensemble), n_parallel

    # Get infill and set constraint handling technique
    infill, n_batch = _get_infill()

    if min_pof is not None and min_pof != .5:
        infill.constraint_strategy = ProbabilityOfFeasibility(min_pof=min_pof, aggregation=g_aggregation)
    else:
        infill.constraint_strategy = MeanConstraintPrediction(aggregation=g_aggregation)

    return infill, n_batch


class SurrogateInfill:
    """Base class for surrogate infill criteria"""

    _exclude = ['surrogate_model']

    def __init__(self):
        self.problem: Optional[Problem] = None
        self.surrogate_model: Optional[Union['SurrogateModel', 'KrgBased']] = None
        self.normalization: Optional[Normalization] = None
        self.n_obj = 0
        self.n_constr = 0
        self.n_f_ic = None

        self.x_train = None
        self.is_active_train = None
        self.y_train = None

        self.f_infill_log = []
        self.g_infill_log = []
        self.n_eval_infill = 0
        self.time_eval_infill = 0.
        self.select_improve_infills = True

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in self._exclude:
            state[key] = None
        return state

    @property
    def needs_variance(self):
        return False

    def get_g_training_set(self, g: np.ndarray) -> np.ndarray:
        return g

    def set_samples(self, x_train: np.ndarray, is_active_train: np.ndarray, y_train: np.ndarray):
        self.x_train = x_train
        self.is_active_train = is_active_train
        self.y_train = y_train

    def predict(self, x: np.ndarray, is_active: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        try:
            kwargs = {'is_acting': is_active.astype(bool)} if self.surrogate_model.supports['x_hierarchy'] else {}
            y = self.surrogate_model.predict_values(self.normalization.forward(x), **kwargs)
        except FloatingPointError:
            y = np.zeros((x.shape[0], self.surrogate_model.ny))*np.nan

        return self._split_f_g(y)

    def predict_variance(self, x: np.ndarray, is_active: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        try:
            kwargs = {'is_acting': is_active.astype(bool)} if self.surrogate_model.supports['x_hierarchy'] else {}
            y_var = self.surrogate_model.predict_variances(self.normalization.forward(x), **kwargs)
        except FloatingPointError:
            y_var = np.zeros((x.shape[0], self.surrogate_model.ny))*np.nan

        return self._split_f_g(y_var)

    def _split_f_g(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.n_constr > 0:
            return y[:, :self.n_obj], y[:, self.n_obj:]
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

    def select_infill(self, population: Population, infill_problem: Problem, n_infill) -> Population:
        """Select infill points and improve the precision using a gradient-based algorithm."""

        sel_pop = self.select_infill_solutions(population, infill_problem, n_infill)
        if not self.select_improve_infills:
            return sel_pop

        # Improve selected points by local optimization
        return self._increase_precision(sel_pop)
        # print(f'SEL POP x    = {sel_pop.get("X")}')
        # print(f'SEL POP f_in = {sel_pop.get("F")}')
        # improved_sel_pop = self._increase_precision(sel_pop)
        # print(f'IMP POP x    = {improved_sel_pop.get("X")}')
        # x, is_active = self.problem.correct_x(improved_sel_pop.get("X"))
        # print(f'IMP POP f_in = {self.evaluate(x, is_active)[0]}')
        # return improved_sel_pop

    def _increase_precision(self, pop: Population) -> Population:
        """Increase the precision of the continuous variables by running a local gradient-based optimization
        on the selected points in the population"""

        # Get continuous design variables
        problem = self.problem
        if not isinstance(problem, ArchOptProblemBase):
            return pop
        is_cont_mask = problem.is_cont_mask
        if not np.any(is_cont_mask):
            return pop

        def get_y_precision_funcs(x_ref: np.ndarray, is_act_ref: np.ndarray, f_ref: np.ndarray, i_opt):
            last_g: Optional[np.ndarray] = None
            x_norm_opt = x_norm[i_opt]
            xl_opt = xl[i_opt]

            def y_precision(x_norm_):
                nonlocal last_g
                x_eval = x_ref.copy()
                x_eval[0, i_opt] = x_norm_*x_norm_opt + xl_opt

                f, g = self.evaluate(x_eval, is_active=is_act_ref)

                # Objective is the improvement in the direction of the negative unit vector
                f_diff = f-f_ref
                f_abs_diff = np.sum(f_diff)

                # Penalize deviation from unit vector to ensure that the design point stays in the same area of the PF
                f_deviation = np.ptp(f_diff)
                f_so = f_abs_diff + 100*f_deviation**2

                # print(f'EVAL {x_norm_}: {f} --> {f_so}, {g}')
                last_g = g[0, :] if g is not None else None
                return f_so

            def get_i_con_fun(i_g):
                def _g(_):
                    if last_g is None:
                        return 0
                    return -last_g[i_g]  # Scipy's minimize formulates ineq constraints as: g(x) >= 0
                return _g

            constraints = [{
                'type': 'ineq',
                'fun': get_i_con_fun(i_g)
            } for i_g in range(self.get_n_infill_constraints())]

            return y_precision, constraints, x_norm_opt, xl_opt

        # Improve selected points
        xl, xu = problem.xl, problem.xu
        x_norm = xu-xl
        x_norm[x_norm < 1e-6] = 1e-6

        x, is_active = problem.correct_x(pop.get('X'))
        f_pop = pop.get('F')
        x_optimized = []
        for i in range(len(pop)):
            # Only optimize active continuous variables
            is_act_i = is_active[i, :]
            i_optimize, = np.where(is_cont_mask & is_act_i)
            x_ref_i = x[i, :]
            if len(i_optimize) == 0:
                x_optimized.append(x_ref_i)
                continue

            # Run optimization
            f_opt, con, xn_, xl_ = get_y_precision_funcs(x[[i], :], is_active[[i], :], f_pop[[i], :], i_optimize)
            bounds = [(0., 1.) for _ in i_optimize]
            x_start_norm = (x_ref_i[i_optimize]-xl_)/xn_
            res = minimize(f_opt, x_start_norm, method='slsqp', bounds=bounds,
                           constraints=con, options={'maxiter': 20, 'eps': 1e-5, 'ftol': 1e-4})
            if res.success:
                x_opt = x_ref_i.copy()
                x_opt[i_optimize] = res.x*xn_ + xl_
                x_optimized.append(x_opt)
            else:
                x_optimized.append(x_ref_i)

        return Population.new(X=np.row_stack(x_optimized))

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

    def evaluate(self, x: np.ndarray, is_active: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Evaluate the surrogate infill objectives (and optionally constraints). Use the predict and predict_variance
        methods to query the surrogate model on its objectives and constraints."""

        s = timeit.default_timer()
        f_infill, g_infill = self._evaluate(x, is_active)
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

    def _evaluate(self, x: np.ndarray, is_active: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Evaluate the surrogate infill objectives (and optionally constraints). Use the predict and predict_variance
        methods to query the surrogate model on its objectives and constraints."""
        raise NotImplementedError


class FunctionEstimateInfill(SurrogateInfill):
    """Infill that directly uses the underlying surrogate model prediction."""

    def get_n_infill_objectives(self) -> int:
        return self.problem.n_obj

    def get_n_infill_constraints(self) -> int:
        return self.problem.n_constr

    def _evaluate(self, x: np.ndarray, is_active: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        f, g = self.predict(x, is_active)
        return f, g


class ConstraintAggregation(Enum):
    NONE = 0  # No aggregation
    ELIMINATE = 1  # Automatically eliminate non-relevant
    AGGREGATE = 2  # Aggregate all into 1


class ConstraintStrategy:
    """
    Base class for a strategy for dealing with design constraints.
    Optionally enables constraint aggregation (max) or elimination (only train models for constraints where for at least
    one design point it is violated and all others are satisfied).
    """

    def __init__(self, aggregation: ConstraintAggregation = None):
        self.problem: Optional[Problem] = None
        self.n_trained_g = None
        self.aggregation = ConstraintAggregation.NONE if aggregation is None else aggregation

    def initialize(self, problem: Problem):
        self.problem = problem

    def get_g_training_set(self, g: np.ndarray) -> np.ndarray:
        # Eliminate constraints that are only violated when at least one other constraint is also violated
        if self.aggregation == ConstraintAggregation.ELIMINATE:
            g_ref = g
            while g_ref.shape[1] > 1:
                for i_g in range(g_ref.shape[1]):
                    is_violated = g_ref[:, i_g] > 0
                    g_ref_other = np.delete(g_ref, i_g, axis=1)

                    # No need to train GP if this constraint is never violated
                    if not np.any(is_violated):
                        break

                    all_other_satisfied = np.all(g_ref_other <= 0, axis=1)
                    i_g_only_active = is_violated & all_other_satisfied
                    if not np.any(i_g_only_active):
                        break
                else:
                    break

                g_ref = g_ref_other
            return g_ref

        # Optionally aggregate constraints by taking the maximum value
        if self.aggregation == ConstraintAggregation.AGGREGATE:
            return np.array([np.max(g, axis=1)]).T
        return g

    def set_samples(self, x_train: np.ndarray, y_train: np.ndarray):
        self.n_trained_g = n_trained_g = y_train.shape[1]-self.problem.n_obj

        n_constr = self.problem.n_ieq_constr
        if n_trained_g == 0 and n_constr != 0:
            raise RuntimeError(f'Expecting at least one trained constraint model ({n_constr}), received 0')
        elif n_constr > 0 and (n_trained_g == 0 or n_trained_g > n_constr):
            raise RuntimeError(f'Expecting between 1 and {n_constr} constraint models, received {n_trained_g}')

        self._set_samples(x_train, y_train)

    def _set_samples(self, x_train: np.ndarray, y_train: np.ndarray):
        pass

    def get_n_infill_constraints(self) -> int:
        return self.n_trained_g

    def evaluate(self, x: np.ndarray, g: np.ndarray, g_var: np.ndarray) -> np.ndarray:
        """Evaluate the infill constraint function(s) given x and predicted g and g_var"""
        raise NotImplementedError


class MeanConstraintPrediction(ConstraintStrategy):
    """Simple use the mean prediction of the constraint functions as the infill constraint"""

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

    def __init__(self, min_pof: float = None, aggregation: ConstraintAggregation = None):
        if min_pof is None:
            min_pof = .5
        self.min_pof = min_pof
        super().__init__(aggregation=aggregation)

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

    def get_g_training_set(self, g: np.ndarray) -> np.ndarray:
        return self.constraint_strategy.get_g_training_set(g)

    def set_samples(self, x_train: np.ndarray, is_active_train: np.ndarray, y_train: np.ndarray):
        super().set_samples(x_train, is_active_train, y_train)
        self.constraint_strategy.set_samples(x_train, y_train)

    @property
    def needs_variance(self):
        return True

    def get_n_infill_constraints(self) -> int:
        return self.constraint_strategy.get_n_infill_constraints()

    def _evaluate(self, x: np.ndarray, is_active: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        f, g = self.predict(x, is_active)
        f_var, g_var = self.predict_variance(x, is_active)

        # Apply constraint handling strategy
        g_infill = g
        if self.n_constr > 0:
            g_infill = self.constraint_strategy.evaluate(x, g, g_var)

        f_infill = self.evaluate_f(f, f_var)
        return f_infill, g_infill

    def get_n_infill_objectives(self) -> int:
        raise NotImplementedError

    def evaluate_f(self, f_predict: np.ndarray, f_var_predict: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class FunctionEstimateConstrainedInfill(ConstrainedInfill):
    """Probability of Feasibility combined with direct function estimate for the objectives."""

    def get_n_infill_objectives(self) -> int:
        return self.problem.n_obj

    def evaluate_f(self, f_predict: np.ndarray, f_var_predict: np.ndarray) -> np.ndarray:
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

    def evaluate_f(self, f_predict: np.ndarray, f_var_predict: np.ndarray) -> np.ndarray:
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
        is_active_pf = pop_pf.get('is_active').astype(bool)
        g_pf = pop_pf.get('G')

        # Get variances
        f_var, _ = self.predict_variance(x_pf, is_active_pf)

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


class ProbabilityOfImprovementInfill(ConstrainedInfill):
    """
    Probability of Improvement represents the probability that some point will be better than the current best estimate
    with some offset:

    PoI(x) = Phi((T - y(x))/sqrt(s(x)))
    where
    - Phi is the cumulative distribution function of the normal distribution
    - T is the improvement target (current best estimate minus some offset)
    - y(x) the surrogate model estimate
    - s(x) the surrogate model variance estimate

    PoI was developed for single-objective optimization, and because of the use of the minimum current objective value,
    it tends towards suggesting improvement points only at the edges of the Pareto front. It has been modified to
    evaluate the PoI with respect to the closest Pareto front point instead.

    Implementation based on:
    Hawe, G.I., "An Enhanced Probability of Improvement Utility Function for Locating Pareto Optimal Solutions", 2007
    """

    def __init__(self, f_min_offset: float = 0., **kwargs):
        self.f_min_offset = f_min_offset
        super().__init__(**kwargs)

    def get_n_infill_objectives(self) -> int:
        return self.problem.n_obj

    def evaluate_f(self, f_predict: np.ndarray, f_var_predict: np.ndarray) -> np.ndarray:
        return self._evaluate_f_poi(f_predict, f_var_predict, self.y_train[:, :f_predict.shape[1]], self.f_min_offset)

    @classmethod
    def _evaluate_f_poi(cls, f: np.ndarray, f_var: np.ndarray, f_current: np.ndarray, f_min_offset=0.) -> np.ndarray:
        # Normalize current and predicted objectives
        f_pareto = cls.get_pareto_front(f_current)
        nadir_point, ideal_point = np.max(f_pareto, axis=0), np.min(f_pareto, axis=0)
        nadir_point[nadir_point == ideal_point] = 1.
        f_pareto_norm = (f_pareto-ideal_point)/(nadir_point-ideal_point)
        f_norm, f_var_norm = cls._normalize_f_var(f, f_var, nadir_point, ideal_point)

        # Get PoI for each point using closest point in the Pareto front
        f_poi = np.empty(f.shape)
        for i in range(f.shape[0]):
            i_par_closest = np.argmin(np.sum((f_pareto_norm-f_norm[i, :])**2, axis=1))
            f_par_targets = f_pareto_norm[i_par_closest, :]-f_min_offset
            poi = cls._poi(f_par_targets, f_norm[i, :], f_var_norm[i, :])
            f_poi[i, :] = 1.-poi

        return f_poi

    @staticmethod
    def _normalize_f_var(f: np.ndarray, f_var: np.ndarray, nadir_point, ideal_point):
        f_norm = (f-ideal_point)/(nadir_point-ideal_point)
        f_var_norm = f_var/((nadir_point-ideal_point+1e-30)**2)
        return f_norm, f_var_norm

    @staticmethod
    def _poi(f_targets: np.ndarray, f: np.ndarray, f_var: np.ndarray) -> np.ndarray:
        return norm.cdf((f_targets-f) / np.sqrt(f_var+1e-8))


class LowerConfidenceBoundInfill(ConstrainedInfill):
    """
    The Lower Confidence Bound (LCB) represents the lowest expected value to be found at some point given its standard
    deviation.

    LCB(x) = y(x) - alpha * sqrt(s(x))
    where
    - y(x) the surrogate model estimate
    - alpha is a scaling parameter (typical value is 2) --> lower means more exploitation, higher more exploration
    - s(x) the surrogate model variance estimate

    Implementation based on:
    Cox, D., "A Statistical Method for Global Optimization", 1992, 10.1109/icsmc.1992.271617
    """

    def __init__(self, alpha: float = 2., **kwargs):
        self.alpha = alpha
        super().__init__(**kwargs)

    def get_n_infill_objectives(self) -> int:
        return self.problem.n_obj

    def evaluate_f(self, f_predict: np.ndarray, f_var_predict: np.ndarray) -> np.ndarray:
        lcb = f_predict - self.alpha*np.sqrt(f_var_predict)
        return lcb


class MinimumPoIInfill(ConstrainedInfill):
    """
    The Minimum Probability of Improvement (MPoI) criterion is a multi-objective infill criterion and modifies the
    calculation of the domination probability by only considering one objective dimension at a time. This should reduce
    computational cost.

    Optionally multiplies the MPoI criteria by its first integral moment, to transform it to an EI-like metric. Uses a
    similar implementation as `EuclideanEIInfill`.

    Implementation based on:
    Rahat, A.A.M., "Alternative Infill Strategies for Expensive Multi-Objective Optimisation", 2017,
        10.1145/3071178.3071276
    Parr, J.M., "Improvement Criteria for Constraint Handling and Multiobjective Optimization", 2013
    """

    def __init__(self, euclidean=False, **kwargs):
        self.euclidean = euclidean
        self.f_pareto = None
        super().__init__(**kwargs)

    def get_n_infill_objectives(self) -> int:
        return 1

    def set_samples(self, x_train: np.ndarray, is_active_train: np.ndarray, y_train: np.ndarray):
        super().set_samples(x_train, is_active_train, y_train)
        self.f_pareto = self.get_pareto_front(y_train[:, :self.problem.n_obj])

    def evaluate_f(self, f_predict: np.ndarray, f_var_predict: np.ndarray) -> np.ndarray:
        return self.get_mpoi_f(f_predict, f_var_predict, self.f_pareto, self.euclidean)

    @classmethod
    def get_mpoi_f(cls, f_predict: np.ndarray, f_var_predict: np.ndarray, f_pareto: np.ndarray, euclidean: bool) \
            -> np.ndarray:

        mpoi = np.empty((f_predict.shape[0], 1))
        for i in range(f_predict.shape[0]):
            mpoi[i, 0] = cls._mpoi(f_pareto, f_predict[i, :], f_var_predict[i, :], euclidean=euclidean)

        mpoi[mpoi < 1e-6] = 0.
        return 1.-mpoi

    @classmethod
    def _mpoi(cls, f_pareto: np.ndarray, f_predict: np.ndarray, var_predict: np.ndarray, euclidean: bool) -> float:

        n, n_f = f_pareto.shape

        # Probability of being dominated for each point in the Pareto front along each objective dimension
        def cdf_not_better(f, f_pred, var_pred):  # Rahat 2017, Eq. 11, 12
            return ndtr((f_pred-f)/np.sqrt(var_pred))

        p_is_dom_dim = np.empty((n, n_f))
        for i_f in range(n_f):
            p_is_dom_dim[:, i_f] = cdf_not_better(f_pareto[:, i_f], f_predict[i_f], var_predict[i_f])

        # Probability of being dominated for each point along all dimensions: Rahat 2017, Eq. 10
        p_is_dom = np.prod(p_is_dom_dim, axis=1)

        # Probability of domination for each point: Rahat 2017, Eq. 13
        p_dom = 1-p_is_dom

        # Minimum probability of domination: Rahat 2017, Eq. 14
        min_poi = np.min(p_dom)

        # Multiply by distance to Pareto front if requested
        if euclidean:
            min_poi *= cls._get_euclidean_moment(min_poi, f_pareto, f_predict)

        return min_poi

    @classmethod
    def _get_euclidean_moment(cls, p_dominate: float, f_pareto: np.ndarray, f_predict: np.ndarray) -> float:

        # If the probability of domination is less than 50%, it means we are on the wrong side of the Pareto front
        if p_dominate < .5:
            return 0.

        return np.min(np.sqrt(np.sum((f_predict-f_pareto) ** 2, axis=1)))  # Parr Eq. 6.9


class EnsembleInfill(ConstrainedInfill):
    """
    Infill strategy that optimize multiple underlying infill criteria simultaneously, thereby getting the best
    compromise between what the different infills suggest.

    More information and application:
    Lyu, W. et al., 2018, July. Batch Bayesian optimization via multi-objective acquisition ensemble for automated
    analog circuit design. In International conference on machine learning (pp. 3306-3314). PMLR.

    Inspired by:
    Cowen-Rivers, A.I. et al., 2022. HEBO: pushing the limits of sample-efficient hyper-parameter optimisation. Journal
    of Artificial Intelligence Research, 74, pp.1269-1349.
    """

    def __init__(self, infills: List[ConstrainedInfill] = None, constraint_strategy: ConstraintStrategy = None):
        self.infills = infills
        super().__init__(constraint_strategy=constraint_strategy)

    def _initialize(self):
        # Get set of default infills if none given
        if self.infills is None:
            if self.problem.n_obj == 1:
                self.infills = [FunctionEstimateConstrainedInfill(), LowerConfidenceBoundInfill(),
                                ExpectedImprovementInfill(), ProbabilityOfImprovementInfill()]
            else:
                self.infills = [FunctionEstimateConstrainedInfill(), LowerConfidenceBoundInfill()]

        # Reset the constraint handling strategies of the underlying infills and initialize them
        for infill in self.infills:
            if isinstance(infill, ConstrainedInfill):
                infill.constraint_strategy = IgnoreConstraints()
            infill.initialize(self.problem, self.surrogate_model, self.normalization)

        super()._initialize()

    def set_samples(self, x_train: np.ndarray, is_active_train: np.ndarray, y_train: np.ndarray):
        super().set_samples(x_train, is_active_train, y_train)
        for infill in self.infills:
            infill.set_samples(x_train, is_active_train, y_train)

    def get_n_infill_objectives(self) -> int:
        return sum([infill.get_n_infill_objectives() for infill in self.infills])

    def evaluate_f(self, f_predict: np.ndarray, f_var_predict: np.ndarray) -> np.ndarray:
        # Merge underlying infill criteria
        f_underlying = [infill.evaluate_f(f_predict, f_var_predict) for infill in self.infills]
        return np.column_stack(f_underlying)

    def select_infill_solutions(self, population: Population, infill_problem: Problem, n_infill) -> Population:
        # Get the Pareto front
        opt_pop = filter_optimum(population, least_infeasible=True)

        # If we have less infills available than requested, return all
        if len(opt_pop) <= n_infill:
            return opt_pop

        # If there are less infills than objectives requested, randomly select from the Pareto front
        if n_infill <= self.n_f_ic:
            i_select = np.random.choice(len(opt_pop), n_infill)
            return opt_pop[i_select]

        # Select by repeatedly eliminating crowded points from the Pareto front
        for _ in range(len(opt_pop)-n_infill):
            crowding_of_front = calc_crowding_distance(opt_pop.get('F'))

            min_crowding = np.min(crowding_of_front)
            i_min_crowding = np.where(crowding_of_front == min_crowding)[0]
            i_remove = np.random.choice(i_min_crowding) if len(i_min_crowding) > 1 else i_min_crowding[0]

            i_keep = np.ones((len(opt_pop),), dtype=bool)
            i_keep[i_remove] = False
            opt_pop = opt_pop[i_keep]
        return opt_pop


class IgnoreConstraints(ConstraintStrategy):

    def get_n_infill_constraints(self) -> int:
        return 0

    def evaluate(self, x: np.ndarray, g: np.ndarray, g_var: np.ndarray) -> np.ndarray:
        return np.zeros((x.shape[0], 0))


class HCInfill(SurrogateInfill):
    """Infill that wraps another infill and modifies it for dealing with hidden constraints"""

    def __init__(self, infill: SurrogateInfill, hc_strategy: HiddenConstraintStrategy):
        self._infill = infill
        self._hc_strategy = hc_strategy
        super().__init__()

        self._initialize_from_underlying(infill)

    @property
    def needs_variance(self):
        return self._infill.needs_variance

    def _initialize(self):
        self._infill.initialize(self.problem, self.surrogate_model, self.normalization)

    def set_samples(self, x_train: np.ndarray, is_active_train: np.ndarray, y_train: np.ndarray):
        self._infill.set_samples(x_train, is_active_train, y_train)

    def _initialize_from_underlying(self, infill: SurrogateInfill):
        if infill.problem is not None:
            self.initialize(infill.problem, infill.surrogate_model, infill.normalization)

            if infill.x_train is not None:
                self.set_samples(infill.x_train, infill.is_active_train, infill.y_train)

    def predict(self, x: np.ndarray, is_active: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self._infill.predict(x, is_active)

    def select_infill_solutions(self, population, infill_problem, n_infill):
        return self._infill.select_infill_solutions(population, infill_problem, n_infill)

    def reset_infill_log(self):
        super().reset_infill_log()
        self._infill.reset_infill_log()

    def predict_variance(self, x: np.ndarray, is_active: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self._infill.predict_variance(x, is_active)

    def get_n_infill_objectives(self) -> int:
        return self._infill.get_n_infill_objectives()

    def get_n_infill_constraints(self) -> int:
        n_constr = self._infill.get_n_infill_constraints()
        if self._hc_strategy.adds_infill_constraint():
            n_constr += 1
        return n_constr

    def _evaluate(self, x: np.ndarray, is_active: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        f_infill, g_infill = self._infill.evaluate(x, is_active)
        f_infill = self._hc_strategy.mod_infill_objectives(x, f_infill)

        if self._hc_strategy.adds_infill_constraint():
            g_hc = self._hc_strategy.evaluate_infill_constraint(x)
            g_infill = np.column_stack([g_infill, g_hc]) if g_infill is not None else np.array([g_hc]).T

        return f_infill, g_infill
