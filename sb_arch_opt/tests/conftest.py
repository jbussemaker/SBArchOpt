import pytest
import itertools
import numpy as np
from typing import Optional, Tuple
from sb_arch_opt.sampling import *
from sb_arch_opt.problems.problems_base import *
from pymoo.core.variable import Real, Integer, Choice
from pymoo.problems.multi.zdt import ZDT1

try:
    import openturns as ot
    from sb_arch_opt.uncertainty import *
    from sb_arch_opt.stochastic_problem import *
    HAS_UNCERTAINTY = True
except ImportError:
    HAS_UNCERTAINTY = False

    class StochasticArchOptProblem:
        pass


def check_dependency():
    if not HAS_UNCERTAINTY:
        raise ImportError(
            'Looks like SBArchOpt uncertainty package is not installed! Run: pip install sb-arch-opt[uncertainty]')


class DummyProblem(ArchOptTestProblemBase):

    def __init__(self, only_discrete=False, fail=False):
        self._problem = problem = ZDT1(n_var=2 if only_discrete else 5)
        if only_discrete:
            des_vars = [Choice(options=[str(9-j) for j in range(10)]) if i == 0 else Integer(bounds=(1, 10))
                        for i in range(problem.n_var)]
        else:
            des_vars = [Real(bounds=(0, 1)) if i % 2 == 0 else (
                Choice(options=[str(9-j) for j in range(10)]) if i == 1 else Integer(bounds=(0, 9)))
                         for i in range(problem.n_var)]
        self.only_discrete = only_discrete
        self.fail = fail
        self._provide_all_x = True
        self._i_eval = 0
        super().__init__(des_vars, n_obj=problem.n_obj)

    def might_have_hidden_constraints(self):
        return self.fail

    def _get_n_valid_discrete(self) -> int:
        if self.only_discrete:
            return 10*5 + 5
        return 10*10

    def set_provide_all_x(self, provide_all_x):
        self._provide_all_x = provide_all_x
        if 'all_discrete_x' in self.design_space.__dict__:
            del self.design_space.__dict__['all_discrete_x']

    def _gen_all_discrete_x(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not self._provide_all_x:
            return
        x, is_active = [], []
        cartesian_prod_values = HierarchicalExhaustiveSampling.get_exhaustive_sample_values(self, n_cont=1)
        if self.only_discrete:
            for x_dv in itertools.product(*cartesian_prod_values):
                if x_dv[0] >= 5 and x_dv[1] != cartesian_prod_values[1][0]:
                    continue
                x.append(x_dv)
                is_active.append([True, x_dv[0] < 5])
        else:
            for x_dv in itertools.product(*cartesian_prod_values):
                x.append(x_dv)
                is_active.append([True]*4+[x_dv[1] < 5])
        return np.array(x), np.array(is_active)

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        self._correct_x_impute(x, is_active_out)
        assert np.all(x >= self.xl)
        assert np.all(x <= self.xu)

        i_dv = np.where(self.is_cat_mask)[0][0]
        cat_values = self.get_categorical_values(x, i_dv)
        assert np.all(x[:, i_dv] == [9-int(val) for val in cat_values])
        assert np.all((x[:, i_dv] == 0) == (cat_values == '9'))

        x_eval = x.copy()
        x_eval[:, self.is_discrete_mask] = (x_eval[:, self.is_discrete_mask]-self.xl[self.is_discrete_mask])/9
        out = self._problem.evaluate(x_eval, return_as_dictionary=True)
        f_out[:, :] = out['F']

        if self.fail:
            is_failed = np.zeros((len(x),), dtype=bool)
            is_failed[(self._i_eval % 2)::2] = True
            f_out[is_failed, :] = np.nan
            self._i_eval += len(x)

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        values = x[:, 0 if self.only_discrete else 1]
        is_active[:, -1] = values < 5

    def __repr__(self):
        return f'{self.__class__.__name__}(only_discrete={self.only_discrete})'


@pytest.fixture
def problem():
    return DummyProblem()


@pytest.fixture
def discrete_problem():
    return DummyProblem(only_discrete=True)


@pytest.fixture
def failing_problem():
    return DummyProblem(fail=True)


def make_space(*distributions) -> 'StochasticParameterSpace':
    return StochasticParameterSpace([StochasticParameter(f'u{i}', dist) for i, dist in enumerate(distributions)])


class VectorizedProblem(StochasticArchOptProblem):
    """f = (u - x0)^2 + x1^2, evaluated for all design points at once"""

    def __init__(self, n=100, seed=42, uq_method=None, fail=False, **kwargs):
        check_dependency()
        self.fail = fail
        self.seen_parameters = []
        super().__init__([Real(bounds=(-2., 2.)), Real(bounds=(-2., 2.))],
                         param_space=make_space(ot.Normal(1., .05)),
                         uq_method=uq_method if uq_method is not None else MonteCarlo(n, seed=seed),
                         n_obj=1, **kwargs)

    def _is_conditionally_active(self):
        return [False, False]

    def _correct_x(self, x, is_active):
        pass

    def _arch_evaluate_sample(self, x, is_active, f_out, g_out, h_out, parameters, *args, **kwargs):
        self.seen_parameters.append(np.asarray(parameters).copy())
        f_out[:, 0] = (parameters[0] - x[:, 0])**2 + x[:, 1]**2
        if self.fail and parameters[0] > 1.02:  # some samples fail to evaluate
            f_out[:, 0] = np.nan


class HierarchicalProblem(StochasticArchOptProblem):
    """Implicit (hierarchical) design space, a constraint, and the per-row evaluation pattern"""

    def __init__(self, n=50, seed=1, **kwargs):
        check_dependency()
        super().__init__([Choice(options=['electric', 'hybrid']), Real(bounds=(.2, 1.)), Real(bounds=(.1, .4))],
                         param_space=StochasticParameterSpace([
                             StochasticParameter('payload', ot.Normal(2., .3)),
                             StochasticParameter('bsfc', ot.Normal(.42, .075)),
                         ]),
                         uq_method=MonteCarlo(n, seed=seed), n_obj=1, n_ieq_constr=1, **kwargs)

    def _is_conditionally_active(self):
        return [False, False, True]

    def _correct_x(self, x, is_active):
        is_active[:, 2] = x[:, 0] == 1  # fuel fraction only active for the hybrid branch

    def _arch_evaluate_sample(self, x, is_active, f_out, g_out, h_out, parameters, *args, **kwargs):
        payload, bsfc = parameters
        for i in range(x.shape[0]):
            mass = .5 + x[i, 1] + payload
            energy = x[i, 2]*130./(bsfc+.1) if x[i, 0] == 1 else x[i, 1]*40.
            f_out[i, 0] = mass**1.5 / energy
            g_out[i, 0] = mass - 3.5


class AllResponseKindsProblem(StochasticArchOptProblem):
    """One objective, one inequality and one equality constraint, each with its own scalar"""

    def __init__(self, **kwargs):
        check_dependency()
        kwargs.setdefault('obj_scalar', [Mean()])
        kwargs.setdefault('ieq_constr_scalar', [Mean()])
        kwargs.setdefault('eq_constr_scalar', [Margin(k=3.)])
        super().__init__([Real(bounds=(0., 1.))], param_space=make_space(ot.Normal(1., .2)),
                         uq_method=MonteCarlo(200, seed=5),
                         n_obj=1, n_ieq_constr=1, n_eq_constr=1, **kwargs)

    def _is_conditionally_active(self):
        return [False]

    def _correct_x(self, x, is_active):
        pass

    def _arch_evaluate_sample(self, x, is_active, f_out, g_out, h_out, parameters, *args, **kwargs):
        f_out[:, 0] = parameters[0] + x[:, 0]
        g_out[:, 0] = parameters[0] - x[:, 0]
        h_out[:, 0] = parameters[0] * x[:, 0]


class DeterministicResponseProblem(StochasticArchOptProblem):
    """Two objectives, of which the second does not depend on the stochastic parameters at all"""

    def __init__(self, uq_method=None, **kwargs):
        super().__init__([Real(bounds=(0., 1.))], param_space=make_space(ot.Normal(1., .2)),
                         uq_method=uq_method or MonteCarlo(50, seed=3), n_obj=2, **kwargs)

    def _is_conditionally_active(self):
        return [False]

    def _correct_x(self, x, is_active):
        pass

    def _arch_evaluate_sample(self, x, is_active, f_out, g_out, h_out, parameters, *args, **kwargs):
        f_out[:, 0] = parameters[0] * x[:, 0]
        f_out[:, 1] = 2. * x[:, 0]


@pytest.fixture
def stochastic_problem():
    return VectorizedProblem()


@pytest.fixture
def hierarchical_problem():
    return HierarchicalProblem()


@pytest.fixture
def all_response_kinds_problem():
    return AllResponseKindsProblem()


@pytest.fixture
def deterministic_response_problem():
    return DeterministicResponseProblem()


def pytest_sessionstart(session):
    from sb_arch_opt.util import _prevent_capture
    print('PREVENT CAPTURE')
    _prevent_capture()