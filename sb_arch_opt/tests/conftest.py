import pytest
import itertools
import numpy as np
from typing import Optional, Tuple
from sb_arch_opt.sampling import *
from sb_arch_opt.problems.problems_base import *
from pymoo.core.variable import Real, Integer, Choice
from pymoo.problems.multi.zdt import ZDT1


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


def pytest_sessionstart(session):
    from sb_arch_opt.util import _prevent_capture
    print('PREVENT CAPTURE')
    _prevent_capture()
