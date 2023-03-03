import pytest
import numpy as np
from sb_arch_opt.pareto_front import *
from pymoo.core.variable import Real, Integer, Choice
from pymoo.problems.multi.zdt import ZDT1


class DummyProblem(ArchOptTestProblemBase):

    def __init__(self, only_discrete=False):
        self._problem = problem = ZDT1(n_var=2 if only_discrete else 5)
        if only_discrete:
            var_types = [Choice(options=[str(9-j) for j in range(10)]) if i == 0 else Integer(bounds=(0, 9))
                         for i in range(problem.n_var)]
        else:
            var_types = [Real(bounds=(0, 1)) if i % 2 == 0 else (
                Choice(options=[str(9-j) for j in range(10)]) if i == 1 else Integer(bounds=(0, 9)))
                         for i in range(problem.n_var)]
        self.only_discrete = only_discrete
        super().__init__(var_types, n_obj=problem.n_obj)

    def get_n_valid_discrete(self) -> int:
        if self.only_discrete:
            return 10*5 + 5
        return 10*10

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        self._correct_x(x, is_active_out)

        i_dv = np.where(self.is_cat_mask)[0][0]
        cat_idx = x[:, i_dv]
        cat_values = self.get_categorical_values(i_dv, cat_idx)
        assert np.all(cat_idx == [9-int(val) for val in cat_values])
        assert np.all((cat_idx == 0) == (cat_values == '9'))

        x_eval = x.copy()
        x_eval[:, self.is_discrete_mask] /= 9
        out = self._problem.evaluate(x_eval, return_as_dictionary=True)
        f_out[:, :] = out['F']

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        values = x[:, 0 if self.only_discrete else 1]
        is_active[:, -1] = values < 5
        self._impute_x(x, is_active)

    def __repr__(self):
        return f'{self.__class__.__name__}(only_discrete={self.only_discrete})'


@pytest.fixture
def problem():
    return DummyProblem()


@pytest.fixture
def discrete_problem():
    return DummyProblem(only_discrete=True)
