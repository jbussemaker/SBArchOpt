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
import numpy as np
from pymoo.core.problem import Problem
from pymoo.core.variable import Real, Integer
from sb_arch_opt.pareto_front import ArchOptTestProblemBase

__all__ = ['NoHierarchyProblemBase', 'NoHierarchyWrappedProblem', 'MixedDiscretizerProblemBase']


class NoHierarchyProblemBase(ArchOptTestProblemBase):
    """Base class for test problems that have no decision hierarchy"""

    def get_n_valid_discrete(self) -> int:
        # No hierarchy, so the number of valid points is the same as the number of declared points
        return self.get_n_declared_discrete()

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        pass  # No need to correct anything

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class NoHierarchyWrappedProblem(NoHierarchyProblemBase):
    """Base class for non-hierarchical test problems that wrap an existing Problem class (to add SBArchOpt features)"""

    def __init__(self, problem: Problem):
        self._problem = problem
        des_vars = [Real(bounds=(problem.xl[i], problem.xu[i])) for i in range(problem.n_var)]
        super().__init__(des_vars, n_obj=problem.n_obj, n_ieq_constr=problem.n_ieq_constr)

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        out = self._problem.evaluate(x, return_as_dictionary=True)
        f_out[:, :] = out['F']
        if self.n_ieq_constr > 0:
            g_out[:, :] = out['G']


class MixedDiscretizerProblemBase(NoHierarchyProblemBase):
    """Problem class that turns an existing test problem into a mixed-discrete problem by mapping the first n (if not
    given: all) variables to integers with a given number of options."""

    def __init__(self, problem: Problem, n_opts=10, n_vars_int: int = None):
        self.problem = problem
        self.n_opts = n_opts
        if n_vars_int is None:
            n_vars_int = problem.n_var
        self.n_vars_int = n_vars_int

        if not problem.has_bounds():
            raise ValueError('Underlying problem should have bounds defined!')
        self._xl_orig = problem.xl
        self._xu_orig = problem.xu

        des_vars = [Integer(bounds=(0, n_opts-1)) if i < n_vars_int else Real(bounds=(problem.xl[i], problem.xu[i]))
                    for i in range(problem.n_var)]
        super().__init__(des_vars, n_obj=problem.n_obj, n_ieq_constr=problem.n_ieq_constr)
        self.callback = problem.callback

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        """
        Implement evaluation and write results in the provided output matrices:
        - x (design vectors): discrete variables have integer values, imputed design vectors can be output here
        - is_active (activeness): vector specifying for each design variable whether it was active or not
        - f (objectives): written as a minimization
        - g (inequality constraints): written as "<= 0"
        - h (equality constraints): written as "= 0"
        """

        n = self.n_vars_int
        xl, xu = self.xl, self.xu
        xl_orig, xu_orig = self._xl_orig, self._xu_orig

        x_underlying = x.copy()
        x_underlying[:, :n] = ((x_underlying[:, :n]-xl[:n])/(xu[:n]-xl[:n]))*(xu_orig[:n]-xl_orig[:n])+xl_orig[:n]

        out = self.problem.evaluate(x_underlying, return_as_dictionary=True, *args, **kwargs)
        f_out[:, :] = out['F']
        if 'G' in out:
            g_out[:, :] = out['G']

    def _map_x(self, x: np.ndarray) -> np.ndarray:
        x = np.copy(x)

        xl, xu = self.xl, self.xu
        xl_orig, xu_orig = self._xl_orig, self._xu_orig

        n = self.n_vars_int
        x[:, :n] = ((x[:, :n]-xl[:n])/(xu[:n]-xl[:n]))*(xu_orig[:n]-xl_orig[:n])+xl_orig[:n]
        return x
