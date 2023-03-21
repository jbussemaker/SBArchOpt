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

This test suite contains a set of continuous unconstrained single-objective problems.
"""
import numpy as np
from pymoo.core.variable import Real
from pymoo.problems.single.himmelblau import Himmelblau as HB
from pymoo.problems.single.rosenbrock import Rosenbrock as RB
from pymoo.problems.single.griewank import Griewank as GW
from sb_arch_opt.problems.problems_base import *

__all__ = ['Himmelblau', 'Rosenbrock', 'Griewank', 'Goldstein', 'Branin']


class Himmelblau(NoHierarchyWrappedProblem):

    def __init__(self):
        super().__init__(HB())


class Rosenbrock(NoHierarchyWrappedProblem):

    def __init__(self, n_var=10):
        super().__init__(RB(n_var=n_var))


class Griewank(NoHierarchyWrappedProblem):

    def __init__(self):
        super().__init__(GW(n_var=10))


class Goldstein(NoHierarchyProblemBase):
    """Goldstein-Price test problem, implementation based on
    https://github.com/scipy/scipy/blob/main/benchmarks/benchmarks/go_benchmark_functions/go_funcs_G.py#L88"""

    def __init__(self):
        des_vars = [Real(bounds=(-2, 2)) for _ in range(2)]
        super().__init__(des_vars)

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        a = (1 + (x[:, 0] + x[:, 1] + 1) ** 2
             * (19 - 14 * x[:, 0] + 3 * x[:, 0] ** 2
             - 14 * x[:, 1] + 6 * x[:, 0] * x[:, 1] + 3 * x[:, 1] ** 2))
        b = (30 + (2 * x[:, 0] - 3 * x[:, 1]) ** 2
             * (18 - 32 * x[:, 0] + 12 * x[:, 0] ** 2
             + 48 * x[:, 1] - 36 * x[:, 0] * x[:, 1] + 27 * x[:, 1] ** 2))
        f_out[:, 0] = a*b


class Branin(NoHierarchyProblemBase):
    """
    Branin test function from:
    Forrester, A., Sobester, A., & Keane, A. (2008). Engineering design via surrogate modelling: a practical guide.
    """

    _des_vars = [
        Real(bounds=(0, 1)), Real(bounds=(0, 1)),
    ]

    def __init__(self):
        super().__init__(self._des_vars)

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):

        for i in range(x.shape[0]):
            f_out[i, 0] = self._h(x[i, 0], x[i, 1])

    @staticmethod
    def _h(x1, x2):
        t1 = (15*x2 - (5/(4*np.pi**2))*(15*x1-5)**2 + (5/np.pi)*(15*x1-5) - 6)**2
        t2 = 10*(1-1/(8*np.pi))*np.cos(15*x1-5) + 10
        return ((t1+t2)-54.8104)/51.9496


if __name__ == '__main__':
    Himmelblau().print_stats()
    Rosenbrock().print_stats()
    Griewank().print_stats()
    Goldstein().print_stats()
    Branin().print_stats()
    # Branin().plot_design_space()
