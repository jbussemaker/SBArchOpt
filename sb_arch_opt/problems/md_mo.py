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

This test suite contains a set of mixed-discrete multi-objective problems.
"""
import numpy as np
from pymoo.core.variable import Real
from pymoo.problems.multi.zdt import ZDT1
from pymoo.problems.single.himmelblau import Himmelblau as HB
from sb_arch_opt.problems.continuous import *
from sb_arch_opt.problems.problems_base import *

__all__ = ['MOHimmelblau', 'MDMOHimmelblau', 'DMOHimmelblau', 'MOGoldstein', 'MDMOGoldstein',
           'DMOGoldstein', 'MOZDT1', 'MDZDT1', 'DZDT1']


class MOHimmelblau(NoHierarchyProblemBase):
    """Multi-objective version of the Himmelblau test problem"""

    def __init__(self):
        self._problem = problem = HB()
        des_vars = [Real(bounds=(problem.xl[i], problem.xu[i])) for i in range(problem.n_var)]
        super().__init__(des_vars, n_obj=2)

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):

        f_out[:, 0] = self._problem.evaluate(x, return_as_dictionary=True)['F'][:, 0]
        f_out[:, 1] = self._problem.evaluate(x[:, ::-1], return_as_dictionary=True)['F'][:, 0]


class MDMOHimmelblau(MixedDiscretizerProblemBase):
    """Mixed-discrete version of the multi-objective Himmelblau test problem"""

    def __init__(self):
        super().__init__(MOHimmelblau(), n_vars_int=1)


class DMOHimmelblau(MixedDiscretizerProblemBase):
    """Discrete version of the multi-objective Himmelblau test problem"""

    def __init__(self):
        super().__init__(MOHimmelblau())


class MOGoldstein(NoHierarchyProblemBase):
    """Multi-objective version of the Goldstein test problem"""

    def __init__(self):
        self._problem = problem = Goldstein()
        super().__init__(problem.des_vars, n_obj=2)

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        f_out[:, 0] = self._problem.evaluate(x, return_as_dictionary=True)['F'][:, 0]
        f_out[:, 1] = -self._problem.evaluate(x+.25, return_as_dictionary=True)['F'][:, 0]


class MDMOGoldstein(MixedDiscretizerProblemBase):
    """Mixed-discrete version of the multi-objective Goldstein test problem"""

    def __init__(self):
        super().__init__(MOGoldstein(), n_vars_int=1)


class DMOGoldstein(MixedDiscretizerProblemBase):
    """Discrete version of the multi-objective Goldstein test problem"""

    def __init__(self):
        super().__init__(MOGoldstein())


class MOZDT1(NoHierarchyWrappedProblem):
    """Wrapper for ZDT1 test problem"""

    def __init__(self):
        super().__init__(ZDT1())


class MDZDT1(MixedDiscretizerProblemBase):
    """Mixed-discrete version of the multi-objective ZDT1 test problem"""

    def __init__(self):
        super().__init__(ZDT1(), n_opts=5, n_vars_int=15)


class DZDT1(MixedDiscretizerProblemBase):
    """Discrete version of the multi-objective ZDT1 test problem"""

    def __init__(self):
        super().__init__(ZDT1(), n_opts=5)


if __name__ == '__main__':
    # MOHimmelblau().print_stats()
    # MDMOHimmelblau().print_stats()
    # DMOHimmelblau().print_stats()
    # # MOHimmelblau().plot_pf()
    # # MDMOHimmelblau().plot_pf()
    # DMOHimmelblau().plot_pf()

    # MOGoldstein().print_stats()
    # MDMOGoldstein().print_stats()
    # DMOGoldstein().print_stats()
    # # MOGoldstein().plot_pf()
    # # MDMOGoldstein().plot_pf()
    # DMOGoldstein().plot_pf()

    MOZDT1().print_stats()
    MDZDT1().print_stats()
    DZDT1().print_stats()
    # MOZDT1().plot_pf()
    # MDZDT1().plot_pf()
    DZDT1().plot_pf()
