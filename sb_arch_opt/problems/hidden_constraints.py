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

This test suite contains a set of mixed-discrete, multi-objective, constrained, hierarchical test problems that are
subject to hidden constraints.
"""
import numpy as np
from sb_arch_opt.problems.hierarchical import *

__all__ = ['MOHierarchicalRosenbrockHC', 'HCMOHierarchicalTestProblem']


class MOHierarchicalRosenbrockHC(MOHierarchicalRosenbrock):
    """
    Adaptation of the multi-objective hierarchical Rosenbrock problem, that sets points with a large constraint
    violation to NaN, simulating hidden constraints.
    """

    def __init__(self):
        super().__init__()
        self.n_ieq_constr -= 1

    def might_have_hidden_constraints(self):
        return True

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        self._correct_x_impute(x, is_active_out)

        g = np.empty((x.shape[0], 2))
        self._eval_f_g(x, f_out, g)

        hc_violated = g[:, 1] > 0.
        if self._mo:
            hc_violated |= np.abs(.5 - (f_out[:, 0] / 20) % 1) > .35
            hc_violated |= (f_out[:, 1] > 1000) & (np.abs(.5 - (f_out[:, 1] / 100) % 1) > .35)

        f_out[hc_violated] = np.nan
        g[hc_violated] = np.nan

        g_out[:, 0] = g[:, 0]


class HCMOHierarchicalTestProblem(HierarchicalMetaProblemBase):
    """
    Multi-objective hierarchical test problem with hidden constraints:
    - Only approximately 42% of design variables are active in a DOE
    - Approximately 60% of solutions do not converge in a DOE (i.e. return nan --> hidden constraint)
    """

    def __init__(self):
        super().__init__(MOHierarchicalRosenbrockHC(), n_rep=2, n_maps=2, f_par_range=[100, 100])

    def might_have_hidden_constraints(self):
        return True

    def __repr__(self):
        return f'{self.__class__.__name__}()'


if __name__ == '__main__':
    # MOHierarchicalRosenbrockHC().print_stats()
    # MOHierarchicalRosenbrockHC().plot_pf()

    HCMOHierarchicalTestProblem().print_stats()
    # HCMOHierarchicalTestProblem().plot_pf()
