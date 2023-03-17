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
from pymoo.core.variable import Real
from sb_arch_opt.problems.hierarchical import *
from sb_arch_opt.problems.problems_base import *
from sb_arch_opt.problem import ArchOptProblemBase
from sb_arch_opt.sampling import RepairedLatinHypercubeSampling
from pymoo.problems.single.ackley import Ackley

__all__ = ['SampledFailureRateMixin', 'Mueller01', 'Mueller02', 'Mueller08', 'Alimo', 'MOHierarchicalRosenbrockHC',
           'HCMOHierarchicalTestProblem']


class SampledFailureRateMixin(ArchOptProblemBase):
    """Mixin to determine the failure rate by monte-carlo sampling"""

    n_samples_failure_rate = 10000

    def get_failure_rate(self) -> float:
        x = RepairedLatinHypercubeSampling().do(self, self.n_samples_failure_rate).get('X')
        out = self.evaluate(x, return_as_dictionary=True)
        is_failed = self.get_failed_points(out)
        return np.sum(is_failed)/len(is_failed)

    def might_have_hidden_constraints(self):
        return True


class Mueller01(SampledFailureRateMixin, NoHierarchyProblemBase):
    """
    Test problem 1 (several disconnected failure regions) of:
    https://pubsonline.informs.org/doi/suppl/10.1287/ijoc.2018.0864/suppl_file/ijoc.2018.0864.sm1.pdf

    Citation: Mueller, J., Day, M. "Surrogate Optimization of Computationally Expensive Black-Box Problems with Hidden
    Constraints", 2019, DOI: https://doi.org/10.1287/ijoc.2018.0864
    """

    def __init__(self, n_var=5):
        self._ackley = a = Ackley(n_var=n_var)
        des_vars = [Real(bounds=(-10, 10)) for _ in range(a.n_var)]
        super().__init__(des_vars)

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        f_out[:, :] = self._ackley.evaluate(x, return_as_dictionary=True)['F']

        def _is_failed(ix):
            for i_dv in range(x.shape[1]):
                if -.2 <= x[ix, i_dv] <= .2:
                    return True

            cx = np.sum(x[ix, :]*(np.sin(x[ix, :]) + .1))
            return cx > 0

        for i in range(x.shape[0]):
            if _is_failed(i):
                f_out[i, :] = np.nan


class Mueller02(SampledFailureRateMixin, NoHierarchyProblemBase):
    """
    Test problem 2 (one failure region) of:
    https://pubsonline.informs.org/doi/suppl/10.1287/ijoc.2018.0864/suppl_file/ijoc.2018.0864.sm1.pdf

    Citation: Mueller, J., Day, M. "Surrogate Optimization of Computationally Expensive Black-Box Problems with Hidden
    Constraints", 2019, DOI: https://doi.org/10.1287/ijoc.2018.0864
    """

    def __init__(self):
        des_vars = [Real(bounds=(-3*np.pi, 3*np.pi)) for _ in range(4)]
        super().__init__(des_vars)

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        f_out[:, 0] = (x[:, 0] - x[:, 1])**2 + \
                      np.exp((1 - np.sin(x[:, 0]))**2) * np.cos(x[:, 1]) + \
                      np.exp((1 - np.cos(x[:, 1]))**2) * np.sin(x[:, 0])

        x_min_term = np.sqrt(np.abs(x[:, [0]] - x + 1))
        x_plus_term = np.sqrt(np.abs(x[:, [0]] + x + 1))

        cx = np.sum((x * np.sin(x_min_term) * np.cos(x_plus_term)) +
                    ((x[:, [0]] + 1) * np.sin(x_plus_term) * np.cos(x_min_term)), axis=1) - 5

        f_out[cx > 0, :] = np.nan


class Mueller08(SampledFailureRateMixin, NoHierarchyProblemBase):
    """
    Test problem 8 (one failure region) of:
    https://pubsonline.informs.org/doi/suppl/10.1287/ijoc.2018.0864/suppl_file/ijoc.2018.0864.sm1.pdf

    Equation (8c) is modified a bit to make the non-failed region a larger

    Citation: Mueller, J., Day, M. "Surrogate Optimization of Computationally Expensive Black-Box Problems with Hidden
    Constraints", 2019, DOI: https://doi.org/10.1287/ijoc.2018.0864
    """

    def __init__(self, n_var=10):
        des_vars = [Real(bounds=(-10, 10)) for _ in range(n_var)]
        super().__init__(des_vars)

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        inner_sum = [np.sum(j*np.sin((j+1)*x) + j, axis=1) for j in range(1, 6)]
        f_out[:, 0] = np.sum(np.column_stack(inner_sum), axis=1)

        cx = np.sum(x**4 - 16*x**2 + 5*x, axis=1) - 1000*self.n_var
        f_out[cx > 0, :] = np.nan


class Alimo(SampledFailureRateMixin, NoHierarchyProblemBase):
    """
    Test problem used by:
    Alimo et al. "Delaunay-based global optimization in nonconvex domains defined by hidden constraints", 2018,
    DOI: 10.1007/978-3-319-89890-2_17
    """

    def __init__(self):
        n_var = 2
        des_vars = [Real(bounds=(0, 1)) for _ in range(n_var)]
        super().__init__(des_vars)

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        x0 = [.19, .29]  # In the paper, no other reference point is given
        f_out[:, 0] = np.sum(np.abs(x - x0)**2, axis=1) - .024*self.n_var

        # The term of -.25 is added
        cx = (self.n_var/12) + .1*np.sum(4*(x-.7)**2 - 2*np.cos(4*np.pi*(x-.7)), axis=1) - .25
        f_out[cx >= 0, :] = np.nan


class MOHierarchicalRosenbrockHC(SampledFailureRateMixin, MOHierarchicalRosenbrock):
    """
    Adaptation of the multi-objective hierarchical Rosenbrock problem, that sets points with a large constraint
    violation to NaN, simulating hidden constraints.
    """

    def __init__(self):
        super().__init__()
        self.n_ieq_constr -= 1

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


class HCMOHierarchicalTestProblem(SampledFailureRateMixin, HierarchicalMetaProblemBase):
    """
    Multi-objective hierarchical test problem with hidden constraints:
    - Only approximately 42% of design variables are active in a DOE
    - Approximately 60% of solutions do not converge in a DOE (i.e. return nan --> hidden constraint)
    """

    def __init__(self):
        super().__init__(MOHierarchicalRosenbrockHC(), n_rep=2, n_maps=2, f_par_range=[100, 100])

    def __repr__(self):
        return f'{self.__class__.__name__}()'


if __name__ == '__main__':
    # MOHierarchicalRosenbrockHC().print_stats()
    # MOHierarchicalRosenbrockHC().plot_pf()

    # HCMOHierarchicalTestProblem().print_stats()
    # HCMOHierarchicalTestProblem().plot_pf()

    # Mueller01().print_stats()
    # Mueller01().plot_design_space(x_base=[-.5]*5)
    # Mueller01().plot_pf()
    # Mueller02().print_stats()
    # Mueller02().plot_design_space()
    # Mueller02().plot_pf()
    # Mueller08().print_stats()
    # Mueller08().plot_pf()
    # Mueller08().plot_design_space()

    # Alimo().print_stats()
    Alimo().plot_design_space()
