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
from deprecated import deprecated
from scipy.spatial import distance
from pymoo.core.variable import Real
from sb_arch_opt.problems.constrained import *
from sb_arch_opt.problems.hierarchical import *
from sb_arch_opt.problems.problems_base import *
from sb_arch_opt.problems.continuous import Branin, Rosenbrock
from sb_arch_opt.problem import ArchOptProblemBase
from sb_arch_opt.sampling import HierarchicalSampling
from pymoo.problems.single.ackley import Ackley

__all__ = ['SampledFailureRateMixin', 'Mueller01', 'Mueller02', 'Mueller08', 'MOMueller08', 'Alimo', 'HCBranin',
           'MOHierarchicalRosenbrockHC', 'HCMOHierarchicalTestProblem', 'RandomHiddenConstraintsBase', 'HCSphere',
           'HierarchicalRosenbrockHC', 'ConstraintHiderMetaProblem', 'CantileveredBeamHC', 'MDCantileveredBeamHC',
           'CarsideHC', 'MDCarsideHC', 'CarsideHCLess', 'MDMueller02', 'MDMueller08', 'MDMOMueller08',
           'HierMueller02', 'HierMueller08', 'MOHierMueller08', 'AlimoEdge', 'HierAlimo', 'HierAlimoEdge',
           'Tfaily01', 'Tfaily02', 'Tfaily03', 'Tfaily04']


class SampledFailureRateMixin(ArchOptProblemBase):
    """Mixin to determine the failure rate by monte-carlo sampling"""

    n_samples_failure_rate = 10000

    def get_failure_rate(self) -> float:
        x = HierarchicalSampling().do(self, self.n_samples_failure_rate).get('X')
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


class MDMueller02(SampledFailureRateMixin, MixedDiscretizerProblemBase):
    """Mixed-discrete version of the Mueller 2 problem"""

    def __init__(self):
        super().__init__(Mueller02(), n_opts=6, n_vars_int=2)


class HierMueller02(SampledFailureRateMixin, TunableHierarchicalMetaProblem):
    """Hierarchical Mueller 2 problem"""

    def __init__(self):
        super().__init__(lambda n: Mueller02(), imp_ratio=6., n_subproblem=20, diversity_range=.5)


class Mueller08(SampledFailureRateMixin, NoHierarchyProblemBase):
    """
    Test problem 8 (one failure region) of:
    https://pubsonline.informs.org/doi/suppl/10.1287/ijoc.2018.0864/suppl_file/ijoc.2018.0864.sm1.pdf

    Equation (8c) is modified a bit to make the non-failed region a bit larger.

    Citation: Mueller, J., Day, M. "Surrogate Optimization of Computationally Expensive Black-Box Problems with Hidden
    Constraints", 2019, DOI: https://doi.org/10.1287/ijoc.2018.0864
    """

    _mo = False

    def __init__(self, n_var=10):
        des_vars = [Real(bounds=(-10, 10)) for _ in range(n_var)]
        super().__init__(des_vars, n_obj=2 if self._mo else 1)

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        inner_sum = [np.sum(j*np.sin((j+1)*x) + j, axis=1) for j in range(1, 6)]
        f_out[:, 0] = np.sum(np.column_stack(inner_sum), axis=1)
        if self._mo:
            inner_sum = [np.sum(j*np.cos((j+1)*x[:, :-1]) + j, axis=1) for j in range(1, 6)]
            f_out[:, 1] = np.sum(np.column_stack(inner_sum), axis=1)
            f_out[:, 1] -= .2*f_out[:, 0]

        cx = np.sum(x**4 - 16*x**2 + 5*x, axis=1) - 1000*self.n_var
        f_out[cx > 0, :] = np.nan


class MOMueller08(Mueller08):
    """Multi-objective version of the Mueller 8 test problem"""

    _mo = True


class MDMueller08(SampledFailureRateMixin, MixedDiscretizerProblemBase):
    """Mixed-discrete version of the Mueller 8 problem"""

    def __init__(self):
        super().__init__(Mueller08(), n_opts=10, n_vars_int=2)


class MDMOMueller08(SampledFailureRateMixin, MixedDiscretizerProblemBase):
    """Mixed-discrete, multi-objective version of the Mueller 8 problem"""

    def __init__(self):
        super().__init__(MOMueller08(), n_opts=10, n_vars_int=2)


class HierMueller08(SampledFailureRateMixin, TunableHierarchicalMetaProblem):
    """Hierarchical Mueller 8 problem"""

    def __init__(self):
        super().__init__(lambda n: Mueller08(), imp_ratio=6., n_subproblem=20, diversity_range=.5)


class MOHierMueller08(SampledFailureRateMixin, TunableHierarchicalMetaProblem):
    """Multi-objective hierarchical Mueller 8 problem"""

    def __init__(self):
        super().__init__(lambda n: MOMueller08(), imp_ratio=6., n_subproblem=20, diversity_range=.5)


class Alimo(SampledFailureRateMixin, Branin):
    """
    Modified test problem used by:
    Alimo et al. "Delaunay-based global optimization in nonconvex domains defined by hidden constraints", 2018,
    DOI: 10.1007/978-3-319-89890-2_17

    The underlying problem is replaced by the Branin function.
    """

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        # x0 = [.19, .29]  # In the paper, no other reference point is given
        # f_out[:, 0] = np.sum(np.abs(x - x0)**2, axis=1) - .024*self.n_var
        super()._arch_evaluate(x, is_active_out, f_out, g_out, h_out, *args, **kwargs)

        # The term of -.25 is added
        x_fail = (x-self.xl)/(self.xu-self.xl)
        self._mod_x_fail(x_fail)
        cx = (self.n_var/12) + .1*np.sum(4*(x_fail-.7)**2 - 2*np.cos(4*np.pi*(x_fail-.7)), axis=1) - .25
        f_out[cx >= 0, :] = np.nan

    def _mod_x_fail(self, x_fail):
        x_fail[:, 1] = 1-x_fail[:, 1]
        x_fail[:, 0] += .15


class AlimoEdge(Alimo):
    """
    Modified Alimo/Branin problem where the optimum points lie at the edge of the failed region.
    """

    def _mod_x_fail(self, x_fail):
        x_fail[:, 0] -= .05
        x_fail[:, 1] += .05


class HierAlimo(SampledFailureRateMixin, TunableHierarchicalMetaProblem):
    """Hierarchical Alimo problem"""

    def __init__(self):
        super().__init__(lambda n: Alimo(), imp_ratio=6., n_subproblem=20, diversity_range=.5)


class HierAlimoEdge(SampledFailureRateMixin, TunableHierarchicalMetaProblem):
    """Hierarchical AlimoEdge problem"""

    def __init__(self):
        super().__init__(lambda n: AlimoEdge(), imp_ratio=6., n_subproblem=20, diversity_range=.5)


class HCBranin(SampledFailureRateMixin, Branin):
    """
    Modified Branin problem with infeasibility disk, as used in:
    Gelbart et al., "Bayesian optimization with unknown constraints", arXiv preprint arXiv:1403.5607 (2014).
    """

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        super()._arch_evaluate(x, is_active_out, f_out, g_out, h_out, *args, **kwargs)

        # The original function is scaled differently
        c = (x[:, 0]-.5)**2 + (x[:, 1]-.5)**2
        f_out[c > .22, :] = np.nan


class RandomHiddenConstraintsBase(SampledFailureRateMixin, NoHierarchyProblemBase):
    """
    Base class for randomly adding failed regions to some design space.

    Inspired by:
    Sacher et al., "A classification approach to efficient global optimization in presence of non-computable domains",
    2018, DOI: 10.1007/s00158-018-1981-8
    """

    def __init__(self, des_vars, density=.25, radius=.1, seed=None, **kwargs):
        self.density = density
        self.radius = radius
        self.seed = seed
        super().__init__(des_vars, **kwargs)
        self._x_failed = None
        self._scale = None

    def _set_failed_points(self, x: np.ndarray, f_out: np.ndarray, g_out: np.ndarray, h_out: np.ndarray):
        if self._x_failed is None:
            if self.seed is not None:
                np.random.seed(self.seed)
            x_failed = HierarchicalSampling().do(self, 100).get('X')
            i_selected = np.random.choice(len(x_failed), size=int(self.density*len(x_failed)), replace=False)

            self._scale = 1/(self.xu-self.xl)
            self._x_failed = x_failed[i_selected, :]*self._scale

        is_failed = np.any(distance.cdist(x*self._scale, self._x_failed) < self.radius, axis=1)
        f_out[is_failed, :] = np.nan
        g_out[is_failed, :] = np.nan
        h_out[is_failed, :] = np.nan

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        """At the end of the function, use `_set_failed_points`!"""
        raise NotImplementedError


class HCSphere(RandomHiddenConstraintsBase):
    """Sphere with randomly-added hidden constraints"""

    _n_vars = 2
    _density = .25

    def __init__(self):
        des_vars = [Real(bounds=(0, 1)) for _ in range(self._n_vars)]
        super().__init__(des_vars, density=self._density, radius=.1, seed=0)

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        f_out[:, 0] = np.sum((x - .5*(self.xu-self.xl))**2, axis=1)
        self._set_failed_points(x, f_out, g_out, h_out)


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


class HierarchicalRosenbrockHC(MOHierarchicalRosenbrockHC):
    """Single-objective hierarchical hidden-constraints Rosenbrock problem"""

    _mo = False


@deprecated(reason='Not realistic (see HierarchicalMetaProblemBase docstring)')
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


class ConstraintHiderMetaProblem(SampledFailureRateMixin, ArchOptTestProblemBase):
    """Meta problem that turns one or more constraints of an underlying problem into hidden constraints"""

    def __init__(self, problem: ArchOptTestProblemBase, i_g_hc):
        self._problem = problem
        self._i_g_hc = i_g_hc = np.array(i_g_hc)

        n_constr = problem.n_ieq_constr
        if not np.all(i_g_hc < n_constr):
            raise RuntimeError(f'Unavailable constraints: {i_g_hc}')
        n_constr -= len(i_g_hc)

        super().__init__(problem.design_space, n_obj=problem.n_obj, n_ieq_constr=n_constr,
                         n_eq_constr=problem.n_eq_constr)

    def _get_n_valid_discrete(self) -> int:
        return self._problem.get_n_valid_discrete()

    def _gen_all_discrete_x(self):
        return self._problem.all_discrete_x

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        self._correct_x_impute(x, is_active_out)
        out = self._problem.evaluate(x, return_as_dictionary=True)
        f_out[:, :] = out['F']
        if 'H' in out:
            h_out[:, :] = out['H']

        g = out['G']
        g_out[:, :] = np.delete(g, self._i_g_hc, axis=1)

        is_failed = np.any(g[:, self._i_g_hc] < 0, axis=1)
        f_out[is_failed, :] = np.nan
        g_out[is_failed, :] = np.nan

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        x[:, :], is_active[:, :] = self._problem.correct_x(x)

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class CantileveredBeamHC(ConstraintHiderMetaProblem):

    def __init__(self):
        super().__init__(ArchCantileveredBeam(), i_g_hc=[1])


class MDCantileveredBeamHC(ConstraintHiderMetaProblem):

    def __init__(self):
        super().__init__(MDCantileveredBeam(), i_g_hc=[1])


class CarsideHC(ConstraintHiderMetaProblem):

    def __init__(self):
        super().__init__(ArchCarside(), i_g_hc=[3, 7])


class CarsideHCLess(ConstraintHiderMetaProblem):

    def __init__(self):
        super().__init__(ArchCarside(), i_g_hc=[6])


class MDCarsideHC(ConstraintHiderMetaProblem):

    def __init__(self):
        super().__init__(MDCarside(), i_g_hc=[3, 7])


class Tfaily01(SampledFailureRateMixin, NoHierarchyProblemBase):
    """
    Test problem 1 from:
    Tfaily et al., "Efficient Acquisition Functions for Bayesian Optimization in the Presence of Hidden Constraints",
    AIAA Aviation 2023 Forum
    """

    def __init__(self):
        des_vars = [
            Real(bounds=(-2, 2)),
            Real(bounds=(-2, 2)),
        ]
        super().__init__(des_vars)

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):

        def w(z):
            return np.exp(-(z-1)**2) + np.exp(-.8*(z+1)**2) - .5*np.sin(8*(z+.1))

        f_out[:, 0] = -w(x[:, 0])*w(x[:, 1])

        # Add the 45-deg rotated ellipse as failure region
        alpha = .25*np.pi
        xx_ = np.cos(alpha)*x[:, 0] + np.sin(alpha)*x[:, 1]
        yy_ = np.sin(alpha)*x[:, 0] - np.cos(alpha)*x[:, 1]
        is_failed = (xx_/2)**2 + yy_**2 < 1
        f_out[is_failed, :] = np.nan


class Tfaily02(SampledFailureRateMixin, Branin):
    """
    Test problem 2 from:
    Tfaily et al., "Efficient Acquisition Functions for Bayesian Optimization in the Presence of Hidden Constraints",
    AIAA Aviation 2023 Forum
    """

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        super()._arch_evaluate(x, is_active_out, f_out, g_out, h_out, *args, **kwargs)

        is_failed = (np.abs(x[:, 0] - .5) < .5) & (np.abs(x[:, 1] - .5) < .4)
        f_out[is_failed, :] = np.nan


class Tfaily03(SampledFailureRateMixin, Rosenbrock):
    """
    Test problem 3 from:
    Tfaily et al., "Efficient Acquisition Functions for Bayesian Optimization in the Presence of Hidden Constraints",
    AIAA Aviation 2023 Forum
    """

    def __init__(self):
        super().__init__(n_var=4)

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        super()._arch_evaluate(x, is_active_out, f_out, g_out, h_out, *args, **kwargs)

        is_failed = np.zeros(x.shape, dtype=bool)
        is_failed[:, :2] = (0 < x[:, :2]) & (x[:, :2] < 1)
        is_failed[:, 2:] = (1 < x[:, 2:]) & (x[:, 2:] < 2)
        is_failed = np.any(is_failed, axis=1)
        f_out[is_failed, :] = np.nan


class Tfaily04(SampledFailureRateMixin, NoHierarchyProblemBase):
    """
    Test problem 4 from:
    Tfaily et al., "Efficient Acquisition Functions for Bayesian Optimization in the Presence of Hidden Constraints",
    AIAA Aviation 2023 Forum
    """

    def __init__(self):
        des_vars = [Real(bounds=(-500, 500)) for _ in range(6)]
        super().__init__(des_vars)

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):

        f_out[:, 0] = 2513.895 - np.sum(x*np.sin(np.sqrt(np.abs(x))), axis=1)

        is_failed = np.any((350 < x) & (x < 420), axis=1)
        f_out[is_failed, :] = np.nan


if __name__ == '__main__':
    # MOHierarchicalRosenbrockHC().print_stats()
    # HierarchicalRosenbrockHC().print_stats()
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

    # MOMueller08().print_stats()
    # MOMueller08().plot_pf()
    # MDMueller02().print_stats()
    # MDMueller02().plot_pf()
    # MDMueller02().plot_pf()
    # MDMueller08().print_stats()
    # MDMOMueller08().print_stats()
    # MDMOMueller08().plot_pf()
    # HierMueller02().print_stats()
    # HierMueller08().print_stats()
    # MOHierMueller08().print_stats()

    # Alimo().print_stats()
    # AlimoEdge().print_stats()
    # Alimo().plot_design_space()
    # AlimoEdge().plot_design_space()
    # HierAlimo().print_stats()
    # HierAlimoEdge().print_stats()
    # HCBranin().print_stats()
    # HCBranin().plot_design_space()
    # HCSphere().print_stats()
    # HCSphere().plot_design_space()

    # CantileveredBeamHC().print_stats()
    # MDCantileveredBeamHC().print_stats()
    # CarsideHC().print_stats()
    # CarsideHCLess().print_stats()
    # MDCarsideHC().print_stats()

    Tfaily01().print_stats()
    Tfaily02().print_stats()
    Tfaily03().print_stats()
    Tfaily04().print_stats()
    # Tfaily04().plot_design_space()
