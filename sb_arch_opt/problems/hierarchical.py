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

This test suite contains a set of mixed-discrete, constrained, hierarchical, multi-objective problems.
"""
import enum
import numpy as np
from typing import *
from deprecated import deprecated
from sb_arch_opt.sampling import *
from pymoo.problems.multi.zdt import ZDT1
from sb_arch_opt.problems.discrete import *
from sb_arch_opt.problems.constrained import *
from sb_arch_opt.design_space_explicit import *
from sb_arch_opt.problems.problems_base import *
from pymoo.core.variable import Real, Integer, Choice
from pymoo.util.ref_dirs import get_reference_directions

__all__ = ['HierarchyProblemBase', 'HierarchicalGoldstein', 'HierarchicalRosenbrock', 'ZaeffererHierarchical',
           'ZaeffererProblemMode', 'MOHierarchicalGoldstein', 'MOHierarchicalRosenbrock', 'HierarchicalMetaProblemBase',
           'MOHierarchicalTestProblem', 'Jenatton', 'TunableHierarchicalMetaProblem', 'TunableZDT1', 'HierZDT1',
           'HierZDT1Small', 'HierZDT1Large', 'HierDiscreteZDT1', 'HierBranin', 'HierCantileveredBeam', 'HierCarside',
           'NeuralNetwork']


class HierarchyProblemBase(ArchOptTestProblemBase):
    """Base class for test problems that have decision hierarchy"""
    _force_get_discrete_rates = True

    def _print_extra_stats(self):
        self.get_discrete_rates(force=self._force_get_discrete_rates, show=True)

    def _get_n_valid_discrete(self) -> int:
        raise NotImplementedError

    def _get_n_active_cont_mean(self) -> Optional[float]:
        if np.all(~self.is_conditionally_active[self.is_cont_mask]):
            return float(np.sum(self.is_cont_mask))

    def _get_n_correct_discrete(self) -> int:
        # True if only imputation is ever applied (no correction)
        return self.get_n_declared_discrete()

    def _get_n_active_cont_mean_correct(self) -> Optional[float]:
        # True if only imputation is ever applied (no correction)
        return float(np.sum(self.is_cont_mask))

    def _is_conditionally_active(self) -> List[bool]:
        _, is_act_all = self.all_discrete_x
        if is_act_all is None:
            _, is_act_all = self.design_space.all_discrete_x_by_trial_and_imputation
        return list(np.any(~is_act_all, axis=0))

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class HierarchicalGoldstein(HierarchyProblemBase):
    """
    Variable-size design space Goldstein function from:
    Pelamatti 2020: "Bayesian Optimization of Variable-Size Design Space Problems", section 5.2 and Appendix B

    Properties:
    - 5 continuous variables
    - 4 integer variables
    - 2 categorical variables
    - Depending on the categorical variables, 8 different sub-problems are defined, ranging from 2 cont + 4 integer to
      5 cont + 2 integer variables
    - 1 objective, 1 constraint
    """

    _mo = False

    def __init__(self):
        des_vars = [
            Real(bounds=(0, 100)), Real(bounds=(0, 100)), Real(bounds=(0, 100)), Real(bounds=(0, 100)),
            Real(bounds=(0, 100)),
            Integer(bounds=(0, 2)), Integer(bounds=(0, 2)), Integer(bounds=(0, 2)), Integer(bounds=(0, 2)),
            Choice(options=[0, 1, 2, 3]), Choice(options=[0, 1]),
        ]

        n_obj = 2 if self._mo else 1
        super().__init__(des_vars, n_obj=n_obj, n_ieq_constr=1)

    def _get_n_valid_discrete(self) -> int:
        # w1 and w2 determine activeness, and we can ignore continuous dimensions
        n_valid = np.ones((4, 2), dtype=int)  # w1, w2

        # DV 5 is valid when w1 == 0 or w1 == 2
        n_valid[[0, 2], :] *= 3

        # DV6 is valid when w1 <= 1
        n_valid[:2, :] *= 3

        # DV 7 and 8 are always valid
        n_valid *= 3*3

        return int(np.sum(n_valid))

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        self._correct_x_impute(x, is_active_out)
        f_h_map = self._map_f_h()
        g_map = self._map_g()

        for i in range(x.shape[0]):
            x_i = x[i, :5]
            z_i = np.array([int(z) for z in x[i, 5:9]])
            w_i = np.array([int(w) for w in x[i, 9:]])

            f_idx = int(w_i[0]+w_i[1]*4)
            f_out[i, 0] = self.h(*f_h_map[f_idx](x_i, z_i))
            if self._mo:
                f2 = self.h(*f_h_map[f_idx](x_i+30, z_i))+(f_idx/7.)*5
                f_out[i, 1] = f2

            g_idx = int(w_i[0])
            g_out[i, 0] = self.g(*g_map[g_idx](x_i, z_i))

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        w1 = x[:, 9].astype(int)
        w2 = x[:, 10].astype(int)

        is_active[:, 2] = (w1 == 1) | (w1 == 3)  # x3
        is_active[:, 3] = w1 >= 2  # x4
        is_active[:, 4] = w2 == 1  # x5

        is_active[:, 5] = (w1 == 0) | (w1 == 2)  # z1
        is_active[:, 6] = w1 <= 1  # z2

    @staticmethod
    def h(x1, x2, x3, x4, x5, z3, z4, cos_term: bool) -> float:
        h = MDGoldstein.h(x1, x2, x3, x4, z3, z4)
        if cos_term:
            h += 5.*np.cos(2.*np.pi*(x5/100.))-2.
        return h

    @staticmethod
    def _map_f_h() -> List[Callable[[np.ndarray, np.ndarray], tuple]]:

        # Appendix B, Table 6-11
        _x3 = [20, 50, 80]
        _x4 = [20, 50, 80]

        def _f1(x, z):
            return x[0], x[1], _x3[z[0]], _x4[z[1]], x[4], z[2], z[3], False

        def _f2(x, z):
            return x[0], x[1], x[2],      _x4[z[1]], x[4], z[2], z[3], False

        def _f3(x, z):
            return x[0], x[1], _x3[z[0]], x[3],      x[4], z[2], z[3], False

        def _f4(x, z):
            return x[0], x[1], x[2],      x[3],      x[4], z[2], z[3], False

        def _f5(x, z):
            return x[0], x[1], _x3[z[0]], _x4[z[1]], x[4], z[2], z[3], True

        def _f6(x, z):
            return x[0], x[1], x[2],      _x4[z[1]], x[4], z[2], z[3], True

        def _f7(x, z):
            return x[0], x[1], _x3[z[0]], x[3],      x[4], z[2], z[3], True

        def _f8(x, z):
            return x[0], x[1], x[2],      x[3],      x[4], z[2], z[3], True

        return [_f1, _f2, _f3, _f4, _f5, _f6, _f7, _f8]

    @staticmethod
    def g(x1, x2, c1, c2):
        return -(x1-50.)**2. - (x2-50.)**2. + (20.+c1*c2)**2.

    @staticmethod
    def _map_g() -> List[Callable[[np.ndarray, np.ndarray], tuple]]:

        # Appendix B, Table 12-15
        _c1 = [3., 2., 1.]
        _c2 = [.5, -1., -2.]

        def _g1(x, z):
            return x[0], x[1], _c1[z[0]], _c2[z[1]]

        def _g2(x, z):
            return x[0], x[1], .5,        _c2[z[1]]

        def _g3(x, z):
            return x[0], x[1], _c1[z[0]], .7

        def _g4(x, z):
            return x[0], x[1], _c1[z[2]], _c2[z[3]]

        return [_g1, _g2, _g3, _g4]

    @classmethod
    def validate_ranges(cls, n_samples=5000, show=True):
        """Compare to Pelamatti 2020, Fig. 6"""
        import matplotlib.pyplot as plt
        from sb_arch_opt.sampling import HierarchicalSampling

        problem = cls()
        x = HierarchicalSampling().do(problem, n_samples).get('X')

        f, g = problem.evaluate(x)
        i_feasible = np.max(g, axis=1) <= 0.

        x_plt, y_plt = [], []
        for i in np.where(i_feasible)[0]:
            w_i = [int(w) for w in x[i, 9:]]
            f_idx = int(w_i[0]+w_i[1]*4)

            x_plt.append(f_idx)
            y_plt.append(f[i])

        plt.figure()
        plt.scatter(x_plt, y_plt, s=1)
        plt.xlabel('Sub-problem'), plt.ylabel('Feasible objective values')

        if show:
            plt.show()


class MOHierarchicalGoldstein(HierarchicalGoldstein):
    """
    Multi-objective adaptation of the hierarchical Goldstein problem. The Pareto front consists of a mix of SP6 and SP8,
    however it is difficult to get a consistent result with NSGA2.

    See Pelamatti 2020 Fig. 6 to compare. Colors in plot of run_test match colors of figure.
    """

    _mo = True

    @classmethod
    def run_test(cls):
        from pymoo.optimize import minimize
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.visualization.scatter import Scatter

        res = minimize(cls(), NSGA2(pop_size=200), termination=('n_gen', 200))
        w_idx = res.X[:, 9] + res.X[:, 10] * 4
        Scatter().add(res.F, c=w_idx, cmap='tab10', vmin=0, vmax=10, color=None).show()


class HierarchicalRosenbrock(HierarchyProblemBase):
    """
    Variable-size design space Rosenbrock function from:
    Pelamatti 2020: "Bayesian Optimization of Variable-Size Design Space Problems", section 5.3 and Appendix C

    Properties:
    - 8 continuous variables
    - 3 integer variables
    - 2 categorical variables
    - Depending on the categorical variables, 4 different sub-problems are defined
    - 1 objective, 2 constraints

    To validate, use so_run() and compare to Pelamatti 2020, Fig. 14
    """

    _mo = False  # Multi-objective

    def __init__(self):
        des_vars = [
            Real(bounds=(-1, .5)), Real(bounds=(0, 1.5)),
            Real(bounds=(-1, .5)), Real(bounds=(0, 1.5)),
            Real(bounds=(-1, .5)), Real(bounds=(0, 1.5)),
            Real(bounds=(-1, .5)), Real(bounds=(0, 1.5)),
            Integer(bounds=(0, 1)), Integer(bounds=(0, 1)), Integer(bounds=(0, 2)),
            Choice(options=[0, 1]), Choice(options=[0, 1]),
        ]

        n_obj = 2 if self._mo else 1
        super().__init__(des_vars, n_obj=n_obj, n_constr=2)

    def _get_n_valid_discrete(self) -> int:
        n_valid = np.ones((2, 2), dtype=int)*2*2  # w1, w2 for DV 8 and 9
        n_valid[:, 1] *= 3  # DV 10 is active when w2 == 1
        return int(np.sum(n_valid))

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        self._correct_x_impute(x, is_active_out)
        self._eval_f_g(x, f_out, g_out)

    @classmethod
    def _eval_f_g(cls, x: np.ndarray, f_out: np.ndarray, g: np.ndarray):
        a1 = [7, 7, 10, 10]
        a2 = [9, 6, 9, 6]
        add_z3 = [False, True, False, True]
        x_idx = [[0, 1, 2, 3], [0, 1, 4, 5], [0, 1, 2, 3, 6, 7], [0, 1, 4, 5, 6, 7]]
        x_idx_g2 = [[0, 1, 2, 3], [0, 1, 2, 3, 6, 7]]

        for i in range(x.shape[0]):
            x_i = x[i, :8]
            z_i = [int(z) for z in x[i, 8:11]]

            w_i = [int(w) for w in x[i, 11:]]
            idx = int(w_i[0]*2+w_i[1])

            x_fg = x_i[x_idx[idx]]
            f_out[i, 0] = f1 = cls.f(x_fg, z_i[0], z_i[1], z_i[2], a1[idx], a2[idx], add_z3[idx])
            if cls._mo:
                f2 = abs((400-f1)/40)**2 + np.sum((x_fg[:4]+1)**2*200)
                f_out[i, 1] = f2

            g[i, 0] = cls.g1(x_fg)
            g[i, 1] = cls.g2(x_i[x_idx_g2[idx]]) if idx < 2 else 0.

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        w1 = x[:, 11].astype(int)
        w2 = x[:, 12].astype(int)
        idx = w1*2+w2

        is_active[:, 2] = idx <= 2  # x3
        is_active[:, 3] = idx <= 2  # x4
        is_active[:, 4] = w2 == 1  # x5
        is_active[:, 5] = w2 == 1  # x6
        is_active[:, 6] = idx >= 1  # x7
        is_active[:, 7] = idx >= 1  # x8

        is_active[:, 10] = w2 == 1  # z3

    @staticmethod
    def f(x: np.ndarray, z1, z2, z3, a1, a2, add_z3: bool):
        s = 1. if z2 == 0 else -1.
        pre = 1. if z2 == 0 else .7

        xi, xi1 = x[:-1], x[1:]
        sum_term = np.sum(pre*a1*a2*(xi1-xi)**2 + ((a1+s*a2)/10.)*(1-xi)**2)
        f = 100.*z1 + sum_term
        if add_z3:
            f -= 35.*z3
        return f

    @staticmethod
    def g1(x: np.ndarray):
        xi, xi1 = x[:-1], x[1:]
        return np.sum(-(xi-1)**3 + xi1 - 2.6)

    @staticmethod
    def g2(x: np.ndarray):
        xi, xi1 = x[:-1], x[1:]
        return np.sum(-xi - xi1 + .4)

    @classmethod
    def validate_ranges(cls, n_samples=5000, show=True):
        """Compare to Pelamatti 2020, Fig. 13"""
        import matplotlib.pyplot as plt
        from sb_arch_opt.sampling import HierarchicalSampling

        problem = cls()
        x = HierarchicalSampling().do(problem, n_samples).get('X')

        f, g = problem.evaluate(x)
        i_feasible = np.max(g, axis=1) <= 0.

        x_plt, y_plt = [], []
        for i in np.where(i_feasible)[0]:
            w_i = [int(w) for w in x[i, 11:]]
            f_idx = int(w_i[0]*2+w_i[1])

            x_plt.append(f_idx)
            y_plt.append(f[i])

        plt.figure()
        plt.scatter(x_plt, y_plt, s=1)
        plt.xlabel('Sub-problem'), plt.ylabel('Feasible objective values')

        if show:
            plt.show()

    def _gen_all_discrete_x(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return self.design_space.all_discrete_x_by_trial_and_imputation


class MOHierarchicalRosenbrock(HierarchicalRosenbrock):
    """
    Multi-objective adaptation of the hierarchical Rosenbrock problem.

    See Pelamatti 2020 Fig. 13 to compare. Colors in plot of run_test match colors of figure.
    """

    _mo = True

    @classmethod
    def run_test(cls, show=True):
        from pymoo.optimize import minimize
        from pymoo.algorithms.moo.nsga2 import NSGA2

        res = minimize(cls(), NSGA2(pop_size=200), termination=('n_gen', 200))
        w_idx = res.X[:, 11]*2 + res.X[:, 12]
        HierarchicalMetaProblemBase.plot_sub_problems(w_idx, res.F, show=show)


class ZaeffererProblemMode(enum.Enum):
    A_OPT_INACT_IMP_PROF_UNI = 'A'
    B_OPT_INACT_IMP_UNPR_UNI = 'B'
    C_OPT_ACT_IMP_PROF_BI = 'C'
    D_OPT_ACT_IMP_UNPR_BI = 'D'
    E_OPT_DIS_IMP_UNPR_BI = 'E'


class ZaeffererHierarchical(HierarchyProblemBase):
    """
    Hierarchical test function from:
    Zaefferer 2018: "A First Analysis of Kernels for Kriging-Based Optimization in Hierarchical Search Spaces",
      section 5
    """

    _mode_map = {
        ZaeffererProblemMode.A_OPT_INACT_IMP_PROF_UNI: (.0, .6, .1),
        ZaeffererProblemMode.B_OPT_INACT_IMP_UNPR_UNI: (.1, .6, .1),
        ZaeffererProblemMode.C_OPT_ACT_IMP_PROF_BI: (.0, .4, .7),
        ZaeffererProblemMode.D_OPT_ACT_IMP_UNPR_BI: (.1, .4, .9),
        ZaeffererProblemMode.E_OPT_DIS_IMP_UNPR_BI: (.1, .4, .7),
    }

    def __init__(self, b=.1, c=.4, d=.7):
        self.b = b
        self.c = c
        self.d = d

        des_vars = [Real(bounds=(0, 1)), Real(bounds=(0, 1))]
        super().__init__(des_vars, n_obj=1)

        self.design_space.needs_cont_correction = True

    def _get_n_valid_discrete(self) -> int:
        return 1

    def _get_n_active_cont_mean(self) -> float:
        return 2-self.c

    def _get_n_correct_discrete(self) -> int:
        return 1

    def _get_n_active_cont_mean_correct(self) -> Optional[float]:
        return 2

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        self._correct_x_impute(x, is_active_out)
        f1 = (x[:, 0] - self.d)**2
        f2 = (x[:, 1] - .5)**2 + self.b
        f2[x[:, 0] <= self.c] = 0.
        f_out[:, 0] = f1+f2

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        is_active[:, 1] = x[:, 0] > self.c  # x2 is active if x1 > c

    @classmethod
    def from_mode(cls, problem_mode: ZaeffererProblemMode):
        b, c, d = cls._mode_map[problem_mode]
        return cls(b=b, c=c, d=d)

    def plot(self, show=True):
        import matplotlib.pyplot as plt

        xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
        zz = self.evaluate(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

        plt.figure(), plt.title(f'b = {self.b:.1f}, c = {self.c:.1f}, d = {self.d:.1f}')
        plt.colorbar(plt.contourf(xx, yy, zz, 50, cmap='viridis')).set_label('$f$')
        plt.contour(xx, yy, zz, 5, colors='k')
        plt.xlabel('$x_1$'), plt.ylabel('$x_2$')

        if show:
            plt.show()

    def __repr__(self):
        return f'{self.__class__.__name__}(b={self.b}, c={self.c}, d={self.d})'


@deprecated(reason='Not realistic (see docstring)')
class HierarchicalMetaProblemBase(HierarchyProblemBase):
    """
    Meta problem used for increasing the amount of design variables of an underlying mixed-integer/hierarchical problem.
    The idea is that design variables are repeated, and extra design variables are added for switching between the
    repeated design variables. Objectives are then slightly modified based on the switching variable.

    For correct modification of the objectives, a range of the to-be-expected objective function values at the Pareto
    front for each objective dimension should be provided (f_par_range).

    Note that each map will correspond to a new part of the Pareto front.

    DEPRECATED: this class and derived problems should not be used anymore, as they don't represent realistic
    hierarchical problem behavior:
    - One or more design variables might have options that are never selected
    - The spread between option occurrence of design variables is not realistic
    """

    def __init__(self, problem: ArchOptTestProblemBase, n_rep=2, n_maps=4, f_par_range=None):
        self._problem = problem

        # Create design vector: 1 selection variables and n_rep repetitions of underlying design variables
        des_vars = [Choice(options=list(range(n_maps)))]
        for i in range(n_rep):
            des_vars += problem.des_vars

        super().__init__(des_vars, n_obj=problem.n_obj, n_ieq_constr=problem.n_ieq_constr)

        self.n_maps = n_maps
        self.n_rep = n_rep

        # Create the mappings between repeated design variables and underlying: select_map specifies which of the
        # repeated variables to use to replace the values of the original design variables
        # The mappings are semi-random: different for different problem configurations, but repeatable for same configs
        rng = np.random.RandomState(problem.n_var * problem.n_obj * n_rep * n_maps)
        self.select_map = [rng.randint(0, n_rep, (problem.n_var,)) for _ in range(n_maps)]

        # Determine how to move the existing Pareto fronts: move them along the Pareto front dimensions to create a
        # composed Pareto front
        if f_par_range is None:
            f_par_range = 1.
        f_par_range = np.atleast_1d(f_par_range)
        if len(f_par_range) == 1:
            f_par_range = np.array([f_par_range[0]]*problem.n_obj)
        self.f_par_range = f_par_range

        ref_dirs = get_reference_directions("uniform", problem.n_obj, n_partitions=n_maps-1)
        i_rd = np.linspace(0, ref_dirs.shape[0]-1, n_maps).astype(int)
        self.f_mod = (ref_dirs[i_rd, :]-.5)*f_par_range

    def _get_n_valid_discrete(self) -> int:
        return self._problem.get_n_valid_discrete()*self.n_maps

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        self._correct_x_impute(x, is_active_out)

        xp, _ = self._get_xp_idx(x)
        f_mod = np.empty((x.shape[0], self.n_obj))
        for i in range(x.shape[0]):
            f_mod[i, :] = self.f_mod[int(x[i, 0]), :]

        fp, g = self._problem.evaluate(xp, return_values_of=['F', 'G'])
        f_out[:, :] = fp+f_mod
        if self.n_ieq_constr > 0:
            g_out[:, :] = g

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        is_active[:, 1:] = False

        xp, i_x_u = self._get_xp_idx(x)
        _, is_active_u = self._problem.correct_x(xp)
        for i in range(x.shape[0]):
            is_active[i, i_x_u[i, :]] = is_active_u[i, :]

    def _get_xp_idx(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Select design variables of the underlying problem based on the repeated variables and the selected mapping"""
        xp = np.empty((x.shape[0], self._problem.n_var))
        i_x_u = np.empty((x.shape[0], self._problem.n_var), dtype=int)
        for i in range(x.shape[0]):
            idx = int(x[i, 0])
            select_map = self.select_map[idx]
            i_x_u[i, :] = i_x_underlying = 1+select_map*len(select_map)+np.arange(0, len(select_map))
            xp[i, :] = x[i, i_x_underlying]

        return xp, i_x_u

    def run_test(self, show=True):
        from pymoo.optimize import minimize
        from pymoo.algorithms.moo.nsga2 import NSGA2

        print(f'Running hierarchical metaproblem: {self.n_var} vars ({self.n_rep} rep, {self.n_maps} maps), '
              f'{self.n_obj} obj, {self.n_ieq_constr} constr')
        res = minimize(self, NSGA2(pop_size=200), termination=('n_gen', 200))

        idx_rep = res.X[:, 0]
        xp, _ = self._get_xp_idx(res.X)
        w_idx = xp[:, 11]*2 + xp[:, 12]
        sp_idx = idx_rep * self.n_rep + w_idx
        sp_labels = ['Rep. %d, SP %d' % (i_rep+1, i+1) for i_rep in range(self.n_rep) for i in range(4)]

        self.plot_sub_problems(sp_idx, res.F, sp_labels=sp_labels, show=show)

    @staticmethod
    def plot_sub_problems(sp_idx: np.ndarray, f: np.ndarray, sp_labels=None, show=True):
        import matplotlib.pyplot as plt

        if f.shape[1] != 2:
            raise RuntimeError('Only for bi-objective optimization!')

        plt.figure(figsize=(4, 2))
        colors = plt.get_cmap('tab10')
        for sp_val in np.unique(sp_idx):
            sp_val = int(sp_val)
            sp_idx_mask = sp_idx == sp_val
            label = ('SP %d' % (sp_val+1,)) if sp_labels is None else sp_labels[sp_val]
            plt.scatter(f[sp_idx_mask, 0], f[sp_idx_mask, 1], color=colors.colors[sp_val], s=10, label=label)

        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.legend(frameon=False)
        plt.xlabel('$f_1$'), plt.ylabel('$f_2$')

        if show:
            plt.show()

    def __repr__(self):
        return f'{self.__class__.__name__}({self._problem}, n_rep={self.n_rep}, n_maps={self.n_maps}, ' \
               f'f_par_range={self.f_par_range})'


@deprecated(reason='Not realistic (see HierarchicalMetaProblemBase docstring)')
class MOHierarchicalTestProblem(HierarchicalMetaProblemBase):
    """
    Multi-objective hierarchical test problem based on the hierarchical rosenbrock problem. Increased number of design
    variables and increased sparseness (approx. 42% of design variables are active in a DOE).

    This is the analytical test problem used in:
    J.H. Bussemaker et al. "Effectiveness of Surrogate-Based Optimization Algorithms for System Architecture
    Optimization." AIAA AVIATION 2021 FORUM. 2021. DOI: 10.2514/6.2021-3095
    """

    def __init__(self):
        super().__init__(MOHierarchicalRosenbrock(), n_rep=2, n_maps=2, f_par_range=[100, 100])

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class Jenatton(HierarchyProblemBase):
    """
    Jenatton test function:
    - https://github.com/facebook/Ax/blob/main/ax/benchmark/problems/synthetic/hss/jenatton.py
    - https://github.com/facebook/Ax/blob/main/ax/metrics/jenatton.py
    """

    def __init__(self, explicit=True):
        self._explicit = explicit
        if explicit:
            ds = ExplicitArchDesignSpace([
                CategoricalParam('x1', [0, 1]),
                CategoricalParam('x2', [0, 1]),
                CategoricalParam('x3', [0, 1]),
                ContinuousParam('x4', 0, 1),
                ContinuousParam('x5', 0, 1),
                ContinuousParam('x6', 0, 1),
                ContinuousParam('x7', 0, 1),
                ContinuousParam('r8', 0, 1),
                ContinuousParam('r9', 0, 1),
            ])

            ds.add_conditions([
                # x1 == 0 activates x2, x4, x5, r8
                EqualsCondition(ds['x2'], ds['x1'], 0),
                EqualsCondition(ds['r8'], ds['x1'], 0),

                # x4 and x5 are additionally only activated if x2 == 0 or 1, respectively
                EqualsCondition(ds['x4'], ds['x2'], 0),
                EqualsCondition(ds['x5'], ds['x2'], 1),

                # x1 == 1 activates x3, x6, x7, r9
                EqualsCondition(ds['x3'], ds['x1'], 1),
                EqualsCondition(ds['r9'], ds['x1'], 1),

                # x6 and x7 are additionally only activated if x3 == 0 or 1, respectively
                EqualsCondition(ds['x6'], ds['x3'], 0),
                EqualsCondition(ds['x7'], ds['x3'], 1),
            ])

            ds_or_dv = ds
        else:
            ds_or_dv = [
                Choice(options=[0, 1]),  # x1
                Choice(options=[0, 1]),  # x2
                Choice(options=[0, 1]),  # x3
                Real(bounds=(0, 1)),  # x4
                Real(bounds=(0, 1)),  # x5
                Real(bounds=(0, 1)),  # x6
                Real(bounds=(0, 1)),  # x7
                Real(bounds=(0, 1)),  # r8
                Real(bounds=(0, 1)),  # r9
            ]
        super().__init__(ds_or_dv)

    def _get_n_valid_discrete(self) -> int:
        return 4

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):

        if not self._explicit:
            self._correct_x_impute(x, is_active_out)

        for i, xi in enumerate(x):
            is_active_i = np.zeros((x.shape[1],), dtype=bool)
            is_active_i[0] = True
            if xi[0] == 0:
                is_active_i[1] = True
                if xi[1] == 0:
                    is_active_i[[3, 7]] = True
                    f_out[i, 0] = xi[3]**2 + .1 + xi[7]  # x4^2 + .1 + r8
                else:
                    is_active_i[[4, 7]] = True
                    f_out[i, 0] = xi[4]**2 + .1 + xi[7]  # x5^2 + .1 + r8
            else:
                is_active_i[2] = True
                if xi[2] == 0:
                    is_active_i[[5, 8]] = True
                    f_out[i, 0] = xi[5]**2 + .1 + xi[8]  # x6^2 + .1 + r9
                else:
                    is_active_i[[6, 8]] = True
                    f_out[i, 0] = xi[6]**2 + .1 + xi[8]  # x7^2 + .1 + r9

            # Check if the design space is defined correctly
            assert np.all(is_active_i == is_active_out[i, :])

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        for i in [2, 5, 6, 8]:  # x1 = 0: x3, x6, x7, r9 inactive
            is_active[x[:, 0] == 0, i] = False
        is_active[(x[:, 0] == 0) & (x[:, 1] == 0), 4] = False  # x2 = 0: x5 inactive
        is_active[(x[:, 0] == 0) & (x[:, 1] == 1), 3] = False  # x2 = 1: x4 inactive

        for i in [1, 3, 4, 7]:  # x2, x4, x5, r8 inactive
            is_active[x[:, 0] == 1, i] = False
        is_active[(x[:, 0] == 1) & (x[:, 2] == 0), 6] = False  # x3 = 0: x7 inactive
        is_active[(x[:, 0] == 1) & (x[:, 2] == 1), 5] = False  # x3 = 1: x6 inactive


class TunableHierarchicalMetaProblem(HierarchyProblemBase):
    """
    Meta problem that turns any problem into a realistically-behaving hierarchical optimization problem with directly
    tunable properties:
    - `imp_ratio` [1+ float] Imputation ratio: ratio between declared nr of discrete design points (Cartesian product)
                                               and nr of valid discrete design points
    - `n_subproblem` [1+ int] Nr of discrete subproblems to separate the underlying problem into
    - `n_opts` [2+ int] Options per discrete selection variable
    - `cont_ratio` [0+ float] Ratio between continuous (underlying problem) and discrete (selection) design variables
    - `diversity_range` [0-1 float] Difference between most and least occurring value for the selection variables (note
                                    that increasing this value also moderately increases imputation ratio)

    Note that these settings assume a non-hierarchical underlying problem.

    It does this by:
    - Repeatedly separating the n_subproblem subproblems into n_opts options according to the provided diversity range
    - Splitting design variables until the desired imputation ratio is achieved
    - Initialize the underlying test problem with enough continuous vars to satisfy the continuous-to-discrete ratio
    - Define how each subproblem modifies the underlying problem's objectives and constraints
    """

    def __init__(self, problem_factory: Callable[[int], ArchOptTestProblemBase], imp_ratio: float, n_subproblem: int,
                 diversity_range: float = .25, n_opts=3, cont_ratio=1., repr_str=None):
        self._repr_str = repr_str

        self._imp_ratio = imp_ratio
        self._n_subproblem = n_subproblem
        self._n_opts = n_opts
        self._diversity_range = diversity_range
        self._cont_ratio = cont_ratio

        # Create design vectors by repeatedly separating with the given diversity range
        # Variables are separated according to _sep_group: round(x**sep_power*(n_opts-.01)-.5) with x [0..1]
        # The largest group is selected where x**sep_power*(n_opts-.01) == 1 --> x_lrg ~= (1/n_opts)**(1/sep_power)
        # the smallest group where x**sep_power*(n_opts-.01) == n_opts-1 --> x_sml ~= (1-1/n_opts)**(1/sep_power)
        # Here, we solve for sep_power such that x_lrg-(1-x_sml) == diversity_range
        sep_power_try = np.linspace(1, 10, 1000)
        div_ranges = (1/n_opts)**(1/sep_power_try) + (1-(1/n_opts))**(1/sep_power_try) - 1
        i_power = np.argmin(np.abs(div_ranges-diversity_range))
        sep_power = sep_power_try[i_power]

        def _sep_group(n_values, sep_power_):
            return np.round((np.linspace(0, 1, n_values)**sep_power_)*(n_opts-.01)-.5).astype(int)

        x_columns = []
        is_active_columns = []
        x_groups = [np.arange(n_subproblem)]
        diverse_separation = True
        while True:
            next_x_groups = []
            needed_separation = False
            x_column = np.zeros((n_subproblem,), dtype=int)
            is_active_column = np.zeros((n_subproblem,), dtype=bool)
            for i_in_group in x_groups:
                if len(i_in_group) == 1:
                    continue
                needed_separation = True

                # Distribute values according to the desired diversity range
                x_sep = _sep_group(len(i_in_group), sep_power if diverse_separation else 1.)
                x_sep = np.unique(x_sep, return_inverse=True)[1]
                x_column[i_in_group] = x_sep
                is_active_column[i_in_group] = True

                # Get next groups
                for x_next in np.unique(x_sep):
                    next_x_groups.append(i_in_group[x_sep == x_next])

            # If no separation was needed, we stop separating
            if not needed_separation:
                break

            x_columns.append(x_column)
            is_active_columns.append(is_active_column)
            x_groups = next_x_groups

            # Stop separating according to the desired diversity range if the imputation ratio would become too large
            if diverse_separation:
                n_group_max = max(len(grp) for grp in next_x_groups)
                n_dv_remain = np.ceil(np.log(n_group_max)/np.log(n_opts))+.5  # The .5 is a heuristic
                min_imp_ratio = (n_opts**(len(x_columns) + n_dv_remain))/n_subproblem
                if min_imp_ratio > imp_ratio:
                    diverse_separation = False

        x_discrete = np.column_stack(x_columns)
        is_act_discrete = np.column_stack(is_active_columns)

        # Separate design variables until we meet the imputation ratio requirement
        def _imp_ratio(x_discrete_):
            return np.prod(np.max(x_discrete_, axis=0)+1)/x_discrete_.shape[0]

        i_x_sep_at_min = x_discrete.shape[1]
        i_unique_blocked: List[Set[int]] = [set() for _ in range(x_discrete.shape[1])]
        while _imp_ratio(x_discrete) < imp_ratio:
            # Search from the end for groups to separate
            i_sep_sel = next_sep_mask = i_group_sep = None
            for i_sep in reversed(list(range(1, i_x_sep_at_min))):
                # Get different groups by unique vectors to the left
                x_sel_unique, idx_unique = np.unique(x_discrete[:, :i_sep], axis=0, return_inverse=True)

                for i_unique in range(len(x_sel_unique)):
                    # Check if this subgroup is blocked from separating (because of too high imputation ratio)
                    if i_unique in i_unique_blocked[i_sep]:
                        continue

                    # Check if group has at least 2 elements and is active
                    next_sep_mask = idx_unique == i_unique
                    if np.sum(next_sep_mask) <= 1:
                        continue

                    is_active_sep = np.any(is_act_discrete[next_sep_mask, i_sep])
                    if not is_active_sep:
                        continue

                    # Check if after separation there are any active variables left for this design variable
                    is_active_remain = is_act_discrete[:, i_sep].copy()
                    is_active_remain[next_sep_mask] = False
                    to_remain_not_active = np.all(~is_active_remain)
                    if to_remain_not_active:
                        continue
                    i_group_sep = i_unique
                    break

                # Nothing found for this variable: move to the left
                else:
                    continue

                # Group to separate was found!
                # Next time, start search from this same level
                i_x_sep_at_min = i_sep+1
                i_sep_sel = i_sep
                break
            if i_sep_sel is None:
                break

            # Determine where to move from and where to move to
            i_move_from = np.arange(i_sep_sel, x_discrete.shape[1])
            move_from_any_active = np.any(is_act_discrete[np.ix_(next_sep_mask, i_move_from)], axis=0)
            i_move_from = i_move_from[move_from_any_active]
            i_move_into = np.arange(x_discrete.shape[1]+1-len(i_move_from), x_discrete.shape[1]+1)

            # Separate the selected design variable into a new variable at the end of the vector
            x_discrete_sep = np.column_stack([x_discrete, np.zeros((x_discrete.shape[0],), dtype=int)])
            x_discrete_sep[next_sep_mask, i_sep_sel:] = 0
            x_discrete_sep[np.ix_(next_sep_mask, i_move_into)] = x_discrete[np.ix_(next_sep_mask, i_move_from)]

            is_act_sep = np.column_stack([is_act_discrete, np.zeros((x_discrete.shape[0],), dtype=bool)])
            is_act_sep[next_sep_mask, i_sep_sel:] = False
            is_act_sep[np.ix_(next_sep_mask, i_move_into)] = is_act_discrete[np.ix_(next_sep_mask, i_move_from)]

            # Remove newly created columns with no active variables
            has_active = np.any(is_act_sep, axis=0)
            x_discrete_sep = x_discrete_sep[:, has_active]
            is_act_sep = is_act_sep[:, has_active]

            # Check imputation ratio constraint
            if _imp_ratio(x_discrete_sep) > imp_ratio:
                i_unique_blocked[i_sep_sel].add(i_group_sep)
                continue
            x_discrete = x_discrete_sep
            is_act_discrete = is_act_sep

        self._x_sub = x_discrete
        self._is_act_sub = is_act_discrete

        # Initialize underlying problem
        n_cont = max(0, int(x_discrete.shape[1]*cont_ratio))
        self._problem = problem = problem_factory(max(2, n_cont))
        self._n_cont = n_cont = 0 if n_cont == 0 else problem.n_var
        pf: np.ndarray = problem.pareto_front()
        pf_min, pf_max = np.min(pf, axis=0), np.max(pf, axis=0)
        is_same = np.abs(pf_max-pf_min) < 1e-10
        pf_max[is_same] = pf_min[is_same]+1.
        self._pf_min, self._pf_max = pf_min, pf_max

        # Define subproblem transformations
        n_sub = x_discrete.shape[0]
        self._transform = transform = np.zeros((n_sub, problem.n_obj*2))
        n_trans = transform.shape[1]
        n_cycles = np.arange(n_trans)+1
        offset = .25*np.linspace(1, 0, n_trans+1)[:n_trans]
        for i_trans in range(n_trans):
            func = np.sin if i_trans % 2 == 0 else np.cos
            transform[:, i_trans] = func((np.linspace(0, 1, n_sub+1)[:n_sub]+offset[i_trans])*2*np.pi*n_cycles[i_trans])

        # mutual_distance = distance.cdist(transform, transform, metric='cityblock')
        # np.fill_diagonal(mutual_distance, np.nan)
        # if np.any(mutual_distance < 1e-10):
        #     raise RuntimeError('Duplicate transformations!')

        # Define design variables
        des_vars = []
        for dv_opts in (np.max(x_discrete, axis=0)+1):
            des_vars.append(Choice(options=list(range(int(dv_opts)))))

        for i_dv, des_var in enumerate(problem.des_vars[:n_cont]):
            if isinstance(des_var, Real):
                des_vars.append(Real(bounds=des_var.bounds))
            elif isinstance(des_var, Integer):
                des_vars.append(Integer(bounds=des_var.bounds))
            elif isinstance(des_var, Choice):
                des_vars.append(Choice(options=des_var.options))
            else:
                raise RuntimeError(f'Design variable type not supported: {des_var!r}')

        super().__init__(des_vars, n_obj=problem.n_obj, n_ieq_constr=problem.n_ieq_constr,
                         n_eq_constr=problem.n_eq_constr)

        self.design_space.use_auto_corrector = False
        self._correct_output = {}

    def _get_n_valid_discrete(self) -> int:
        n_discrete_underlying = self._problem.get_n_valid_discrete()
        return n_discrete_underlying*self._x_sub.shape[0]

    def _get_n_correct_discrete(self) -> int:
        n_correct_underlying = self._problem.get_n_correct_discrete()

        n_sub = self._x_sub.shape[1]
        n_opts_sub = self.xu[:n_sub]-self.xl[:n_sub]+1
        n_correct = np.ones(self._x_sub.shape)
        for j in range(n_sub):
            n_correct[~self._is_act_sub[:, j], j] = n_opts_sub[j]

        n_correct_sub = np.sum(np.prod(n_correct, axis=1))
        return int(n_correct_underlying*n_correct_sub)

    def might_have_hidden_constraints(self):
        return self._problem.might_have_hidden_constraints()

    def get_failure_rate(self) -> float:
        return self._problem.get_failure_rate()

    def get_n_batch_evaluate(self) -> Optional[int]:
        return self._problem.get_n_batch_evaluate()

    def _gen_all_discrete_x(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        x_sub, is_act_sub = self._x_sub, self._is_act_sub
        if self._n_cont == 0:
            return x_sub, is_act_sub

        x_problem, is_act_problem = HierarchicalExhaustiveSampling().get_all_x_discrete(self._problem)

        n_sub_select = x_sub.shape[0]
        x_sub = np.repeat(x_sub, x_problem.shape[0], axis=0)
        is_act_sub = np.repeat(is_act_sub, x_problem.shape[0], axis=0)
        x_problem = np.tile(x_problem, (n_sub_select, 1))
        is_act_problem = np.tile(is_act_problem, (n_sub_select, 1))

        x_all = np.column_stack([x_sub, x_problem])
        is_act_all = np.column_stack([is_act_sub, is_act_problem])
        return x_all, is_act_all

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        # Correct and impute
        self._correct_x_impute(x, is_active_out)
        i_sub_selected = self._correct_output['i_sub_sel']
        n_sub = self._x_sub.shape[1]

        # Evaluate underlying problem
        x_underlying = self._get_x_underlying(x[:, n_sub:])
        out = self._problem.evaluate(x_underlying, return_as_dictionary=True)
        if 'G' in out:
            g_out[:, :] = out['G']
        if 'H' in out:
            h_out[:, :] = out['H']

        # Transform outputs
        f_out[:, :] = self._transform_out(out['F'], i_sub_selected)

    def _transform_out(self, f: np.ndarray, i_sub_selected: np.ndarray) -> np.ndarray:
        pf_min, pf_max = self._pf_min, self._pf_max
        trans = self._transform
        f_norm = (f-pf_min)/(pf_max-pf_min)
        for i_obj in range(f.shape[1]):
            f_other = f_norm.copy()
            f_other[:, i_obj] = 0

            translate_shear = trans[i_sub_selected, i_obj::f.shape[1]]
            translate, scale = translate_shear.T
            fi_norm = f_norm[:, i_obj]
            fi_norm += .2*translate
            fi_norm = (fi_norm-.5)*(.5+.4*scale)+.5
            f_norm[:, i_obj] = fi_norm

        return f_norm*(pf_max-pf_min) + pf_min

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        # Match/correct sub-problem selection design variables
        i_sub_selected = self._get_corrected_sub_sel_idx(x)

        x_sub, is_act_sub = self._x_sub, self._is_act_sub
        n_sub = x_sub.shape[1]
        x[:, :n_sub] = x_sub[i_sub_selected, :].copy()
        is_active[:, :n_sub] = is_act_sub[i_sub_selected, :].copy()

        # Correct design vectors of underlying problem
        n_cont = self._n_cont
        x_underlying = self._get_x_underlying(x[:, n_sub:])
        x_problem, is_act_problem = self._problem.correct_x(x_underlying)
        x[:, n_sub:] = x_problem[:, :n_cont]
        is_active[:, n_sub:] = is_act_problem[:, :n_cont]

        self._correct_output = {'i_sub_sel': i_sub_selected}

    def _get_corrected_sub_sel_idx(self, x: np.ndarray):
        x_sub, is_active_sub = self._x_sub, self._is_act_sub

        corrected_sub_sel_idx = np.zeros((x.shape[0],), dtype=int)
        for j, xi in enumerate(x):
            matched_dv_idx = np.arange(x_sub.shape[0])
            x_valid_matched, is_active_valid_matched = x_sub, is_active_sub
            for i, is_discrete in enumerate(self.is_discrete_mask):
                # Ignore continuous vars
                if not is_discrete:
                    continue

                # Match active valid x to value or inactive valid x
                is_active_valid_i = is_active_valid_matched[:, i]
                matched = (is_active_valid_i & (x_valid_matched[:, i] == xi[i])) | (~is_active_valid_i)

                # If there are no matches, match the closest value
                if not np.any(matched):
                    x_val_dist = np.abs(x_valid_matched[:, i] - xi[i])
                    matched = x_val_dist == np.min(x_val_dist)

                # Select vectors and check if there are any vectors left to choose from
                matched_dv_idx = matched_dv_idx[matched]
                x_valid_matched = x_valid_matched[matched, :]
                is_active_valid_matched = is_active_valid_matched[matched, :]

                # If there is only one matched vector left, there is no need to continue checking
                if len(matched_dv_idx) == 1:
                    break
            corrected_sub_sel_idx[j] = matched_dv_idx[0]
        return corrected_sub_sel_idx

    def _get_x_underlying(self, x_underlying):
        if self._n_cont == 0:
            return np.ones((x_underlying.shape[0], self._problem.n_var))*.5*(self._problem.xl+self._problem.xu)
        return x_underlying

    def plot_i_sub_problem(self, x: np.ndarray = None, show=True):
        import matplotlib.pyplot as plt
        if x is None:
            x = self.pareto_set()
        x, _ = self.correct_x(x)
        f = self.evaluate(x, return_as_dictionary=True)['F']
        i_sub_selected = self._correct_output['i_sub_sel']

        plt.figure()
        f0 = f[:, 0]
        f1 = f[:, 1] if f.shape[1] > 1 else f0
        for i_sp in np.unique(i_sub_selected):
            mask = i_sub_selected == i_sp
            plt.scatter(f0[mask], f1[mask], s=10, marker='o', label=f'#{i_sp+1}')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        plt.tight_layout()
        if show:
            plt.show()

    def plot_transformation(self, show=True):
        import matplotlib.pyplot as plt
        plt.figure()
        if1 = 0 if self._problem.n_obj < 2 else 1

        pf = self._problem.pareto_front()
        plt.scatter(pf[:, 0], pf[:, if1], s=10, marker='o', label='Orig')

        for i_sub in range(self._x_sub.shape[0]):
            pf_transformed = self._transform_out(pf, np.ones((pf.shape[0],), dtype=int)*i_sub)
            plt.scatter(pf_transformed[:, 0], pf_transformed[:, if1], s=10, marker='o', label=f'#{i_sub+1}')

        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        plt.tight_layout()
        if show:
            plt.show()

    def __repr__(self):
        if self._repr_str is not None:
            return self._repr_str
        return f'{self.__class__.__name__}({self._problem!r}, imp_ratio={self._imp_ratio}, ' \
               f'n_subproblem={self._n_subproblem}, n_opts={self._n_opts}, diversity_range={self._diversity_range}, ' \
               f'cont_ratio={self._cont_ratio})'


class HierBranin(TunableHierarchicalMetaProblem):

    def __init__(self):
        def factory(n):
            return MDBranin()
        super().__init__(factory, imp_ratio=5., n_subproblem=50, diversity_range=.5, n_opts=3)


class TunableZDT1(TunableHierarchicalMetaProblem):

    def __init__(self, imp_ratio=1., n_subproblem=100, diversity_range=.5, n_opts=3, cont_ratio=1.):
        def factory(n):
            return NoHierarchyWrappedProblem(ZDT1(n_var=n))
        super().__init__(factory, imp_ratio=imp_ratio, n_subproblem=n_subproblem, diversity_range=diversity_range,
                         n_opts=n_opts, cont_ratio=cont_ratio)


class HierZDT1Small(TunableZDT1):

    def __init__(self):
        super().__init__(imp_ratio=2., n_subproblem=10, diversity_range=.25, n_opts=3, cont_ratio=1)


class HierZDT1(TunableZDT1):

    def __init__(self):
        super().__init__(imp_ratio=5., n_subproblem=200, n_opts=3, cont_ratio=.5)


class HierZDT1Large(TunableZDT1):

    def __init__(self):
        super().__init__(imp_ratio=10., n_subproblem=2000, n_opts=4, cont_ratio=1.)


class HierDiscreteZDT1(TunableZDT1):

    def __init__(self):
        super().__init__(imp_ratio=5., n_subproblem=2000, n_opts=4, cont_ratio=0)


class HierCantileveredBeam(TunableHierarchicalMetaProblem):

    def __init__(self):
        def factory(n):
            return ArchCantileveredBeam()
        super().__init__(factory, imp_ratio=6., n_subproblem=20, diversity_range=.5)


class HierCarside(TunableHierarchicalMetaProblem):

    def __init__(self):
        def factory(n):
            return ArchCarside()
        super().__init__(factory, imp_ratio=7., n_subproblem=50, diversity_range=.5)


class NeuralNetwork(HierarchyProblemBase):
    """
    Multi-layer perceptron test problem from:
    Audet, C., Hallé-Hannan, E. and Le Digabel, S., 2023, March. A general mathematical framework for constrained
    mixed-variable blackbox optimization problems with meta and categorical variables. In Operations Research Forum
    (Vol. 4, No. 1, pp. 1-37). Springer International Publishing.

    Implementation based on:
    https://github.com/SMTorg/smt/blob/master/smt/problems/neural_network.py
    """

    def __init__(self):
        des_vars = [
            Integer(bounds=(1, 3)),
            Real(bounds=(-5, -2)),
            Real(bounds=(-5, -1)),
            Integer(bounds=(3, 8)),
            Choice(options=['ReLU', 'SELU', 'ISRLU']),
            Integer(bounds=(0, 5)),
            Integer(bounds=(0, 5)),
            Integer(bounds=(0, 5)),
        ]
        super().__init__(des_vars)

    def _get_n_valid_discrete(self) -> int:
        n_base = 6*3  # x3, x4
        return sum([
            n_base*6,  # x0 == 0
            n_base*6**2,  # x0 == 1
            n_base*6**3,  # x0 == 2
        ])

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        self._correct_x_impute(x, is_active_out)

        _f_factors = [
            np.array([2,  1, -.5]),
            np.array([-1, 2, -.5]),
            np.array([-1, 1,  .5]),
        ]

        def f(x1, x2, x3, x4, x5, x6=None, x7=None):
            f_value = np.sum(_f_factors[int(x4)]*np.array([x1, x2, 2**x3])) + x5**2
            if x6 is not None:
                f_value += .3*x6
            if x7 is not None:
                f_value -= .1*x7**3
            return f_value

        for i, xi in enumerate(x):
            f_out[i, 0] = f(xi[1], xi[2], xi[3], xi[4], xi[5],
                            x6=xi[6] if xi[0] in [2, 3] else None,
                            x7=xi[7] if xi[0] == 3 else None)

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        is_active[:, [6, 7]] = False
        is_active[x[:, 0] == 2, 6] = True
        is_active[x[:, 0] == 3, 6] = True
        is_active[x[:, 0] == 3, 7] = True

    def plot_doe(self, n=1000):
        """Compare to: https://smt.readthedocs.io/en/latest/_images/neuralnetwork_Test_test_hier_neural_network.png"""
        import matplotlib.pyplot as plt

        x = HierarchicalSampling().do(self, n).get('X')
        f = self.evaluate(x, return_as_dictionary=True)['F']
        plt.figure()
        plt.scatter(x[:, 0], f[:, 0], s=5)
        plt.xlabel('x0'), plt.ylabel('y')
        plt.show()


if __name__ == '__main__':
    # HierarchicalGoldstein().print_stats()
    # MOHierarchicalGoldstein().print_stats()
    # # HierarchicalGoldstein().plot_pf()
    # MOHierarchicalGoldstein().plot_pf()

    # HierarchicalRosenbrock().print_stats()
    # MOHierarchicalRosenbrock().print_stats()
    # # HierarchicalRosenbrock().plot_pf()
    # MOHierarchicalRosenbrock().plot_pf()

    ZaeffererHierarchical.from_mode(ZaeffererProblemMode.A_OPT_INACT_IMP_PROF_UNI).print_stats()
    # ZaeffererHierarchical.from_mode(ZaeffererProblemMode.A_OPT_INACT_IMP_PROF_UNI).plot_pf()
    # ZaeffererHierarchical.from_mode(ZaeffererProblemMode.A_OPT_INACT_IMP_PROF_UNI).plot_design_space()

    # MOHierarchicalTestProblem().print_stats()
    # MOHierarchicalTestProblem().plot_pf()

    # Jenatton().print_stats()
    # # Jenatton().plot_pf()

    # p = HierBranin()
    # p = HierZDT1Small()
    # p = HierZDT1()
    # p = HierZDT1Large()
    # p = HierDiscreteZDT1()
    # p = HierCantileveredBeam()
    # p = HierCarside()
    # p.print_stats()
    # p.reset_pf_cache()
    # p.plot_pf()
    # p.plot_transformation(show=False)
    # p.plot_i_sub_problem()

    NeuralNetwork().print_stats()
    # NeuralNetwork().plot_pf()
