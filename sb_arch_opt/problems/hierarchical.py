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
import string
import itertools
import numpy as np
from typing import *
from deprecated import deprecated
from sb_arch_opt.problems.md_mo import *
from pymoo.problems.multi.zdt import ZDT1
from sb_arch_opt.problems.discrete import *
from sb_arch_opt.problems.problems_base import *
from pymoo.core.variable import Real, Integer, Choice
from pymoo.util.ref_dirs import get_reference_directions

__all__ = ['HierarchyProblemBase', 'HierarchicalGoldstein', 'HierarchicalRosenbrock', 'ZaeffererHierarchical',
           'ZaeffererProblemMode', 'MOHierarchicalGoldstein', 'MOHierarchicalRosenbrock', 'HierarchicalMetaProblemBase',
           'MOHierarchicalTestProblem', 'Jenatton', 'CombinatorialHierarchicalMetaProblem', 'CombHierBranin',
           'CombHierRosenbrock']


class HierarchyProblemBase(ArchOptTestProblemBase):
    """Base class for test problems that have decision hierarchy"""

    def get_n_valid_discrete(self) -> int:
        raise NotImplementedError

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

    def get_n_valid_discrete(self) -> int:
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
        from sb_arch_opt.sampling import RepairedLatinHypercubeSampling

        problem = cls()
        x = RepairedLatinHypercubeSampling().do(problem, n_samples).get('X')

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

    def get_n_valid_discrete(self) -> int:
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
        from sb_arch_opt.sampling import RepairedLatinHypercubeSampling

        problem = cls()
        x = RepairedLatinHypercubeSampling().do(problem, n_samples).get('X')

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

    def get_n_valid_discrete(self) -> int:
        return 1

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

    def get_n_valid_discrete(self) -> int:
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

    def __init__(self):
        des_vars = [
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
        super().__init__(des_vars)

    def get_n_valid_discrete(self) -> int:
        return 4

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        for i, xi in enumerate(x):
            if xi[0] == 0:
                if xi[1] == 0:
                    f_out[i, 0] = xi[3]**2 + .1 + xi[7]  # x4^2 + .1 + r8
                else:
                    f_out[i, 0] = xi[4]**2 + .1 + xi[7]  # x5^2 + .1 + r8
            else:
                if xi[2] == 0:
                    f_out[i, 0] = xi[5]**2 + .1 + xi[8]  # x6^2 + .1 + r9
                else:
                    f_out[i, 0] = xi[6]**2 + .1 + xi[8]  # x7^2 + .1 + r9

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        for i in [2, 5, 6, 8]:  # x1 = 0: x3, x6, x7, r9 inactive
            is_active[x[:, 0] == 0, i] = False
        is_active[(x[:, 0] == 0) & (x[:, 1] == 0), 4] = False  # x2 = 0: x5 inactive
        is_active[(x[:, 0] == 0) & (x[:, 1] == 1), 3] = False  # x2 = 1: x4 inactive

        for i in [1, 4, 5, 7]:  # x2, x4, x5, r8 inactive
            is_active[x[:, 0] == 1, i] = False
        is_active[(x[:, 0] == 1) & (x[:, 2] == 0), 6] = False  # x3 = 0: x7 inactive
        is_active[(x[:, 0] == 1) & (x[:, 2] == 1), 5] = False  # x3 = 1: x6 inactive


class CombinatorialHierarchicalMetaProblem(HierarchyProblemBase):
    """
    Meta problem that turns any (mixed-discrete, multi-objective) problem into a realistically-behaving hierarchical
    optimization problem:
    - The problem is separated into n_parts per dimension:
      - Continuous variables are simply separated linearly
      - Discrete variables are separated such that there are at least 2 options in each part
    - Categorical selection variables are added to select subparts to apply divided variables to
      - Parts are selected ordinally (i.e. not per dimension, but from "one big list"); orders are randomized
      - For each subpart one or more original dimensions are deactivated
      - Selection variables are added for selecting which subpart to evaluate, but repeated separation into groups

    Note: the underlying problem should not be hierarchical!

    Settings:
    - n_parts: number of parts to separate each dimension; increases the nr of possible discrete design points
    - n_sel_dv: number of design variables to use for selecting which part to evaluate; higher values increase the
                imputation ratio, reduce the nr of options for each selection design variable
    - sep_power: controls how non-uniform the selection subdivisions are: higher than 1 increases imputation ratio and
                 difference between occurrence rates
    - target_n_opts_ratio: controls the nr of options of the last separation variable: higher reduces the imp ratio
    """

    def __init__(self, problem: ArchOptTestProblemBase, n_parts=2, n_sel_dv=4, sep_power=1.2, target_n_opts_ratio=5.):
        self._problem = problem
        self._n_parts = n_parts

        # Separate the underlying design space in different parts
        parts = []
        is_cont_mask, xl, xu = problem.is_cont_mask, problem.xl, problem.xu
        div_bounds = 1/n_parts
        n_opts_max = np.zeros((problem.n_var,), dtype=int)
        for i_div in itertools.product(*[range(n_parts) for _ in range(problem.n_var)]):
            part = []
            for i_dv, i in enumerate(i_div):
                if is_cont_mask[i_dv]:
                    # Map continuous variable to a subrange
                    xl_i, xu_i = xl[i_dv], xu[i_dv]
                    bounds = tuple(np.array([i, i+1])*div_bounds*(xu_i-xl_i)+xl_i)
                    part.append((False, bounds))

                else:
                    # Map discrete variable to a subrange that exists of at least 2 options
                    n_opts = int(xu[i_dv]+1)
                    n_opt_per_div = max(2, np.ceil(n_opts / n_parts))
                    i_opts = np.arange(n_opt_per_div*i, n_opt_per_div*(i+1))

                    # Ensure that no options outside the bounds can be selected
                    i_opts = tuple(i_opts[i_opts < n_opts])
                    if len(i_opts) == 0:
                        # If too far outside the bounds, retain the last option only
                        i_opts = (n_opts-1,)

                    # Track the maximum nr of options
                    if len(i_opts) > n_opts_max[i_dv]:
                        n_opts_max[i_dv] = len(i_opts)

                    part.append((True, i_opts))
            parts.append(part)

        # Shuffle the parts
        n_sel_dv = max(2, n_sel_dv)
        rng = np.random.default_rng(problem.n_var * problem.n_obj * n_parts * n_sel_dv)
        self._parts = parts = [parts[i] for i in rng.permutation(np.arange(len(parts)))]

        # Define which mapped design variables are active for each part
        self._parts_is_active = parts_is_active = np.ones((len(parts), problem.n_var), dtype=bool)
        osc_period = max(5, int(len(parts)/5))
        idx_cont = np.where(is_cont_mask)[0]
        n_inactive = np.floor(len(idx_cont)*(.5-.5*np.cos(osc_period*np.arange(len(parts))/np.pi))).astype(int)
        for i, part in enumerate(parts):
            # Discrete variables with one option only are inactive
            for i_dv, (is_discrete, settings) in enumerate(part):
                if is_discrete and len(settings) < 2:
                    parts_is_active[i, i_dv] = False

            # Deactivate continuous variables based on an oscillating equation
            cont_inactive_idx = idx_cont[len(idx_cont)-n_inactive[i]:]
            parts_is_active[i, cont_inactive_idx] = False

        # Define selection design variables
        # Repeatedly separate the number of parts to be selected
        init_range = .65  # Directly controls the range (= difference between min and max occurrence rates) of x0

        def _sep_group(n_values, n_sep):
            return np.round((np.linspace(0, 1, n_values)**sep_power)*(n_sep-.01)-.5).astype(int)

        n = len(parts)

        # We define one design variable that makes the initial separation
        init_sep_frac = .5+.5*init_range-.05+.1*rng.random()

        # Determine in how many groups we should separate at each step
        # Repeatedly subdividing the largest group determines how many values the last largest group has. We calculate
        # for several separation numbers what the fraction of the largest group is; then calculate how big the latest
        # largest group is. If this number is equal to the corresponding nr of separations, it means that each design
        # variable will have the same nr of options. We choose the nr of separations where the ratio is a bit higher
        # than 1 to introduce another source of non-uniformity in the problem formulation
        n_sep_possible = np.arange(2, max(2, np.floor(.25*n))+1, dtype=int)
        frac_largest_group = np.array([np.sum(_sep_group(100, n_sep) == 0)/100 for n_sep in n_sep_possible])
        n_rel_last_group = (init_sep_frac*frac_largest_group**(n_sel_dv-2))*n/n_sep_possible
        n_rel_lg_idx = np.where(n_rel_last_group > target_n_opts_ratio)[0]
        n_sep_per_dv = n_sep_possible[n_rel_lg_idx[-1]] if len(n_rel_lg_idx) > 0 else n_sep_possible[0]

        x_sel = np.zeros((n, n_sel_dv), dtype=int)
        is_active_sel = np.ones((n, n_sel_dv), dtype=bool)
        dv_groups = [np.arange(n, dtype=int)]
        for i in range(n_sel_dv):
            # Separate current selection groups
            next_dv_groups = []
            needed_separation = False
            for group_idx in dv_groups:
                if len(group_idx) == 1:
                    is_active_sel[group_idx[0], i:] = False
                    continue
                needed_separation = True

                # For the last group just add uniformly increasing values to avoid needing additional groups
                if i == n_sel_dv-1:
                    x_sel[group_idx, i] = np.arange(len(group_idx))
                    continue

                # For the first group separate based on the initial separation fraction
                if i == 0:
                    x_next_group = np.ones((len(group_idx),), dtype=int)
                    n_first_group = int(np.ceil(init_sep_frac*len(x_next_group)))
                    x_next_group[:n_first_group] = 0
                else:
                    # Distribute values unevenly (raising the power results in a more uneven distribution)
                    x_next_group = _sep_group(len(group_idx), n_sep_per_dv)

                x_sel[group_idx, i] = x_next_group

                # Determine next groups
                for x_next in np.unique(x_next_group):
                    next_dv_groups.append(group_idx[x_next_group == x_next])

            dv_groups = next_dv_groups
            if not needed_separation:
                x_sel = x_sel[:, :i]
                is_active_sel = is_active_sel[:, :i]
                break

        self._x_sel = x_sel
        self._is_active_sel = is_active_sel
        des_vars = []
        for i in range(x_sel.shape[1]):
            des_vars.append(Choice(options=list(sorted(np.unique(x_sel[:, i])))))

        # Add mapped design variables
        for i_dv, des_var in enumerate(problem.des_vars):
            if isinstance(des_var, Real):
                des_vars.append(Real(bounds=des_var.bounds))
            elif isinstance(des_var, Integer):
                des_vars.append(Integer(bounds=(0, n_opts_max[i_dv]-1)))
            elif isinstance(des_var, Choice):
                des_vars.append(Choice(options=list(range(n_opts_max[i_dv]))))
            else:
                raise RuntimeError(f'Design variable type not supported: {des_var!r}')

        super().__init__(des_vars, n_obj=problem.n_obj, n_ieq_constr=problem.n_ieq_constr,
                         n_eq_constr=problem.n_eq_constr)

        self.__correct_output = {}

    def get_n_valid_discrete(self) -> int:
        # Get nr of combinations for each part
        parts_is_active = self._parts_is_active
        n_part_combs = np.ones(parts_is_active.shape, dtype=int)
        for i, part in enumerate(self._parts):
            for i_dv, (is_discrete, settings) in enumerate(part):
                if not parts_is_active[i, i_dv] or not is_discrete:
                    continue
                n_part_combs[i, i_dv] = len(settings)

        # Sum the nr of combinations for the parts
        return int(np.sum(np.prod(n_part_combs, axis=1)))

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        # Correct and impute
        self._correct_x_impute(x, is_active_out)
        i_part_selected = self.__correct_output['i_part_sel']

        parts = self._parts
        parts_is_active = self._parts_is_active
        n_dv_map = self._x_sel.shape[1]
        xl, xu = self._problem.xl, self._problem.xu

        # Map design variables to underlying problem
        x_underlying = x[:, n_dv_map:].copy()
        for i, i_part in enumerate(i_part_selected):
            is_active_i = parts_is_active[i_part, :]
            x_part = x_underlying[i, :]
            for i_dv, (is_discrete, settings) in enumerate(parts[i_part]):
                if is_discrete:
                    if is_active_i[i_dv]:
                        i_x_mapped = int(x_part[i_dv])
                        x_part[i_dv] = settings[i_x_mapped] if i_x_mapped < len(settings) else settings[-1]
                    else:
                        x_part[i_dv] = 0
                else:
                    bnd = settings
                    if is_active_i[i_dv]:
                        x_part[i_dv] = bnd[0]+(bnd[1]-bnd[0])*((x_part[i_dv]-xl[i_dv])/(xu[i_dv]-xl[i_dv]))
                    else:
                        x_part[i_dv] = .5*np.sum(bnd)

        # Evaluate underlying problem
        out = self._problem.evaluate(x_underlying, return_as_dictionary=True)
        if np.any(out['is_active'] == 0):
            raise RuntimeError('Underlying problem should not be hierarchical!')

        f_out[:, :] = out['F']
        if 'G' in out:
            g_out[:, :] = out['G']
        if 'H' in out:
            h_out[:, :] = out['H']

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        # Match to selection design vector
        x_sel = self._x_sel
        is_act_sel = self._is_active_sel
        n_dv_sel = x_sel.shape[1]
        i_part_selected = np.zeros((x.shape[0],), dtype=int)
        for i, xi in enumerate(x):

            # Recursively select design vectors matching ours
            idx_match = np.arange(x_sel.shape[0], dtype=int)
            for i_sel in range(n_dv_sel):
                idx_match_i = idx_match[x_sel[idx_match, i_sel] == xi[i_sel]]

                # If none found, we impute
                if len(idx_match_i) == 0:
                    xi[i_sel] = imp_dv_value = x_sel[idx_match[-1], i_sel]
                    idx_match_i = idx_match[x_sel[idx_match, i_sel] == imp_dv_value]

                # If one found, we have a match!
                if len(idx_match_i) == 1:
                    i_part = idx_match_i[0]
                    i_part_selected[i] = i_part
                    xi[:n_dv_sel] = x_sel[i_part, :]
                    is_active[i, :n_dv_sel] = is_act_sel[i_part, :]
                    break

                # Otherwise, we continue searching
                idx_match = idx_match_i
            else:
                raise RuntimeError(f'Could not match design vectors: {xi[:n_dv_sel]}')

        # Correct DVs of underlying problem and set activeness
        n_dv_map = x_sel.shape[1]
        part_is_active = self._parts_is_active
        for i, i_part in enumerate(i_part_selected):
            is_active[i, n_dv_map:] = part_is_active[i_part, :]

        self.__correct_output = {'i_part_sel': i_part_selected}


class CombHierBranin(CombinatorialHierarchicalMetaProblem):

    def __init__(self):
        super().__init__(MDBranin(), n_parts=4, n_sel_dv=4, sep_power=1.2, target_n_opts_ratio=5.)


class CombHierRosenbrock(CombinatorialHierarchicalMetaProblem):

    def __init__(self):
        problem = MixedDiscretizerProblemBase(MORosenbrock(n_var=6), n_opts=3, n_vars_int=2)
        super().__init__(problem, n_parts=3, n_sel_dv=5, sep_power=1.1, target_n_opts_ratio=1.)


if __name__ == '__main__':
    # HierarchicalGoldstein().print_stats()
    # MOHierarchicalGoldstein().print_stats()
    # # HierarchicalGoldstein().plot_pf()
    # MOHierarchicalGoldstein().plot_pf()

    # HierarchicalRosenbrock().print_stats()
    # MOHierarchicalRosenbrock().print_stats()
    # # HierarchicalRosenbrock().plot_pf()
    # MOHierarchicalRosenbrock().plot_pf()

    # ZaeffererHierarchical.from_mode(ZaeffererProblemMode.A_OPT_INACT_IMP_PROF_UNI).print_stats()
    # ZaeffererHierarchical.from_mode(ZaeffererProblemMode.A_OPT_INACT_IMP_PROF_UNI).plot_pf()
    # ZaeffererHierarchical.from_mode(ZaeffererProblemMode.A_OPT_INACT_IMP_PROF_UNI).plot_design_space()

    # MOHierarchicalTestProblem().print_stats()
    # MOHierarchicalTestProblem().plot_pf()

    # Jenatton().print_stats()
    # # Jenatton().plot_pf()

    CombHierBranin().print_stats()
    # CombHierBranin().plot_pf()
    CombHierRosenbrock().print_stats()
    # CombHierRosenbrock().plot_pf()
