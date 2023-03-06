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
import enum
import numpy as np
from typing import *
from sb_arch_opt.pareto_front import *
from sb_arch_opt.problems.discrete import *
from pymoo.core.variable import Real, Integer, Choice
from pymoo.util.ref_dirs import get_reference_directions

__all__ = ['HierarchyProblemBase', 'HierarchicalGoldstein', 'HierarchicalRosenbrock', 'ZaeffererHierarchical',
           'ZaeffererProblemMode', 'MOHierarchicalGoldstein', 'MOHierarchicalRosenbrock', 'HierarchicalMetaProblemBase',
           'MOHierarchicalTestProblem']


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


class HierarchicalMetaProblemBase(HierarchyProblemBase):
    """
    Meta problem used for increasing the amount of design variables of an underlying mixed-integer/hierarchical problem.
    The idea is that design variables are repeated, and extra design variables are added for switching between the
    repeated design variables. Objectives are then slightly modified based on the switching variable.

    For correct modification of the objectives, a range of the to-be-expected objective function values at the Pareto
    front for each objective dimension should be provided (f_par_range).

    Note that each map will correspond to a new part of the Pareto front.
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


class MOHierarchicalTestProblem(HierarchicalMetaProblemBase):
    """
    Multi-objective hierarchical test problem based on the hierarchical rosenbrock problem. Increased number of design
    variables and increased sparseness (approx. 42% of design variables are active in a DOE).
    """

    def __init__(self):
        super().__init__(MOHierarchicalRosenbrock(), n_rep=2, n_maps=2, f_par_range=[100, 100])

    def __repr__(self):
        return f'{self.__class__.__name__}()'


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

    MOHierarchicalTestProblem().print_stats()
    # MOHierarchicalTestProblem().plot_pf()
