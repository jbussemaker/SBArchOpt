"""
MIT License

Copyright: (c) 2023, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import itertools
import numpy as np
from typing import Optional, Tuple
from pymoo.core.problem import Problem
from pymoo.core.variable import Real, Integer, Choice
from sb_arch_opt.problem import ArchOptProblemBase
from sb_arch_opt.pareto_front import CachedParetoFrontMixin
from sb_arch_opt.sampling import HierarchicalExhaustiveSampling

__all__ = ['ArchOptTestProblemBase', 'NoHierarchyProblemBase', 'NoHierarchyWrappedProblem', 'MixedDiscretizerProblemBase']


class ArchOptTestProblemBase(CachedParetoFrontMixin, ArchOptProblemBase):
    """Helper class to extend the ArchOptProblemBase with Pareto front caching"""

    def might_have_hidden_constraints(self):
        """For the test problems we know which ones have hidden constraints"""
        return False

    def plot_design_space(self, ix_plot=None, x_base=None, n=200, show=True):
        import matplotlib.pyplot as plt
        from matplotlib.colors import CenteredNorm
        if ix_plot is None:
            ix_plot = (0, 1)
        ix, iy = ix_plot
        x_name, y_name = f'$x_{ix}$', f'$x_{iy}$'

        x_lim, y_lim = (self.xl[ix], self.xu[ix]), (self.xl[iy], self.xu[iy])
        x, y = np.linspace(x_lim[0], x_lim[1], n), np.linspace(y_lim[0], y_lim[1], n)
        xx, yy = np.meshgrid(x, y)

        if x_base is None:
            x_base = .5*(self.xl+self.xu)
        x_eval = np.zeros((xx.size, len(x_base)))
        x_eval[:, :] = x_base
        x_eval[:, ix] = xx.ravel()
        if self.n_var > 1:
            x_eval[:, iy] = yy.ravel()
        out = self.evaluate(x_eval, return_as_dictionary=True)

        def _plot_out(z, z_name, is_constraint=False):
            zz = z.reshape(xx.shape)
            plt.figure(), plt.title(f'{self!r}\nmin = {np.nanmin(z)}')

            plt.fill_between(x_lim, y_lim[0], y_lim[1], facecolor='none', hatch='X', edgecolor='r', linewidth=0)

            cmap = 'RdBu_r' if is_constraint else 'summer'
            kwargs = {}
            if is_constraint:
                kwargs['norm'] = CenteredNorm()
            c = plt.contourf(xx, yy, zz, 50, cmap=cmap, **kwargs)
            plt.contour(xx, yy, zz, 10, colors='k', linewidths=.5, **kwargs)
            if is_constraint:
                plt.contour(xx, yy, zz, [0], colors='k', linewidths=3)
            plt.colorbar(c).set_label(z_name)
            plt.xlabel(x_name), plt.xlim(x_lim)
            plt.ylabel(y_name), plt.ylim(y_lim)
            plt.tight_layout()

        for if_ in range(self.n_obj):
            _plot_out(out['F'][:, if_], f'$f_{if_}$')
        for ig in range(self.n_ieq_constr):
            _plot_out(out['G'][:, ig], f'$g_{ig}$', is_constraint=True)
        for ih in range(self.n_eq_constr):
            _plot_out(out['H'][:, ih], f'$h_{ih}$', is_constraint=True)

        if show:
            plt.show()


class NoHierarchyProblemBase(ArchOptTestProblemBase):
    """Base class for test problems that have no decision hierarchy"""

    def _get_n_valid_discrete(self) -> int:
        # No hierarchy, so the number of valid points is the same as the number of declared points
        return self.get_n_declared_discrete()

    def _get_n_active_cont_mean(self) -> float:
        # No hierarchy, so the mean nr of active continuous dimensions is the same as the nr of continuous dimensions
        return float(np.sum(self.is_cont_mask))

    def _get_n_correct_discrete(self) -> int:
        # No hierarchy, so the number of correct points is the same as the number of declared points
        return self.get_n_declared_discrete()

    def _get_n_active_cont_mean_correct(self) -> float:
        # No hierarchy, so the mean nr of active continuous dimensions is the same as the nr of continuous dimensions
        return float(np.sum(self.is_cont_mask))

    def _is_conditionally_active(self):
        return [False]*self.n_var

    def _gen_all_discrete_x(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        # No hierarchy, so we can just get the Cartesian product of discrete variables
        x_values = HierarchicalExhaustiveSampling.get_exhaustive_sample_values(self, n_cont=1)

        # Set some limit to what we want to generate
        if np.prod([len(values) for values in x_values], dtype=float) > 1e6:
            return

        x_discrete = np.array(list(itertools.product(*x_values)))
        is_active = np.ones(x_discrete.shape, dtype=bool)
        return x_discrete, is_active

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class NoHierarchyWrappedProblem(NoHierarchyProblemBase):
    """Base class for non-hierarchical test problems that wrap an existing Problem class (to add SBArchOpt features)"""

    def __init__(self, problem: Problem, repr_str=None):
        self._problem = problem
        self._repr_str = repr_str
        des_vars = [Real(bounds=(problem.xl[i], problem.xu[i])) for i in range(problem.n_var)]
        super().__init__(des_vars, n_obj=problem.n_obj, n_ieq_constr=problem.n_ieq_constr)

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        out = self._problem.evaluate(x, return_as_dictionary=True)
        f_out[:, :] = out['F']
        if self.n_ieq_constr > 0:
            g_out[:, :] = out['G']

    def __repr__(self):
        if self._repr_str is not None:
            return self._repr_str
        return f'{self.__class__.__name__}()'


class MixedDiscretizerProblemBase(NoHierarchyProblemBase):
    """Problem class that turns an existing test problem into a mixed-discrete problem by mapping the first n (if not
    given: all) variables to integers with a given number of options."""

    def __init__(self, problem: Problem, n_opts=10, n_vars_int: int = None, cat=False):
        self.problem = problem
        self.n_opts = n_opts
        if n_vars_int is None:
            n_vars_int = problem.n_var
        self.n_vars_int = n_vars_int

        if not problem.has_bounds():
            raise ValueError('Underlying problem should have bounds defined!')
        self._xl_orig = problem.xl
        self._xu_orig = problem.xu

        def _get_var():
            return Choice(options=list(range(n_opts))) if cat else Integer(bounds=(0, n_opts-1))

        des_vars = [_get_var() if i < n_vars_int else Real(bounds=(problem.xl[i], problem.xu[i]))
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
