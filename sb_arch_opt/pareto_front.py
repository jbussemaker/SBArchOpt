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
import os
import re
import pickle
import hashlib
import numpy as np
from typing import *
import concurrent.futures
import matplotlib.pyplot as plt

from pymoo.optimize import minimize
from pymoo.core.variable import Real
from pymoo.core.problem import Problem
from pymoo.core.evaluator import Evaluator
from pymoo.visualization.scatter import Scatter
from pymoo.core.initialization import Initialization
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.termination.default import DefaultMultiObjectiveTermination, DefaultSingleObjectiveTermination
from sb_arch_opt.sampling import HierarchicalExhaustiveSampling, HierarchicalSampling

try:
    # pymoo < 0.6.1
    from pymoo.algorithms.moo.nsga2 import calc_crowding_distance
except ImportError:
    # pymoo >= 0.6.1
    from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance

__all__ = ['CachedParetoFrontMixin']


class CachedParetoFrontMixin(Problem):
    """Mixin to calculate the Pareto front once by simply running the problem several times using NSGA2, meant for test
    problems. Stores the results based on the repr of the main class, so make sure that one is set."""

    default_enable_pf_calc = True

    def reset_pf_cache(self):
        cache_path = self._pf_cache_path()
        if os.path.exists(cache_path):
            os.remove(cache_path)

        # pymoo's implementation of function result caching
        if 'cache' in self.__dict__:
            for key in ['pareto_front', 'pareto_set']:
                if key in self.__dict__['cache']:
                    del self.__dict__['cache'][key]

    def calc_pareto_front(self, **kwargs):
        return self._calc_pareto_front(force=True, **kwargs)

    def _calc_pareto_front(self, *_, **kwargs):
        _, pf = self._calc_pareto_set_front(**kwargs)
        return pf

    def calc_pareto_set(self, *_, **kwargs):
        return self._calc_pareto_set(force=True, **kwargs)

    def _calc_pareto_set(self, *_, **kwargs):
        ps, _ = self._calc_pareto_set_front(**kwargs)
        return ps

    def _calc_pareto_set_front(self, *_, pop_size=None, n_gen_min=10, n_repeat=4, n_pts_keep=100, force=False, **__):
        if not force and not self.default_enable_pf_calc:
            raise RuntimeError('On-demand PF calc is disabled, use calc_pareto_front instead')

        # Check if Pareto front has already been cached
        cache_path = self._pf_cache_path()
        if not force and os.path.exists(cache_path):
            with open(cache_path, 'rb') as fp:
                ps, pf = pickle.load(fp)

            # Sort by first objective dimension to ensure Pareto front and set points match
            # (because pymoo sorts the Pareto front but not the Pareto set)
            i_sorted = np.argsort(pf[:, 0])
            ps = ps[i_sorted, :]
            pf = pf[i_sorted, :]

            return ps, pf

        # Get population size
        if pop_size is None:
            pop_size = 10*self.n_var

        # Get an approximation of the combinatorial design space size, only relevant if there are no continuous vars
        n = 1
        xl, xu = self.bounds()
        for i, var in enumerate(self.vars.values()):
            if isinstance(var, Real):
                n = None
                break
            n *= int(xu[i]-xl[i]+1)

        # If the design space is smaller than the number of requested evaluations, simply evaluate all points
        if n is not None and n < pop_size*n_gen_min*n_repeat:
            pop = HierarchicalExhaustiveSampling().do(self, n)
            Evaluator().eval(self, pop)

            ps = pop.get('X')
            pf = pop.get('F')
            i_non_dom = NonDominatedSorting().do(pf, only_non_dominated_front=True)
            ps = ps[i_non_dom, :]
            pf = pf[i_non_dom, :]

        # Otherwise, execute NSGA2 in parallel and merge resulting Pareto fronts
        else:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(self._run_minimize, pop_size, n_gen_min, i, n_repeat)
                           for i in range(n_repeat)]
                concurrent.futures.wait(futures)

                ps = pf = None
                for i in range(n_repeat):
                    res = futures[i].result()
                    if res.F is None:
                        continue
                    if pf is None:
                        ps = res.X
                        pf = res.F
                    else:
                        pf_merged = np.row_stack([pf, res.F])
                        i_non_dom = NonDominatedSorting().do(pf_merged, only_non_dominated_front=True)
                        ps = np.row_stack([ps, res.X])[i_non_dom, :]
                        pf = pf_merged[i_non_dom, :]

        # Reduce size of Pareto front to a predetermined amount to ease Pareto-front-related calculations
        if pf is None or pf.shape[0] == 0:
            raise RuntimeError('Could not find Pareto front')
        pf, i_unique = np.unique(pf, axis=0, return_index=True)
        ps = ps[i_unique, :]
        if n_pts_keep is not None and pf.shape[0] > n_pts_keep:
            for _ in range(pf.shape[0]-n_pts_keep):
                crowding_of_front = calc_crowding_distance(pf)
                i_max_crowding = np.argsort(crowding_of_front)[1:]
                ps = ps[i_max_crowding, :]
                pf = pf[i_max_crowding, :]

        # Sort by first objective dimension to ensure Pareto front and set points match
        # (because pymoo sorts the Pareto front but not the Pareto set)
        i_sorted = np.argsort(pf[:, 0])
        ps = ps[i_sorted, :]
        pf = pf[i_sorted, :]

        # Store in cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as fp:
            pickle.dump((ps, pf), fp)
        return ps, pf

    def _run_minimize(self, pop_size, n_gen, i, n):
        from sb_arch_opt.algo.pymoo_interface import get_nsga2

        robust_period = n_gen
        n_max_gen = n_gen*10
        n_max_eval = n_max_gen*pop_size
        print(f'Discovering Pareto front {i+1}/{n} ({pop_size} pop, {n_gen} <= gen <= {n_max_gen}): {self!r}')
        if self.n_obj > 1:
            termination = DefaultMultiObjectiveTermination(
                xtol=5e-4, cvtol=1e-8, ftol=1e-4, n_skip=n_gen, period=robust_period, n_max_gen=n_max_gen,
                n_max_evals=n_max_eval)
        else:
            termination = DefaultSingleObjectiveTermination(
                xtol=1e-8, cvtol=1e-8, ftol=1e-6, period=robust_period, n_max_gen=n_max_gen, n_max_evals=n_max_eval)

        result = minimize(self, get_nsga2(pop_size=pop_size), termination=termination, copy_termination=False)
        result.history = None
        result.algorithm = None
        return result

    def plot_pf(self: Union[Problem, 'CachedParetoFrontMixin'], show_approx_f_range=True, n_sample=100,
                filename=None, show=True, **kwargs):
        """Plot the Pareto front, optionally including randomly sampled points from the design space"""
        pf = self.pareto_front(**kwargs)
        scatter = Scatter(close_on_destroy=False)
        if show_approx_f_range:
            scatter.add(self.get_approx_f_range(), s=.1, color='white')

            pop = Initialization(HierarchicalSampling()).do(self, n_sample)
            Evaluator().eval(self, pop)
            scatter.add(pop.get('F'), s=5)

        scatter.add(pf)
        if filename is not None:
            scatter.save(filename)
        if show:
            scatter.show()
        plt.close(scatter.fig)

    def get_approx_f_range(self, n_sample=100):
        pop = Initialization(HierarchicalSampling()).do(self, n_sample)
        Evaluator().eval(self, pop)
        f = pop.get('F')
        f_max = np.max(f, axis=0)
        f_min = np.min(f, axis=0)
        return np.array([f_min, f_max])

    def _pf_cache_path(self):
        class_str = repr(self)
        if class_str.startswith('<'):
            class_str = self.__class__.__name__
        class_str = re.sub('[^0-9a-z]', '_', class_str.lower().strip())

        if len(class_str) > 20:
            class_str = hashlib.md5(class_str.encode('utf-8')).hexdigest()[:20]

        return os.path.expanduser(os.path.join('~', '.arch_opt_pf_cache', '2_'+class_str+'.pkl'))
