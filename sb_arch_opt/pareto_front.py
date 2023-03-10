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
import os
import re
import pickle
import hashlib
import numpy as np
from typing import *
import concurrent.futures
import matplotlib.pyplot as plt
from assign_pymoo.sampling import *

from pymoo.optimize import minimize
from pymoo.core.variable import Real
from pymoo.core.problem import Problem
from pymoo.core.evaluator import Evaluator
from pymoo.visualization.scatter import Scatter
from pymoo.core.initialization import Initialization
from pymoo.algorithms.moo.nsga2 import calc_crowding_distance
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.termination.default import DefaultMultiObjectiveTermination, DefaultSingleObjectiveTermination

from sb_arch_opt.util import patch_ftol_bug
from sb_arch_opt.problem import ArchOptProblemBase
from sb_arch_opt.sampling import RepairedExhaustiveSampling, RepairedRandomSampling

__all__ = ['CachedParetoFrontMixin', 'ArchOptTestProblemBase']


class CachedParetoFrontMixin(Problem):
    """Mixin to calculate the Pareto front once by simply running the problem several times using NSGA2, meant for test
    problems. Stores the results based on the repr of the main class, so make sure that one is set."""

    default_enable_pf_calc = True

    def reset_pf_cache(self):
        cache_path = self._pf_cache_path()
        if os.path.exists(cache_path):
            os.remove(cache_path)

    def calc_pareto_front(self, **kwargs):
        return self._calc_pareto_front(force=True, **kwargs)

    def _calc_pareto_front(self, *_, pop_size=200, n_gen_min=10, n_repeat=12, n_pts_keep=100, force=False, **kwargs):
        if not force and not self.default_enable_pf_calc:
            raise RuntimeError('On-demand PF calc is disabled, use calc_pareto_front instead')

        # Check if Pareto front has already been cached
        cache_path = self._pf_cache_path()
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as fp:
                return pickle.load(fp)

        # Get an approximation of the combinatorial design space size, only relevant if there are no continuous vars
        n = 1
        xl, xu = self.bounds()
        for i, var in enumerate(self.vars):
            if isinstance(var, Real):
                n = None
                break
            n *= int(xu[i]-xl[i]+1)

        # If the design space is smaller than the number of requested evaluations, simply evaluate all points
        if n is not None and n < pop_size*n_gen_min*n_repeat:
            pop = RepairedExhaustiveSampling().do(self, n)
            Evaluator().eval(self, pop)

            pf = pop.get('F')
            i_non_dom = NonDominatedSorting().do(pf, only_non_dominated_front=True)
            pf = pf[i_non_dom, :]

        # Otherwise, execute NSGA2 in parallel and merge resulting Pareto fronts
        else:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(self._run_minimize, pop_size, n_gen_min, i, n_repeat)
                           for i in range(n_repeat)]
                concurrent.futures.wait(futures)

                pf = None
                for i in range(n_repeat):
                    res = futures[i].result()
                    if res.F is None:
                        continue
                    if pf is None:
                        pf = res.F
                    else:
                        pf_merged = np.row_stack([pf, res.F])
                        i_non_dom = NonDominatedSorting().do(pf_merged, only_non_dominated_front=True)
                        pf = pf_merged[i_non_dom, :]

        # Reduce size of Pareto front to a predetermined amount to ease Pareto-front-related calculations
        if pf is None or pf.shape[0] == 0:
            raise RuntimeError('Could not find Pareto front')
        pf = np.unique(pf, axis=0)
        if n_pts_keep is not None and pf.shape[0] > n_pts_keep:
            for _ in range(pf.shape[0]-n_pts_keep):
                crowding_of_front = calc_crowding_distance(pf)
                i_max_crowding = np.argsort(crowding_of_front)[1:]
                pf = pf[i_max_crowding, :]

        # Store in cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as fp:
            pickle.dump(pf, fp)
        return pf

    def _run_minimize(self, pop_size, n_gen, i, n):
        from sb_arch_opt.algo.pymoo_interface import get_nsga2
        print(f'Running Pareto front discovery {i+1}/{n} ({pop_size} pop, {n_gen} gen): {self.name()}')

        robust_period = n_gen
        n_max_gen = n_gen*10
        n_max_eval = n_max_gen*pop_size
        if self.n_obj > 1:
            termination = DefaultMultiObjectiveTermination(
                xtol=5e-4, cvtol=1e-8, ftol=5e-3, n_skip=5, period=robust_period, n_max_gen=n_max_gen,
                n_max_evals=n_max_eval)
        else:
            termination = DefaultSingleObjectiveTermination(
                xtol=1e-8, cvtol=1e-8, ftol=1e-6, period=robust_period, n_max_gen=n_max_gen, n_max_evals=n_max_eval)

        patch_ftol_bug(termination)
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

            pop = Initialization(RepairedRandomSampling()).do(self, n_sample)
            Evaluator().eval(self, pop)
            scatter.add(pop.get('F'), s=5)

        scatter.add(pf)
        if filename is not None:
            scatter.save(filename)
        if show:
            scatter.show()
        plt.close(scatter.fig)

    def get_approx_f_range(self, n_sample=100):
        pop = Initialization(RepairedRandomSampling()).do(self, n_sample)
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

        return os.path.expanduser(os.path.join('~', '.arch_opt_pf_cache', class_str+'.pkl'))


class ArchOptTestProblemBase(CachedParetoFrontMixin, ArchOptProblemBase):
    """Helper class to extend the ArchOptProblemBase with Pareto front caching"""
    pass
