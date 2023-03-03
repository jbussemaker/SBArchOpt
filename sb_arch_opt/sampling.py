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
import itertools
import numpy as np

from pymoo.core.repair import Repair
from pymoo.core.variable import Real
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.population import Population
from pymoo.core.initialization import Initialization
from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.sampling.lhs import LatinHypercubeSampling, sampling_lhs_unit

from sb_arch_opt.problem import ArchOptRepair

__all__ = ['RepairedExhaustiveSampling', 'RepairedLatinHypercubeSampling', 'RepairedRandomSampling', 'get_init_sampler']


def get_init_sampler(repair: Repair = None, lhs=True, remove_duplicates=True, **kwargs):
    """Helper function to get an Initialization class with repair (LHS) sampling"""

    if repair is None:
        repair = ArchOptRepair()

    sampling = RepairedLatinHypercubeSampling(repair=repair, **kwargs) \
        if lhs else RepairedExhaustiveSampling(repair=repair, **kwargs)

    # Samples are already repaired because we're using the repaired samplers
    eliminate_duplicates = DefaultDuplicateElimination() if remove_duplicates else None
    return Initialization(sampling, eliminate_duplicates=eliminate_duplicates)


class RepairedExhaustiveSampling(Sampling):
    """Exhaustively samples the design space, taking n_cont samples for each continuous variable.
    Can take a long time if the design space is large."""

    def __init__(self, repair: Repair = None, n_cont=5, remove_duplicates=True):
        super().__init__()
        if repair is None:
            repair = ArchOptRepair()
        self._repair = repair
        self._n_cont = n_cont
        self._remove_duplicates = remove_duplicates

    def _do(self, problem: Problem, n_samples, **kwargs):
        # Get values to be sampled for each design variable
        opt_values = self.get_exhaustive_sample_values(problem, self._n_cont)

        # Get cartesian product of all values
        x = np.array([np.array(dv) for dv in itertools.product(*opt_values)])

        # Create and repair the population
        pop = Population.new(X=x)
        pop = self._repair.do(problem, pop)

        if self._remove_duplicates:
            pop = self.safe_remove_duplicates(pop)

        return pop.get('X')

    @staticmethod
    def get_exhaustive_sample_values(problem: Problem, n_cont=5):
        # Determine bounds and which design variables are discrete
        xl, xu = problem.bounds()
        is_cont = np.ones((len(xl),), dtype=bool)
        if problem.vars is not None:
            for i, var in enumerate(problem.vars):
                if not isinstance(var, Real):
                    is_cont[i] = False

        # Get values to be sampled for each design variable
        return [np.linspace(xl[i], xu[i], n_cont) if is_cont[i] else np.arange(xl[i], xu[i]+1) for i in range(len(xl))]

    @classmethod
    def get_n_sample_exhaustive(cls, problem: Problem, n_cont=5):
        values = cls.get_exhaustive_sample_values(problem, n_cont=n_cont)
        return int(np.prod([len(opts) for opts in values], dtype=np.float))

    @staticmethod
    def safe_remove_duplicates(pop: Population) -> Population:
        gb_needed = ((len(pop)**2)*8)/(1024**3)
        if gb_needed < 2:
            pop = DefaultDuplicateElimination().do(pop)
        return pop

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class RepairedLatinHypercubeSampling(LatinHypercubeSampling):
    """Latin hypercube sampling only returning repaired samples."""

    def __init__(self, repair: Repair = None, **kwargs):
        super().__init__(**kwargs)
        if repair is None:
            repair = ArchOptRepair()
        self._repair = repair

    def _do(self, problem: Problem, n_samples, **kwargs):
        if self._repair is None:
            return super()._do(problem, n_samples, **kwargs)

        # Get and repair initial LHS round
        xl, xu = problem.bounds()
        x = sampling_lhs_unit(n_samples, problem.n_var, smooth=self.smooth)
        x = self.repair_x(problem, x, xl, xu)

        # Subsequent rounds to improve the LHS score
        if self.criterion is not None:
            score = self.criterion(x)
            for j in range(1, self.iterations):

                _X = sampling_lhs_unit(n_samples, problem.n_var, smooth=self.smooth)
                _X = self.repair_x(problem, _X, xl, xu)
                _score = self.criterion(_X)

                if _score > score:
                    x, score = _X, _score

        # De-normalize
        return xl + x * (xu - xl)

    def repair_x(self, problem, x, xl, xu):
        # De-normalize before repairing
        x_abs = x*(xu-xl)+xl
        x_abs = self._repair.do(problem, Population.new(X=x_abs)).get("X")
        return (x_abs-xl)/(xu-xl)

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class RepairedRandomSampling(FloatRandomSampling):
    """Repaired float sampling with architecture repair"""

    _n_comb_gen_all_max = 100e3

    def __init__(self, repair: Repair = None):
        if repair is None:
            repair = ArchOptRepair()
        self._repair = repair
        super().__init__()

    def _do(self, problem, n_samples, **kwargs):
        # Get values to be sampled for each design variable
        opt_values = RepairedExhaustiveSampling.get_exhaustive_sample_values(problem, n_cont=5)

        # Get the number of samples in the cartesian product
        n_opt_values = int(np.prod([len(values) for values in opt_values], dtype=float))

        # If less than some threshold, sample all and then select (this gives a better distribution)
        if n_opt_values < self._n_comb_gen_all_max:
            try:
                # Get cartesian product of all values
                x = np.array([np.array(dv) for dv in itertools.product(*opt_values)])

                # Repair sampled population and remove duplicates
                pop = Population.new(X=x)
                pop = self._repair.do(problem, pop)
                pop = RepairedExhaustiveSampling.safe_remove_duplicates(pop)
                x = pop.get('X')

                # Randomly select values
                if n_samples < x.shape[0]:
                    i_x = np.random.choice(x.shape[0], size=n_samples, replace=False)
                    x = x[i_x, :]
                return x

            except MemoryError:
                pass

        # If above the threshold (or a memory error occurred), sample randomly
        x = np.empty((n_samples, len(opt_values)))
        for i_x in range(n_samples):
            x[i_x, :] = [np.random.choice(opt_values_i) for opt_values_i in opt_values]

        # Repair
        x = self._repair.do(problem, Population.new(X=x)).get("X")
        return x

    def __repr__(self):
        return f'{self.__class__.__name__}()'
