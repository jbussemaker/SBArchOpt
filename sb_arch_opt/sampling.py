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
from typing import Optional
from scipy.spatial import distance

from pymoo.core.repair import Repair
from pymoo.core.variable import Real
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.population import Population
from pymoo.core.initialization import Initialization
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.operators.sampling.lhs import LatinHypercubeSampling, sampling_lhs_unit

from sb_arch_opt.problem import ArchOptRepair

__all__ = ['RepairedExhaustiveSampling', 'RepairedLatinHypercubeSampling', 'RepairedRandomSampling', 'get_init_sampler',
           'LargeDuplicateElimination']


def get_init_sampler(repair: Repair = None, lhs=True, remove_duplicates=True, **kwargs):
    """Helper function to get an Initialization class with repair (LHS) sampling"""

    if repair is None:
        repair = ArchOptRepair()

    sampling = RepairedLatinHypercubeSampling(repair=repair, **kwargs) \
        if lhs else RepairedExhaustiveSampling(repair=repair, **kwargs)

    # Samples are already repaired because we're using the repaired samplers
    eliminate_duplicates = LargeDuplicateElimination() if remove_duplicates else None
    return Initialization(sampling, eliminate_duplicates=eliminate_duplicates)


class RepairedExhaustiveSampling(Sampling):
    """Exhaustively samples the design space, taking n_cont samples for each continuous variable.
    Can take a long time if the design space is large, and doesn't work well for purely continuous problems."""

    def __init__(self, repair: Repair = None, n_cont=5):
        super().__init__()
        if repair is None:
            repair = ArchOptRepair()
        self._repair = repair
        self._n_cont = n_cont

    def _do(self, problem: Problem, n_samples, **kwargs):
        # First sample only discrete dimensions
        opt_values = self.get_exhaustive_sample_values(problem, 1)
        x = np.array(list(itertools.product(*opt_values)))

        is_cont_mask = self.get_is_cont_mask(problem)
        is_discrete_mask = ~is_cont_mask

        # Create and repair the sampled design vectors in batches
        n_batch = 1000
        x_later = x
        x_repaired = x[:0, :]
        is_active_repaired = np.zeros(x_repaired.shape, dtype=bool)
        while x_later.shape[0] > 0:
            # Repair current batch
            x_repair = x_later[:n_batch, :]
            x_later = x_later[x_repair.shape[0]:, :]

            x_repair = self._repair.do(problem, x_repair)
            if isinstance(self._repair, ArchOptRepair):
                is_active = self._repair.latest_is_active
            else:
                is_active = np.ones(x_repair.shape, dtype=bool)

            # Remove duplicates
            is_dup_mask = LargeDuplicateElimination.eliminate(x_repair)
            x_repair = x_repair[~is_dup_mask, :]
            is_active = is_active[~is_dup_mask, :]

            # Remove duplicates compared to previous
            is_dup_mask = LargeDuplicateElimination.eliminate(x_repair, x_repaired)
            x_repair = x_repair[~is_dup_mask, :]
            is_active = is_active[~is_dup_mask, :]

            x_repaired = np.row_stack([x_repaired, x_repair])

            # If we also have activeness information, remove duplicate design vectors from to-be-repaired vectors
            is_active = is_active.astype(bool)
            is_active_repaired = np.row_stack([is_active_repaired, is_active])
            if x_later.shape[0] == 0:
                continue

            for i, xi in enumerate(x_repair):
                is_act_i = is_active[i, :]
                if np.all(is_act_i[is_discrete_mask]):
                    continue

                # Remove design vectors that are duplicate for the active design variables
                is_dup_mask = LargeDuplicateElimination.eliminate(x_later[:, is_act_i & is_discrete_mask],
                                                                  np.array([xi[is_act_i & is_discrete_mask]]))

                x_later = x_later[~is_dup_mask, :]
                if x_later.shape[0] == 0:
                    break

        # Expand along continuous dimensions
        x_discr = np.row_stack(x_repaired)
        is_act_discr = np.row_stack(is_active_repaired)
        n_cont = self._n_cont
        if n_cont > 1 and np.any(is_cont_mask):
            x = x_discr
            is_act = is_act_discr
            for i_dv in np.where(is_cont_mask)[0]:
                # Expand when continuous variable is active
                is_act_i = is_act[:, i_dv]
                n_repeat = np.ones(len(is_act_i), dtype=int)
                n_repeat[is_act_i] = n_cont

                x = np.repeat(x, n_repeat, axis=0)
                is_act = np.repeat(is_act, n_repeat, axis=0)

                # Fill sampled values
                rep_idx = np.cumsum([0]+list(n_repeat))[:-1]
                dv_sampled = np.linspace(problem.xl[i_dv], problem.xu[i_dv], n_cont)
                for i in np.where(is_act_i)[0]:
                    x[rep_idx[i]:rep_idx[i]+n_cont, i_dv] = dv_sampled

        else:
            x = x_discr

        return x

    @classmethod
    def get_exhaustive_sample_values(cls, problem: Problem, n_cont=5):
        # Determine bounds and which design variables are discrete
        xl, xu = problem.bounds()
        is_cont = cls.get_is_cont_mask(problem)

        # Get values to be sampled for each design variable
        return [np.linspace(xl[i], xu[i], n_cont) if is_cont[i] else np.arange(xl[i], xu[i]+1) for i in range(len(xl))]

    @staticmethod
    def get_is_cont_mask(problem: Problem):
        is_cont = np.ones((problem.n_var,), dtype=bool)
        if problem.vars is not None:
            for i, var in enumerate(problem.vars.values()):
                if not isinstance(var, Real):
                    is_cont[i] = False
        return is_cont

    @classmethod
    def get_n_sample_exhaustive(cls, problem: Problem, n_cont=5):
        values = cls.get_exhaustive_sample_values(problem, n_cont=n_cont)
        return int(np.prod([len(opts) for opts in values], dtype=np.float))

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
        x_abs = self._repair.do(problem, x_abs)
        return (x_abs-xl)/(xu-xl)

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class RepairedRandomSampling(FloatRandomSampling):
    """
    Repaired float sampling with architecture repair. There are two ways the random sampling is done:
    A: Generate and select:
       1. Generate the Cartesian product of all discrete values
       2. Repair/impute design vectors
       3. Randomly select from remaining design vectors
       4. Randomize continuous variables
       5. Repair again to impute continuous variables
    B: One-shot
       1. Randomly select design variable values
       2. Repair/impute design vectors

    The first way yields better results, as there is an even chance of selecting every valid discrete design vector,
    however it takes more memory and might be too much for very large design spaces.
    """

    _n_comb_gen_all_max = 10e3

    def __init__(self, repair: Repair = None):
        if repair is None:
            repair = ArchOptRepair()
        self._repair = repair
        super().__init__()

    def _do(self, problem, n_samples, **kwargs):
        # Get Cartesian product of all discrete design variables (only available if design space is not too large)
        x = self.get_repaired_cartesian_product(problem, self._repair)

        return self.randomly_sample(problem, n_samples, self._repair, x)

    @classmethod
    def get_repaired_cartesian_product(cls, problem: Problem, repair: Repair) -> Optional[np.ndarray]:
        # Get values to be sampled for each discrete design variable
        opt_values = RepairedExhaustiveSampling.get_exhaustive_sample_values(problem, n_cont=1)

        # Get the number of samples in the cartesian product
        n_opt_values = int(np.prod([len(values) for values in opt_values], dtype=float))

        # If less than some threshold, sample all and then select (this gives a better distribution)
        if n_opt_values < cls._n_comb_gen_all_max:
            try:
                # Get cartesian product of all values
                x = np.array([np.array(dv) for dv in itertools.product(*opt_values)])

                # Repair sampled population and remove duplicates
                x = repair.do(problem, x)
                is_duplicate = LargeDuplicateElimination.eliminate(x)
                return x[~is_duplicate, :]

            except MemoryError:
                pass

    @staticmethod
    def randomly_sample(problem, n_samples, repair: Repair, x_all: Optional[np.ndarray]):
        is_cont_mask = RepairedExhaustiveSampling.get_is_cont_mask(problem)
        xl, xu = problem.xl, problem.xu

        # If the population of all available discrete design vectors is available, sample from there
        if x_all is not None:
            # Randomly select values
            x = x_all
            if n_samples < x.shape[0]:
                i_x = np.random.choice(x.shape[0], size=n_samples, replace=False)
            else:
                i_x = np.sort(np.concatenate([
                    np.arange(x.shape[0]), np.random.choice(x.shape[0], size=n_samples-x.shape[0], replace=True)]))
            x = x[i_x, :]

            # Randomize continuous variables
            if np.any(is_cont_mask):
                for i_dv, is_cont in enumerate(is_cont_mask):
                    if is_cont:
                        x[:, i_dv] = (np.random.random((x.shape[0],)))*(xu[i_dv]-xl[i_dv])+xl[i_dv]

                x = repair.do(problem, x)

            return x

        # If above the threshold (or a memory error occurred), sample randomly
        opt_values = RepairedExhaustiveSampling.get_exhaustive_sample_values(problem, n_cont=1)
        x = np.empty((n_samples, problem.n_var))
        for i_dv in range(problem.n_var):
            if is_cont_mask[i_dv]:
                x[:, i_dv] = np.random.random((n_samples,))*(xu[i_dv]-xl[i_dv]) + xl[i_dv]
            else:
                x[:, i_dv] = np.random.choice(opt_values[i_dv], (n_samples,))

        # Repair
        x = repair.do(problem, x)
        return x

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class LargeDuplicateElimination(DefaultDuplicateElimination):
    """
    Duplicate elimination that can deal with a large amount of individuals in a population: instead of creating one big
    n_pop x n_pop cdist matrix, it does so in batches, thereby staying fast and saving in memory at the same time.
    """
    _n_per_batch = 200

    def _do(self, pop, other, is_duplicate):
        x = self.func(pop)
        other = self.func(other) if other is not None else None
        return self.eliminate(x, other, is_duplicate, self.epsilon)

    @classmethod
    def eliminate(cls, x, other=None, is_duplicate=None, epsilon=1e-16):
        # Either compare x to itself or to another x
        x = x.copy().astype(float)
        if is_duplicate is None:
            is_duplicate = np.zeros((x.shape[0],), dtype=bool)

        to_itself = False
        if other is None:
            x_ = x
            to_itself = True
        else:
            x_ = other.copy().astype(float)

        # Determine how many batches we need
        n_per_batch = cls._n_per_batch
        nx = x.shape[0]
        n = (x_.shape[0] - 1) if to_itself else x_.shape[0]
        if n == 0:
            return is_duplicate
        n_batches = int(np.ceil(n / n_per_batch))
        n_in_batch = np.ones((n_batches,), dtype=np.int16)*n_per_batch
        n_in_batch[-1] = n - (n_batches-1)*n_per_batch

        for ib, n_batch in enumerate(n_in_batch):
            i_compare_to = np.arange(n_batch)+ib*n_per_batch
            i = i_compare_to[0]

            # Only compare points in the population to other points that are not already marked as duplicate
            non_dup = ~is_duplicate
            x_check = x[i+1:, ][non_dup[i+1:], :] if to_itself else x[non_dup, :]
            if x_check.shape[0] == 0:
                break

            # Do the comparison: the result is an n_x_check x n_i_compare_to boolean matrix
            i_is_dup = distance.cdist(x_check, x_[i_compare_to, :], metric='cityblock') < epsilon

            if to_itself:
                # Expand to all indices from non-duplicate indices
                i_is_dup_expanded = np.zeros((nx-i-1, n_batch), dtype=bool)
                i_is_dup_expanded[non_dup[i+1:], :] = i_is_dup

                # If we compare to ourselves, we will have a diagonal that is always true, and we should ignore anything
                # above the triangle, otherwise the indices are off
                i_is_dup_expanded[np.triu_indices(n_batch, k=1)] = False

                # Mark as duplicate rows where any of the columns is true
                is_duplicate[i+1:][np.any(i_is_dup_expanded, axis=1)] = True

            else:
                # Expand to all indices from non-duplicate indices
                i_is_dup_expanded = np.zeros((nx, n_batch), dtype=bool)
                i_is_dup_expanded[non_dup, :] = i_is_dup

                # Mark as duplicate rows where any of the columns is true
                is_duplicate[np.any(i_is_dup_expanded, axis=1)] = True

        return is_duplicate
