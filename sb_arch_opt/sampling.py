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
import logging
import warnings
import numpy as np
from typing import Optional, Tuple, List
from scipy.stats.qmc import Sobol
from scipy.spatial import distance

from pymoo.core.repair import Repair
from pymoo.core.variable import Real
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.initialization import Initialization
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.operators.sampling.lhs import sampling_lhs_unit

from sb_arch_opt.problem import ArchOptProblemBase, ArchOptRepair
from sb_arch_opt.util import get_np_random_singleton

__all__ = ['HierarchicalExhaustiveSampling', 'HierarchicalSampling',
           'get_init_sampler', 'LargeDuplicateElimination', 'TrailRepairWarning']

log = logging.getLogger('sb_arch_opt.sampling')


def get_init_sampler(repair: Repair = None, remove_duplicates=True):
    """Helper function to get an Initialization class with hierarchical sampling"""

    if repair is None:
        repair = ArchOptRepair()
    sampling = HierarchicalSampling(repair=repair, sobol=True)

    # Samples are already repaired because we're using the hierarchical samplers
    eliminate_duplicates = LargeDuplicateElimination() if remove_duplicates else None
    return Initialization(sampling, eliminate_duplicates=eliminate_duplicates)


class TrailRepairWarning(RuntimeWarning):
    pass


class HierarchicalExhaustiveSampling(Sampling):
    """Exhaustively samples the design space, taking n_cont samples for each continuous variable.
    Can take a long time if the design space is large and the problem doesn't provide a way to generate all discrete
    design vectors, and doesn't work well for purely continuous problems."""

    def __init__(self, repair: Repair = None, n_cont=5):
        super().__init__()
        if repair is None:
            repair = ArchOptRepair()
        self._repair = repair
        self._n_cont = n_cont

    def _do(self, problem: Problem, n_samples, **kwargs):
        return self.do_sample(problem)

    def do_sample(self, problem: Problem):
        # First sample only discrete dimensions
        x_discr, is_act_discr = self.get_all_x_discrete(problem)

        # Expand along continuous dimensions
        n_cont = self._n_cont
        is_cont_mask = self.get_is_cont_mask(problem)
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
                dv_sampled = np.linspace(problem.xl[i_dv], problem.xu[i_dv], n_cont)

                n_dv_rep = np.sum(is_act_i)
                dv_sampled_rep = np.tile(dv_sampled, n_dv_rep)
                rep_idx = np.cumsum([0]+list(n_repeat))[:-1]
                i_repeated_at = np.repeat(rep_idx[is_act_i], n_repeat[is_act_i]) + np.tile(np.arange(n_cont), n_dv_rep)
                x[i_repeated_at, i_dv] = dv_sampled_rep

        else:
            x = x_discr

        return x

    @staticmethod
    def has_cheap_all_x_discrete(problem: Problem):
        if isinstance(problem, ArchOptProblemBase):
            # Check if the problem itself provides all discrete design vectors
            x_discrete, _ = problem.all_discrete_x
            if x_discrete is not None:
                return True

        return False

    def get_all_x_discrete(self, problem: Problem):
        # Check if the problem itself can provide all discrete design vectors
        if isinstance(problem, ArchOptProblemBase):
            x_discr, is_act_discr = problem.all_discrete_x
            if x_discr is not None:
                return x_discr, is_act_discr

        # Otherwise, use trail and repair (costly!)
        warnings.warn(f'Generating hierarchical discrete samples by trial and repair for {problem!r}! '
                      f'Consider implementing `_gen_all_discrete_x`', TrailRepairWarning)
        return self.get_all_x_discrete_by_trial_and_repair(problem)

    @staticmethod
    def get_all_x_discrete_by_trial_and_repair(problem: Problem):
        if not isinstance(problem, ArchOptProblemBase):
            raise RuntimeError('Not implemented for generic Problems!')
        return problem.design_space.all_discrete_x_by_trial_and_imputation

    @classmethod
    def get_exhaustive_sample_values(cls, problem: Problem, n_cont=5):
        if isinstance(problem, ArchOptProblemBase):
            return problem.design_space.get_exhaustive_sample_values(n_cont=n_cont)

        # Determine bounds and which design variables are discrete
        xl, xu = problem.bounds()
        is_cont = cls.get_is_cont_mask(problem)

        # Get values to be sampled for each design variable
        return [np.linspace(xl[i], xu[i], n_cont) if is_cont[i] else np.arange(xl[i], xu[i]+1) for i in range(len(xl))]

    @staticmethod
    def get_is_cont_mask(problem: Problem):
        if isinstance(problem, ArchOptProblemBase):
            return problem.is_cont_mask

        is_cont = np.ones((problem.n_var,), dtype=bool)
        if problem.vars is not None:
            for i, var in enumerate(problem.vars.values()):
                if not isinstance(var, Real):
                    is_cont[i] = False
        return is_cont

    @classmethod
    def get_n_sample_exhaustive(cls, problem: Problem, n_cont=5):
        values = cls.get_exhaustive_sample_values(problem, n_cont=n_cont)
        return int(np.prod([len(opts) for opts in values], dtype=float))

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class HierarchicalSampling(FloatRandomSampling):
    """
    Hierarchical mixed-discrete sampling. There are two ways the random sampling is performed:
    A: Generate and select:
       1. Generate all possible discrete design vectors
       2. Separate discrete design vectors based on discrete rate diversity
       3. Within each group, uniformly sample discrete design vectors
       4. Concatenate and randomize active continuous variables
    B: One-shot:
       1. Randomly select design variable values
       2. Repair/impute design vectors

    The first way yields better results, as there is an even chance of selecting every valid discrete design vector,
    however it takes more memory and might be too much for very large design spaces.
    """

    _n_comb_gen_all_max = 100e3

    def __init__(self, repair: Repair = None, sobol=True, seed=None):
        if repair is None:
            repair = ArchOptRepair()
        self._repair = repair
        self.sobol = sobol
        self.n_iter = 10
        # self.high_rd_split = .8
        # self.low_rd_split = None
        super().__init__()

        # Simply set the seed on the global numpy instance
        if seed is not None:
            np.random.seed(seed)

    def _do(self, problem, n_samples, **kwargs):
        x_sampled, _ = self.sample_get_x(problem, n_samples)
        return x_sampled

    def sample_get_x(self, problem: ArchOptProblemBase, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample design points using the hierarchical sampling algorithm and return is_active information.
        """

        # Get Cartesian product of all discrete design variables (only available if design space is not too large)
        x, is_active = self.get_hierarchical_cartesian_product(problem, self._repair)

        x_sampled, is_active = self.randomly_sample(problem, n_samples, self._repair, x, is_active)
        return x_sampled, is_active

    @classmethod
    def get_hierarchical_cartesian_product(cls, problem: Problem, repair: Repair) \
            -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        # Get values to be sampled for each discrete design variable
        exhaustive_sampling = HierarchicalExhaustiveSampling(repair=repair, n_cont=1)
        opt_values = exhaustive_sampling.get_exhaustive_sample_values(problem, n_cont=1)

        # Get the number of samples in the cartesian product
        n_opt_values = int(np.prod([len(values) for values in opt_values], dtype=float))

        # If less than some threshold, sample all and then select (this gives a better distribution)
        if n_opt_values < cls._n_comb_gen_all_max or exhaustive_sampling.has_cheap_all_x_discrete(problem):
            try:
                x, is_active = exhaustive_sampling.get_all_x_discrete(problem)
                return x, is_active
            except MemoryError:
                pass

        warnings.warn(f'Hierarchical sampling is not possible for {problem!r}, falling back to non-hierarchical '
                      f'sampling! Consider implementing `_gen_all_discrete_x`', TrailRepairWarning)
        return None, None

    def randomly_sample(self, problem, n_samples, repair: Repair, x_all: Optional[np.ndarray],
                        is_act_all: Optional[np.ndarray], lhs=False) -> Tuple[np.ndarray, np.ndarray]:
        is_cont_mask = HierarchicalExhaustiveSampling.get_is_cont_mask(problem)
        has_x_cont = np.any(is_cont_mask)
        xl, xu = problem.xl, problem.xu
        sobol = self.sobol

        # If the population of all available discrete design vectors is available, sample from there
        if x_all is not None:
            x, is_active = self._sample_discrete_x(n_samples, is_cont_mask, x_all, is_act_all, sobol=sobol)

        # Otherwise, sample discrete vectors randomly
        else:
            x, is_active = self._sample_discrete_x_random(problem, repair, n_samples, is_cont_mask, sobol=sobol)

        # Randomize continuous variables
        if has_x_cont:
            if is_active is None:
                is_active = np.ones(x.shape, dtype=bool)

            nx_cont = len(np.where(is_cont_mask)[0])
            if lhs:
                x_unit = sampling_lhs_unit(x.shape[0], nx_cont)
            elif sobol:
                x_unit = self._sobol(x.shape[0], nx_cont)
            else:
                x_unit = np.random.random((x.shape[0], nx_cont))

            x_unit_abs = x_unit*(xu[is_cont_mask]-xl[is_cont_mask])+xl[is_cont_mask]

            # Do not overwrite inactive imputed continuous variables
            is_inactive_override = ~is_active[:, is_cont_mask]
            x_unit_abs[is_inactive_override] = x[:, is_cont_mask][is_inactive_override]

            x[:, is_cont_mask] = x_unit_abs

            # Correct variables
            x, is_active = self._correct(problem, repair, x)

        if x.shape[0] != n_samples:
            log.info(f'Generated {x.shape[0]} samples, {n_samples} requested')

        return x, is_active

    @staticmethod
    def _correct(problem: Problem, repair: Repair, x):
        if isinstance(problem, ArchOptProblemBase):
            return problem.correct_x(x)

        x_corr = repair.do(problem, x)
        is_active_corr = None
        if isinstance(repair, ArchOptRepair) and repair.latest_is_active is not None:
            is_active_corr = repair.latest_is_active
        if is_active_corr is None:
            raise ValueError('Unexpectedly empty is_active!')
        return x_corr, is_active_corr

    def _sample_discrete_x(self, n_samples: int, is_cont_mask, x_all: np.ndarray, is_act_all: np.ndarray, sobol=False):
        if x_all.shape[0] == 0:
            raise ValueError('Set of discrete vectors cannot be empty!')

        def _choice(n_choose, n_from, replace=True):
            return self._choice(n_choose, n_from, replace=replace, sobol=sobol)

        # Separate design vectors into groups
        groups = self.group_design_vectors(x_all, is_act_all, is_cont_mask)

        # Apply weights to the different groups
        weights = np.array(self._get_group_weights(groups, is_act_all))

        # Uniformly choose from which group to sample
        if len(groups) == 1:
            selected_groups = np.zeros((n_samples,), dtype=int)
        else:
            unit_weights = weights/np.sum(weights)
            selected_groups = np.zeros((n_samples,), dtype=int)
            selected_pos = np.sort(self._sobol(n_samples))
            for cum_weight in np.cumsum(unit_weights)[:-1]:
                selected_groups[selected_pos > cum_weight] += 1

        x = []
        is_active = []
        has_x_cont = np.any(is_cont_mask)
        i_x_sampled = np.zeros((x_all.shape[0],), dtype=bool)
        for i_grp in range(len(groups)):
            i_x_tgt = np.where(selected_groups == i_grp)[0]
            if len(i_x_tgt) == 0:
                continue

            i_x_group = groups[i_grp]
            i_from_group = self._sample_discrete_from_group(
                x_all[i_x_group, :], is_act_all[i_x_group, :], len(i_x_tgt), _choice, has_x_cont)

            x_all_choose = i_x_group[i_from_group]
            x.append(x_all[x_all_choose, :])
            is_active.append(is_act_all[x_all_choose, :])
            i_x_sampled[x_all_choose] = True

        x = np.row_stack(x)
        is_active = np.row_stack(is_active)

        # Uniformly add discrete vectors if there are not enough (can happen if some groups are very small and there
        # are no continuous dimensions)
        if x.shape[0] < n_samples:
            n_add = n_samples-x.shape[0]
            x_available = x_all[~i_x_sampled, :]
            is_act_available = is_act_all[~i_x_sampled, :]

            if n_add < x_available.shape[0]:
                i_from_group = _choice(n_add, x_available.shape[0], replace=False)
            else:
                i_from_group = np.arange(x_available.shape[0])

            x = np.row_stack([x, x_available[i_from_group, :]])
            is_active = np.row_stack([is_active, is_act_available[i_from_group, :]])

        return x, is_active

    def _sample_discrete_from_group(self, x_group: np.ndarray, is_act_group: np.ndarray, n_sel: int, choice_func,
                                    has_x_cont: bool) -> np.ndarray:
        # Get the number of design points to sample
        n_in_group = x_group.shape[0]
        n_sel = n_sel
        i_x_selected = np.array([], dtype=int)
        while n_sel >= n_in_group:
            # If we have to sample a multiple of the available points or if we cannot sample duplicate points (because
            # there are no continuous variables), return all points
            if n_sel == n_in_group or not has_x_cont:
                return np.concatenate([i_x_selected, np.arange(n_in_group)])

            # Pre-select all points once
            i_x_selected = np.concatenate([i_x_selected, np.arange(n_in_group)])
            n_sel = n_sel-n_in_group

        if n_sel == 1:
            i_x_take = choice_func(1, n_in_group, replace=False)
            return np.concatenate([i_x_selected, i_x_take])

        # Randomly sample several times to get the best distribution of points
        i_x_tries = []
        metrics = []
        for _ in range(self.n_iter):
            i_x_try = choice_func(n_sel, n_in_group, replace=False)
            i_x_tries.append(i_x_try)

            x_try = x_group[i_x_try, :]
            dist = distance.cdist(x_try, x_try, metric='cityblock')
            np.fill_diagonal(dist, np.nan)

            min_dist = np.nanmin(dist)
            median_dist = np.nanmean(dist)
            metrics.append((min_dist, median_dist))

        # Get the distribution with max minimum distance and max mean distance
        i_best = sorted(range(len(metrics)), key=metrics.__getitem__)[-1]
        return np.concatenate([i_x_selected, i_x_tries[i_best]])

    def group_design_vectors(self, x_all: np.ndarray, is_act_all: np.ndarray, is_cont_mask) -> List[np.ndarray]:
        # Group by active design variables
        is_active_unique, unique_indices = np.unique(is_act_all, axis=0, return_inverse=True)
        return [np.where(unique_indices == i)[0] for i in range(len(is_active_unique))]

        # # Group by rate diversity (difference between discrete value occurrences)
        # is_discrete_mask = ~is_cont_mask
        # high_rd_split = self.high_rd_split
        # low_rd_split = self.low_rd_split
        # i_low_rd_split = None
        #
        # def recursive_get_groups(group_i: np.ndarray) -> List[np.ndarray]:
        #     nonlocal i_low_rd_split
        #     if len(group_i) == 0:
        #         return []
        #
        #     # For current group, get rate diversity information
        #     x_grp = x_all[group_i, :]
        #     x_min = np.min(x_grp, axis=0).astype(int)
        #     is_act_grp = is_act_all[group_i, :]
        #     counts, diversity, active_diversity, i_opts = \
        #         ArchDesignSpace.calculate_discrete_rates_raw(x_grp - x_min, is_act_grp, is_discrete_mask)
        #
        #     # Split on low split rate
        #     xi_split = None
        #     if low_rd_split is not None:
        #         rd_split_rates, = np.where(active_diversity >= low_rd_split)
        #         if i_low_rd_split is None:  # If no low-split variable has been set
        #             if len(rd_split_rates) == 0:
        #                 i_low_rd_split = -1  # Set to "no low-split var"
        #             else:
        #                 i_low_rd_split = rd_split_rates[0]  # Choose first var
        #                 xi_split = rd_split_rates[0]
        #
        #         elif i_low_rd_split != -1 and len(rd_split_rates) > 0 and rd_split_rates[0] == i_low_rd_split:
        #             # Split on same low-split variable
        #             xi_split = rd_split_rates[0]
        #
        #     # Check high split rate
        #     if xi_split is None:
        #         rd_split_rates, = np.where(active_diversity >= high_rd_split)
        #         if len(rd_split_rates) == 0:
        #             return [group_i]
        #         # Split on first variable
        #         xi_split = rd_split_rates[0]
        #
        #     opt_rates = counts[1:, xi_split]
        #     i_opt_min = np.nanargmin(opt_rates) + x_min[xi_split]
        #
        #     min_rate_group = x_grp[:, xi_split] == i_opt_min
        #     group_i_min = group_i[min_rate_group]
        #     group_i_other = group_i[~min_rate_group]
        #
        #     # Recursively define groups within split groups
        #     return recursive_get_groups(group_i_min) + recursive_get_groups(group_i_other)
        #
        # return recursive_get_groups(np.arange(x_all.shape[0]))

    def _get_group_weights(self, groups: List[np.ndarray], is_act_all: np.ndarray) -> List[float]:
        # Uniform sampling
        return [1.]*len(groups)

        # # Weight subgroups by nr of active variables
        # nr_active = np.sum(is_act_all, axis=1)
        # avg_nr_active = [np.sum(nr_active[group])/len(group) for group in groups]
        # return avg_nr_active

    def _sample_discrete_x_random(self, problem: Problem, repair: Repair, n_samples: int, is_cont_mask, sobol=False):
        has_x_cont = np.any(is_cont_mask)

        def _choice(n_choose, n_from, replace=True):
            return self._choice(n_choose, n_from, replace=replace, sobol=sobol)

        n_x = 0
        x = is_active = None
        for _ in range(3):
            # Determine how many samples to request in this round:
            # add a little margin to correct for potential duplicate vectors
            n_add = min(n_samples, (n_samples-n_x)*5)

            if isinstance(problem, ArchOptProblemBase):
                x_add, is_active_add = problem.design_space.quick_sample_discrete_x(n_add)

            else:
                opt_values = HierarchicalExhaustiveSampling.get_exhaustive_sample_values(problem, n_cont=1)
                x_add = np.empty((n_add, problem.n_var))
                for i_dv in range(problem.n_var):
                    if not is_cont_mask[i_dv]:
                        i_opt_sampled = _choice(n_add, len(opt_values[i_dv]))
                        x_add[:, i_dv] = opt_values[i_dv][i_opt_sampled]

            x = x_add if x is None else np.row_stack([x, x_add])

            # Correct and remove duplicates
            x, is_active = self._correct(problem, repair, x)
            is_unique = ~LargeDuplicateElimination.eliminate(x)
            x = x[is_unique, :]
            is_active = is_active[is_unique, :]

            n_x = x.shape[0]
            if n_x > n_samples:
                x = x[:n_samples, :]
                is_active = is_active[:n_samples, :]
                break
            if n_x == n_samples:
                break

        # Duplicate discrete vectors if needed and if there are continuous dimensions
        if x.shape[0] < n_samples and has_x_cont:
            n_add = n_samples-x.shape[0]
            i_select_dup = _choice(n_add, x.shape[0])
            x = np.row_stack(x, x[i_select_dup, :])
            is_active = np.row_stack(is_active, is_active[i_select_dup, :])

        return x, is_active

    @staticmethod
    def _sobol(n_samples, n_dims=None) -> np.ndarray:
        """
        Sample using a Sobol sequence, which supposedly gives a better distribution of points in a hypercube.
        More info: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html
        """

        # Get the power of 2 for generating samples (generating a power of 2 gives points with the lowest discrepancy)
        pow2 = int(np.ceil(np.log2(n_samples)))

        # Sample points and only return the amount needed
        global_rng = get_np_random_singleton()
        x = Sobol(d=n_dims or 1, seed=global_rng).random_base2(m=pow2)
        x = x[:n_samples, :]
        return x[:, 0] if n_dims is None else x

    @classmethod
    def _choice(cls, n_choose, n_from, replace=True, sobol=False):
        if sobol:
            return cls._sobol_choice(n_choose, n_from, replace=replace)
        return np.random.choice(n_from, n_choose, replace=replace)

    @classmethod
    def _sobol_choice(cls, n_choose, n_from, replace=True):
        """
        Randomly choose n_choose from n_from values, optionally replacing (i.e. allow choosing values multiple times).
        If n_choose > n_from
        """
        if n_choose <= 0:
            return np.zeros((0,), dtype=int)

        # If replace (i.e. values can be chosen multiple times)
        if replace:
            # Generate unit samples
            x_unit = cls._sobol(n_choose)

            # Scale to nr of possible values and round
            return np.round(x_unit*(n_from-.01)-.5).astype(int)

        # If we cannot replace, we cannot choose more values than available
        if n_choose > n_from:
            raise ValueError(f'Nr of values to choose should be lower than available values: {n_choose} > {n_from}')

        # Generate unit samples from total nr available
        x_unit = cls._sobol(n_from)

        # Get sorting arguments: this places each float value on an increasing integer scale
        x_unit = x_unit.argsort()

        # Only return the amount that we actually want to choose
        return x_unit[:n_choose]

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
        n_in_batch = np.ones((n_batches,), dtype=int)*n_per_batch
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
