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
import pickle
import logging

import numpy as np
import pandas as pd
from typing import Optional
from pymoo.core.result import Result
from pymoo.core.callback import Callback
from pymoo.core.algorithm import Algorithm
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population, Individual
from pymoo.core.initialization import Initialization

from sb_arch_opt.util import capture_log
from sb_arch_opt.problem import ArchOptProblemBase
from sb_arch_opt.sampling import LargeDuplicateElimination

__all__ = ['load_from_previous_results', 'initialize_from_previous_results', 'ResultsStorageCallback',
           'ArchOptEvaluator']

log = logging.getLogger('sb_arch_opt.pymoo')


def load_from_previous_results(problem: ArchOptProblemBase, result_folder: str, cumulative=False) -> Optional[Population]:
    """Load a Population from previously-stored results"""
    capture_log()

    # Try to load using problem-specific function first
    population = problem.load_previous_results(result_folder) if not cumulative else None
    if population is not None and len(population) > 0:
        log.info(f'Previous results loaded from problem results: {len(population)} design points')

    # Try to load from pymoo storage
    else:
        population = ResultsStorageCallback.load_pop(result_folder, cumulative=cumulative)
        if population is not None and len(population) > 0:
            log.info(f'Previous results loaded from pymoo results: {len(population)} design points')
        else:
            return

    # Set evaluated flag
    def _set_eval(ind: Individual):
        # Assume evaluated but failed points have Inf as output values
        is_eval = ~np.all(np.isnan(ind.get('F')))
        if is_eval:
            ind.evaluated.update({'X', 'F', 'G', 'H'})

    population.apply(_set_eval)

    return population


def initialize_from_previous_results(algorithm: Algorithm, problem: ArchOptProblemBase, result_folder: str) -> bool:
    """Initialize an Algorithm from previously stored results"""
    capture_log()

    if not hasattr(algorithm, 'initialization'):
        raise RuntimeError(f'Algorithm has no initialization step, cannot set initial population: {algorithm!r}')

    # Try to load from previous results
    population = load_from_previous_results(problem, result_folder)
    if population is None:
        log.info(f'No previous population found, not changing initialization strategy')
        return False

    # Set static initialization on the algorithm to start from the loaded population
    algorithm.initialization = Initialization(population)
    return True


class ResultsStorageCallback(Callback):
    """
    Optimization callback that stores intermediate and final optimization results:
    - intermediate and final Population in pymoo_population.pkl
    - final Result object in pymoo_results.pkl
    - any problem-specific intermediate and final results
    """

    def __init__(self, results_folder: str, callback=None):
        self.results_folder = results_folder
        os.makedirs(results_folder, exist_ok=True)
        self.callback = callback
        self.cumulative_pop = None
        super().__init__()

    def initialize(self, algorithm: Algorithm):
        self.cumulative_pop = None

        # Hook into the results function to store final results
        result_func = algorithm.result

        def wrapped_result():
            result = result_func()
            self.store_intermediate(algorithm, final=True)

            # Store pymoo results
            if result.algorithm is not None:
                result.algorithm.callback = None
            try:
                self._store_results(result)
            except MemoryError:

                result.history = None
                result.algorithm = None

                try:
                    self._store_results(result)
                except MemoryError:
                    log.info('Could not store pymoo result object: MemoryError')

            return result

        algorithm.result = wrapped_result

    def _update(self, algorithm: Algorithm):
        # Store intermediate results after each optimizer iteration
        self.store_intermediate(algorithm)

    def store_intermediate(self, algorithm: Algorithm, final=False):
        # Store pymoo population
        if not hasattr(algorithm, 'pop'):
            raise RuntimeError(f'Algorithm has no population (pop property): {algorithm!r}')
        pop: Population = algorithm.pop
        self.store_pop(pop)

        # Store cumulative pymoo population
        if self.cumulative_pop is None:
            self.cumulative_pop = pop
        else:
            self.cumulative_pop = LargeDuplicateElimination().do(Population.merge(self.cumulative_pop, pop))
        self.store_pop(self.cumulative_pop, cumulative=True)

        # Store problem-specific results
        self.store_intermediate_problem(algorithm.problem, final=final)

    def store_pop(self, pop: Population, cumulative=False):
        with open(self._get_pop_file_path(self.results_folder, cumulative=cumulative), 'wb') as fp:
            pickle.dump(pop, fp)

        if len(pop) > 0:
            cumulative_str = '_cumulative' if cumulative else ''
            csv_path = os.path.join(self.results_folder, f'pymoo_population{cumulative_str}.csv')
            self.get_pop_as_df(pop).to_csv(csv_path)

    @staticmethod
    def get_pop_as_df(pop: Population) -> pd.DataFrame:
        cols = []
        all_data = []
        for symbol in ['x', 'f', 'g', 'h']:
            data = pop.get(symbol.upper())
            all_data.append(data)
            cols += [f'{symbol}{i}' for i in range(data.shape[1])]

        return pd.DataFrame(columns=cols, data=np.column_stack(all_data))

    def store_intermediate_problem(self, problem, final=False):
        if isinstance(problem, ArchOptProblemBase):
            problem.store_results(self.results_folder, final=final)

    def _store_results(self, result: Result):
        with open(os.path.join(self.results_folder, 'pymoo_results.pkl'), 'wb') as fp:
            pickle.dump(result, fp)

    @classmethod
    def load_pop(cls, results_folder: str, cumulative=False) -> Optional[Population]:
        pop_path = cls._get_pop_file_path(results_folder, cumulative=cumulative)
        if not os.path.exists(pop_path):
            return

        with open(pop_path, 'rb') as fp:
            pop = pickle.load(fp)

        if not isinstance(pop, Population):
            raise ValueError(f'Loaded population not of type Population ({pop_path}): {pop!r}')
        return pop

    @staticmethod
    def _get_pop_file_path(results_folder, cumulative=False) -> str:
        cumulative_str = '_cumulative' if cumulative else ''
        return os.path.join(results_folder, f'pymoo_population{cumulative_str}.pkl')

    def __call__(self, *args, **kwargs):
        super().__call__(*args, **kwargs)
        if self.callback is not None:
            self.callback(*args, **kwargs)


class ArchOptEvaluator(Evaluator):
    """
    Evaluate that adds some optional functionalities useful for architecture optimization:
    - It implements the extreme barrier approach for dealing with hidden constraints: NaN outputs are replaced by Inf
    - It stores intermediate results during evaluation, which also allows results to be stored during a large DoE for
      example, instead of only when an algorithm makes a new iteration

    Batch process size is determined using `get_n_batch_evaluate` if not specified explicitly!

    Also using the ResultsStorageCallback ensures that also final problem-specific results are stored.
    """

    def __init__(self, *args, results_folder: str = None, n_batch=None, **kwargs):
        self.extreme_barrier = True
        self.results_folder = results_folder
        self.n_batch = n_batch
        super().__init__(*args, **kwargs)
        self._skipping_pop = None

    def eval(self, problem, pop: Population, skip_already_evaluated: bool = None, evaluate_values_of: list = None,
             count_evals: bool = True, **kwargs):

        evaluate_values_of = self.evaluate_values_of if evaluate_values_of is None else evaluate_values_of
        skip_already_evaluated = self.skip_already_evaluated if skip_already_evaluated is None else skip_already_evaluated

        self._skipping_pop = None
        if skip_already_evaluated:
            i_skipped = [i for i, ind in enumerate(pop) if all([e in ind.evaluated for e in evaluate_values_of])]
            self._skipping_pop = pop[i_skipped]

        return super().eval(problem, pop, skip_already_evaluated=skip_already_evaluated,
                            evaluate_values_of=evaluate_values_of, count_evals=count_evals, **kwargs)

    def _eval(self, problem, pop, evaluate_values_of, **kwargs):
        if self.results_folder is None:
            super()._eval(problem, pop, evaluate_values_of, **kwargs)

        else:
            # Evaluate in batch and store intermediate results
            callback = ResultsStorageCallback(self.results_folder)

            n_batch = self.n_batch
            if n_batch is None and isinstance(problem, ArchOptProblemBase):
                n_batch = problem.get_n_batch_evaluate()
            if n_batch is None:
                n_batch = 1  # Assume there is no batch processing, and we want to save after every evaluation

            for i_batch in range(0, len(pop), n_batch):
                batch_pop = pop[i_batch:i_batch+n_batch]
                super()._eval(problem, batch_pop, evaluate_values_of, **kwargs)

                callback.store_pop(self._normalize_pop(pop, evaluate_values_of, skipping_pop=self._skipping_pop))
                callback.store_intermediate_problem(problem)

        # Apply extreme barrier: replace NaN with Inf
        if self.extreme_barrier:
            for key in ['F', 'G', 'H']:
                values = pop.get(key)
                values[np.isnan(values)] = np.inf
                pop.set(key, values)

        return pop

    @staticmethod
    def _normalize_pop(pop: Population, evaluate_values_of, nan_as_inf=True, skipping_pop: Population = None) -> Population:
        """Ensure that the matrices in a Population are two-dimensional"""
        if skipping_pop is not None:
            pop = Population.merge(skipping_pop, pop)

        pop_data = {}
        for key in (['X']+evaluate_values_of):
            data = pop.get(key)

            if len(data.shape) == 1:
                partial_data = np.zeros((len(data), len(data[0])))*np.nan
                for i, row in enumerate(data):
                    if row is not None and len(row) > 0:
                        if nan_as_inf:
                            row = row.copy()
                            row[np.isnan(row)] = np.inf
                        partial_data[i, :] = row
                data = partial_data

            pop_data[key] = data
        return Population.new(**pop_data)
