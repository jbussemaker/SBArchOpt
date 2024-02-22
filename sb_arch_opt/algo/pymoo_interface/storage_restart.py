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
           'ArchOptEvaluator', 'set_initial_population']

log = logging.getLogger('sb_arch_opt.pymoo')


def load_from_previous_results(problem: ArchOptProblemBase, result_folder: str, load_from_problem=True,
                               cumulative=True) -> Optional[Population]:
    """Load a (cumulative) Population from previously-stored results"""
    capture_log()

    # Try to load using problem-specific function first
    population = None
    if load_from_problem:
        population = problem.load_previous_results(result_folder)
        if population is not None and len(population) > 0:
            log.info(f'Previous results loaded from problem results: {len(population)} design points')

    # Additionally try to load from pymoo storage to merge with non-evaluated design points
    pymoo_population = ArchOptEvaluator.load_pop(result_folder, cumulative=cumulative)
    if pymoo_population is not None and len(pymoo_population) > 0:

        if population is None:
            log.info(f'Previous results loaded from pymoo results: {len(pymoo_population)} design points')
            population = pymoo_population

        elif len(pymoo_population) > len(population):
            unique_points = LargeDuplicateElimination().do(pymoo_population, population, to_itself=False)
            if len(unique_points) > 0:
                log.info(f'Merged additional design points from pymoo results: {len(unique_points)} design points')
                population = Population.merge(population, unique_points)

    if population is None:
        return

    # Set evaluated flags
    n_evaluated = ArchOptEvaluator.pop_set_eval_flags(population)
    log.info(f'Evaluation status: {n_evaluated} of {len(population)} ({(n_evaluated/len(population))*100:.1f}%) '
             f'are already evaluated')

    return population


def initialize_from_previous_results(algorithm: Algorithm, problem: ArchOptProblemBase, result_folder: str, **kwargs) \
        -> bool:
    """Initialize an Algorithm from previously stored results"""
    capture_log()

    if not hasattr(algorithm, 'initialization'):
        raise RuntimeError(f'Algorithm has no initialization step, cannot set initial population: {algorithm!r}')

    # Try to load from previous results
    population = load_from_previous_results(problem, result_folder, **kwargs)
    if population is None:
        log.info('No previous population found, not changing initialization strategy')
        return False

    # Set static initialization on the algorithm to start from the loaded population
    set_initial_population(algorithm, population)

    return True


def set_initial_population(algorithm: Algorithm, population: Population):
    """Set the initial population on an Algorithm"""

    # Set static initialization on the algorithm to start from the loaded population
    algorithm.initialization = Initialization(population)

    # Initialize cumulative population
    if isinstance(algorithm.evaluator, ArchOptEvaluator):
        algorithm.evaluator.initialize_cumulative(population)


class ResultsStorageCallback(Callback):
    """
    Optimization callback that stores final optimization results in pymoo_results.pkl
    """

    def __init__(self, results_folder: str, callback=None):
        self.results_folder = results_folder
        os.makedirs(results_folder, exist_ok=True)
        self.callback = callback
        super().__init__()

    def initialize(self, algorithm: Algorithm):
        # Hook into the results function to store final results
        result_func = algorithm.result

        def wrapped_result():
            result = result_func()

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

    def _store_results(self, result: Result):
        with open(os.path.join(self.results_folder, 'pymoo_results.pkl'), 'wb') as fp:
            pickle.dump(result, fp)

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

    You can also use this class with intermediate (partial) results storage in an ask-tell setting:
    >>> algo: Algorithm
    >>> problem: ArchOptProblemBase

    >>> pop = algo.ask()
    >>> evaluator: ArchOptEvaluator = algo.evaluator

    >>> pop_to_eval = evaluator.eval_pre(pop)
    >>> for batch_pop in evaluator.iter_pop_batch(problem, pop_to_eval):
    >>>     out = ...  # Return output in dict format as returned from problem.evaluate
    >>>     evaluator.eval_apply_to_pop(batch_pop, out)
    >>>     evaluator.eval_batch_post(problem, pop_to_eval, batch_pop)

    >>> evaluator.eval_post(problem, pop)
    >>> algo.tell(pop)
    """

    def __init__(self, *args, results_folder: str = None, n_batch=None, **kwargs):
        self.results_folder = results_folder
        if results_folder is not None:
            os.makedirs(results_folder, exist_ok=True)
        self.n_batch = n_batch
        self._cumulative_pop = None

        super().__init__(*args, **kwargs)

        self._evaluated_pop = None
        self._non_eval_cumulative_pop = None

    def initialize_cumulative(self, cumulative_pop: Population):
        # Set cumulative population and correct the nr of evaluations
        self._cumulative_pop = cumulative_pop
        self.n_eval = len(self._get_idx_evaluated(cumulative_pop))

    def eval(self, problem, pop: Population, skip_already_evaluated: bool = None, evaluate_values_of: list = None,
             count_evals: bool = True, **kwargs):

        self.eval_pre(pop, skip_already_evaluated=skip_already_evaluated, evaluate_values_of=evaluate_values_of)

        results = super().eval(problem, pop, skip_already_evaluated=skip_already_evaluated,
                               evaluate_values_of=evaluate_values_of, count_evals=count_evals, **kwargs)

        self.eval_post(problem, pop, advance_n_eval=False)
        return results

    def eval_pre(self, pop: Population, skip_already_evaluated=None, evaluate_values_of: list = None):
        # Get pop being skipped in order to complete the intermediate storage
        skip_already_evaluated = \
            self.skip_already_evaluated if skip_already_evaluated is None else skip_already_evaluated

        self._evaluated_pop = None
        pop_to_evaluate = pop
        if skip_already_evaluated:
            i_evaluated = self._get_idx_evaluated(pop, evaluate_values_of=evaluate_values_of)
            self._evaluated_pop = pop[i_evaluated]
            pop_to_evaluate = np.delete(pop, i_evaluated)

        # Get portion of the cumulative population that is currently not under evaluation
        self._non_eval_cumulative_pop = None
        if self._cumulative_pop is not None:
            is_duplicate = LargeDuplicateElimination.eliminate(self._cumulative_pop.get('X'), pop.get('X'))
            self._non_eval_cumulative_pop = self._cumulative_pop[~is_duplicate]

        return pop_to_evaluate

    def eval_post(self, problem, pop, advance_n_eval=True, evaluate_values_of: list = None):
        # Advance evaluation count
        if advance_n_eval:
            i_evaluated = self._get_idx_evaluated(pop, evaluate_values_of=evaluate_values_of)
            n_evaluated = len(i_evaluated) - len(self._evaluated_pop)
            if n_evaluated > 0:
                self.n_eval += n_evaluated

        # Apply extreme barrier: replace NaN with Inf
        self._apply_extreme_barrier(pop, evaluate_values_of=evaluate_values_of)

        # Post-evaluation storage
        if self.results_folder is not None:
            self._store_intermediate(problem, pop)
        self._non_eval_cumulative_pop = None

    def _get_idx_evaluated(self, pop: Population, evaluate_values_of: list = None):
        evaluate_values_of = self.evaluate_values_of if evaluate_values_of is None else evaluate_values_of
        return [i for i, ind in enumerate(pop) if all([e in ind.evaluated for e in evaluate_values_of])]

    def get_n_batch(self, problem):
        n_batch = self.n_batch
        if n_batch is None and isinstance(problem, ArchOptProblemBase):
            n_batch = problem.get_n_batch_evaluate()
        if n_batch is None:
            n_batch = 1  # Assume there is no batch processing, and we want to save after every evaluation
        return n_batch

    def iter_pop_batch(self, problem, pop: Population, n_batch=None):
        if n_batch is None:
            n_batch = self.get_n_batch(problem)

        for i_batch in range(0, len(pop), n_batch):
            yield pop[i_batch:i_batch+n_batch]

    def _eval(self, problem, pop, evaluate_values_of, **kwargs):
        if self.results_folder is None:
            super()._eval(problem, pop, evaluate_values_of, **kwargs)

        else:
            # Evaluate in batch and store intermediate results
            for batch_pop in self.iter_pop_batch(problem, pop):
                super()._eval(problem, batch_pop, evaluate_values_of, **kwargs)
                self.eval_batch_post(problem, pop, batch_pop, evaluate_values_of=evaluate_values_of)

        # Apply extreme barrier: replace NaN with Inf
        self._apply_extreme_barrier(pop)

        return pop

    @staticmethod
    def eval_apply_to_pop(pop: Population, out: dict):
        # for each of the attributes set it to the problem
        for key, val in out.items():
            if val is not None:
                pop.set(key, val)

        # finally set all the attributes to be evaluated for all individuals
        pop.apply(lambda ind: ind.evaluated.update(out.keys()))

    def eval_batch_post(self, problem, pop: Population, batch_pop: Population, evaluate_values_of: list = None):
        if evaluate_values_of is None:
            evaluate_values_of = self.evaluate_values_of

        self._apply_extreme_barrier(batch_pop, evaluate_values_of=evaluate_values_of)
        intermediate_pop = self._normalize_pop(pop, evaluate_values_of, evaluated_pop=self._evaluated_pop)

        if self.results_folder is not None:
            self._store_intermediate(problem, intermediate_pop)
        return intermediate_pop

    def _apply_extreme_barrier(self, pop: Population, evaluate_values_of: list = None):
        if evaluate_values_of is None:
            evaluate_values_of = self.evaluate_values_of

        for key in evaluate_values_of:
            values = pop.get(key)
            values[np.isnan(values)] = np.inf
            pop.set(key, values)

        for ind in pop:
            ind.evaluated.update(evaluate_values_of)

    @staticmethod
    def _normalize_pop(pop: Population, evaluate_values_of, evaluated_pop: Population = None) -> Population:
        """Ensure that the matrices in a Population are two-dimensional"""
        pop_data = {}
        is_evaluated = [False for _ in range(len(pop))]
        for key in (['X']+evaluate_values_of):
            data = pop.get(key, to_numpy=False)

            partial_data = np.zeros((len(data), len(data[0])))*np.nan
            for i, row in enumerate(data):
                if row is not None and len(row) > 0:
                    partial_data[i, :] = row
                    if key != 'X' and not np.any(np.isnan(row)):
                        is_evaluated[i] = True
            data = partial_data

            pop_data[key] = data

        normalized_pop = Population.new(**pop_data)
        for i, is_eval in enumerate(is_evaluated):
            if is_eval:
                normalized_pop[i].evaluated.add('X')
                normalized_pop[i].evaluated.update(evaluate_values_of)

        if evaluated_pop is not None:
            normalized_pop = Population.merge(evaluated_pop, normalized_pop)
        return normalized_pop

    def _store_intermediate(self, problem, pop: Population):
        # Store pymoo population
        self.store_pop(pop)

        # Store cumulative pymoo population
        if self._non_eval_cumulative_pop is not None:
            unique_non_eval_pop = LargeDuplicateElimination().do(self._non_eval_cumulative_pop, pop, to_itself=False)
            self._cumulative_pop = Population.merge(unique_non_eval_pop, pop)
        else:
            self._cumulative_pop = pop
        self.store_pop(self._cumulative_pop, cumulative=True)

        # Store problem-specific results
        self._store_intermediate_problem(problem)

    def store_pop(self, pop: Population, cumulative=False):
        with open(self._get_pop_file_path(self.results_folder, cumulative=cumulative), 'wb') as fp:
            pickle.dump(pop, fp)

        if len(pop) > 0:
            cumulative_str = '_cumulative' if cumulative else ''
            csv_path = os.path.join(self.results_folder, f'pymoo_population{cumulative_str}.csv')
            self.get_pop_as_df(pop).to_csv(csv_path)

    @classmethod
    def get_pop_as_df(cls, pop: Population) -> pd.DataFrame:
        pop = cls._normalize_pop(pop, ['F', 'G', 'H'])

        cols = []
        all_data = []
        for symbol in ['x', 'f', 'g', 'h']:
            data = pop.get(symbol.upper())
            all_data.append(data)
            cols += [f'{symbol}{i}' for i in range(data.shape[1])]

        return pd.DataFrame(columns=cols, data=np.column_stack(all_data))

    @classmethod
    def get_pop_from_df(cls, df: pd.DataFrame):
        symbols = list(set(col[0] for col in df.columns))

        def _get_symbol_data(symbol):
            cols = [col for col in df.columns if col.startswith(symbol)]
            symbol_data = np.zeros((len(df), len(cols)))*np.nan
            for col in cols:
                i_col = int(col[1:])
                symbol_data[:, i_col] = df[col]
            return symbol_data

        data = {sym.upper(): _get_symbol_data(sym) for sym in symbols}
        pop = Population.new(**data)
        cls.pop_set_eval_flags(pop)
        return pop

    @staticmethod
    def pop_set_eval_flags(pop: Population):
        # Set evaluated flags
        def _set_eval(ind: Individual):
            nonlocal n_evaluated
            # Assume evaluated but failed points have Inf as output values
            is_eval = ~np.all(np.isnan(ind.get('F')))
            if is_eval:
                ind.evaluated.update({'X', 'F', 'G', 'H'})
                n_evaluated += 1

        n_evaluated = 0
        pop.apply(_set_eval)
        return n_evaluated

    def _store_intermediate_problem(self, problem):
        if isinstance(problem, ArchOptProblemBase):
            problem.store_results(self.results_folder)

    @classmethod
    def load_pop(cls, results_folder: str, cumulative=True) -> Optional[Population]:
        pop_path = cls._get_pop_file_path(results_folder, cumulative=cumulative)
        if cumulative and not os.path.exists(pop_path):
            pop_path = cls._get_pop_file_path(results_folder)
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
