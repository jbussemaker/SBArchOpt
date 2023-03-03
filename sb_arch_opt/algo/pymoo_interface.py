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
from typing import Optional
from pymoo.core.result import Result
from pymoo.core.callback import Callback
from pymoo.core.algorithm import Algorithm
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.initialization import Initialization

from sb_arch_opt.util import capture_log
from sb_arch_opt.sampling import get_init_sampler
from sb_arch_opt.problem import ArchOptRepair, ArchOptProblemBase

__all__ = ['get_repair', 'provision_pymoo', 'get_nsga2', 'initialize_from_previous_results', 'ResultsStorageCallback',
           'ExtremeBarrierEvaluator']

log = logging.getLogger('sb_arch_opt.pymoo')


def get_repair() -> ArchOptRepair:
    """Helper function to get the architecture optimization repair operator"""
    return ArchOptRepair()


def provision_pymoo(algorithm: Algorithm, init_use_lhs=True, set_init=True, results_folder=None,
                    enable_extreme_barrier=True):
    """
    Provisions a pymoo Algorithm to work correctly for architecture optimization:
    - Sets initializer using a repaired sampler (if `set_init = True`)
    - Sets a repair operator
    - Optionally stores intermediate and final results in some results folder
    - Optionally enables extreme-barrier for dealing with hidden constraints (replace NaN with Inf)
    """
    capture_log()

    if set_init and hasattr(algorithm, 'initialization'):
        algorithm.initialization = get_init_sampler(lhs=init_use_lhs)

    if hasattr(algorithm, 'repair'):
        algorithm.repair = ArchOptRepair()

    if results_folder is not None:
        algorithm.callback = ResultsStorageCallback(results_folder, callback=algorithm.callback)

    if enable_extreme_barrier:
        algorithm.evaluator = ExtremeBarrierEvaluator()

    return algorithm


def get_nsga2(pop_size: int, results_folder=None) -> NSGA2:
    """Returns a preconfigured NSGA2 algorithm"""
    algorithm = NSGA2(pop_size=pop_size, repair=ArchOptRepair())
    provision_pymoo(algorithm, results_folder=results_folder)
    return algorithm


def initialize_from_previous_results(algorithm: Algorithm, problem: ArchOptProblemBase, result_folder: str) -> bool:
    """Initialize an Algorithm from previously stored results"""
    capture_log()

    if not hasattr(algorithm, 'initialization'):
        raise RuntimeError(f'Algorithm has no initialization step, cannot set initial population: {algorithm!r}')

    # Try to load using problem-specific function first
    population = problem.load_previous_results(result_folder)
    if population is not None and len(population) > 0:
        log.info(f'Previous results loaded from problem results: {len(population)} design points')

    # Try to load from pymoo storage
    else:
        population = ResultsStorageCallback.load_pop(result_folder)
        if population is not None and len(population) > 0:
            log.info(f'Previous results loaded from pymoo results: {len(population)} design points')
        else:
            log.info(f'No previous population found, not changing initialization strategy')
            return False

    # Set evaluated flag
    population.apply(lambda ind: ind.evaluated.update({'X', 'F', 'G', 'H'}))

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
        super().__init__()

    def initialize(self, algorithm: Algorithm):
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
        self.store_pop(algorithm)

        # Store problem-specific results
        problem = algorithm.problem
        if isinstance(problem, ArchOptProblemBase):
            problem.store_results(self.results_folder, final=final)

    def store_pop(self, algorithm: Algorithm):
        if not hasattr(algorithm, 'pop'):
            raise RuntimeError(f'Algorithm has no population (pop property): {algorithm!r}')
        pop: Population = algorithm.pop

        with open(self._get_pop_file_path(self.results_folder), 'wb') as fp:
            pickle.dump(pop, fp)

    def _store_results(self, result: Result):
        with open(os.path.join(self.results_folder, 'pymoo_results.pkl'), 'wb') as fp:
            pickle.dump(result, fp)

    @classmethod
    def load_pop(cls, results_folder: str) -> Optional[Population]:
        pop_path = cls._get_pop_file_path(results_folder)
        if not os.path.exists(pop_path):
            return

        with open(pop_path, 'rb') as fp:
            pop = pickle.load(fp)

        if not isinstance(pop, Population):
            raise ValueError(f'Loaded population not of type Population ({pop_path}): {pop!r}')
        return pop

    @staticmethod
    def _get_pop_file_path(results_folder) -> str:
        return os.path.join(results_folder, 'pymoo_population.pkl')

    def __call__(self, *args, **kwargs):
        super().__call__(*args, **kwargs)
        if self.callback is not None:
            self.callback(*args, **kwargs)


class ExtremeBarrierEvaluator(Evaluator):
    """Evaluator that applies the extreme barrier approach for dealing with hidden constraints: replace NaN with Inf"""

    def _eval(self, problem, pop, evaluate_values_of, **kwargs):
        super()._eval(problem, pop, evaluate_values_of, **kwargs)

        for key in ['F', 'G', 'H']:
            values = pop.get(key)
            values[np.isnan(values)] = np.inf
            pop.set(key, values)

        return pop
