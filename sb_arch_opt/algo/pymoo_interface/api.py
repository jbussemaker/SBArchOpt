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
from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.termination.max_eval import MaximumFunctionCallTermination

from sb_arch_opt.sampling import *
from sb_arch_opt.util import capture_log
from sb_arch_opt.problem import ArchOptRepair
from sb_arch_opt.algo.pymoo_interface.metrics import *
from sb_arch_opt.algo.pymoo_interface.md_mating import *
from sb_arch_opt.algo.pymoo_interface.storage_restart import *

__all__ = ['provision_pymoo', 'ArchOptNSGA2', 'get_nsga2', 'initialize_from_previous_results', 'ResultsStorageCallback',
           'ArchOptEvaluator', 'get_default_termination', 'DeltaHVTermination', 'ArchOptEvaluator',
           'load_from_previous_results', 'get_doe_algo', 'DOEAlgorithm']

log = logging.getLogger('sb_arch_opt.pymoo')


def provision_pymoo(algorithm: Algorithm, set_init=True, results_folder=None):
    """
    Provisions a pymoo Algorithm to work correctly for architecture optimization:
    - Sets initializer using a repaired sampler (if `set_init = True`)
    - Sets a repair operator
    - Optionally stores intermediate and final results in some results folder
    - Replace NaN outputs with Inf
    """
    capture_log()

    if set_init and hasattr(algorithm, 'initialization'):
        algorithm.initialization = get_init_sampler()

    if hasattr(algorithm, 'repair'):
        algorithm.repair = ArchOptRepair()

    if results_folder is not None:
        algorithm.callback = ResultsStorageCallback(results_folder, callback=algorithm.callback)

    algorithm.evaluator = ArchOptEvaluator(results_folder=results_folder)

    return algorithm


class ArchOptNSGA2(NSGA2):
    """NSGA2 preconfigured with mixed-variable operators and other architecture optimization measures"""

    def __init__(self,
                 pop_size=100,
                 sampling=HierarchicalSampling(),
                 repair=ArchOptRepair(),
                 mating=MixedDiscreteMating(repair=ArchOptRepair(), eliminate_duplicates=LargeDuplicateElimination()),
                 eliminate_duplicates=LargeDuplicateElimination(),
                 survival=RankAndCrowdingSurvival(),
                 output=EHVMultiObjectiveOutput(),
                 results_folder=None,
                 **kwargs):

        evaluator = ArchOptEvaluator(results_folder=results_folder)
        callback = ResultsStorageCallback(results_folder) if results_folder is not None else None

        super().__init__(pop_size=pop_size, sampling=sampling, repair=repair, mating=mating,
                         eliminate_duplicates=eliminate_duplicates, survival=survival, output=output,
                         evaluator=evaluator, callback=callback, **kwargs)


def get_nsga2(pop_size: int, results_folder=None, **kwargs):
    """Returns a NSGA2 algorithm preconfigured to work with mixed-discrete variables and other architecture optimization
    measures"""
    capture_log()
    return ArchOptNSGA2(pop_size=pop_size, results_folder=results_folder, **kwargs)


class DOEAlgorithm(ArchOptNSGA2):
    """Algorithm that stops after initialization"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_termination()

        self._init_sampling = self.initialization.sampling

    def set_doe_size(self, problem, doe_size: int, **kwargs) -> bool:
        """
        Set the DOE size, also if the algo is already initialized with a prior population.
        Returns whether the DOE size was increased.
        """
        was_increased = False
        self.pop_size = doe_size
        self._set_termination()

        # Modify an existing initial population if set
        if isinstance(self.initialization.sampling, Population):
            init_pop = self.initialization.sampling

            if doe_size < len(init_pop):
                log.info(f'Reducing initialized population size from {len(init_pop)} to {doe_size}; '
                         f'discarding {len(init_pop)-doe_size} points!')
                self.initialization.sampling = init_pop[:doe_size]

            elif doe_size > len(init_pop):
                n_add = doe_size-len(init_pop)

                log.info(f'Adding {n_add} samples to increase DOE size from {len(init_pop)} to {doe_size}')
                add_pop = self._init_sampling(problem, n_add, **kwargs)
                add_pop = self.repair(problem, add_pop)

                self.initialization.sampling = Population.merge(init_pop, add_pop)
                was_increased = True

            log.info(f'New DOE size: {len(self.initialization.sampling)}')

        return was_increased

    def _set_termination(self):
        self.termination = MaximumFunctionCallTermination(n_max_evals=self.pop_size)

    def has_next(self):
        return not self.is_initialized

    def _infill(self):
        raise RuntimeError('Infill should not be called!')


def get_doe_algo(doe_size: int, results_folder=None, **kwargs):
    """Returns an algorithm preconfigured for architecture optimization that will only run a DOE. Useful when
    evaluations is expensive and more inspection is needed before continuing with optimization"""
    capture_log()
    return DOEAlgorithm(pop_size=doe_size, results_folder=results_folder, **kwargs)
