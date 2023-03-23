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
import logging
from pymoo.core.algorithm import Algorithm
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
           'load_from_previous_results', 'get_doe_algo']

log = logging.getLogger('sb_arch_opt.pymoo')


def provision_pymoo(algorithm: Algorithm, set_init=True, results_folder=None, enable_extreme_barrier=True):
    """
    Provisions a pymoo Algorithm to work correctly for architecture optimization:
    - Sets initializer using a repaired sampler (if `set_init = True`)
    - Sets a repair operator
    - Optionally stores intermediate and final results in some results folder
    - Optionally enables extreme-barrier for dealing with hidden constraints (replace NaN with Inf)
    """
    capture_log()

    if set_init and hasattr(algorithm, 'initialization'):
        algorithm.initialization = get_init_sampler()

    if hasattr(algorithm, 'repair'):
        algorithm.repair = ArchOptRepair()

    if results_folder is not None:
        algorithm.callback = ResultsStorageCallback(results_folder, callback=algorithm.callback)

    if results_folder is not None or enable_extreme_barrier:
        algorithm.evaluator = ArchOptEvaluator(extreme_barrier=enable_extreme_barrier, results_folder=results_folder)

    return algorithm


class ArchOptNSGA2(NSGA2):
    """NSGA2 preconfigured with mixed-variable operators and other architecture optimization measures"""

    def __init__(self,
                 pop_size=100,
                 sampling=HierarchicalRandomSampling(),
                 repair=ArchOptRepair(),
                 mating=MixedDiscreteMating(repair=ArchOptRepair(), eliminate_duplicates=LargeDuplicateElimination()),
                 eliminate_duplicates=LargeDuplicateElimination(),
                 survival=RankAndCrowdingSurvival(),
                 output=EHVMultiObjectiveOutput(),
                 results_folder=None,
                 **kwargs):

        evaluator = ArchOptEvaluator(extreme_barrier=True, results_folder=results_folder)
        callback = ResultsStorageCallback(results_folder) if results_folder is not None else None

        super().__init__(pop_size=pop_size, sampling=sampling, repair=repair, mating=mating,
                         eliminate_duplicates=eliminate_duplicates, survival=survival, output=output,
                         evaluator=evaluator, callback=callback, **kwargs)


def get_nsga2(pop_size: int, results_folder=None, **kwargs):
    """Returns a NSGA2 algorithm preconfigured to work with mixed-discrete variables and other architecture optimization
    measures"""
    capture_log()
    return ArchOptNSGA2(pop_size=pop_size, results_folder=results_folder, **kwargs)


def get_doe_algo(doe_size: int, results_folder=None, **kwargs):
    """Returns an algorithm preconfigured for architecture optimization that will only run a DOE. Useful when
    evaluations is expensive and more inspection is needed before continuing with optimization"""
    algo = get_nsga2(pop_size=doe_size, results_folder=results_folder, **kwargs)
    algo.termination = MaximumFunctionCallTermination(n_max_evals=doe_size)
    return algo
