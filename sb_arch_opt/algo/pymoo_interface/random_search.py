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
from sb_arch_opt.problem import *
from sb_arch_opt.algo.pymoo_interface.api import ArchOptEvaluator
from sb_arch_opt.algo.pymoo_interface.metrics import EHVMultiObjectiveOutput
from sb_arch_opt.sampling import LargeDuplicateElimination, HierarchicalSampling

from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population
from pymoo.util.optimum import filter_optimum
from pymoo.core.initialization import Initialization

try:
    from tpe.optimizer import TPEOptimizer
    HAS_TPE = True
except ImportError:
    HAS_TPE = False

__all__ = ['HAS_TPE', 'RandomSearchAlgorithm']

log = logging.getLogger('sb_arch_opt.random')


class RandomSearchAlgorithm(Algorithm):
    """
    A random search algorithm for benchmarking purposes.
    """

    def __init__(self, n_init: int, **kwargs):
        super().__init__(
            outputs=EHVMultiObjectiveOutput(),
            evaluator=ArchOptEvaluator(extreme_barrier=True),
            **kwargs)
        self.n_init = n_init
        self.sampling = HierarchicalSampling()
        self.initialization = Initialization(
            self.sampling, repair=ArchOptRepair(), eliminate_duplicates=LargeDuplicateElimination())

    def _initialize_infill(self):
        return self.initialization.do(self.problem, self.n_init)

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills

    def _infill(self):
        return self.sampling.do(self.problem, 1)

    def _advance(self, infills=None, is_init=False, **kwargs):
        self.pop = Population.merge(self.pop, infills)

    def _set_optimum(self):
        pop = self.pop
        if self.opt is not None:
            pop = Population.merge(self.opt, pop)
        self.opt = filter_optimum(pop, least_infeasible=True)
