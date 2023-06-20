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
            evaluator=ArchOptEvaluator(),
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
