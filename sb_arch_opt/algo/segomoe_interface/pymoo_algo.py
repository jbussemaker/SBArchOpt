"""
MIT License

Copyright: (c) 2024, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
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
from sb_arch_opt.algo.pymoo_interface import *
from sb_arch_opt.algo.pymoo_interface.metrics import EHVMultiObjectiveOutput

from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population
from pymoo.util.optimum import filter_optimum
from pymoo.termination.max_eval import MaximumFunctionCallTermination

from sb_arch_opt.algo.segomoe_interface.algo import SEGOMOEInterface, check_dependencies

__all__ = ['SEGOMOEAlgorithm']

log = logging.getLogger('sb_arch_opt.segomoe')


class SEGOMOEAlgorithm(Algorithm):
    """
    Algorithm that wraps the SEGOMOE interface.

    The population state is managed here, and each time infill points are asked for the SEGOMOE population is updated
    from the algorithm population.
    """

    def __init__(self, segomoe: SEGOMOEInterface, output=EHVMultiObjectiveOutput(), **kwargs):
        check_dependencies()
        super().__init__(output=output, **kwargs)
        self.segomoe = segomoe

        self.termination = MaximumFunctionCallTermination(self.segomoe.n_init + self.segomoe.n_infill)
        self._store_intermediate_results()

        self.initialization = None  # Enable DOE override

    def _initialize_infill(self):
        if self.initialization is not None:
            return self.initialization.do(self.problem, self.segomoe.n_init, algorithm=self)
        return self._infill()

    def _initialize_advance(self, infills=None, **kwargs):
        self._advance(infills=infills, **kwargs)

    def has_next(self):
        if not super().has_next():
            return False

        self._infill_set_pop()
        if not self.segomoe.optimization_has_ask():
            return False
        return True

    def _infill(self):
        self._infill_set_pop()
        x_infill = self.segomoe.optimization_ask()
        off = Population.new(X=x_infill) if x_infill is not None else Population.new()

        # Stop if no new offspring is generated
        if len(off) == 0:
            self.termination.force_termination = True

        return off

    def _infill_set_pop(self):
        if self.pop is None or len(self.pop) == 0:
            self.segomoe.set_pop(pop=None)
        else:
            self.segomoe.set_pop(self.pop)

    def _advance(self, infills=None, **kwargs):
        if infills is not None:
            self.segomoe.optimization_tell_pop(infills)
        self.pop = self.segomoe.pop

    def _set_optimum(self):
        pop = self.pop
        i_failed = ArchOptProblemBase.get_failed_points(pop)
        valid_pop = pop[~i_failed]
        if len(valid_pop) == 0:
            self.opt = Population.new(X=[None])
        else:
            self.opt = filter_optimum(valid_pop, least_infeasible=True)

    def _store_intermediate_results(self):
        """Enable intermediate results storage to support restarting"""
        results_folder = self.segomoe.results_folder
        self.evaluator = ArchOptEvaluator(results_folder=results_folder)
        self.callback = ResultsStorageCallback(results_folder, callback=self.callback)

    def initialize_from_previous_results(self, problem: ArchOptProblemBase, results_folder: str = None) -> bool:
        """Initialize the SBO algorithm from previously stored results"""
        if results_folder is None:
            results_folder = self.segomoe.results_folder

        population = load_from_previous_results(problem, results_folder)
        if population is None:
            return False

        self.pop = population
        self._set_optimum()
        return True
