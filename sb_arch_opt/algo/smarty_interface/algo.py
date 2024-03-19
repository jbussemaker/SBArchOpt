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
import numpy as np
from pymoo.core.population import Population
from sb_arch_opt.problem import ArchOptProblemBase

try:
    from smarty.problem.optimizationProblem import CustomOptProb
    from smarty.optimize.sbo import SBO
    from smarty.optimize import convergenceCriteria as CC
    from smarty import Log

    HAS_SMARTY = True
except ImportError:
    HAS_SMARTY = False

__all__ = ['HAS_SMARTY', 'check_dependencies', 'SMARTyArchOptInterface']

log = logging.getLogger('sb_arch_opt.smarty')


def check_dependencies():
    if not HAS_SMARTY:
        raise ImportError('SMARTy not installed!')


class SMARTyArchOptInterface:
    """
    Interface class to SMARTy SBO.
    """

    def __init__(self, problem: ArchOptProblemBase, n_init: int, n_infill: int):
        check_dependencies()
        Log.SetLogLevel(1)

        self._problem = problem
        self._n_init = n_init
        self._n_infill = n_infill
        self._has_g = problem.n_ieq_constr > 0

        self._opt_prob = None
        self._optimizer = None

    @property
    def problem(self):
        return self._problem

    @property
    def opt_prob(self):
        if self._opt_prob is None:
            bounds = np.column_stack([self._problem.xl, self._problem.xu])

            problem_structure = {'objFuncs': {f'f{i}': 'F' for i in range(self._problem.n_obj)}}
            if self._has_g:
                problem_structure['constrFuncs'] = {f'g{i}': 'F' for i in range(self._problem.n_ieq_constr)}

            self._opt_prob = CustomOptProb(bounds=bounds, problemStructure=problem_structure,
                                           customFunctionHandler=self._evaluate, vectorized=True,
                                           problemName=repr(self._problem))
        return self._opt_prob

    @property
    def optimizer(self) -> 'SBO':
        if self._optimizer is None:
            self._optimizer = sbo = SBO(self.opt_prob)

            for key, settings in sbo._settingsDOE.items():
                settings['nSamples'] = self._n_init

        return self._optimizer

    def _evaluate(self, x, _):
        out = self._problem.evaluate(x, return_as_dictionary=True)

        outputs = {}
        for i in range(self._problem.n_obj):
            outputs[f'objFuncs/f{i}/F'] = out['F'][:, i]
        for i in range(self._problem.n_ieq_constr):
            outputs[f'constrFuncs/g{i}/F'] = out['G'][:, i]
        return outputs

    @property
    def pop(self) -> Population:
        f, g, idx = self.opt_prob.CreateObjAndConstrMatrices()
        x = self.opt_prob.inputMatrix[idx]

        kwargs = {'X': x, 'F': f}
        if self._problem.n_ieq_constr > 0:
            kwargs['G'] = g
        return Population.new(**kwargs)

    def _get_infill(self):
        if self._problem.n_obj == 1:
            return 'EI'

        elif self._problem.n_obj == 2:
            return 'EHVI2D'
        return 'WFGEHVI'

    def _get_convergence(self):
        if self._problem.n_obj == 1:
            return [
                CC.AbsOptXChange(1e-8, 5),
                CC.MinInfillValue(1e-6, 4),
            ]

        return [
            CC.StallIterations(5),
        ]

    def optimize(self):
        """Run the optimization loop for n_infill infill points (on top on the initialization points)"""
        optimizer = self.optimizer
        optimizer.Optimize(
            nMaxIters=self._n_infill,
            listConvCrit=self._get_convergence(),
            infillMethod=self._get_infill(),
        )
