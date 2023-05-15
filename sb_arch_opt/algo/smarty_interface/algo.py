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
        raise ImportError(f'SMARTy not installed!')


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
