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
from typing import *
from dataclasses import dataclass
import pymoo.core.variable as var
from sb_arch_opt.problem import ArchOptProblemBase

try:
    from smt.surrogate_models.rbf import RBF
    from smt.surrogate_models.krg import KRG
    from smt.surrogate_models.surrogate_model import SurrogateModel
    from smt.applications.mixed_integer import FLOAT, INT, ENUM

    from sb_arch_opt.algo.simple_sbo.algo import *
    from sb_arch_opt.algo.simple_sbo.infill import *
    from sb_arch_opt.algo.simple_sbo.metrics import *

    HAS_SIMPLE_SBO = True
except ImportError:
    HAS_SIMPLE_SBO = False
    get_sbo_termination = lambda *_, **__: None

__all__ = ['check_dependencies', 'HAS_SIMPLE_SBO', 'ModelFactory']


def check_dependencies():
    if not HAS_SIMPLE_SBO:
        raise ImportError(f'simple_sbo dependencies not installed: python setup.py install[simple_sbo]')


@dataclass
class SMTDesignSpaceSpec:
    var_defs: List[dict]  # [{'name': name, 'lb': lb, 'ub', ub}, ...]
    var_types: List[Union[str, Tuple[str, int]]]  # FLOAT, INT, ENUM
    var_limits: List[Union[Tuple[float, float], list]]  # Bounds (options for an enum)
    is_mixed_discrete: bool


class ModelFactory:

    def __init__(self, problem: ArchOptProblemBase):
        self.problem = problem

    def get_smt_design_space_spec(self) -> SMTDesignSpaceSpec:
        """Get information about the design space as needed by SMT and SEGOMOE"""
        var_defs = []
        var_types = []
        var_limits = []
        is_mixed_discrete = False
        xl, xu = self.problem.xl, self.problem.xu
        for i, var_def in enumerate(self.problem.des_vars):
            name = f'x{i}'
            var_defs.append({'name': name, 'lb': xl[i], 'ub': xu[i]})

            if isinstance(var_def, var.Real):
                var_types.append(FLOAT)
                var_limits.append(var_def.bounds)

            elif isinstance(var_def, var.Integer):
                is_mixed_discrete = True
                var_types.append(INT)
                var_limits.append(var_def.bounds)

            elif isinstance(var_def, var.Binary):
                is_mixed_discrete = True
                var_types.append(INT)
                var_limits.append([0, 1])

            elif isinstance(var_def, var.Choice):
                is_mixed_discrete = True
                var_types.append((ENUM, len(var_def.options)))
                var_limits.append(var_def.options)

            else:
                raise RuntimeError(f'Unsupported design variable type: {var_def!r}')

        return SMTDesignSpaceSpec(
            var_defs=var_defs,
            var_types=var_types,
            var_limits=var_limits,
            is_mixed_discrete=is_mixed_discrete,
        )

    @staticmethod
    def get_rbf_model():
        return RBF(print_global=False, d0=1., poly_degree=-1, reg=1e-10)

    @staticmethod
    def get_kriging_model():
        return KRG(print_global=False)
