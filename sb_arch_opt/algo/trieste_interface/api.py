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
from sb_arch_opt.problem import ArchOptProblemBase
from sb_arch_opt.algo.trieste_interface.algo import *


__all__ = ['get_trieste_optimizer', 'HAS_TRIESTE']


def get_trieste_optimizer(problem: ArchOptProblemBase, n_init: int, n_infill: int, pof: float = .5):
    """
    Gets the main interface to Trieste. Use the `run_optimization` method to run the DOE and infill loops.
    """
    check_dependencies()
    return ArchOptBayesianOptimizer(problem, n_init, n_infill, pof=pof)
