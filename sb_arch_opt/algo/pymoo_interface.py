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
from pymoo.core.algorithm import Algorithm
from pymoo.algorithms.moo.nsga2 import NSGA2
from sb_arch_opt.problem import ArchOptRepair
from sb_arch_opt.sampling import get_init_sampler

__all__ = ['get_repair', 'provision_pymoo', 'get_nsga2']


def get_repair() -> ArchOptRepair:
    """Helper function to get the architecture optimization repair operator"""
    return ArchOptRepair()


def provision_pymoo(algorithm: Algorithm, init_use_lhs=True, set_init=True):
    """
    Provisions a pymoo Algorithm to work correctly for architecture optimization:
    - Sets initializer using a repaired sampler (if `set_init = True`)
    - Sets a repair operator
    """
    if set_init and hasattr(algorithm, 'initialization'):
        algorithm.initialization = get_init_sampler(lhs=init_use_lhs)
    if hasattr(algorithm, 'repair'):
        algorithm.repair = ArchOptRepair()
    return algorithm


def get_nsga2(pop_size: int) -> NSGA2:
    """Returns a preconfigured NSGA2 algorithm"""
    algorithm = NSGA2(pop_size=pop_size, repair=ArchOptRepair())
    provision_pymoo(algorithm)
    return algorithm

