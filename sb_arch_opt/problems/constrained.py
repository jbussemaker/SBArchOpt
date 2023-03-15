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

This test suite contains a set of continuous and mixed-discrete, multi-objective, constrained problems.
"""
from pymoo.problems.multi.osy import OSY
from pymoo.problems.multi.carside import Carside
from pymoo.problems.multi.welded_beam import WeldedBeam
from pymoo.problems.multi.dascmop import DASCMOP7, DIFFICULTIES
from pymoo.problems.single.cantilevered_beam import CantileveredBeam
from sb_arch_opt.problems.problems_base import *

__all__ = ['ArchCantileveredBeam', 'MDCantileveredBeam', 'ArchWeldedBeam', 'MDWeldedBeam', 'ArchCarside', 'MDCarside',
           'ArchOSY', 'MDOSY', 'MODASCMOP', 'MDDASCMOP']


class ArchCantileveredBeam(NoHierarchyWrappedProblem):

    def __init__(self):
        super().__init__(CantileveredBeam())


class MDCantileveredBeam(MixedDiscretizerProblemBase):

    def __init__(self):
        super().__init__(ArchCantileveredBeam(), n_vars_int=2)


class ArchWeldedBeam(NoHierarchyWrappedProblem):
    """Welded beam test problem: https://pymoo.org/problems/multi/welded_beam.html"""

    def __init__(self):
        super().__init__(WeldedBeam())


class MDWeldedBeam(MixedDiscretizerProblemBase):
    """Mixed-discrete version of the welded beam test problem"""

    def __init__(self):
        super().__init__(ArchWeldedBeam(), n_vars_int=2)


class ArchCarside(NoHierarchyWrappedProblem):
    """Carside test problem"""

    def __init__(self):
        super().__init__(Carside())


class MDCarside(MixedDiscretizerProblemBase):
    """Mixed-discrete version of the Carside test problem"""

    def __init__(self):
        super().__init__(ArchCarside(), n_vars_int=4)


class ArchOSY(NoHierarchyWrappedProblem):
    """OSY test problem: https://pymoo.org/problems/multi/osy.html"""

    def __init__(self):
        super().__init__(OSY())


class MDOSY(MixedDiscretizerProblemBase):
    """Mixed-discrete version of the OSY test problem"""

    def __init__(self):
        super().__init__(ArchOSY(), n_vars_int=3)


class MODASCMOP(NoHierarchyWrappedProblem):
    """A particular instance of the DAS-CMOP 3-objective test problem:
    https://pymoo.org/problems/constrained/dascmop.html"""

    def __init__(self):
        super().__init__(DASCMOP7(DIFFICULTIES[0]))


class MDDASCMOP(MixedDiscretizerProblemBase):
    """Mixed-discrete version of the DAS-CMOP test problem"""

    def __init__(self):
        super().__init__(MODASCMOP(), n_opts=3, n_vars_int=15)


if __name__ == '__main__':
    ArchCantileveredBeam().print_stats()
    MDCantileveredBeam().print_stats()
    # ArchWeldedBeam().print_stats()
    # MDWeldedBeam().print_stats()
    # # ArchWeldedBeam().plot_pf()
    # # MDWeldedBeam().plot_pf()

    # ArchCarside().print_stats()
    # MDCarside().print_stats()
    # # ArchCarside().plot_pf()
    # MDCarside().plot_pf()

    # ArchOSY().print_stats()
    # MDOSY().print_stats()
    # # ArchOSY().plot_pf()
    # MDOSY().plot_pf()

    # MODASCMOP().print_stats()
    # MDDASCMOP().print_stats()
    # # MODASCMOP().plot_pf()
    # MDDASCMOP().plot_pf()
