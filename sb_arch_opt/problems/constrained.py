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
import numpy as np
from pymoo.core.variable import Real
from pymoo.problems.multi.osy import OSY
from pymoo.problems.multi.carside import Carside
from pymoo.problems.multi.welded_beam import WeldedBeam
from pymoo.problems.multi.dascmop import DASCMOP7, DIFFICULTIES
from pymoo.problems.single.cantilevered_beam import CantileveredBeam
from sb_arch_opt.problems.problems_base import *

__all__ = ['ArchCantileveredBeam', 'MDCantileveredBeam', 'ArchWeldedBeam', 'MDWeldedBeam', 'ArchCarside', 'MDCarside',
           'ArchOSY', 'MDOSY', 'MODASCMOP', 'MDDASCMOP', 'ConBraninProd', 'ConBraninGomez']


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


class ConBraninBase(NoHierarchyProblemBase):
    """
    Constrained Branin function from:
    Parr, J., Holden, C.M., Forrester, A.I. and Keane, A.J., 2010. Review of efficient surrogate infill sampling
    criteria with constraint handling.
    """

    def __init__(self):
        des_vars = [
            Real(bounds=(-5, 10)),
            Real(bounds=(0, 15)),
        ]
        super().__init__(des_vars, n_ieq_constr=1)

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):

        x_norm = (x+[5, 0])/15
        for i in range(x.shape[0]):
            f_out[i, 0] = self._h(x[i, 0], x[i, 1])
            g_out[i, 0] = self._g(x_norm[i, 0], x_norm[i, 1])

    @staticmethod
    def _h(x1, x2):
        t1 = (x2 - (5.1/(4*np.pi**2))*x1**2 + (5/np.pi)*x1 - 6)**2
        t2 = 10*(1-1/(8*np.pi))*np.cos(x1) + 10
        return t1 + t2 + 5*x2

    def plot(self, show=True):
        import matplotlib.pyplot as plt

        xx1, xx2 = np.meshgrid(np.linspace(-5, 10, 100), np.linspace(0, 15, 100))
        out = self.evaluate(np.column_stack([xx1.ravel(), xx2.ravel()]), return_as_dictionary=True)
        zz = out['F'][:, 0]
        zz[out['G'][:, 0] > 0] = np.nan

        plt.figure(), plt.title(f'{self.__class__.__name__}')
        plt.colorbar(plt.contourf(xx1, xx2, zz.reshape(xx1.shape), 50, cmap='inferno'))
        plt.xlabel('$x_1$'), plt.ylabel('$x_2$')

        if show:
            plt.show()

    def _g(self, x1, x2):
        raise NotImplementedError


class ConBraninProd(ConBraninBase):
    """Constrained Branin problem with the product constraint (Eq. 14)"""

    def _g(self, x1, x2):
        return .2 - x1*x2


class ConBraninGomez(ConBraninBase):
    """Constrained Branin problem with the Gomez#3 constraint (Eq. 15)"""

    def _g(self, x1, x2):
        x1 = x1*2-1
        x2 = x2*2-1
        g = (4 - 2.1*x1**2 + (x1**4)/3)*x1**2 + x1*x2 + (-4 + 4*x2**2)*x2**2 + 3*np.sin(6*(1-x1)) + 3*np.sin(6*(1-x2))
        return 6-g


if __name__ == '__main__':
    # ArchCantileveredBeam().print_stats()
    # ArchCantileveredBeam().plot_design_space()
    # MDCantileveredBeam().print_stats()
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

    # ConBraninProd().plot()
    ConBraninProd().print_stats()
    # ConBraninGomez().plot()
    ConBraninGomez().print_stats()
