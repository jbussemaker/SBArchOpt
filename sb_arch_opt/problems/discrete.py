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

This test suite contains a set of mixed-discrete single-objective problems.
"""
import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.population import Population
from pymoo.core.variable import Real, Choice, Integer
from sb_arch_opt.problems.continuous import Branin
from sb_arch_opt.problems.problems_base import *

__all__ = ['MDBranin', 'AugmentedMDBranin', 'MDGoldstein', 'MunozZunigaToy', 'Halstrup04']


class MDBranin(Branin):
    """
    Mixed-discrete version of the Branin problem that introduces two discrete variables that transform the original
    Branin space in different ways.

    Implementation based on:
    Pelamatti 2020: "Overview and Comparison of Gaussian Process-Based Surrogate Models for Mixed Continuous and
    Discrete Variables", section 4.1
    """

    _des_vars = [
        Real(bounds=(0, 1)), Real(bounds=(0, 1)),
        Choice(options=[0, 1]), Choice(options=[0, 1]),
    ]

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):

        for i in range(x.shape[0]):
            h = self._h(x[i, 0], x[i, 1])

            z1, z2 = x[i, 2], x[i, 3]
            if z1 == 0:
                f_out[i, 0] = h if z2 == 0 else (.4*h + 1.1)
            else:
                f_out[i, 0] = (-.75*h + 5.2) if z2 == 0 else (-.5*h - 2.1)

    def plot(self, z1=0, z2=0, show=True):
        xx, yy = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
        x = np.column_stack([xx.ravel(), yy.ravel(), np.ones((xx.size,))*z1, np.ones((xx.size,))*z2])

        out = Population.new(X=x)
        out = self.evaluate(x, out)
        ff = out.reshape(xx.shape)

        plt.figure(), plt.title('Discrete Branin: $z_1$ = %d, $z_2$ = %d' % (z1, z2))
        plt.colorbar(plt.contourf(xx, yy, ff, 50, cmap='viridis'))
        plt.xlabel('$x_1$'), plt.ylabel('$x_2$')
        plt.xlim([0, 1]), plt.ylim([0, 1])

        if show:
            plt.show()


class AugmentedMDBranin(MDBranin):
    """
    Mixed-discrete version of the Branin function with more continuous input dimensions.

    Implementation based on:
    Pelamatti 2020: "Overview and Comparison of Gaussian Process-Based Surrogate Models for Mixed Continuous and
    Discrete Variables", section 4.2
    """

    _des_vars = [Real(bounds=(0, 1)) if i < 10 else Choice(options=[0, 1]) for i in range(12)]

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        for i in range(x.shape[0]):
            h = sum([self._h(x[i, j], x[i, j+1]) for j in range(0, 10, 2)])

            z1, z2 = x[i, 2], x[i, 3]
            if z1 == 0:
                f_out[i, 0] = h if z2 == 0 else (.4*h + 1.1)
            else:
                f_out[i, 0] = (-.75*h + 5.2) if z2 == 0 else (-.5*h - 2.1)


class MDGoldstein(NoHierarchyProblemBase):
    """
    Mixed-discrete version of the Goldstein problem that introduces two discrete variables that transform the original
    design space in different ways.

    Implementation based on:
    Pelamatti 2020: "Overview and Comparison of Gaussian Process-Based Surrogate Models for Mixed Continuous and
    Discrete Variables", section 4.1
    """

    def __init__(self):
        des_vars = [
            Real(bounds=(0, 100)), Real(bounds=(0, 100)),
            Integer(bounds=(0, 2)), Integer(bounds=(0, 2)),
        ]
        super().__init__(des_vars)

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        _x3 = [20, 50, 80]
        _x4 = [20, 50, 80]

        for i in range(x.shape[0]):
            x3, x4 = _x3[int(x[i, 2])], _x4[int(x[i, 3])]
            f_out[i, 0] = self.h(x[i, 0], x[i, 1], x3, x4)

    @staticmethod
    def h(x1, x2, x3, x4, z3=4, z4=3):
        return sum([
            53.3108,
            .184901 * x1,
            -5.02914 * x1**3 * 1e-6,
            7.72522 * x1**z3 * 1e-8,
            0.0870775 * x2,
            -0.106959 * x3,
            7.98772 * x3**z4 * 1e-6,
            0.00242482 * x4,
            1.32851 * x4**3 * 1e-6,
            -0.00146393 * x1 * x2,
            -0.00301588 * x1 * x3,
            -0.00272291 * x1 * x4,
            0.0017004 * x2 * x3,
            0.0038428 * x2 * x4,
            -0.000198969 * x3 * x4,
            1.86025 * x1 * x2 * x3 * 1e-5,
            -1.88719 * x1 * x2 * x4 * 1e-6,
            2.50923 * x1 * x3 * x4 * 1e-5,
            -5.62199 * x2 * x3 * x4 * 1e-5,
        ])

    def plot(self, z1=0, z2=0, show=True):
        xx, yy = np.meshgrid(np.linspace(0, 100, 50), np.linspace(0, 100, 50))
        x = np.column_stack([xx.ravel(), yy.ravel(), np.ones((xx.size,))*z1, np.ones((xx.size,))*z2])

        out = Population.new(X=x)
        out = self.evaluate(x, out)
        ff = out.reshape(xx.shape)

        plt.figure(), plt.title('Discrete Goldstein: $z_1$ = %d, $z_2$ = %d' % (z1, z2))
        plt.colorbar(plt.contourf(xx, yy, ff, 50, cmap='viridis'))
        plt.xlabel('$x_1$'), plt.ylabel('$x_2$')
        plt.xlim([0, 100]), plt.ylim([0, 100])

        if show:
            plt.show()


class MunozZunigaToy(NoHierarchyProblemBase):
    """
    Toy problem from:
    Munoz Zuniga 2020: "Global optimization for mixed categorical-continuous variables based on Gaussian process models
    with a randomized categorical space exploration step", 10.1080/03155986.2020.1730677

    Minimum: -2.329605
    """

    def __init__(self):
        des_vars = [Real(bounds=(0, 1)), Integer(bounds=(0, 9))]
        super().__init__(des_vars)

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        f = [
            lambda x_: np.cos(3.6 * np.pi * (x_ - 2)) + x_ - 1,
            lambda x_: 2 * np.cos(1.1 * np.pi * np.exp(x_)) - .5 * x_ + 2,
            lambda x_: np.cos(2 * np.pi * x_) + .5 * x_,
            lambda x_: x_ * (np.cos(3.4 * np.pi * (x_ - 1)) - .5 * (x_ - 1)),
            lambda x_: -.5 * x_ ** 2,
            lambda x_: 2 * np.cos(.25 * np.pi * np.exp(-x_ ** 4)) ** 2 - .5 * x_ + 1,
            lambda x_: x_ * np.cos(3.4 * np.pi * x_) - .5 * x_ + 1,
            lambda x_: x_ * (-np.cos(7 * .5 * np.pi * x_) - .5 * x_) + 2,
            lambda x_: -.5 * x_ ** 5 + 1,
            lambda x_: -np.cos(5 * .5 * np.pi * x_) ** 2 * np.sqrt(x_) - .5 * np.log(x_ + .5) - 1.3,
        ]

        for i in range(10):
            i_x = x[:, 1] == i
            if len(np.where(i_x)[0] > 0):
                f_out[i_x, 0] = f[i](x[i_x, 0])

    def plot(self, show=True):
        x = np.linspace(0, 1, 100)
        z = np.array(list(range(10)))
        xx, zz = np.meshgrid(x, z)
        xx = xx.ravel()
        zz = zz.ravel()

        out = Population.new(X=np.column_stack([xx, zz]))
        out = self.evaluate(out.get('X'), out)
        f = out[:]

        plt.figure(), plt.title('Munoz-Zuniga Toy Problem')
        for i in z:
            i_x = zz == i
            plt.plot(x, f[i_x], linewidth=1, label='$z = %d$' % (i+1,))
        plt.xlim([0, 1]), plt.xlabel('$x$'), plt.ylabel('$f$'), plt.legend()

        if show:
            plt.show()


class Halstrup04(NoHierarchyProblemBase):
    """
    Fourth mixed-discrete test problem from:
    Halstrup 2016, "Black-Box Optimization of Mixed Discrete-Continuous Optimization Problems"

    Minimum: 1.7025 (https://mixed-optimization-benchmark.github.io/cases/hal04/)
    Original report contains an error
    """

    f_aux_mod = [
        [  # a
            [(1., .0), (1., .2)],  # d
            [(.9, .0), (1., .25)],  # e
        ],
        [  # b
            [(1., .5), (.8, .0)],  # d
            [(.5, .0), (1., .8)],  # e
        ],
        [  # c
            [(1., .9), (.5, .0)],  # d
            [(1., 1.), (1., 1.25)],  # e
        ],
    ]

    def __init__(self):
        des_vars = [
            Real(bounds=(0, 1)), Real(bounds=(0, 1)), Real(bounds=(0, 1)), Real(bounds=(0, 1)), Real(bounds=(0, 1)),
            Choice(options=[0, 1, 2]), Choice(options=[0, 1]), Choice(options=[0, 1]),
        ]
        super().__init__(des_vars)

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        x_ = x[:, :5]
        z_ = x[:, 5:].astype(int)

        d = 8
        x2_term = 2**(np.arange(5)/(d-1))
        f_aux = np.sum((5*x_+(1-x_))**2*x2_term, axis=1)-2.75

        for i in range(z_.shape[0]):
            f_aux_mul, f_aux_add = self.f_aux_mod[z_[i, 0]][z_[i, 1]][z_[i, 2]]
            f_out[i, 0] = f_aux[i]*f_aux_mul + f_aux_add


if __name__ == '__main__':
    MDBranin().print_stats()
    AugmentedMDBranin().print_stats()
    MDGoldstein().print_stats()
    MunozZunigaToy().print_stats()
    Halstrup04().print_stats()

    # MDBranin().plot_pf()
    MDBranin().plot_design_space()
    # AugmentedMDBranin().plot_pf()
    # MDGoldstein().plot_pf()
    # MunozZunigaToy().plot_pf()
    # Halstrup04().plot_pf()
