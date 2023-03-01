import numpy as np
from sb_arch_opt.problem import *
from sb_arch_opt.sampling import *
from sb_arch_opt.algo.pymoo_interface import *

from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA


def test_provision():
    ga = GA()
    provision_pymoo(ga)
    assert isinstance(ga.repair, ArchOptRepair)
    assert isinstance(ga.initialization.sampling, RepairedLatinHypercubeSampling)


def test_nsga2(problem: ArchOptProblemBase):
    nsga2 = get_nsga2(pop_size=100)
    result = minimize(problem, nsga2, termination=('n_gen', 10))
    pop = result.pop

    x_imp, _ = problem.correct_x(pop.get('X'))
    assert np.all(pop.get('X') == x_imp)
