import os
import pytest
from sb_arch_opt.problem import *
from sb_arch_opt.algo.hebo_interface import *
from sb_arch_opt.problems.md_mo import MOZDT1
from sb_arch_opt.problems.constrained import ArchCantileveredBeam
from sb_arch_opt.algo.hebo_interface.algo import HEBOArchOptInterface

def check_dependency():
    return pytest.mark.skipif(not HAS_HEBO, reason='HEBO dependencies not installed')


@pytest.mark.skipif(int(os.getenv('RUN_SLOW_TESTS', 0)) != 1, reason='Set RUN_SLOW_TESTS=1 to run slow tests')
def test_slow_tests():
    assert HAS_HEBO


@check_dependency()
def test_design_space(problem: ArchOptProblemBase):
    hebo = HEBOArchOptInterface(problem, n_init=10)
    design_space = hebo.design_space
    assert len(design_space.paras) == problem.n_var


@check_dependency()
def test_simple():
    assert HAS_HEBO

    n_init = 30
    hebo = get_hebo_optimizer(MOZDT1(), n_init=30, seed=42)
    hebo.optimize(n_infill=2)

    pop = hebo.pop
    assert len(pop) == n_init+2


@check_dependency()
def test_constrained():
    opt = get_hebo_optimizer(ArchCantileveredBeam(), n_init=20)
    opt.optimize(n_infill=1)
    assert len(opt.pop) == 21


@check_dependency()
def test_simple_failing(failing_problem: ArchOptProblemBase):
    hebo = get_hebo_optimizer(failing_problem, n_init=20)
    hebo.optimize(n_infill=1)

    pop = hebo.pop
    assert len(pop) == 10
