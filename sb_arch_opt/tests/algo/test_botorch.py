import os
import pytest
from sb_arch_opt.problem import *
from sb_arch_opt.algo.botorch_interface import *
from sb_arch_opt.problems.discrete import MDBranin
try:
    from ax.service.utils.best_point import get_pareto_optimal_parameters
    from botorch.exceptions.errors import InputDataError
except ImportError:
    pass

def check_dependency():
    return pytest.mark.skipif(not HAS_BOTORCH, reason='BoTorch/Ax dependencies not installed')


@pytest.mark.skipif(int(os.getenv('RUN_SLOW_TESTS', 0)) != 1, reason='Set RUN_SLOW_TESTS=1 to run slow tests')
def test_slow_tests():
    assert HAS_BOTORCH


@check_dependency()
def test_simple(problem: ArchOptProblemBase):
    interface = get_botorch_interface(problem)
    opt = interface.get_optimization_loop(n_init=10, n_infill=1, seed=42)
    opt.full_run()

    pop = interface.get_population(opt)
    assert len(pop) == 11


@check_dependency()
def test_simple_so():
    opt = get_botorch_interface(MDBranin()).get_optimization_loop(n_init=10, n_infill=1)
    opt.full_run()


@check_dependency()
def test_simple_failing(failing_problem: ArchOptProblemBase):
    interface = get_botorch_interface(failing_problem)
    opt = interface.get_optimization_loop(n_init=10, n_infill=1)
    opt.full_run()

    pop = interface.get_population(opt)
    assert len(pop) == 10
