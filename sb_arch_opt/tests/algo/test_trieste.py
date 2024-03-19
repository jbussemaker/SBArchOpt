import os
import pytest
import tempfile
from sb_arch_opt.problem import *
from sb_arch_opt.algo.trieste_interface import *
from sb_arch_opt.problems.constrained import ArchCantileveredBeam
from sb_arch_opt.algo.trieste_interface.algo import ArchOptBayesianOptimizer

def check_dependency():
    return pytest.mark.skipif(not HAS_TRIESTE, reason='Trieste dependencies not installed')


@pytest.mark.skipif(int(os.getenv('RUN_SLOW_TESTS', 0)) != 1, reason='Set RUN_SLOW_TESTS=1 to run slow tests')
def test_slow_tests():
    assert HAS_TRIESTE


@check_dependency()
def test_search_space(problem: ArchOptProblemBase):
    search_space = ArchOptBayesianOptimizer.get_search_space(problem)
    assert search_space.dimension == problem.n_var


@check_dependency()
def test_simple(problem: ArchOptProblemBase):
    assert HAS_TRIESTE

    opt = get_trieste_optimizer(problem, n_init=10, n_infill=1, seed=42)
    assert repr(opt)
    result = opt.run_optimization()

    pop = opt.to_population(result.datasets)
    assert len(pop) == 11


@pytest.mark.skipif(int(os.getenv('RUN_SLOW_TESTS', 0)) != 1, reason='Set RUN_SLOW_TESTS=1 to run slow tests')
@check_dependency()
def test_constrained():
    opt = get_trieste_optimizer(ArchCantileveredBeam(), n_init=10, n_infill=1)
    assert opt.run_optimization()


# @pytest.mark.skipif(int(os.getenv('RUN_SLOW_TESTS', 0)) != 1, reason='Set RUN_SLOW_TESTS=1 to run slow tests')
@pytest.mark.skip('TensorFlow FuncGraph cannot be pickled')
@check_dependency()
def test_store_results_restart(problem: ArchOptProblemBase):
    with tempfile.TemporaryDirectory() as tmp_folder:
        for i in range(2):
            opt = get_trieste_optimizer(problem, n_init=10, n_infill=1+i)
            opt.initialize_from_previous(tmp_folder)
            result = opt.run_optimization(results_folder=tmp_folder)

            pop = opt.to_population(result.datasets)
            assert len(pop) == 11+i


@pytest.mark.skipif(int(os.getenv('RUN_SLOW_TESTS', 0)) != 1, reason='Set RUN_SLOW_TESTS=1 to run slow tests')
@check_dependency()
def test_simple_failing(failing_problem: ArchOptProblemBase):
    opt = get_trieste_optimizer(failing_problem, n_init=10, n_infill=1)
    result = opt.run_optimization()

    pop = opt.to_population(result.datasets)
    assert len(pop) == 5
