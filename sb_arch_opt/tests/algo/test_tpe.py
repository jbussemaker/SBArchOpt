import os
import pytest
import tempfile
from pymoo.optimize import minimize
from sb_arch_opt.algo.tpe_interface import *
from sb_arch_opt.problems.discrete import MDBranin
from sb_arch_opt.problems.hidden_constraints import Alimo

def check_dependency():
    return pytest.mark.skipif(not HAS_TPE, reason='TPE dependencies not installed')


@pytest.mark.skipif(int(os.getenv('RUN_SLOW_TESTS', 0)) != 1, reason='Set RUN_SLOW_TESTS=1 to run slow tests')
def test_slow_tests():
    assert HAS_TPE


@check_dependency()
def test_simple():
    assert HAS_TPE

    tpe = ArchTPEInterface(MDBranin())
    x, f = tpe.optimize(n_init=1, n_infill=1)
    assert x.shape == (2, 4)
    assert f.shape == (2,)


@check_dependency()
def test_md_branin():
    tpe = ArchTPEInterface(MDBranin())
    x, f = tpe.optimize(n_init=20, n_infill=10)
    assert x.shape == (30, 4)
    assert f.shape == (30,)


@check_dependency()
def test_algorithm():
    algo = TPEAlgorithm(n_init=20)
    result = minimize(MDBranin(), algo, ('n_eval', 30), copy_algorithm=False)
    assert len(result.pop) == 30
    assert algo.opt is not None


@check_dependency()
def test_failed_evaluations():
    minimize(Alimo(), TPEAlgorithm(n_init=20), ('n_eval', 100))


@check_dependency()
def test_store_results_restart():
    problem = MDBranin()
    with tempfile.TemporaryDirectory() as tmp_folder:
        for i in range(2):
            tpe = TPEAlgorithm(n_init=10, results_folder=tmp_folder)
            initialize_from_previous_results(tpe, problem, tmp_folder)

            n_eval = 11+i
            result = minimize(problem, tpe, termination=('n_eval', n_eval))
            assert len(result.pop) == 10+(i+1)
