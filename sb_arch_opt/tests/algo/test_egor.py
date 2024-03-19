import os
import pytest
import tempfile
from sb_arch_opt.problem import *
from sb_arch_opt.algo.egor_interface import *
from sb_arch_opt.problems.discrete import MDBranin
from sb_arch_opt.problems.constrained import ArchCantileveredBeam, MDCantileveredBeam
from sb_arch_opt.algo.egor_interface.algo import EgorArchOptInterface

def check_dependency():
    return pytest.mark.skipif(not HAS_EGOBOX, reason="Egor dependencies not installed")


@pytest.mark.skipif(int(os.getenv('RUN_SLOW_TESTS', 0)) != 1, reason='Set RUN_SLOW_TESTS=1 to run slow tests')
def test_slow_tests():
    assert HAS_EGOBOX


@check_dependency()
def test_design_space(problem: ArchOptProblemBase):
    egor = EgorArchOptInterface(problem, n_init=10)
    design_space = egor.design_space
    assert len(design_space) == problem.n_var


@check_dependency()
def test_simple():
    assert HAS_EGOBOX
    n_init = 30
    n_infill = 2
    egor = get_egor_optimizer(MDBranin(), n_init=n_init, seed=42)
    egor.minimize(n_infill=n_infill)
    pop = egor.pop
    assert len(pop) == n_init + n_infill


@check_dependency()
def test_restart():
    with tempfile.TemporaryDirectory() as temp_dir:
        n_init = 30
        n_infill = 2
        egor = get_egor_optimizer(MDBranin(), n_init=n_init, results_folder=temp_dir, seed=42)
        egor.minimize(n_infill=n_infill)
        pop = egor.pop
        assert len(pop) == n_init + n_infill

        egor2 = get_egor_optimizer(MDBranin(), n_init=n_init, results_folder=temp_dir, seed=42)
        egor2.initialize_from_previous(results_folder=temp_dir)
        pop = egor2.pop
        assert len(pop) == n_init + n_infill

        n_infill2 = 10
        egor2.minimize(n_infill=n_infill2)
        pop = egor2.pop
        assert len(pop) == n_init + n_infill2


@check_dependency()
def test_constrained():
    opt = get_egor_optimizer(ArchCantileveredBeam(), n_init=20)
    opt.minimize(n_infill=1)
    assert len(opt.pop) == 21


@check_dependency()
def test_md_constrained():
    opt = get_egor_optimizer(MDCantileveredBeam(), n_init=20)
    opt.minimize(n_infill=1)
    assert len(opt.pop) == 21
