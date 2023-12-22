import pytest
from sb_arch_opt.problem import *
from sb_arch_opt.algo.egor_interface import *
from sb_arch_opt.problems.discrete import MDBranin
from sb_arch_opt.problems.constrained import ArchCantileveredBeam, MDCantileveredBeam
from sb_arch_opt.algo.egor_interface.algo import EgorArchOptInterface

check_dependency = lambda: pytest.mark.skipif(
    not HAS_EGOBOX, reason="Egor dependencies not installed"
)


@check_dependency()
def test_design_space(problem: ArchOptProblemBase):
    egor = EgorArchOptInterface(problem, "./", n_init=10)
    design_space = egor.design_space
    assert len(design_space) == problem.n_var


@check_dependency()
def test_simple():
    assert HAS_EGOBOX

    n_init = 30
    egor = get_egor_optimizer(MDBranin(), n_init=30, seed=42)
    egor.minimize(n_infill=2)

    pop = egor.pop
    assert len(pop) == n_init + 2


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
