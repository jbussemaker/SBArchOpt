import pytest
from sb_arch_opt.problem import *
from sb_arch_opt.algo.smarty_interface import *
from sb_arch_opt.problems.continuous import Branin
from sb_arch_opt.problems.constrained import ArchCantileveredBeam
from sb_arch_opt.algo.smarty_interface.algo import SMARTyArchOptInterface

def check_dependency():
    return pytest.mark.skipif(not HAS_SMARTY, reason='SMARTy dependencies not installed')


@check_dependency()
def test_opt_prob(problem: ArchOptProblemBase):
    smarty = SMARTyArchOptInterface(problem, n_init=10, n_infill=1)
    opt_prob = smarty.opt_prob
    assert opt_prob.bounds.shape[0] == problem.n_var


@check_dependency()
def test_simple():
    assert HAS_SMARTY

    smarty = get_smarty_optimizer(Branin(), n_init=10, n_infill=2)
    smarty.optimize()

    pop = smarty.pop
    assert len(pop) == 12


@check_dependency()
def test_simple_mo(problem: ArchOptProblemBase):
    assert HAS_SMARTY

    smarty = get_smarty_optimizer(problem, n_init=10, n_infill=2)
    smarty.optimize()

    pop = smarty.pop
    assert len(pop) == 12


@check_dependency()
def test_constrained():
    smarty = get_smarty_optimizer(ArchCantileveredBeam(), n_init=10, n_infill=1)
    smarty.optimize()
    assert len(smarty.pop) == 11
