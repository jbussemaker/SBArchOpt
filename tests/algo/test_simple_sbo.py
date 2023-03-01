import pytest
from sb_arch_opt.problem import *
from sb_arch_opt.algo.simple_sbo import *
from pymoo.optimize import minimize

check_dependency = lambda: pytest.mark.skipif(not HAS_SIMPLE_SBO, reason='Simple SBO dependencies not installed')


@check_dependency()
def test_simple_sbo_rbf(problem: ArchOptProblemBase):
    assert HAS_SIMPLE_SBO

    sbo = get_simple_sbo_rbf(init_size=10)
    result = minimize(problem, sbo, termination=('n_eval', 12))
    assert len(result.pop) == 12


@check_dependency()
def test_simple_sbo_krg(problem: ArchOptProblemBase):
    assert HAS_SIMPLE_SBO

    sbo = get_simple_sbo_krg(init_size=10)
    result = minimize(problem, sbo, termination=('n_eval', 12))
    assert len(result.pop) == 12


@check_dependency()
def test_simple_sbo_krg_y(problem: ArchOptProblemBase):
    assert HAS_SIMPLE_SBO

    sbo = get_simple_sbo_krg(init_size=10, use_mvpf=False)
    result = minimize(problem, sbo, termination=('n_eval', 12))
    assert len(result.pop) == 12


@check_dependency()
def test_simple_sbo_krg_ei(problem: ArchOptProblemBase):
    assert HAS_SIMPLE_SBO

    sbo = get_simple_sbo_krg(init_size=10, use_ei=True)
    result = minimize(problem, sbo, termination=('n_eval', 12))
    assert len(result.pop) == 12
