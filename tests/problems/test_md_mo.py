from sb_arch_opt.sampling import *
from sb_arch_opt.problems.md_mo import *
from sb_arch_opt.problems.md_base import MixedDiscretizerProblemBase
from pymoo.problems.multi.zdt import ZDT1
from pymoo.core.evaluator import Evaluator


def test_md_base():
    problem = MixedDiscretizerProblemBase(ZDT1(n_var=4), n_opts=4, n_vars_int=2)
    assert problem.get_n_declared_discrete() == 4**2
    assert problem.get_imputation_ratio() == 1
    problem.print_stats()

    pop = RepairedExhaustiveSampling(n_cont=3).do(problem, 0)
    assert len(pop) == (4**2)*(3**2)
    Evaluator().eval(problem, pop)


def _test_no_hierarchy(problem):
    assert problem.get_imputation_ratio() == 1
    problem.print_stats()

    if RepairedExhaustiveSampling.get_n_sample_exhaustive(problem, n_cont=3) < 1e3:
        pop = RepairedExhaustiveSampling(n_cont=3).do(problem, 0)
    else:
        pop = RepairedRandomSampling().do(problem, 100)
    Evaluator().eval(problem, pop)
    return pop


def test_mo_himmelblau():
    pop = _test_no_hierarchy(MOHimmelblau())
    assert len(pop) == 3**2


def test_md_mo_himmelblau():
    pop = _test_no_hierarchy(MDMOHimmelblau())
    assert len(pop) == 10*3


def test_discrete_mo_himmelblau():
    pop = _test_no_hierarchy(DMOHimmelblau())
    assert len(pop) == 10*10


def test_mo_goldstein():
    _test_no_hierarchy(MOGoldstein())


def test_md_mo_goldstein():
    _test_no_hierarchy(MDMOGoldstein())


def test_discrete_mo_goldstein():
    _test_no_hierarchy(DMOGoldstein())


def test_mo_zdt1():
    _test_no_hierarchy(MOZDT1())


def test_md_mo_zdt1():
    _test_no_hierarchy(MDZDT1())


def test_discrete_mo_zdt1():
    _test_no_hierarchy(DZDT1())
