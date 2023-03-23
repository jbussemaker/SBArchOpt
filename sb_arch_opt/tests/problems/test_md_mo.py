from sb_arch_opt.sampling import *
from sb_arch_opt.problems.md_mo import *
from sb_arch_opt.problems.problems_base import MixedDiscretizerProblemBase
from pymoo.problems.multi.zdt import ZDT1
from pymoo.core.evaluator import Evaluator


def test_md_base():
    problem = MixedDiscretizerProblemBase(ZDT1(n_var=4), n_opts=4, n_vars_int=2)
    assert problem.get_n_declared_discrete() == 4**2
    assert problem.get_imputation_ratio() == 1
    problem.print_stats()

    pop = HierarchicalExhaustiveSampling(n_cont=3).do(problem, 0)
    assert len(pop) == (4**2)*(3**2)
    Evaluator().eval(problem, pop)


def run_test_no_hierarchy(problem):
    assert problem.get_imputation_ratio() == 1
    problem.print_stats()

    if HierarchicalExhaustiveSampling.get_n_sample_exhaustive(problem, n_cont=3) < 1e3:
        pop = HierarchicalExhaustiveSampling(n_cont=3).do(problem, 0)
    else:
        pop = HierarchicalRandomSampling().do(problem, 100)
    Evaluator().eval(problem, pop)
    return pop


def test_mo_himmelblau():
    pop = run_test_no_hierarchy(MOHimmelblau())
    assert len(pop) == 3**2


def test_md_mo_himmelblau():
    pop = run_test_no_hierarchy(MDMOHimmelblau())
    assert len(pop) == 10*3


def test_discrete_mo_himmelblau():
    pop = run_test_no_hierarchy(DMOHimmelblau())
    assert len(pop) == 10*10


def test_mo_goldstein():
    run_test_no_hierarchy(MOGoldstein())


def test_md_mo_goldstein():
    run_test_no_hierarchy(MDMOGoldstein())


def test_discrete_mo_goldstein():
    run_test_no_hierarchy(DMOGoldstein())


def test_mo_rosenbrock():
    run_test_no_hierarchy(MORosenbrock())


def test_md_mo_rosenbrock():
    run_test_no_hierarchy(MDMORosenbrock())


def test_mo_zdt1():
    run_test_no_hierarchy(MOZDT1())


def test_md_mo_zdt1_small():
    run_test_no_hierarchy(MDZDT1Small())


def test_md_mo_zdt1_mid():
    run_test_no_hierarchy(MDZDT1Mid())


def test_md_mo_zdt1():
    run_test_no_hierarchy(MDZDT1())


def test_discrete_mo_zdt1():
    run_test_no_hierarchy(DZDT1())
