import numpy as np
from sb_arch_opt.sampling import *
from sb_arch_opt.problems.md_mo import *
from sb_arch_opt.problems.problems_base import MixedDiscretizerProblemBase
from pymoo.problems.multi.zdt import ZDT1
from pymoo.core.evaluator import Evaluator


def test_md_base():
    problem = MixedDiscretizerProblemBase(ZDT1(n_var=4), n_opts=4, n_vars_int=2)
    assert problem.get_n_declared_discrete() == 4**2
    assert problem.get_imputation_ratio() == 1
    assert problem.get_correction_ratio() == 1
    problem.print_stats()

    pop = HierarchicalExhaustiveSampling(n_cont=3).do(problem, 0)
    assert len(pop) == (4**2)*(3**2)
    Evaluator().eval(problem, pop)


def run_test_no_hierarchy(problem, exh_n_cont=3):
    assert problem.get_imputation_ratio() == 1
    assert problem.get_correction_ratio() == 1
    assert np.all(~problem.is_conditionally_active)
    problem.print_stats()

    x_discrete, is_act_discrete = problem.all_discrete_x
    if x_discrete is not None:
        assert x_discrete.shape[0] == problem.get_n_valid_discrete()
        if x_discrete.shape[0] < 1000:
            assert np.all(~LargeDuplicateElimination.eliminate(x_discrete))

    pop = None
    if exh_n_cont != -1 and (HierarchicalExhaustiveSampling.get_n_sample_exhaustive(problem, n_cont=exh_n_cont) < 1e3
                             or x_discrete is not None):
        try:
            pop = HierarchicalExhaustiveSampling(n_cont=exh_n_cont).do(problem, 0)
        except MemoryError:
            pass
    if pop is None:
        pop = HierarchicalSampling().do(problem, 100)
    Evaluator().eval(problem, pop)
    problem.get_population_statistics(pop, show=True)
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
    run_test_no_hierarchy(MOZDT1(), exh_n_cont=1)


def test_md_mo_zdt1_small():
    run_test_no_hierarchy(MDZDT1Small(), exh_n_cont=1)


def test_md_mo_zdt1_mid():
    run_test_no_hierarchy(MDZDT1Mid(), exh_n_cont=1)


def test_md_mo_zdt1():
    run_test_no_hierarchy(MDZDT1(), exh_n_cont=1)


def test_discrete_mo_zdt1():
    run_test_no_hierarchy(DZDT1(), exh_n_cont=1)
