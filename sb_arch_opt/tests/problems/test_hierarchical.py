import pytest
from sb_arch_opt.sampling import *
from sb_arch_opt.problems.hierarchical import *
from pymoo.core.evaluator import Evaluator


def run_test_hierarchy(problem, imp_ratio, check_n_valid=True):
    if check_n_valid:
        pop = RepairedExhaustiveSampling(n_cont=1).do(problem, 0)
        assert len(pop) == problem.get_n_valid_discrete()

    assert problem.get_imputation_ratio() == pytest.approx(imp_ratio, rel=.02)
    problem.print_stats()

    if RepairedExhaustiveSampling.get_n_sample_exhaustive(problem, n_cont=3) < 1e3:
        pop = RepairedExhaustiveSampling(n_cont=3).do(problem, 0)
    else:
        pop = RepairedRandomSampling().do(problem, 100)
    Evaluator().eval(problem, pop)
    return pop


def test_hier_goldstein():
    run_test_hierarchy(HierarchicalGoldstein(), 2.25)


def test_mo_hier_goldstein():
    run_test_hierarchy(MOHierarchicalGoldstein(), 2.25)


def test_hier_rosenbrock():
    run_test_hierarchy(HierarchicalRosenbrock(), 1.5)


def test_mo_hier_rosenbrock():
    run_test_hierarchy(MOHierarchicalRosenbrock(), 1.5)


def test_hier_zaefferer():
    run_test_hierarchy(ZaeffererHierarchical.from_mode(ZaeffererProblemMode.A_OPT_INACT_IMP_PROF_UNI), 1)


def test_hier_test_problem():
    run_test_hierarchy(MOHierarchicalTestProblem(), 72)


def test_jenatton():
    run_test_hierarchy(Jenatton(), 2)
