import os
import pytest
import numpy as np
from sb_arch_opt.sampling import *
from sb_arch_opt.problems.continuous import *
from sb_arch_opt.problems.hierarchical import *
from pymoo.core.evaluator import Evaluator


def run_test_hierarchy(problem, imp_ratio, check_n_valid=True, validate_exhaustive=False, exh_n_cont=3,
                       corr_ratio=None):
    x_discrete, is_act_discrete = problem.all_discrete_x
    if check_n_valid or x_discrete is not None:
        if x_discrete is not None:
            assert np.all(~LargeDuplicateElimination.eliminate(x_discrete))
            assert x_discrete.shape[0] == problem.get_n_valid_discrete()

            if validate_exhaustive:
                x_trail_repair, _ = HierarchicalExhaustiveSampling.get_all_x_discrete_by_trial_and_repair(problem)
                assert {tuple(ix) for ix in x_trail_repair} == {tuple(ix) for ix in x_discrete}

            is_cond_act = problem.is_conditionally_active
            assert np.all(is_cond_act == np.any(~is_act_discrete, axis=0))

        pop = HierarchicalExhaustiveSampling(n_cont=1).do(problem, 0)
        assert len(pop) == problem.get_n_valid_discrete()

    x_discrete, is_act_discrete = problem.all_discrete_x
    assert HierarchicalExhaustiveSampling.has_cheap_all_x_discrete(problem) == (x_discrete is not None)

    assert problem.get_discrete_imputation_ratio() == pytest.approx(imp_ratio, rel=.02)
    if corr_ratio is None:
        corr_ratio = 1  # Assume no correction
    if np.isnan(corr_ratio):
        assert np.isnan(problem.get_discrete_correction_ratio())
    else:
        assert problem.get_discrete_correction_ratio() == pytest.approx(corr_ratio, rel=.02)
        assert problem.get_discrete_correction_ratio() <= problem.get_discrete_imputation_ratio()
    problem.print_stats()

    pop = None
    if exh_n_cont != -1 and HierarchicalExhaustiveSampling.get_n_sample_exhaustive(problem, n_cont=exh_n_cont) < 1e3:
        try:
            pop = HierarchicalExhaustiveSampling(n_cont=exh_n_cont).do(problem, 0)
        except MemoryError:
            pass
    if pop is None:
        pop = HierarchicalSampling().do(problem, 100)
    Evaluator().eval(problem, pop)
    problem.get_population_statistics(pop, show=True)
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

    problem = ZaeffererHierarchical.from_mode(ZaeffererProblemMode.A_OPT_INACT_IMP_PROF_UNI)
    x, is_act = problem.correct_x(np.array([[0, .75]]))
    assert np.all(x == [[0, .5]])
    assert np.all(is_act == [[True, False]])

    assert problem.get_imputation_ratio() == 2/(2-problem.c)
    assert problem.get_correction_ratio() == 1


def test_hier_test_problem():
    run_test_hierarchy(MOHierarchicalTestProblem(), 72)


def test_jenatton():
    problem = Jenatton()
    run_test_hierarchy(problem, 2)
    run_test_hierarchy(problem, 2)

    run_test_hierarchy(Jenatton(explicit=False), 2)


def test_hier_branin():
    run_test_hierarchy(HierBranin(), 3.24, validate_exhaustive=True, corr_ratio=1.05)


@pytest.mark.skipif(int(os.getenv('RUN_SLOW_TESTS', 0)) != 1, reason='Set RUN_SLOW_TESTS=1 to run slow tests')
def test_hier_zdt1():
    run_test_hierarchy(HierZDT1Small(), 1.8, validate_exhaustive=True, corr_ratio=1.125)
    run_test_hierarchy(HierZDT1(), 4.86, validate_exhaustive=True, corr_ratio=1.07)
    run_test_hierarchy(HierZDT1Large(), 8.19, corr_ratio=1.12)
    run_test_hierarchy(HierDiscreteZDT1(), 4.1, corr_ratio=1.12)


def test_hier_cantilevered_beam():
    run_test_hierarchy(HierCantileveredBeam(), 5.4, corr_ratio=1.04)


def test_hier_carside():
    run_test_hierarchy(HierCarside(), 6.48, corr_ratio=1.05)


def test_hier_nn():
    run_test_hierarchy(NeuralNetwork(), 2.51)


def test_tunable_hierarchical_meta_problem():
    prob1 = TunableHierarchicalMetaProblem(lambda n: Branin(), imp_ratio=10, n_subproblem=4, diversity_range=.5)
    prob1.print_stats()

    prob2 = TunableHierarchicalMetaProblem(lambda n: Branin(), imp_ratio=10, n_subproblem=5, diversity_range=.25)
    prob2.print_stats()
    assert prob1._pf_cache_path() != prob2._pf_cache_path()
