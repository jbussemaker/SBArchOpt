import os
import pytest
import numpy as np
from sb_arch_opt.problems.assignment import *
from sb_arch_opt.tests.problems.test_hierarchical import run_test_hierarchy

def check_dependency():
    return pytest.mark.skipif(not HAS_ASSIGN_ENC, reason='assign_enc dependencies not installed')


@pytest.mark.skipif(int(os.getenv('RUN_SLOW_TESTS', 0)) != 1, reason='Set RUN_SLOW_TESTS=1 to run slow tests')
def test_slow_tests():
    assert HAS_ASSIGN_ENC


@check_dependency()
def test_assignment():
    assignment = Assignment()
    run_test_hierarchy(assignment, 1)
    run_test_hierarchy(AssignmentLarge(), 1, check_n_valid=False)

    AssignmentInj().print_stats()
    AssignmentInjLarge().print_stats()
    AssignmentRepeat().print_stats()
    AssignmentRepeatLarge().print_stats()

    x_all, is_act_all = assignment.all_discrete_x
    assert x_all is not None

    is_cond_act = assignment.is_conditionally_active
    assert np.all(is_cond_act == np.any(~is_act_all, axis=0))


@check_dependency()
def test_partitioning():
    Partitioning().print_stats()
    run_test_hierarchy(PartitioningCovering(), 1.71, corr_ratio=1.71)

    _ = PartitioningCovering().is_conditionally_active


@check_dependency()
def test_unordered():
    run_test_hierarchy(UnordNonReplComb(), 2.55, corr_ratio=2.55)
    UnordNonReplCombLarge().print_stats()
    UnorderedComb().print_stats()

    _ = UnorderedComb().is_conditionally_active


@check_dependency()
def test_assign_enc_gnc():
    for problem, n_valid, imp_ratio, corr_ratio in [
        (AssignmentGNCNoActType(), 327, 14.1, 3.01),
        (AssignmentGNCNoAct(), 29857, 39.5, np.nan),
        (AssignmentGNCNoType(), 85779, 82.5, np.nan),
        (AssignmentGNC(), 79091323, 367, np.nan),
    ]:
        run_test_hierarchy(problem, imp_ratio, check_n_valid=n_valid < 400, corr_ratio=corr_ratio)
        assert problem.get_n_valid_discrete() == n_valid

        x_all, _ = problem.all_discrete_x
        _ = problem.is_conditionally_active
