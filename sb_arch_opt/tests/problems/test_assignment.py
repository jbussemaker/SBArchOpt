import pytest
from sb_arch_opt.problems.assignment import *
from sb_arch_opt.tests.problems.test_hierarchical import run_test_hierarchy

check_dependency = lambda: pytest.mark.skipif(not HAS_ASSIGN_ENC, reason='assign_enc dependencies not installed')


@check_dependency()
def test_assignment():
    run_test_hierarchy(Assignment(), 1)
    run_test_hierarchy(AssignmentLarge(), 1, check_n_valid=False)

    AssignmentInj().print_stats()
    AssignmentInjLarge().print_stats()
    AssignmentRepeat().print_stats()
    AssignmentRepeatLarge().print_stats()


@check_dependency()
def test_partitioning():
    Partitioning().print_stats()
    run_test_hierarchy(PartitioningCovering(), 1.71)


@check_dependency()
def test_unordered():
    run_test_hierarchy(UnordNonReplComb(), 5.09)
    UnordNonReplCombLarge().print_stats()
    UnorderedComb().print_stats()


@check_dependency()
def test_assign_enc_gnc():
    for problem, n_valid, imp_ratio in [
        (AssignmentGNCNoActType(), 327, 14.1),
        (AssignmentGNCNoAct(), 29857, 39.5),
        (AssignmentGNCNoType(), 85779, 82.5),
        (AssignmentGNC(), 79091323, 367),
    ]:
        run_test_hierarchy(problem, imp_ratio, check_n_valid=False)
        assert problem.get_n_valid_discrete() == n_valid
