from sb_arch_opt.problems.hidden_constraints import *
from sb_arch_opt.tests.problems.test_discrete import run_test_no_hierarchy
from sb_arch_opt.tests.problems.test_hierarchical import run_test_hierarchy


def test_mueller_01():
    run_test_no_hierarchy(Mueller01())


def test_mueller_02():
    run_test_no_hierarchy(Mueller02())


def test_mueller_08():
    run_test_no_hierarchy(Mueller08())


def test_alimo():
    run_test_no_hierarchy(Alimo())


def test_hc_branin():
    run_test_no_hierarchy(HCBranin())


def test_hier_rosenbrock_hc():
    run_test_hierarchy(MOHierarchicalRosenbrockHC(), 1.5)


def test_hier_test_problem_hc():
    run_test_hierarchy(HCMOHierarchicalTestProblem(), 72)
