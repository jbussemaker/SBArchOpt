from sb_arch_opt.problems.hidden_constraints import *
from tests.problems.test_hierarchical import run_test_hierarchy


def test_hier_rosenbrock_hc():
    run_test_hierarchy(MOHierarchicalRosenbrockHC(), 1.5)


def test_hier_test_problem_hc():
    run_test_hierarchy(HCMOHierarchicalTestProblem(), 72)
