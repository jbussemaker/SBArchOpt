from sb_arch_opt.problems.hidden_constraints import *
from sb_arch_opt.tests.problems.test_discrete import run_test_no_hierarchy
from sb_arch_opt.tests.problems.test_hierarchical import run_test_hierarchy


def test_mueller_01():
    run_test_no_hierarchy(Mueller01())


def test_mueller_02():
    run_test_no_hierarchy(Mueller02())
    run_test_no_hierarchy(MDMueller02())
    run_test_hierarchy(HierMueller02(), 5.4)


def test_mueller_08():
    run_test_no_hierarchy(Mueller08())
    run_test_no_hierarchy(MOMueller08())
    run_test_no_hierarchy(MDMueller08())
    run_test_no_hierarchy(MDMOMueller08())
    run_test_hierarchy(HierMueller08(), 5.4)
    run_test_hierarchy(MOHierMueller08(), 5.4)


def test_alimo():
    run_test_no_hierarchy(Alimo())


def test_hc_branin():
    run_test_no_hierarchy(HCBranin())


def test_hc_sphere():
    run_test_no_hierarchy(HCSphere())


def test_hier_rosenbrock_hc():
    run_test_hierarchy(MOHierarchicalRosenbrockHC(), 1.5)


def test_hier_test_problem_hc():
    run_test_hierarchy(HCMOHierarchicalTestProblem(), 72)


def test_cantilevered_beam():
    run_test_no_hierarchy(CantileveredBeamHC())
    run_test_no_hierarchy(MDCantileveredBeamHC())


def test_carside():
    run_test_no_hierarchy(CarsideHC())
    run_test_no_hierarchy(CarsideHCLess())
    run_test_no_hierarchy(MDCarsideHC())
