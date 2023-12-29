from sb_arch_opt.problems.continuous import *
from sb_arch_opt.tests.problems.test_md_mo import run_test_no_hierarchy


def test_himmelblau():
    run_test_no_hierarchy(Himmelblau())
    Himmelblau().get_discrete_rates(show=True)


def test_rosenbrock():
    run_test_no_hierarchy(Rosenbrock())


def test_griewank():
    run_test_no_hierarchy(Griewank())


def test_goldstein():
    run_test_no_hierarchy(Goldstein())


def test_branin():
    run_test_no_hierarchy(Branin())
