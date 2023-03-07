from sb_arch_opt.problems.discrete import *
from sb_arch_opt.tests.problems.test_md_mo import run_test_no_hierarchy


def test_md_branin():
    run_test_no_hierarchy(MDBranin())


def test_aug_md_branin():
    run_test_no_hierarchy(AugmentedMDBranin())


def test_md_goldstein():
    run_test_no_hierarchy(MDGoldstein())


def test_munoz_zuniga():
    run_test_no_hierarchy(MunozZunigaToy())


def test_halstrup4():
    run_test_no_hierarchy(Halstrup04())
