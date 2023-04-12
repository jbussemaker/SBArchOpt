from sb_arch_opt.problems.constrained import *
from sb_arch_opt.tests.problems.test_md_mo import run_test_no_hierarchy


def test_cantilevered_beam():
    run_test_no_hierarchy(ArchCantileveredBeam())


def test_welded_beam():
    run_test_no_hierarchy(ArchWeldedBeam())


def test_md_welded_beam():
    run_test_no_hierarchy(MDWeldedBeam())


def test_carside():
    run_test_no_hierarchy(ArchCarside())


def test_md_carside():
    run_test_no_hierarchy(MDCarside())


def test_osy():
    run_test_no_hierarchy(ArchOSY())


def test_md_osy():
    run_test_no_hierarchy(MDOSY())


def test_das_cmop():
    run_test_no_hierarchy(MODASCMOP(), exh_n_cont=1)


def test_md_das_cmop():
    run_test_no_hierarchy(MDDASCMOP(), exh_n_cont=-1)
