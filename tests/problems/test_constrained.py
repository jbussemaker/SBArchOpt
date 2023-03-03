from sb_arch_opt.sampling import *
from sb_arch_opt.problems.constrained import *
from pymoo.core.evaluator import Evaluator


def _test_no_hierarchy(problem):
    assert problem.get_imputation_ratio() == 1
    problem.print_stats()

    if RepairedExhaustiveSampling.get_n_sample_exhaustive(problem, n_cont=3) < 1e3:
        pop = RepairedExhaustiveSampling(n_cont=3).do(problem, 0)
    else:
        pop = RepairedRandomSampling().do(problem, 100)
    Evaluator().eval(problem, pop)
    return pop


def test_welded_beam():
    _test_no_hierarchy(ArchWeldedBeam())


def test_md_welded_beam():
    _test_no_hierarchy(MDWeldedBeam())


def test_carside():
    _test_no_hierarchy(ArchCarside())


def test_md_carside():
    _test_no_hierarchy(MDCarside())


def test_osy():
    _test_no_hierarchy(ArchOSY())


def test_md_osy():
    _test_no_hierarchy(MDOSY())


def test_das_cmop():
    _test_no_hierarchy(MODASCMOP())


def test_md_das_cmop():
    _test_no_hierarchy(MDDASCMOP())
